//! Unit Task Aggregation
//!
//! Clusters related actions into meaningful Unit Tasks.

use crate::analysis::intent_binding::Action;
use crate::capture::types::EnrichedEvent;
use std::collections::VecDeque;

/// A unit task representing a semantic action
#[derive(Debug, Clone)]
pub struct UnitTask {
    /// Unique identifier
    pub id: uuid::Uuid,
    /// Human-readable name
    pub name: String,
    /// Description of the task
    pub description: String,
    /// Primary action
    pub primary_action: Action,
    /// All events in this task
    pub events: Vec<EnrichedEvent>,
    /// Start timestamp
    pub start_time: u64,
    /// End timestamp
    pub end_time: u64,
    /// Nesting level (0 = root, 1+ = nested)
    pub nesting_level: usize,
    /// Parent task ID (if nested)
    pub parent_id: Option<uuid::Uuid>,
}

/// Configuration for action clustering
#[derive(Debug, Clone)]
pub struct ClusteringConfig {
    /// Maximum time gap within a cluster (ms)
    pub max_gap_ms: u64,
    /// Minimum events for a cluster
    pub min_events: usize,
    /// Whether to merge transient interactions
    pub merge_transient: bool,
    /// Minimum movement to consider (pixels)
    pub min_movement_px: f64,
}

impl Default for ClusteringConfig {
    fn default() -> Self {
        Self {
            max_gap_ms: 1000,     // 1 second
            min_events: 2,
            merge_transient: true,
            min_movement_px: 5.0,
        }
    }
}

impl ClusteringConfig {
    /// Validate configuration values and return errors for invalid settings.
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if self.max_gap_ms == 0 {
            errors.push("max_gap_ms must be > 0".to_string());
        }
        if self.min_events == 0 {
            errors.push("min_events must be > 0".to_string());
        }
        if self.min_movement_px < 0.0 {
            errors.push(format!(
                "min_movement_px must be >= 0.0, got {}",
                self.min_movement_px
            ));
        }
        errors
    }
}

/// Action clusterer
pub struct ActionClusterer {
    /// Configuration
    pub config: ClusteringConfig,
    /// Pending events for incremental processing
    pending: VecDeque<EnrichedEvent>,
}

impl ActionClusterer {
    /// Create with default config
    pub fn new() -> Self {
        Self {
            config: ClusteringConfig::default(),
            pending: VecDeque::new(),
        }
    }

    /// Create with custom config
    pub fn with_config(config: ClusteringConfig) -> Self {
        Self {
            config,
            pending: VecDeque::new(),
        }
    }

    /// Add an event for incremental processing
    /// Returns completed tasks if any clusters are ready to emit
    pub fn add_event(&mut self, event: EnrichedEvent) -> Vec<UnitTask> {
        let mut completed_tasks = Vec::new();
        
        // Check if we should flush existing pending events
        if let Some(last) = self.pending.back() {
            if self.should_split_streaming(last, &event) {
                // Flush pending events as a task
                let events: Vec<EnrichedEvent> = self.pending.drain(..).collect();
                if let Some(task) = self.create_task(&events, None) {
                    completed_tasks.push(task);
                }
            }
        }
        
        self.pending.push_back(event);
        completed_tasks
    }

    /// Flush all pending events and return any remaining tasks
    pub fn flush(&mut self) -> Vec<UnitTask> {
        let mut tasks = Vec::new();
        if !self.pending.is_empty() {
            let events: Vec<EnrichedEvent> = self.pending.drain(..).collect();
            if let Some(task) = self.create_task(&events, None) {
                tasks.push(task);
            }
        }
        tasks
    }

    /// Get the number of pending events
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Clear all pending events without creating tasks
    pub fn clear_pending(&mut self) {
        self.pending.clear();
    }

    /// Check if should split for streaming processing (simplified version)
    fn should_split_streaming(&self, last: &EnrichedEvent, next: &EnrichedEvent) -> bool {
        // Time gap check
        let time_gap = if next.raw.timestamp.ticks() > last.raw.timestamp.ticks() {
            crate::time::timebase::MachTimebase::elapsed_millis(
                last.raw.timestamp.ticks(),
                next.raw.timestamp.ticks(),
            )
        } else {
            0
        };

        time_gap > self.config.max_gap_ms
    }

    /// Cluster events into unit tasks
    pub fn cluster(&self, events: &[EnrichedEvent], actions: &[Action]) -> Vec<UnitTask> {
        let mut tasks = Vec::new();
        let mut current_events: Vec<EnrichedEvent> = Vec::new();
        let mut current_action: Option<&Action> = None;
        let mut action_idx = 0;

        for event in events {
            // Check if we should start a new cluster
            let should_split = if current_events.is_empty() {
                false
            } else {
                self.should_split(&current_events, event)
            };

            if should_split && !current_events.is_empty() {
                // Create task from current cluster
                if let Some(task) = self.create_task(&current_events, current_action) {
                    tasks.push(task);
                }
                current_events.clear();
                action_idx = (action_idx + 1).min(actions.len().saturating_sub(1));
            }

            current_events.push(event.clone());

            // Update current action if this is a click event
            if event.raw.event_type.is_click() && action_idx < actions.len() {
                current_action = Some(&actions[action_idx]);
            }
        }

        // Don't forget the last cluster
        if !current_events.is_empty() {
            if let Some(task) = self.create_task(&current_events, current_action) {
                tasks.push(task);
            }
        }

        // Post-process: merge transient interactions
        if self.config.merge_transient {
            tasks = self.merge_transient_tasks(tasks);
        }

        // Filter out micro-operations
        tasks = self.filter_micro_operations(tasks);

        tasks
    }

    /// Check if we should split at this event
    fn should_split(&self, current: &[EnrichedEvent], next: &EnrichedEvent) -> bool {
        let last = match current.last() {
            Some(l) => l,
            None => return false,
        };

        // Time gap check
        let time_gap = if next.raw.timestamp.ticks() > last.raw.timestamp.ticks() {
            crate::time::timebase::MachTimebase::elapsed_millis(
                last.raw.timestamp.ticks(),
                next.raw.timestamp.ticks(),
            )
        } else {
            0
        };

        if time_gap > self.config.max_gap_ms {
            return true;
        }

        // Context change (window switch)
        if let (Some(last_sem), Some(next_sem)) = (&last.semantic, &next.semantic) {
            if last_sem.window_id != next_sem.window_id {
                return true;
            }
        }

        // Event type transitions that suggest new action
        let last_type = &last.raw.event_type;
        let next_type = &next.raw.event_type;

        // Click followed by keyboard = new action (filling a field)
        if last_type.is_click() && next_type.is_keyboard() {
            return false; // Keep together - likely form filling
        }

        // Keyboard followed by click = new action
        if last_type.is_keyboard() && next_type.is_click() {
            // Check if it's been a while
            return time_gap > 500;
        }

        false
    }

    /// Create a unit task from a cluster of events
    fn create_task(&self, events: &[EnrichedEvent], action: Option<&Action>) -> Option<UnitTask> {
        if events.len() < self.config.min_events {
            return None;
        }

        let first = events.first()?;
        let last = events.last()?;

        let (name, description, primary_action) = if let Some(act) = action {
            (
                self.action_to_name(act),
                act.description(),
                act.clone(),
            )
        } else {
            let name = self.infer_task_name(events);
            let desc = format!("{} events", events.len());
            let action = Action::Unknown {
                description: desc.clone(),
                confidence: 0.5,
            };
            (name, desc, action)
        };

        Some(UnitTask {
            id: uuid::Uuid::new_v4(),
            name,
            description,
            primary_action,
            events: events.to_vec(),
            start_time: first.raw.timestamp.ticks(),
            end_time: last.raw.timestamp.ticks(),
            nesting_level: 0,
            parent_id: None,
        })
    }

    /// Convert action to task name
    fn action_to_name(&self, action: &Action) -> String {
        match action {
            Action::Click { element_name, .. } => format!("Click {}", element_name),
            Action::Fill { field_name, .. } => format!("Fill {}", field_name),
            Action::Select { option, .. } => format!("Select {}", option),
            Action::Type { .. } => "Type text".to_string(),
            Action::Scroll { direction, .. } => format!("Scroll {:?}", direction),
            Action::Shortcut { keys, .. } => format!("Press {}", keys.join("+")),
            _ => "Action".to_string(),
        }
    }

    /// Infer task name from events using semantic context
    fn infer_task_name(&self, events: &[EnrichedEvent]) -> String {
        // Try to find a named click target
        for event in events {
            if event.raw.event_type.is_click() {
                if let Some(sem) = &event.semantic {
                    if let Some(title) = &sem.title {
                        if !title.is_empty() {
                            return format!("Click {}", title);
                        }
                    }
                    // Fallback: use role + identifier
                    if let Some(role) = &sem.ax_role {
                        if let Some(id) = &sem.identifier {
                            return format!("Click {} ({})", role, id);
                        }
                        // Use role + parent context
                        if let Some(parent) = &sem.parent_title {
                            if !parent.is_empty() {
                                return format!("Click {} in {}", role, parent);
                            }
                        }
                    }
                }
            }
        }

        // Collect typed text for keyboard-heavy tasks
        let mut typed_text = String::new();
        for event in events {
            if event.raw.event_type.is_keyboard() {
                if let Some(ch) = event.raw.character {
                    if !ch.is_control() {
                        typed_text.push(ch);
                    }
                }
            }
        }

        // Count event types
        let clicks = events.iter().filter(|e| e.raw.event_type.is_click()).count();
        let keys = events.iter().filter(|e| e.raw.event_type.is_keyboard()).count();

        // Try to get window context for more descriptive names
        let window_context = events.iter()
            .find_map(|e| e.semantic.as_ref()?.window_title.as_ref())
            .map(|t| t.as_str())
            .unwrap_or("");

        let app_name = events.iter()
            .find_map(|e| e.semantic.as_ref()?.app_name.as_ref())
            .map(|t| t.as_str())
            .unwrap_or("");

        if keys > clicks {
            if !typed_text.is_empty() {
                let preview = if typed_text.len() > 30 {
                    format!("{}...", &typed_text[..27])
                } else {
                    typed_text
                };
                if !app_name.is_empty() {
                    return format!("Type \"{}\" in {}", preview, app_name);
                }
                return format!("Type \"{}\"", preview);
            }
            if !app_name.is_empty() {
                return format!("Type in {}", app_name);
            }
            "Type text".to_string()
        } else if clicks > 0 {
            if !window_context.is_empty() {
                format!("Click in {}", window_context)
            } else if !app_name.is_empty() {
                format!("Click in {}", app_name)
            } else {
                "Click".to_string()
            }
        } else {
            "Action".to_string()
        }
    }

    /// Merge transient UI interactions
    fn merge_transient_tasks(&self, tasks: Vec<UnitTask>) -> Vec<UnitTask> {
        // Patterns to merge:
        // - Click dropdown + Click option = Select
        // - Click field + Type = Fill
        // - Multiple small mouse movements = single move

        let mut result = Vec::new();
        let mut i = 0;

        while i < tasks.len() {
            let current = &tasks[i];

            // Check for dropdown pattern
            if i + 1 < tasks.len() {
                let next = &tasks[i + 1];

                // Dropdown pattern: click on combo/popup followed by click on menu item
                if self.is_dropdown_pattern(current, next) {
                    let merged = self.merge_dropdown(current, next);
                    result.push(merged);
                    i += 2;
                    continue;
                }
            }

            result.push(current.clone());
            i += 1;
        }

        result
    }

    /// Check if two tasks form a dropdown selection pattern
    fn is_dropdown_pattern(&self, t1: &UnitTask, t2: &UnitTask) -> bool {
        // First task should be a click
        if !matches!(t1.primary_action, Action::Click { .. }) {
            return false;
        }

        // Check semantic info
        for event in &t1.events {
            if let Some(sem) = &event.semantic {
                if let Some(role) = &sem.ax_role {
                    if role.contains("ComboBox") || role.contains("PopUp") || role.contains("Menu") {
                        // Second task should also be a click (on menu item)
                        return matches!(t2.primary_action, Action::Click { .. } | Action::Select { .. });
                    }
                }
            }
        }

        false
    }

    /// Merge dropdown tasks
    fn merge_dropdown(&self, t1: &UnitTask, t2: &UnitTask) -> UnitTask {
        let menu_name = match &t1.primary_action {
            Action::Click { element_name, .. } => element_name.clone(),
            _ => "menu".to_string(),
        };

        let option = match &t2.primary_action {
            Action::Click { element_name, .. } => element_name.clone(),
            Action::Select { option, .. } => option.clone(),
            _ => "option".to_string(),
        };

        let mut events = t1.events.clone();
        events.extend(t2.events.clone());

        UnitTask {
            id: uuid::Uuid::new_v4(),
            name: format!("Select {} from {}", option, menu_name),
            description: format!("Select '{}' from '{}'", option, menu_name),
            primary_action: Action::Select {
                menu_name,
                option,
                confidence: 0.9,
            },
            events,
            start_time: t1.start_time,
            end_time: t2.end_time,
            nesting_level: t1.nesting_level,
            parent_id: t1.parent_id,
        }
    }

    /// Filter out micro-operations
    fn filter_micro_operations(&self, tasks: Vec<UnitTask>) -> Vec<UnitTask> {
        tasks
            .into_iter()
            .filter(|task| {
                // Keep tasks with significant actions
                if task.events.len() >= self.config.min_events {
                    return true;
                }

                // Keep click tasks
                if matches!(task.primary_action, Action::Click { .. } | Action::Select { .. }) {
                    return true;
                }

                // Keep typing tasks
                if matches!(task.primary_action, Action::Type { .. }) {
                    return true;
                }

                false
            })
            .collect()
    }
}

impl Default for ActionClusterer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capture::types::{RawEvent, EventType, CursorState, ModifierFlags};
    use crate::time::timebase::Timestamp;

    fn make_enriched_event(timestamp_ticks: u64, event_type: EventType, x: f64, y: f64) -> EnrichedEvent {
        let raw = RawEvent::mouse(
            Timestamp::from_ticks(timestamp_ticks),
            event_type,
            x,
            y,
            CursorState::Arrow,
            ModifierFlags::default(),
            1,
        );
        EnrichedEvent::new(raw, 0)
    }

    fn make_click_action(element: &str) -> Action {
        Action::Click {
            element_name: element.to_string(),
            element_role: "AXButton".to_string(),
            confidence: 0.9,
        }
    }

    #[test]
    fn test_clustering_config() {
        let config = ClusteringConfig::default();
        assert!(config.max_gap_ms > 0);
        assert!(config.min_events > 0);
    }

    #[test]
    fn test_action_clusterer_creation() {
        let clusterer = ActionClusterer::new();
        assert!(clusterer.config.merge_transient);
    }

    #[test]
    fn test_action_to_name() {
        let clusterer = ActionClusterer::new();

        let click = Action::Click {
            element_name: "Submit".to_string(),
            element_role: "AXButton".to_string(),
            confidence: 0.9,
        };

        let name = clusterer.action_to_name(&click);
        assert_eq!(name, "Click Submit");
    }

    #[test]
    fn test_unit_task_creation() {
        let task = UnitTask {
            id: uuid::Uuid::new_v4(),
            name: "Click Submit".to_string(),
            description: "Click 'Submit'".to_string(),
            primary_action: make_click_action("Submit"),
            events: vec![],
            start_time: 1000,
            end_time: 2000,
            nesting_level: 0,
            parent_id: None,
        };

        assert_eq!(task.name, "Click Submit");
        assert_eq!(task.nesting_level, 0);
        assert!(task.parent_id.is_none());
        assert_eq!(task.start_time, 1000);
        assert_eq!(task.end_time, 2000);
    }

    #[test]
    fn test_nested_unit_task() {
        let parent_id = uuid::Uuid::new_v4();
        let task = UnitTask {
            id: uuid::Uuid::new_v4(),
            name: "Fill Field".to_string(),
            description: "Fill 'Email' with 'test@example.com'".to_string(),
            primary_action: Action::Fill {
                field_name: "Email".to_string(),
                value: "test@example.com".to_string(),
                confidence: 0.9,
            },
            events: vec![],
            start_time: 1000,
            end_time: 2000,
            nesting_level: 1,
            parent_id: Some(parent_id),
        };

        assert_eq!(task.nesting_level, 1);
        assert_eq!(task.parent_id, Some(parent_id));
    }

    #[test]
    fn test_clustering_config_custom() {
        let config = ClusteringConfig {
            max_gap_ms: 2000,
            min_events: 3,
            merge_transient: false,
            min_movement_px: 10.0,
        };

        assert_eq!(config.max_gap_ms, 2000);
        assert_eq!(config.min_events, 3);
        assert!(!config.merge_transient);
        assert_eq!(config.min_movement_px, 10.0);
    }

    #[test]
    fn test_clusterer_with_custom_config() {
        let config = ClusteringConfig {
            max_gap_ms: 500,
            min_events: 1,
            merge_transient: false,
            min_movement_px: 1.0,
        };

        let clusterer = ActionClusterer::with_config(config);
        assert_eq!(clusterer.config.max_gap_ms, 500);
        assert_eq!(clusterer.config.min_events, 1);
        assert!(!clusterer.config.merge_transient);
    }

    #[test]
    fn test_cluster_empty_events() {
        let clusterer = ActionClusterer::new();
        let events: Vec<EnrichedEvent> = vec![];
        let actions: Vec<Action> = vec![];

        let tasks = clusterer.cluster(&events, &actions);
        assert!(tasks.is_empty());
    }

    #[test]
    fn test_cluster_single_event_below_minimum() {
        let clusterer = ActionClusterer::new();
        let events = vec![make_enriched_event(1000, EventType::LeftMouseDown, 100.0, 100.0)];
        let actions: Vec<Action> = vec![];

        let tasks = clusterer.cluster(&events, &actions);
        // Default min_events is 2, so single event should not create a task
        assert!(tasks.is_empty());
    }

    #[test]
    fn test_cluster_multiple_events_creates_task() {
        use crate::time::timebase::MachTimebase;
        MachTimebase::init();

        let clusterer = ActionClusterer::with_config(ClusteringConfig {
            max_gap_ms: 1000,
            min_events: 2,
            merge_transient: false,
            min_movement_px: 5.0,
        });

        let events = vec![
            make_enriched_event(1000, EventType::MouseMoved, 100.0, 100.0),
            make_enriched_event(1100, EventType::LeftMouseDown, 150.0, 150.0),
        ];
        let actions = vec![make_click_action("Button")];

        let tasks = clusterer.cluster(&events, &actions);
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].events.len(), 2);
    }

    #[test]
    fn test_cluster_time_gap_splits_tasks() {
        use crate::time::timebase::MachTimebase;
        MachTimebase::init();

        let clusterer = ActionClusterer::with_config(ClusteringConfig {
            max_gap_ms: 500,
            min_events: 2,
            merge_transient: false,
            min_movement_px: 5.0,
        });

        // First cluster
        let e1 = make_enriched_event(1000, EventType::MouseMoved, 100.0, 100.0);
        let e2 = make_enriched_event(1200, EventType::LeftMouseDown, 150.0, 150.0);
        // Large time gap (1000ms > 500ms threshold)
        let e3 = make_enriched_event(2500, EventType::MouseMoved, 200.0, 200.0);
        let e4 = make_enriched_event(2700, EventType::LeftMouseDown, 250.0, 250.0);

        let events = vec![e1, e2, e3, e4];
        let actions = vec![make_click_action("Button1"), make_click_action("Button2")];

        let tasks = clusterer.cluster(&events, &actions);
        assert!(!tasks.is_empty());
    }

    #[test]
    fn test_infer_task_name_from_clicks() {
        let clusterer = ActionClusterer::new();

        let events = vec![
            make_enriched_event(1000, EventType::MouseMoved, 100.0, 100.0),
            make_enriched_event(1100, EventType::LeftMouseDown, 150.0, 150.0),
            make_enriched_event(1150, EventType::LeftMouseUp, 150.0, 150.0),
        ];

        let name = clusterer.infer_task_name(&events);
        assert_eq!(name, "Click");
    }

    #[test]
    fn test_infer_task_name_from_keyboard() {
        let clusterer = ActionClusterer::new();

        let raw1 = RawEvent::keyboard(
            Timestamp::from_ticks(1000),
            EventType::KeyDown,
            0,
            Some('h'),
            ModifierFlags::default(),
            (100.0, 100.0),
        );
        let raw2 = RawEvent::keyboard(
            Timestamp::from_ticks(1100),
            EventType::KeyDown,
            1,
            Some('i'),
            ModifierFlags::default(),
            (100.0, 100.0),
        );

        let events = vec![
            EnrichedEvent::new(raw1, 0),
            EnrichedEvent::new(raw2, 1),
        ];

        let name = clusterer.infer_task_name(&events);
        assert_eq!(name, "Type \"hi\"");
    }

    #[test]
    fn test_action_to_name_fill() {
        let clusterer = ActionClusterer::new();

        let fill = Action::Fill {
            field_name: "Username".to_string(),
            value: "test_user".to_string(),
            confidence: 0.9,
        };

        let name = clusterer.action_to_name(&fill);
        assert_eq!(name, "Fill Username");
    }

    #[test]
    fn test_action_to_name_select() {
        let clusterer = ActionClusterer::new();

        let select = Action::Select {
            menu_name: "File".to_string(),
            option: "Save".to_string(),
            confidence: 0.9,
        };

        let name = clusterer.action_to_name(&select);
        assert_eq!(name, "Select Save");
    }

    #[test]
    fn test_action_to_name_scroll() {
        use crate::analysis::intent_binding::ScrollDirection;
        let clusterer = ActionClusterer::new();

        let scroll = Action::Scroll {
            direction: ScrollDirection::Down,
            magnitude: 100.0,
            confidence: 0.9,
        };

        let name = clusterer.action_to_name(&scroll);
        assert!(name.contains("Scroll"));
    }

    #[test]
    fn test_action_to_name_shortcut() {
        let clusterer = ActionClusterer::new();

        let shortcut = Action::Shortcut {
            keys: vec!["Cmd".to_string(), "S".to_string()],
            confidence: 0.9,
        };

        let name = clusterer.action_to_name(&shortcut);
        assert_eq!(name, "Press Cmd+S");
    }

    #[test]
    fn test_filter_micro_operations() {
        let clusterer = ActionClusterer::with_config(ClusteringConfig {
            max_gap_ms: 1000,
            min_events: 3,
            merge_transient: false,
            min_movement_px: 5.0,
        });

        let tasks = vec![
            UnitTask {
                id: uuid::Uuid::new_v4(),
                name: "Click".to_string(),
                description: "Click button".to_string(),
                primary_action: make_click_action("Button"),
                events: vec![make_enriched_event(1000, EventType::LeftMouseDown, 100.0, 100.0)],
                start_time: 1000,
                end_time: 1100,
                nesting_level: 0,
                parent_id: None,
            },
            UnitTask {
                id: uuid::Uuid::new_v4(),
                name: "Action".to_string(),
                description: "Unknown action".to_string(),
                primary_action: Action::Unknown {
                    description: "Unknown".to_string(),
                    confidence: 0.5,
                },
                events: vec![make_enriched_event(2000, EventType::MouseMoved, 200.0, 200.0)],
                start_time: 2000,
                end_time: 2100,
                nesting_level: 0,
                parent_id: None,
            },
        ];

        let filtered = clusterer.filter_micro_operations(tasks);
        // Only click task should remain (has significant action)
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].name, "Click");
    }

    #[test]
    fn test_is_dropdown_pattern_positive() {
        let clusterer = ActionClusterer::new();

        let t1 = UnitTask {
            id: uuid::Uuid::new_v4(),
            name: "Click Menu".to_string(),
            description: "Click combo box".to_string(),
            primary_action: make_click_action("ComboBox"),
            events: vec![{
                let mut e = make_enriched_event(1000, EventType::LeftMouseDown, 100.0, 100.0);
                e.semantic = Some(crate::capture::types::SemanticContext {
                    ax_role: Some("AXComboBox".to_string()),
                    title: Some("Select Option".to_string()),
                    ..Default::default()
                });
                e
            }],
            start_time: 1000,
            end_time: 1100,
            nesting_level: 0,
            parent_id: None,
        };

        let t2 = UnitTask {
            id: uuid::Uuid::new_v4(),
            name: "Click Option".to_string(),
            description: "Click menu item".to_string(),
            primary_action: make_click_action("Option1"),
            events: vec![make_enriched_event(1200, EventType::LeftMouseDown, 100.0, 120.0)],
            start_time: 1200,
            end_time: 1300,
            nesting_level: 0,
            parent_id: None,
        };

        assert!(clusterer.is_dropdown_pattern(&t1, &t2));
    }

    #[test]
    fn test_is_dropdown_pattern_negative() {
        let clusterer = ActionClusterer::new();

        let t1 = UnitTask {
            id: uuid::Uuid::new_v4(),
            name: "Click Button".to_string(),
            description: "Click button".to_string(),
            primary_action: make_click_action("Button"),
            events: vec![make_enriched_event(1000, EventType::LeftMouseDown, 100.0, 100.0)],
            start_time: 1000,
            end_time: 1100,
            nesting_level: 0,
            parent_id: None,
        };

        let t2 = UnitTask {
            id: uuid::Uuid::new_v4(),
            name: "Click Another".to_string(),
            description: "Click another button".to_string(),
            primary_action: make_click_action("Another"),
            events: vec![make_enriched_event(1200, EventType::LeftMouseDown, 200.0, 200.0)],
            start_time: 1200,
            end_time: 1300,
            nesting_level: 0,
            parent_id: None,
        };

        assert!(!clusterer.is_dropdown_pattern(&t1, &t2));
    }

    #[test]
    fn test_merge_dropdown() {
        let clusterer = ActionClusterer::new();

        let t1 = UnitTask {
            id: uuid::Uuid::new_v4(),
            name: "Click Menu".to_string(),
            description: "Click menu".to_string(),
            primary_action: make_click_action("Menu"),
            events: vec![make_enriched_event(1000, EventType::LeftMouseDown, 100.0, 100.0)],
            start_time: 1000,
            end_time: 1100,
            nesting_level: 0,
            parent_id: None,
        };

        let t2 = UnitTask {
            id: uuid::Uuid::new_v4(),
            name: "Click Option".to_string(),
            description: "Click option".to_string(),
            primary_action: make_click_action("Option1"),
            events: vec![make_enriched_event(1200, EventType::LeftMouseDown, 100.0, 120.0)],
            start_time: 1200,
            end_time: 1300,
            nesting_level: 0,
            parent_id: None,
        };

        let merged = clusterer.merge_dropdown(&t1, &t2);

        assert!(merged.name.contains("Select"));
        assert!(merged.name.contains("Option1"));
        assert!(merged.name.contains("Menu"));
        assert_eq!(merged.events.len(), 2);
        assert_eq!(merged.start_time, 1000);
        assert_eq!(merged.end_time, 1300);
        assert!(matches!(merged.primary_action, Action::Select { .. }));
    }

    #[test]
    fn test_default_action_clusterer() {
        let clusterer = ActionClusterer::default();
        assert_eq!(clusterer.config.max_gap_ms, 1000);
        assert_eq!(clusterer.config.min_events, 2);
        assert!(clusterer.config.merge_transient);
    }

    #[test]
    fn test_streaming_add_event() {
        use crate::time::timebase::MachTimebase;
        MachTimebase::init();

        let mut clusterer = ActionClusterer::with_config(ClusteringConfig {
            max_gap_ms: 500,
            min_events: 2,
            merge_transient: false,
            min_movement_px: 5.0,
        });

        // Add first event - should not complete a task
        let e1 = make_enriched_event(1000, EventType::LeftMouseDown, 100.0, 100.0);
        let tasks = clusterer.add_event(e1);
        assert!(tasks.is_empty());
        assert_eq!(clusterer.pending_count(), 1);

        // Add second event close in time - should not complete
        let e2 = make_enriched_event(1200, EventType::LeftMouseUp, 100.0, 100.0);
        let tasks = clusterer.add_event(e2);
        assert!(tasks.is_empty());
        assert_eq!(clusterer.pending_count(), 2);
    }

    #[test]
    fn test_streaming_time_gap_emits_task() {
        use crate::time::timebase::MachTimebase;
        MachTimebase::init();

        let mut clusterer = ActionClusterer::with_config(ClusteringConfig {
            max_gap_ms: 500,
            min_events: 2,
            merge_transient: false,
            min_movement_px: 5.0,
        });

        // Add two events close together
        clusterer.add_event(make_enriched_event(1000, EventType::LeftMouseDown, 100.0, 100.0));
        clusterer.add_event(make_enriched_event(1200, EventType::LeftMouseUp, 100.0, 100.0));

        // Add event with large time gap - should emit previous cluster
        let e3 = make_enriched_event(2000000000, EventType::LeftMouseDown, 200.0, 200.0);
        let tasks = clusterer.add_event(e3);
        
        // Previous cluster should be emitted
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].events.len(), 2);
        
        // New event should be pending
        assert_eq!(clusterer.pending_count(), 1);
    }

    #[test]
    fn test_streaming_flush() {
        use crate::time::timebase::MachTimebase;
        MachTimebase::init();

        let mut clusterer = ActionClusterer::with_config(ClusteringConfig {
            max_gap_ms: 1000,
            min_events: 2,
            merge_transient: false,
            min_movement_px: 5.0,
        });

        // Add events
        clusterer.add_event(make_enriched_event(1000, EventType::LeftMouseDown, 100.0, 100.0));
        clusterer.add_event(make_enriched_event(1200, EventType::LeftMouseUp, 100.0, 100.0));
        
        assert_eq!(clusterer.pending_count(), 2);

        // Flush remaining
        let tasks = clusterer.flush();
        assert_eq!(tasks.len(), 1);
        assert_eq!(clusterer.pending_count(), 0);
    }

    #[test]
    fn test_streaming_clear_pending() {
        let mut clusterer = ActionClusterer::new();
        
        clusterer.add_event(make_enriched_event(1000, EventType::LeftMouseDown, 100.0, 100.0));
        assert_eq!(clusterer.pending_count(), 1);
        
        clusterer.clear_pending();
        assert_eq!(clusterer.pending_count(), 0);
    }

    #[test]
    fn test_streaming_flush_empty() {
        let mut clusterer = ActionClusterer::new();
        let tasks = clusterer.flush();
        assert!(tasks.is_empty());
    }

    #[test]
    fn test_validate_default_config() {
        let config = ClusteringConfig::default();
        assert!(config.validate().is_empty());
    }

    #[test]
    fn test_validate_zero_max_gap() {
        let config = ClusteringConfig { max_gap_ms: 0, ..Default::default() };
        let errors = config.validate();
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("max_gap_ms"));
    }

    #[test]
    fn test_validate_zero_min_events() {
        let config = ClusteringConfig { min_events: 0, ..Default::default() };
        let errors = config.validate();
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("min_events"));
    }

    #[test]
    fn test_validate_negative_movement() {
        let config = ClusteringConfig { min_movement_px: -1.0, ..Default::default() };
        let errors = config.validate();
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("min_movement_px"));
    }

    #[test]
    fn test_validate_multiple_errors() {
        let config = ClusteringConfig {
            max_gap_ms: 0,
            min_events: 0,
            merge_transient: true,
            min_movement_px: -5.0,
        };
        let errors = config.validate();
        assert_eq!(errors.len(), 3);
    }

    #[test]
    fn test_validate_zero_movement_is_ok() {
        let config = ClusteringConfig { min_movement_px: 0.0, ..Default::default() };
        assert!(config.validate().is_empty());
    }
}
