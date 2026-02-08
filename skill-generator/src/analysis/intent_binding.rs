//! Intent Inference from Trajectory + Semantics
//!
//! Combines kinematic analysis with semantic context to infer
//! the user's intended action.

use super::kinematic_segmentation::{KinematicAnalysis, MovementPattern};
use super::rdp_simplification::SimplifiedTrajectory;
use crate::capture::types::SemanticContext;
use serde::{Deserialize, Serialize};

/// Inferred user action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Action {
    /// Click on a UI element
    Click {
        element_name: String,
        element_role: String,
        confidence: f32,
    },
    /// Double-click
    DoubleClick {
        element_name: String,
        element_role: String,
        confidence: f32,
    },
    /// Right-click (context menu)
    RightClick {
        element_name: String,
        element_role: String,
        confidence: f32,
    },
    /// Fill a text field
    Fill {
        field_name: String,
        value: String,
        confidence: f32,
    },
    /// Search/explore (user looking for something)
    Search {
        candidate_elements: Vec<String>,
        confidence: f32,
    },
    /// Select from dropdown/menu
    Select {
        menu_name: String,
        option: String,
        confidence: f32,
    },
    /// Drag and drop
    DragDrop {
        source: String,
        target: String,
        confidence: f32,
    },
    /// Scroll
    Scroll {
        direction: ScrollDirection,
        magnitude: f64,
        confidence: f32,
    },
    /// Type text
    Type {
        text: String,
        confidence: f32,
    },
    /// Keyboard shortcut
    Shortcut {
        keys: Vec<String>,
        confidence: f32,
    },
    /// Unknown action
    Unknown {
        description: String,
        confidence: f32,
    },
}

/// Scroll direction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScrollDirection {
    Up,
    Down,
    Left,
    Right,
}

impl Action {
    /// Get the confidence score
    pub fn confidence(&self) -> f32 {
        match self {
            Action::Click { confidence, .. } => *confidence,
            Action::DoubleClick { confidence, .. } => *confidence,
            Action::RightClick { confidence, .. } => *confidence,
            Action::Fill { confidence, .. } => *confidence,
            Action::Search { confidence, .. } => *confidence,
            Action::Select { confidence, .. } => *confidence,
            Action::DragDrop { confidence, .. } => *confidence,
            Action::Scroll { confidence, .. } => *confidence,
            Action::Type { confidence, .. } => *confidence,
            Action::Shortcut { confidence, .. } => *confidence,
            Action::Unknown { confidence, .. } => *confidence,
        }
    }

    /// Get a human-readable description
    pub fn description(&self) -> String {
        match self {
            Action::Click { element_name, .. } => format!("Click '{}'", element_name),
            Action::DoubleClick { element_name, .. } => format!("Double-click '{}'", element_name),
            Action::RightClick { element_name, .. } => format!("Right-click '{}'", element_name),
            Action::Fill { field_name, value, .. } => format!("Fill '{}' with '{}'", field_name, value),
            Action::Search { candidate_elements, .. } => {
                format!("Search among: {}", candidate_elements.join(", "))
            }
            Action::Select { menu_name, option, .. } => {
                format!("Select '{}' from '{}'", option, menu_name)
            }
            Action::DragDrop { source, target, .. } => format!("Drag '{}' to '{}'", source, target),
            Action::Scroll { direction, .. } => format!("Scroll {:?}", direction),
            Action::Type { text, .. } => format!("Type '{}'", text),
            Action::Shortcut { keys, .. } => format!("Press {}", keys.join("+")),
            Action::Unknown { description, .. } => description.clone(),
        }
    }
}

/// Intent inference engine
pub struct IntentBinder {
    /// Confidence threshold for accepting an inference
    pub confidence_threshold: f32,
}

impl IntentBinder {
    /// Create with default settings
    pub fn new() -> Self {
        Self {
            confidence_threshold: 0.7,
        }
    }

    /// Infer action from trajectory and semantics
    pub fn infer_action(
        &self,
        _trajectory: &SimplifiedTrajectory,
        kinematics: &KinematicAnalysis,
        target_semantic: Option<&SemanticContext>,
        click_count: u8,
        is_right_click: bool,
    ) -> Action {
        // Get element info from semantics
        let (element_name, element_role) = if let Some(semantic) = target_semantic {
            (
                semantic.title.clone().or(semantic.identifier.clone()).unwrap_or_else(|| "unknown".to_string()),
                semantic.ax_role.clone().unwrap_or_else(|| "unknown".to_string()),
            )
        } else {
            ("unknown".to_string(), "unknown".to_string())
        };

        // Calculate base confidence from kinematics and semantics
        let kinematic_confidence = self.kinematic_confidence(kinematics);
        let semantic_confidence = if target_semantic.is_some() { 1.0 } else { 0.5 };
        let base_confidence = (kinematic_confidence + semantic_confidence) / 2.0;

        // Handle right-click
        if is_right_click {
            return Action::RightClick {
                element_name,
                element_role,
                confidence: base_confidence * 0.95,
            };
        }

        // Handle double-click
        if click_count >= 2 {
            return Action::DoubleClick {
                element_name,
                element_role,
                confidence: base_confidence * 0.95,
            };
        }

        // Infer based on element role and movement pattern
        match (element_role.as_str(), kinematics.pattern) {
            // Text field + ballistic = likely filling
            ("AXTextField" | "AXTextArea" | "AXComboBox", MovementPattern::Ballistic) => {
                Action::Click {
                    element_name,
                    element_role,
                    confidence: base_confidence * 0.9,
                }
            }

            // Button + ballistic = confident click
            ("AXButton" | "AXLink", MovementPattern::Ballistic) => {
                Action::Click {
                    element_name,
                    element_role,
                    confidence: base_confidence * 0.95,
                }
            }

            // Menu item
            ("AXMenuItem" | "AXMenuBarItem", _) => {
                Action::Select {
                    menu_name: target_semantic
                        .and_then(|s| s.parent_title.clone())
                        .unwrap_or_else(|| "menu".to_string()),
                    option: element_name,
                    confidence: base_confidence * 0.9,
                }
            }

            // Searching pattern
            (_, MovementPattern::Searching) => {
                Action::Search {
                    candidate_elements: vec![element_name],
                    confidence: base_confidence * 0.7,
                }
            }

            // Corrective pattern suggests uncertainty
            (_, MovementPattern::Corrective) => {
                Action::Click {
                    element_name,
                    element_role,
                    confidence: base_confidence * 0.6,
                }
            }

            // Default to click
            _ => {
                Action::Click {
                    element_name,
                    element_role,
                    confidence: base_confidence * 0.8,
                }
            }
        }
    }

    /// Infer action from a sequence of keyboard events
    pub fn infer_keyboard_action(&self, keys: &[(u16, Option<char>, bool)]) -> Action {
        // Check for shortcuts (modifier + key)
        let has_modifier = keys.iter().any(|(_, _, mods)| *mods);

        if has_modifier && keys.len() <= 3 {
            let key_names: Vec<String> = keys
                .iter()
                .filter_map(|(code, ch, _)| {
                    ch.map(|c| c.to_string())
                        .or_else(|| Some(format!("Key{}", code)))
                })
                .collect();

            return Action::Shortcut {
                keys: key_names,
                confidence: 0.9,
            };
        }

        // Otherwise it's typing
        let text: String = keys
            .iter()
            .filter_map(|(_, ch, _)| *ch)
            .collect();

        Action::Type {
            text,
            confidence: 0.95,
        }
    }

    /// Infer scroll action
    pub fn infer_scroll_action(&self, delta_x: f64, delta_y: f64) -> Action {
        let direction = if delta_y.abs() > delta_x.abs() {
            if delta_y > 0.0 {
                ScrollDirection::Up
            } else {
                ScrollDirection::Down
            }
        } else if delta_x > 0.0 {
            ScrollDirection::Left
        } else {
            ScrollDirection::Right
        };

        let magnitude = (delta_x * delta_x + delta_y * delta_y).sqrt();

        Action::Scroll {
            direction,
            magnitude,
            confidence: 0.95,
        }
    }

    /// Calculate confidence based on kinematic analysis
    fn kinematic_confidence(&self, kinematics: &KinematicAnalysis) -> f32 {
        match kinematics.pattern {
            MovementPattern::Ballistic => 0.95,
            MovementPattern::Searching => 0.6,
            MovementPattern::Corrective => 0.5,
            MovementPattern::Stationary => 0.8,
        }
    }
}

impl Default for IntentBinder {
    fn default() -> Self {
        Self::new()
    }
}

/// Bound action with full context
#[derive(Debug, Clone)]
pub struct BoundAction {
    /// The inferred action
    pub action: Action,
    /// Original events that triggered this action
    pub event_ids: Vec<uuid::Uuid>,
    /// Target semantic context
    pub target_semantic: Option<SemanticContext>,
    /// Trajectory analysis
    pub trajectory_summary: Option<TrajectorySummary>,
    /// Timestamp of the triggering event
    pub timestamp_ticks: u64,
}

/// Summary of trajectory for an action
#[derive(Debug, Clone)]
pub struct TrajectorySummary {
    /// Start position
    pub start: (f64, f64),
    /// End position
    pub end: (f64, f64),
    /// Path length
    pub path_length: f64,
    /// Duration in ms
    pub duration_ms: u64,
    /// Movement pattern
    pub pattern: MovementPattern,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_action_confidence() {
        let action = Action::Click {
            element_name: "Submit".to_string(),
            element_role: "AXButton".to_string(),
            confidence: 0.9,
        };

        assert!((action.confidence() - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_action_description() {
        let action = Action::Click {
            element_name: "Submit".to_string(),
            element_role: "AXButton".to_string(),
            confidence: 0.9,
        };

        assert_eq!(action.description(), "Click 'Submit'");
    }

    #[test]
    fn test_scroll_direction() {
        let binder = IntentBinder::new();

        // Scroll down
        let action = binder.infer_scroll_action(0.0, -10.0);
        if let Action::Scroll { direction, .. } = action {
            assert_eq!(direction as u8, ScrollDirection::Down as u8);
        } else {
            panic!("Expected Scroll action");
        }
    }

    #[test]
    fn test_keyboard_shortcut() {
        let binder = IntentBinder::new();

        // Cmd+S style shortcut
        let keys = vec![
            (55u16, Some('s'), true), // Command key present
        ];

        let action = binder.infer_keyboard_action(&keys);
        assert!(matches!(action, Action::Shortcut { .. }));
    }

    #[test]
    fn test_typing() {
        let binder = IntentBinder::new();

        let keys = vec![
            (0u16, Some('h'), false),
            (1u16, Some('e'), false),
            (2u16, Some('l'), false),
            (3u16, Some('l'), false),
            (4u16, Some('o'), false),
        ];

        let action = binder.infer_keyboard_action(&keys);
        if let Action::Type { text, .. } = action {
            assert_eq!(text, "hello");
        } else {
            panic!("Expected Type action");
        }
    }

    #[test]
    fn test_right_click_inference() {
        use super::super::kinematic_segmentation::{KinematicAnalysis, MovementPattern};
        use super::super::rdp_simplification::SimplifiedTrajectory;
        use crate::capture::types::{RawEvent, EventType, CursorState, ModifierFlags};
        use crate::time::timebase::Timestamp;

        let binder = IntentBinder::new();

        let events = vec![
            RawEvent::mouse(
                Timestamp::from_ticks(0),
                EventType::MouseMoved,
                0.0,
                0.0,
                CursorState::Arrow,
                ModifierFlags::default(),
                0,
            ),
        ];

        let trajectory = SimplifiedTrajectory::from_events(&events, 1.0);
        let kinematics = KinematicAnalysis {
            velocity_profile: vec![],
            segments: vec![],
            peak_velocity: 0.0,
            total_duration_ms: 0,
            ballistic_vector: None,
            homing_dwell_ms: 0,
            pattern: MovementPattern::Ballistic,
        };

        let action = binder.infer_action(&trajectory, &kinematics, None, 1, true);
        assert!(matches!(action, Action::RightClick { .. }));
        assert!(action.confidence() > 0.5);
    }

    #[test]
    fn test_double_click_inference() {
        use super::super::kinematic_segmentation::{KinematicAnalysis, MovementPattern};
        use super::super::rdp_simplification::SimplifiedTrajectory;
        use crate::capture::types::{RawEvent, EventType, CursorState, ModifierFlags};
        use crate::time::timebase::Timestamp;

        let binder = IntentBinder::new();

        let events = vec![
            RawEvent::mouse(
                Timestamp::from_ticks(0),
                EventType::MouseMoved,
                0.0,
                0.0,
                CursorState::Arrow,
                ModifierFlags::default(),
                0,
            ),
        ];

        let trajectory = SimplifiedTrajectory::from_events(&events, 1.0);
        let kinematics = KinematicAnalysis {
            velocity_profile: vec![],
            segments: vec![],
            peak_velocity: 0.0,
            total_duration_ms: 0,
            ballistic_vector: None,
            homing_dwell_ms: 0,
            pattern: MovementPattern::Ballistic,
        };

        let action = binder.infer_action(&trajectory, &kinematics, None, 2, false);
        assert!(matches!(action, Action::DoubleClick { .. }));
    }

    #[test]
    fn test_button_click_with_semantic() {
        use super::super::kinematic_segmentation::{KinematicAnalysis, MovementPattern};
        use super::super::rdp_simplification::SimplifiedTrajectory;
        use crate::capture::types::{RawEvent, EventType, SemanticContext, SemanticSource, CursorState, ModifierFlags};
        use crate::time::timebase::Timestamp;

        let binder = IntentBinder::new();

        let events = vec![
            RawEvent::mouse(
                Timestamp::from_ticks(0),
                EventType::MouseMoved,
                0.0,
                0.0,
                CursorState::Arrow,
                ModifierFlags::default(),
                0,
            ),
        ];

        let trajectory = SimplifiedTrajectory::from_events(&events, 1.0);
        let kinematics = KinematicAnalysis {
            velocity_profile: vec![],
            segments: vec![],
            peak_velocity: 0.0,
            total_duration_ms: 0,
            ballistic_vector: None,
            homing_dwell_ms: 0,
            pattern: MovementPattern::Ballistic,
        };

        let semantic = SemanticContext {
            ax_role: Some("AXButton".to_string()),
            title: Some("Submit".to_string()),
            identifier: None,
            value: None,
            parent_role: None,
            parent_title: None,
            window_title: None,
            app_bundle_id: None,
            app_name: None,
            pid: None,
            window_id: None,
            frame: None,
            source: SemanticSource::Accessibility,
            confidence: 1.0,
            ocr_text: None,
            ancestors: vec![],
        };

        let action = binder.infer_action(&trajectory, &kinematics, Some(&semantic), 1, false);

        if let Action::Click { element_name, element_role, confidence } = action {
            assert_eq!(element_name, "Submit");
            assert_eq!(element_role, "AXButton");
            assert!(confidence > 0.7); // High confidence for button with ballistic movement
        } else {
            panic!("Expected Click action");
        }
    }

    #[test]
    fn test_searching_pattern_inference() {
        use super::super::kinematic_segmentation::{KinematicAnalysis, MovementPattern};
        use super::super::rdp_simplification::SimplifiedTrajectory;
        use crate::capture::types::{RawEvent, EventType, CursorState, ModifierFlags};
        use crate::time::timebase::Timestamp;

        let binder = IntentBinder::new();

        let events = vec![
            RawEvent::mouse(
                Timestamp::from_ticks(0),
                EventType::MouseMoved,
                0.0,
                0.0,
                CursorState::Arrow,
                ModifierFlags::default(),
                0,
            ),
        ];

        let trajectory = SimplifiedTrajectory::from_events(&events, 1.0);
        let kinematics = KinematicAnalysis {
            velocity_profile: vec![],
            segments: vec![],
            peak_velocity: 100.0,
            total_duration_ms: 1000,
            ballistic_vector: None,
            homing_dwell_ms: 0,
            pattern: MovementPattern::Searching,
        };

        let action = binder.infer_action(&trajectory, &kinematics, None, 1, false);
        assert!(matches!(action, Action::Search { .. }));
    }

    #[test]
    fn test_menu_item_selection() {
        use super::super::kinematic_segmentation::{KinematicAnalysis, MovementPattern};
        use super::super::rdp_simplification::SimplifiedTrajectory;
        use crate::capture::types::{RawEvent, EventType, SemanticContext, SemanticSource, CursorState, ModifierFlags};
        use crate::time::timebase::Timestamp;

        let binder = IntentBinder::new();

        let events = vec![
            RawEvent::mouse(
                Timestamp::from_ticks(0),
                EventType::MouseMoved,
                0.0,
                0.0,
                CursorState::Arrow,
                ModifierFlags::default(),
                0,
            ),
        ];

        let trajectory = SimplifiedTrajectory::from_events(&events, 1.0);
        let kinematics = KinematicAnalysis {
            velocity_profile: vec![],
            segments: vec![],
            peak_velocity: 0.0,
            total_duration_ms: 0,
            ballistic_vector: None,
            homing_dwell_ms: 0,
            pattern: MovementPattern::Ballistic,
        };

        let semantic = SemanticContext {
            ax_role: Some("AXMenuItem".to_string()),
            title: Some("Save".to_string()),
            identifier: None,
            value: None,
            parent_role: None,
            parent_title: Some("File".to_string()),
            window_title: None,
            app_bundle_id: None,
            app_name: None,
            pid: None,
            window_id: None,
            frame: None,
            source: SemanticSource::Accessibility,
            confidence: 1.0,
            ocr_text: None,
            ancestors: vec![],
        };

        let action = binder.infer_action(&trajectory, &kinematics, Some(&semantic), 1, false);

        if let Action::Select { menu_name, option, confidence } = action {
            assert_eq!(menu_name, "File");
            assert_eq!(option, "Save");
            assert!(confidence > 0.7);
        } else {
            panic!("Expected Select action");
        }
    }

    #[test]
    fn test_scroll_up_inference() {
        let binder = IntentBinder::new();

        let action = binder.infer_scroll_action(0.0, 10.0);
        if let Action::Scroll { direction, magnitude, .. } = action {
            assert!(matches!(direction, ScrollDirection::Up));
            assert!((magnitude - 10.0).abs() < 0.001);
        } else {
            panic!("Expected Scroll action");
        }
    }

    #[test]
    fn test_scroll_horizontal() {
        let binder = IntentBinder::new();

        // Scroll right (negative delta_x)
        let action = binder.infer_scroll_action(-15.0, 0.0);
        if let Action::Scroll { direction, magnitude, .. } = action {
            assert!(matches!(direction, ScrollDirection::Right));
            assert!((magnitude - 15.0).abs() < 0.001);
        } else {
            panic!("Expected Scroll action");
        }
    }

    #[test]
    fn test_keyboard_typing_empty() {
        let binder = IntentBinder::new();

        let keys = vec![];
        let action = binder.infer_keyboard_action(&keys);

        if let Action::Type { text, .. } = action {
            assert!(text.is_empty());
        } else {
            panic!("Expected Type action");
        }
    }

    #[test]
    fn test_corrective_pattern_low_confidence() {
        use super::super::kinematic_segmentation::{KinematicAnalysis, MovementPattern};
        use super::super::rdp_simplification::SimplifiedTrajectory;
        use crate::capture::types::{RawEvent, EventType, CursorState, ModifierFlags};
        use crate::time::timebase::Timestamp;

        let binder = IntentBinder::new();

        let events = vec![
            RawEvent::mouse(
                Timestamp::from_ticks(0),
                EventType::MouseMoved,
                0.0,
                0.0,
                CursorState::Arrow,
                ModifierFlags::default(),
                0,
            ),
        ];

        let trajectory = SimplifiedTrajectory::from_events(&events, 1.0);
        let kinematics = KinematicAnalysis {
            velocity_profile: vec![],
            segments: vec![],
            peak_velocity: 100.0,
            total_duration_ms: 1000,
            ballistic_vector: None,
            homing_dwell_ms: 0,
            pattern: MovementPattern::Corrective,
        };

        let action = binder.infer_action(&trajectory, &kinematics, None, 1, false);
        // Corrective pattern should result in lower confidence
        assert!(action.confidence() < 0.7);
    }

    #[test]
    fn test_action_description_fill() {
        let action = Action::Fill {
            field_name: "Email".to_string(),
            value: "test@example.com".to_string(),
            confidence: 0.9,
        };

        assert_eq!(action.description(), "Fill 'Email' with 'test@example.com'");
    }

    #[test]
    fn test_action_description_drag_drop() {
        let action = Action::DragDrop {
            source: "File.txt".to_string(),
            target: "Trash".to_string(),
            confidence: 0.85,
        };

        assert_eq!(action.description(), "Drag 'File.txt' to 'Trash'");
    }

    #[test]
    fn test_kinematic_confidence_levels() {
        use super::super::kinematic_segmentation::{KinematicAnalysis, MovementPattern};

        let binder = IntentBinder::new();

        // Test ballistic pattern (high confidence)
        let ballistic = KinematicAnalysis {
            velocity_profile: vec![],
            segments: vec![],
            peak_velocity: 0.0,
            total_duration_ms: 0,
            ballistic_vector: None,
            homing_dwell_ms: 0,
            pattern: MovementPattern::Ballistic,
        };
        assert_eq!(binder.kinematic_confidence(&ballistic), 0.95);

        // Test searching pattern (lower confidence)
        let searching = KinematicAnalysis {
            velocity_profile: vec![],
            segments: vec![],
            peak_velocity: 0.0,
            total_duration_ms: 0,
            ballistic_vector: None,
            homing_dwell_ms: 0,
            pattern: MovementPattern::Searching,
        };
        assert_eq!(binder.kinematic_confidence(&searching), 0.6);
    }
}
