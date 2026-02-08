//! Sparse Signal Interpolation
//!
//! When a significant portion of actions have null semantic data,
//! use context and LLM inference to reconstruct missing labels.

use crate::capture::types::{EnrichedEvent, SemanticContext, SemanticSource};

/// Reconstruction configuration
#[derive(Debug, Clone)]
pub struct ReconstructionConfig {
    /// Threshold for triggering reconstruction (fraction of null events)
    pub null_threshold: f64,
    /// Maximum events to process in one batch
    pub batch_size: usize,
    /// Minimum confidence to accept inferred label
    pub min_confidence: f32,
}

impl Default for ReconstructionConfig {
    fn default() -> Self {
        Self {
            null_threshold: 0.4, // 40% null triggers reconstruction
            batch_size: 50,
            min_confidence: 0.6,
        }
    }
}

/// Reconstruction result
#[derive(Debug, Clone)]
pub struct ReconstructionResult {
    /// Number of events processed
    pub events_processed: usize,
    /// Number of successful inferences
    pub inferences_made: usize,
    /// Events that couldn't be inferred
    pub failed_indices: Vec<usize>,
}

/// Context reconstructor using LLM inference
pub struct ContextReconstructor {
    /// Configuration
    pub config: ReconstructionConfig,
    /// Shared HTTP client for LLM API calls
    http_client: reqwest::Client,
}

impl ContextReconstructor {
    /// Create with default config
    pub fn new() -> Self {
        Self {
            config: ReconstructionConfig::default(),
            http_client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .pool_max_idle_per_host(2)
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
        }
    }

    /// Create with custom config
    pub fn with_config(config: ReconstructionConfig) -> Self {
        Self {
            config,
            http_client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .pool_max_idle_per_host(2)
                .build()
                .unwrap_or_else(|_| reqwest::Client::new()),
        }
    }

    /// Check if reconstruction should be triggered
    pub fn should_reconstruct(&self, events: &[EnrichedEvent]) -> bool {
        if events.is_empty() {
            return false;
        }

        let null_count = events.iter().filter(|e| e.semantic.is_none()).count();
        let null_ratio = null_count as f64 / events.len() as f64;

        null_ratio >= self.config.null_threshold
    }

    /// Reconstruct missing semantic data
    pub fn reconstruct(
        &self,
        events: &mut [EnrichedEvent],
        goal: &str,
        app_context: Option<&str>,
    ) -> ReconstructionResult {
        let mut inferences_made = 0;
        let mut failed_indices = Vec::new();

        // Find events with null semantic data
        let null_indices: Vec<usize> = events
            .iter()
            .enumerate()
            .filter(|(_, e)| e.semantic.is_none())
            .map(|(i, _)| i)
            .collect();

        for idx in null_indices {
            // Gather context from surrounding events
            let context = self.gather_context(events, idx);

            // Attempt inference
            if let Some(inferred) = self.infer_semantic(&events[idx], goal, app_context, &context) {
                events[idx].semantic = Some(inferred);
                inferences_made += 1;
            } else {
                failed_indices.push(idx);
            }
        }

        ReconstructionResult {
            events_processed: events.len(),
            inferences_made,
            failed_indices,
        }
    }

    /// Gather context from surrounding events
    fn gather_context(&self, events: &[EnrichedEvent], index: usize) -> EventContext {
        let window = 5; // Look at 5 events before and after

        let before: Vec<&EnrichedEvent> = events[..index]
            .iter()
            .rev()
            .take(window)
            .collect();

        let after: Vec<&EnrichedEvent> = events[index + 1..]
            .iter()
            .take(window)
            .collect();

        // Extract known semantic info from context
        let known_elements: Vec<String> = before
            .iter()
            .chain(after.iter())
            .filter_map(|e| e.semantic.as_ref())
            .filter_map(|s| s.title.clone())
            .collect();

        let known_roles: Vec<String> = before
            .iter()
            .chain(after.iter())
            .filter_map(|e| e.semantic.as_ref())
            .filter_map(|s| s.ax_role.clone())
            .collect();

        EventContext {
            known_elements,
            known_roles,
            before_count: before.len(),
            after_count: after.len(),
        }
    }

    /// Infer semantic data using heuristics and patterns
    fn infer_semantic(
        &self,
        event: &EnrichedEvent,
        goal: &str,
        _app_context: Option<&str>,
        context: &EventContext,
    ) -> Option<SemanticContext> {
        // Calculate base confidence adjustment from context strength
        let context_boost = context.context_strength() * 0.1; // Up to 10% boost
        
        // Heuristic 1: Keyboard events after text field click
        if event.raw.event_type.is_keyboard()
            && context.known_roles.iter().any(|r| r.contains("TextField") || r.contains("TextArea")) {
                let base_confidence = 0.7;
                return Some(SemanticContext {
                    ax_role: Some("AXTextField".to_string()),
                    title: Some("Text Input".to_string()),
                    source: SemanticSource::Inferred,
                    confidence: (base_confidence + context_boost).min(0.95),
                    ..Default::default()
                });
            }

        // Heuristic 2: Click after known menu element
        if event.raw.event_type.is_click()
            && context.known_roles.iter().any(|r| r.contains("Menu") || r.contains("PopUp")) {
                let base_confidence = 0.65;
                return Some(SemanticContext {
                    ax_role: Some("AXMenuItem".to_string()),
                    title: Some("Menu Option".to_string()),
                    source: SemanticSource::Inferred,
                    confidence: (base_confidence + context_boost).min(0.95),
                    ..Default::default()
                });
            }

        // Heuristic 3: Use goal keywords
        let goal_keywords: Vec<&str> = goal.split_whitespace().collect();
        for keyword in goal_keywords {
            if context.known_elements.iter().any(|e| e.to_lowercase().contains(&keyword.to_lowercase())) {
                let base_confidence = 0.6;
                return Some(SemanticContext {
                    ax_role: Some("AXButton".to_string()),
                    title: Some(keyword.to_string()),
                    source: SemanticSource::Inferred,
                    confidence: (base_confidence + context_boost).min(0.95),
                    ..Default::default()
                });
            }
        }

        // Could not infer
        None
    }

    /// Infer semantic context using LLM for complex cases
    pub async fn llm_infer(
        &self,
        event: &EnrichedEvent,
        goal: &str,
        context: &str,
        api_key: Option<&str>,
    ) -> Option<SemanticContext> {
        // Resolve API key: parameter takes priority, then env var
        let api_key = match api_key.map(String::from).or_else(|| std::env::var("ANTHROPIC_API_KEY").ok()) {
            Some(key) => key,
            None => {
                tracing::debug!("No API key available, skipping LLM inference");
                return None;
            }
        };

        // Build prompt
        let prompt = format!(
            r#"Analyze this UI event and infer the semantic context.

Goal: {}

Event Details:
- Type: {:?}
- Coordinates: ({:.0}, {:.0})
- Timestamp: {}

Context from surrounding events:
{}

Based on this information, infer what UI element the user was interacting with.
Respond in JSON format:
{{
  "ax_role": "AXRole (e.g., AXButton, AXTextField)",
  "title": "Element title or label",
  "identifier": "Unique identifier if apparent",
  "confidence": 0.0-1.0
}}"#,
            goal,
            event.raw.event_type,
            event.raw.coordinates.0,
            event.raw.coordinates.1,
            event.raw.timestamp.ticks(),
            context
        );

        // Call Anthropic API with retry logic
        let body = serde_json::json!({
            "model": "claude-sonnet-4-5-20250929",
            "max_tokens": 512,
            "messages": [{"role": "user", "content": prompt}]
        });

        let response = match super::http_retry::send_with_retry(
            &self.http_client,
            |c| c.post("https://api.anthropic.com/v1/messages")
                .header("x-api-key", &api_key)
                .header("anthropic-version", "2023-06-01")
                .header("content-type", "application/json")
                .json(&body),
            3,
            "LLM inference",
        ).await {
            Some(r) => r,
            None => return None,
        };

        // Parse response
        #[derive(serde::Deserialize)]
        struct ApiResponse {
            content: Vec<ContentBlock>,
        }

        #[derive(serde::Deserialize)]
        struct ContentBlock {
            text: String,
        }

        #[derive(serde::Deserialize)]
        struct InferredContext {
            ax_role: Option<String>,
            title: Option<String>,
            identifier: Option<String>,
            confidence: Option<f32>,
        }

        let api_response: ApiResponse = match response.json().await {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!("Failed to parse LLM response: {}", e);
                return None;
            }
        };

        let text = api_response.content.first()?.text.clone();

        // Extract JSON from response
        let json_start = text.find('{')?;
        let json_end = text.rfind('}')?;
        let json_text = &text[json_start..=json_end];

        let inferred: InferredContext = match serde_json::from_str(json_text) {
            Ok(i) => i,
            Err(e) => {
                tracing::warn!("Failed to parse inferred context: {}", e);
                return None;
            }
        };

        Some(SemanticContext {
            ax_role: inferred.ax_role,
            title: inferred.title,
            identifier: inferred.identifier,
            source: SemanticSource::Inferred,
            confidence: inferred.confidence.unwrap_or(0.5),
            ..Default::default()
        })
    }
}

impl Default for ContextReconstructor {
    fn default() -> Self {
        Self::new()
    }
}

/// Context gathered from surrounding events for inference
#[derive(Debug)]
struct EventContext {
    /// Known element titles from surrounding events
    known_elements: Vec<String>,
    /// Known AX roles from surrounding events
    known_roles: Vec<String>,
    /// Number of events with semantic data before this event
    before_count: usize,
    /// Number of events with semantic data after this event
    after_count: usize,
}

impl EventContext {
    /// Calculate context strength (0.0-1.0) based on available surrounding context
    fn context_strength(&self) -> f32 {
        let total_context = self.before_count + self.after_count;
        if total_context == 0 {
            return 0.0;
        }
        
        // More context = higher confidence
        // Having both before and after context is better than just one side
        let balance_factor = if self.before_count > 0 && self.after_count > 0 {
            1.0
        } else {
            0.7 // Penalize one-sided context
        };
        
        let count_factor = (total_context as f32 / 10.0).min(1.0);
        count_factor * balance_factor
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capture::types::{CursorState, EventType, ModifierFlags, RawEvent};
    use crate::time::timebase::Timestamp;

    fn make_test_event(
        timestamp: u64,
        event_type: EventType,
        semantic: Option<SemanticContext>,
    ) -> EnrichedEvent {
        let raw = if event_type.is_keyboard() {
            RawEvent::keyboard(
                Timestamp::from_ticks(timestamp),
                event_type,
                0,
                None,
                ModifierFlags::default(),
                (100.0, 100.0),
            )
        } else {
            RawEvent::mouse(
                Timestamp::from_ticks(timestamp),
                event_type,
                100.0,
                100.0,
                CursorState::Arrow,
                ModifierFlags::default(),
                1,
            )
        };

        let mut event = EnrichedEvent::new(raw, 0);
        event.semantic = semantic;
        event
    }

    fn make_semantic(ax_role: &str, title: &str) -> SemanticContext {
        SemanticContext {
            ax_role: Some(ax_role.to_string()),
            title: Some(title.to_string()),
            source: SemanticSource::Accessibility,
            confidence: 1.0,
            ..Default::default()
        }
    }

    #[test]
    fn test_event_context_strength() {
        // No context
        let ctx = EventContext {
            known_elements: vec![],
            known_roles: vec![],
            before_count: 0,
            after_count: 0,
        };
        assert_eq!(ctx.context_strength(), 0.0);

        // One-sided context (only before)
        let ctx = EventContext {
            known_elements: vec!["Button".to_string()],
            known_roles: vec!["AXButton".to_string()],
            before_count: 3,
            after_count: 0,
        };
        let strength = ctx.context_strength();
        assert!(strength > 0.0);
        assert!(strength < 0.5); // Penalized for one-sided

        // Balanced context
        let ctx = EventContext {
            known_elements: vec!["Button".to_string()],
            known_roles: vec!["AXButton".to_string()],
            before_count: 3,
            after_count: 3,
        };
        let strength = ctx.context_strength();
        assert!(strength > 0.5); // No penalty, good strength

        // Strong balanced context
        let ctx = EventContext {
            known_elements: vec!["Button".to_string()],
            known_roles: vec!["AXButton".to_string()],
            before_count: 5,
            after_count: 5,
        };
        let strength = ctx.context_strength();
        assert_eq!(strength, 1.0); // Max strength
    }

    #[test]
    fn test_config_default() {
        let config = ReconstructionConfig::default();
        assert!(config.null_threshold > 0.0);
        assert!(config.null_threshold < 1.0);
    }

    #[test]
    fn test_should_reconstruct() {
        let reconstructor = ContextReconstructor::new();

        // Empty events - no reconstruction
        let empty: Vec<EnrichedEvent> = vec![];
        assert!(!reconstructor.should_reconstruct(&empty));
    }

    #[test]
    fn test_reconstructor_creation() {
        let reconstructor = ContextReconstructor::new();
        assert!(reconstructor.config.batch_size > 0);
        assert!(reconstructor.config.min_confidence > 0.0);
    }

    #[test]
    fn test_default_reconstructor() {
        let reconstructor = ContextReconstructor::default();
        assert!(reconstructor.config.null_threshold > 0.0);
    }

    #[test]
    fn test_should_reconstruct_with_high_null_ratio() {
        let reconstructor = ContextReconstructor::new();

        // Create events with 50% null semantic data
        let mut events = vec![];
        for i in 0..10 {
            let semantic = if i % 2 == 0 {
                None
            } else {
                Some(make_semantic("AXButton", "Test"))
            };
            events.push(make_test_event(i * 100, EventType::LeftMouseDown, semantic));
        }

        // 50% null ratio > 40% threshold
        assert!(reconstructor.should_reconstruct(&events));
    }

    #[test]
    fn test_should_not_reconstruct_with_low_null_ratio() {
        let reconstructor = ContextReconstructor::new();

        // Create events with 20% null semantic data
        let mut events = vec![];
        for i in 0..10 {
            let semantic = if i < 2 {
                None
            } else {
                Some(make_semantic("AXButton", "Test"))
            };
            events.push(make_test_event(i * 100, EventType::LeftMouseDown, semantic));
        }

        // 20% null ratio < 40% threshold
        assert!(!reconstructor.should_reconstruct(&events));
    }

    #[test]
    fn test_reconstruct_keyboard_after_textfield() {
        let reconstructor = ContextReconstructor::new();

        let mut events = vec![
            make_test_event(100, EventType::LeftMouseDown, Some(make_semantic("AXTextField", "Username"))),
            make_test_event(200, EventType::KeyDown, None), // This should be inferred
            make_test_event(300, EventType::KeyDown, None), // This should be inferred
        ];

        let result = reconstructor.reconstruct(&mut events, "login to website", None);

        // Should have inferred 2 keyboard events
        assert_eq!(result.inferences_made, 2);
        assert_eq!(result.events_processed, 3);

        // Check the inferred events
        assert!(events[1].semantic.is_some());
        let sem = events[1].semantic.as_ref().unwrap();
        assert_eq!(sem.source, SemanticSource::Inferred);
        assert!(sem.ax_role.as_ref().unwrap().contains("TextField"));
    }

    #[test]
    fn test_reconstruct_click_after_menu() {
        let reconstructor = ContextReconstructor::new();

        let mut events = vec![
            make_test_event(100, EventType::LeftMouseDown, Some(make_semantic("AXMenu", "File"))),
            make_test_event(200, EventType::LeftMouseDown, None), // Should be inferred as menu item
        ];

        let result = reconstructor.reconstruct(&mut events, "open file", None);

        assert_eq!(result.inferences_made, 1);
        assert!(events[1].semantic.is_some());

        let sem = events[1].semantic.as_ref().unwrap();
        assert_eq!(sem.source, SemanticSource::Inferred);
        assert!(sem.ax_role.as_ref().unwrap().contains("MenuItem"));
    }

    #[test]
    fn test_reconstruct_using_goal_keywords() {
        let reconstructor = ContextReconstructor::new();

        let mut events = vec![
            make_test_event(100, EventType::LeftMouseDown, Some(make_semantic("AXButton", "Submit"))),
            make_test_event(200, EventType::LeftMouseDown, None), // Should use "submit" from goal
        ];

        let result = reconstructor.reconstruct(&mut events, "click submit button", None);

        assert_eq!(result.inferences_made, 1);
        let sem = events[1].semantic.as_ref().unwrap();
        assert!(sem.title.as_ref().unwrap().to_lowercase().contains("submit") ||
                sem.title.as_ref().unwrap().to_lowercase().contains("click"));
    }

    #[test]
    fn test_reconstruct_with_custom_config() {
        let config = ReconstructionConfig {
            null_threshold: 0.3,
            batch_size: 25,
            min_confidence: 0.7,
        };

        let reconstructor = ContextReconstructor::with_config(config);

        assert_eq!(reconstructor.config.null_threshold, 0.3);
        assert_eq!(reconstructor.config.batch_size, 25);
        assert_eq!(reconstructor.config.min_confidence, 0.7);
    }

    #[test]
    fn test_gather_context_window() {
        let reconstructor = ContextReconstructor::new();

        let events = vec![
            make_test_event(100, EventType::LeftMouseDown, Some(make_semantic("AXButton", "Button1"))),
            make_test_event(200, EventType::LeftMouseDown, Some(make_semantic("AXButton", "Button2"))),
            make_test_event(300, EventType::LeftMouseDown, Some(make_semantic("AXButton", "Button3"))),
            make_test_event(400, EventType::LeftMouseDown, None), // Index 3 - target
            make_test_event(500, EventType::LeftMouseDown, Some(make_semantic("AXButton", "Button4"))),
            make_test_event(600, EventType::LeftMouseDown, Some(make_semantic("AXButton", "Button5"))),
        ];

        let context = reconstructor.gather_context(&events, 3);

        // Should have gathered context from surrounding events
        assert!(context.before_count > 0);
        assert!(context.after_count > 0);
        assert!(!context.known_elements.is_empty());
        assert!(!context.known_roles.is_empty());
    }

    #[test]
    fn test_reconstruction_result_with_failures() {
        let reconstructor = ContextReconstructor::new();

        // Create events that won't match any heuristics
        let mut events = vec![
            make_test_event(100, EventType::MouseMoved, None),
            make_test_event(200, EventType::MouseMoved, None),
        ];

        let result = reconstructor.reconstruct(&mut events, "random goal", None);

        // These events won't be inferred by current heuristics
        assert_eq!(result.events_processed, 2);
        assert_eq!(result.inferences_made + result.failed_indices.len(), 2);
    }

    #[test]
    fn test_no_reconstruction_when_all_have_semantic() {
        let reconstructor = ContextReconstructor::new();

        let mut events = vec![
            make_test_event(100, EventType::LeftMouseDown, Some(make_semantic("AXButton", "Button1"))),
            make_test_event(200, EventType::LeftMouseDown, Some(make_semantic("AXButton", "Button2"))),
        ];

        let result = reconstructor.reconstruct(&mut events, "test goal", None);

        // No null events, so no inferences
        assert_eq!(result.inferences_made, 0);
        assert_eq!(result.failed_indices.len(), 0);
    }

    #[test]
    fn test_reconstruction_confidence_levels() {
        let reconstructor = ContextReconstructor::new();

        let mut events = vec![
            make_test_event(100, EventType::LeftMouseDown, Some(make_semantic("AXTextField", "Input"))),
            make_test_event(200, EventType::KeyDown, None),
        ];

        reconstructor.reconstruct(&mut events, "enter text", None);

        let sem = events[1].semantic.as_ref().unwrap();
        assert!(sem.confidence >= 0.6); // min_confidence from default config
        assert!(sem.confidence <= 1.0);
    }
}
