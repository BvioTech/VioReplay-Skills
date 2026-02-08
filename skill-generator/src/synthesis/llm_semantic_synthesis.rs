//! LLM-Driven Generalization

use crate::capture::types::EnrichedEvent;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, warn};

/// LLM synthesis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisResult {
    /// Variable type
    pub variable_type: String,
    /// Inferred variable name
    pub inferred_name: String,
    /// Derivation/explanation
    pub derivation: String,
    /// Confidence score
    pub confidence: f32,
}

/// Cache entry for LLM responses
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CacheEntry {
    result: SynthesisResult,
    timestamp: u64,
}

/// Anthropic API request body
#[derive(Debug, Serialize)]
struct AnthropicRequest {
    model: String,
    max_tokens: u32,
    messages: Vec<Message>,
}

#[derive(Debug, Serialize)]
struct Message {
    role: String,
    content: String,
}

/// Anthropic API response body
#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    content: Vec<ContentBlock>,
}

#[derive(Debug, Deserialize)]
struct ContentBlock {
    text: String,
}

/// LLM synthesizer for complex variable inference
pub struct LlmSynthesizer {
    /// API endpoint
    pub api_endpoint: String,
    /// Model to use
    pub model: String,
    /// Temperature for generation
    pub temperature: f32,
    /// API key (read from environment)
    api_key: Option<String>,
    /// HTTP client
    client: Client,
    /// Response cache
    cache: HashMap<String, CacheEntry>,
    /// Cache TTL in seconds
    cache_ttl: u64,
}

impl LlmSynthesizer {
    /// Create with default settings
    pub fn new() -> Self {
        Self {
            api_endpoint: "https://api.anthropic.com/v1/messages".to_string(),
            model: "claude-sonnet-4-5-20250929".to_string(),
            temperature: 0.3,
            api_key: std::env::var("ANTHROPIC_API_KEY").ok(),
            client: Client::new(),
            cache: HashMap::new(),
            cache_ttl: 3600, // 1 hour
        }
    }

    /// Create with custom API key
    pub fn with_api_key(api_key: &str) -> Self {
        Self {
            api_endpoint: "https://api.anthropic.com/v1/messages".to_string(),
            model: "claude-sonnet-4-5-20250929".to_string(),
            temperature: 0.3,
            api_key: Some(api_key.to_string()),
            client: Client::new(),
            cache: HashMap::new(),
            cache_ttl: 3600,
        }
    }

    /// Check if API key is configured
    pub fn is_configured(&self) -> bool {
        self.api_key.is_some()
    }

    /// Synthesize variable meaning from context
    pub async fn synthesize(
        &mut self,
        goal: &str,
        trace_snippet: &[EnrichedEvent],
        observed_input: &str,
        question: &str,
    ) -> Option<SynthesisResult> {
        // Generate cache key
        let cache_key = self.generate_cache_key(goal, observed_input, question);

        // Check cache
        if let Some(cached) = self.get_cached(&cache_key) {
            return Some(cached);
        }

        // Build prompt
        let prompt = self.build_prompt(goal, trace_snippet, observed_input, question);

        // Call LLM API
        let result = self.call_llm(&prompt).await?;

        // Cache result
        self.cache_result(&cache_key, &result);

        Some(result)
    }

    /// Generate cache key
    fn generate_cache_key(&self, goal: &str, input: &str, question: &str) -> String {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        goal.hash(&mut hasher);
        input.hash(&mut hasher);
        question.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }

    /// Get cached result if valid
    fn get_cached(&self, key: &str) -> Option<SynthesisResult> {
        self.cache.get(key).and_then(|entry| {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();
            if now - entry.timestamp < self.cache_ttl {
                Some(entry.result.clone())
            } else {
                None
            }
        })
    }

    /// Cache a result
    fn cache_result(&mut self, key: &str, result: &SynthesisResult) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        self.cache.insert(
            key.to_string(),
            CacheEntry {
                result: result.clone(),
                timestamp: now,
            },
        );
    }

    /// Build prompt for LLM
    fn build_prompt(
        &self,
        goal: &str,
        trace_snippet: &[EnrichedEvent],
        observed_input: &str,
        question: &str,
    ) -> String {
        let trace_summary = self.summarize_trace(trace_snippet);

        format!(
            r#"You are analyzing a user interaction recording to identify variables for skill generation.

Goal: {}

Recent actions:
{}

Observed user input: "{}"

Question: {}

Analyze this and provide:
1. The type of variable (e.g., "airport_code", "date", "email")
2. A good variable name for the skill template
3. An explanation of the relationship

Respond in JSON format:
{{
  "variable_type": "type_name",
  "inferred_name": "variable_name",
  "derivation": "explanation",
  "confidence": 0.0-1.0
}}"#,
            goal, trace_summary, observed_input, question
        )
    }

    /// Summarize trace for prompt
    fn summarize_trace(&self, trace: &[EnrichedEvent]) -> String {
        trace
            .iter()
            .take(10)
            .map(|e| {
                let action = if e.raw.event_type.is_click() {
                    "click"
                } else if e.raw.event_type.is_keyboard() {
                    "type"
                } else {
                    "action"
                };
                let target = e
                    .semantic
                    .as_ref()
                    .and_then(|s| s.title.clone())
                    .unwrap_or_else(|| "unknown".to_string());
                format!("- {} on '{}'", action, target)
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Call LLM API
    async fn call_llm(&self, prompt: &str) -> Option<SynthesisResult> {
        let api_key = match &self.api_key {
            Some(key) => key,
            None => {
                warn!("No ANTHROPIC_API_KEY configured, skipping LLM call");
                return None;
            }
        };

        debug!("Calling Anthropic API with prompt length: {}", prompt.len());

        let request_body = AnthropicRequest {
            model: self.model.clone(),
            max_tokens: 1024,
            messages: vec![Message {
                role: "user".to_string(),
                content: prompt.to_string(),
            }],
        };

        let response = match self
            .client
            .post(&self.api_endpoint)
            .header("x-api-key", api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&request_body)
            .send()
            .await
        {
            Ok(resp) => resp,
            Err(e) => {
                warn!("Failed to call Anthropic API: {}", e);
                return None;
            }
        };

        if !response.status().is_success() {
            warn!(
                "Anthropic API returned error status: {}",
                response.status()
            );
            return None;
        }

        let response_body: AnthropicResponse = match response.json().await {
            Ok(body) => body,
            Err(e) => {
                warn!("Failed to parse Anthropic response: {}", e);
                return None;
            }
        };

        // Extract text from response
        let text = response_body.content.first()?.text.clone();

        // Parse JSON from response text
        // Try to find JSON in the response (handle markdown code blocks)
        let json_text = if let Some(start) = text.find('{') {
            if let Some(end) = text.rfind('}') {
                &text[start..=end]
            } else {
                &text
            }
        } else {
            &text
        };

        match serde_json::from_str::<SynthesisResult>(json_text) {
            Ok(result) => {
                debug!("Successfully parsed LLM synthesis result");
                Some(result)
            }
            Err(e) => {
                warn!("Failed to parse synthesis result JSON: {}", e);
                None
            }
        }
    }

    /// Clear expired cache entries
    pub fn cleanup_cache(&mut self) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        self.cache.retain(|_, entry| now - entry.timestamp < self.cache_ttl);
    }
}

impl Default for LlmSynthesizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_key_generation() {
        let synthesizer = LlmSynthesizer::new();

        let key1 = synthesizer.generate_cache_key("goal", "input", "question");
        let key2 = synthesizer.generate_cache_key("goal", "input", "question");
        let key3 = synthesizer.generate_cache_key("goal2", "input", "question");

        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
    }

    #[test]
    fn test_prompt_building() {
        let synthesizer = LlmSynthesizer::new();

        let prompt = synthesizer.build_prompt(
            "Book a flight",
            &[],
            "CDG",
            "What does CDG represent?",
        );

        assert!(prompt.contains("Book a flight"));
        assert!(prompt.contains("CDG"));
    }

    #[test]
    fn test_synthesizer_without_api_key() {
        // Create without env var set
        let synthesizer = LlmSynthesizer::new();
        // May or may not be configured depending on env
        // Just verify it doesn't panic
        let _ = synthesizer.is_configured();
    }

    #[test]
    fn test_synthesizer_with_api_key() {
        let synthesizer = LlmSynthesizer::with_api_key("test-key");
        assert!(synthesizer.is_configured());
    }

    #[test]
    fn test_synthesis_result_structure() {
        let result = SynthesisResult {
            variable_type: "airport_code".to_string(),
            inferred_name: "destination_airport".to_string(),
            derivation: "User entered a 3-letter IATA code".to_string(),
            confidence: 0.95,
        };

        assert_eq!(result.variable_type, "airport_code");
        assert!(result.confidence > 0.9);
    }

    #[test]
    fn test_cache_operations() {
        let mut synthesizer = LlmSynthesizer::new();
        let key = synthesizer.generate_cache_key("goal", "input", "question");

        // Initially no cache
        assert!(synthesizer.get_cached(&key).is_none());

        // Cache a result
        let result = SynthesisResult {
            variable_type: "email".to_string(),
            inferred_name: "user_email".to_string(),
            derivation: "Email format detected".to_string(),
            confidence: 0.9,
        };
        synthesizer.cache_result(&key, &result);

        // Should be cached now
        let cached = synthesizer.get_cached(&key);
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().variable_type, "email");
    }

    #[test]
    fn test_cleanup_cache() {
        let mut synthesizer = LlmSynthesizer::new();
        synthesizer.cache_ttl = 0; // Expire immediately

        let key = synthesizer.generate_cache_key("goal", "input", "question");
        let result = SynthesisResult {
            variable_type: "test".to_string(),
            inferred_name: "test".to_string(),
            derivation: "test".to_string(),
            confidence: 0.5,
        };
        synthesizer.cache_result(&key, &result);

        // Wait a moment then cleanup
        std::thread::sleep(std::time::Duration::from_millis(10));
        synthesizer.cleanup_cache();

        // Cache should be expired
        assert!(synthesizer.cache.is_empty());
    }

    #[test]
    fn test_default_implementation() {
        let synthesizer = LlmSynthesizer::default();
        assert!(synthesizer.api_endpoint.contains("anthropic.com"));
    }

    #[test]
    fn test_anthropic_request_serialization() {
        let request = AnthropicRequest {
            model: "claude-sonnet-4-5-20250929".to_string(),
            max_tokens: 1024,
            messages: vec![Message {
                role: "user".to_string(),
                content: "Hello".to_string(),
            }],
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("claude-sonnet-4-5-20250929"));
        assert!(json.contains("max_tokens"));
    }

    #[test]
    fn test_summarize_trace_click_events() {
        use crate::capture::types::{
            CursorState, EventType, ModifierFlags, RawEvent, SemanticContext, SemanticSource,
        };
        use crate::time::timebase::Timestamp;

        let synthesizer = LlmSynthesizer::new();

        // Create events with semantic data
        let mut events = vec![];
        for i in 0..3 {
            let raw = RawEvent::mouse(
                Timestamp::from_ticks(1000 * (i + 1)),
                EventType::LeftMouseDown,
                100.0,
                200.0,
                CursorState::Arrow,
                ModifierFlags::default(),
                1,
            );

            let mut semantic = SemanticContext::default();
            semantic.title = Some(format!("Button {}", i + 1));
            semantic.ax_role = Some("AXButton".to_string());
            semantic.source = SemanticSource::Accessibility;

            events.push(EnrichedEvent::new(raw, i).with_semantic(semantic));
        }

        let summary = synthesizer.summarize_trace(&events);

        assert!(summary.contains("click"));
        assert!(summary.contains("Button 1"));
        assert!(summary.contains("Button 2"));
        assert!(summary.contains("Button 3"));
    }

    #[test]
    fn test_summarize_trace_keyboard_events() {
        use crate::capture::types::{
            EventType, ModifierFlags, RawEvent, SemanticContext, SemanticSource,
        };
        use crate::time::timebase::Timestamp;

        let synthesizer = LlmSynthesizer::new();

        // Create keyboard events
        let mut events = vec![];
        for i in 0..2 {
            let raw = RawEvent::keyboard(
                Timestamp::from_ticks(1000 * (i + 1)),
                EventType::KeyDown,
                42,
                Some('a'),
                ModifierFlags::default(),
                (100.0, 200.0),
            );

            let mut semantic = SemanticContext::default();
            semantic.value = Some(format!("Text {}", i + 1));
            semantic.source = SemanticSource::Accessibility;

            events.push(EnrichedEvent::new(raw, i).with_semantic(semantic));
        }

        let summary = synthesizer.summarize_trace(&events);

        assert!(summary.contains("type"));
    }

    #[test]
    fn test_summarize_trace_truncates_at_10_events() {
        use crate::capture::types::{
            CursorState, EventType, ModifierFlags, RawEvent, SemanticContext,
        };
        use crate::time::timebase::Timestamp;

        let synthesizer = LlmSynthesizer::new();

        // Create 15 events
        let mut events = vec![];
        for i in 0..15 {
            let raw = RawEvent::mouse(
                Timestamp::from_ticks(1000 * (i + 1)),
                EventType::LeftMouseDown,
                100.0,
                200.0,
                CursorState::Arrow,
                ModifierFlags::default(),
                1,
            );

            let mut semantic = SemanticContext::default();
            semantic.title = Some(format!("Item {}", i + 1));

            events.push(EnrichedEvent::new(raw, i).with_semantic(semantic));
        }

        let summary = synthesizer.summarize_trace(&events);

        // Should only have first 10
        let line_count = summary.lines().count();
        assert_eq!(line_count, 10);
        assert!(summary.contains("Item 1"));
        assert!(summary.contains("Item 10"));
        assert!(!summary.contains("Item 11"));
    }

    #[test]
    fn test_prompt_includes_all_components() {
        let synthesizer = LlmSynthesizer::new();

        let prompt = synthesizer.build_prompt(
            "Book a flight from SFO to LAX",
            &[],
            "SFO",
            "What type of variable is this?",
        );

        // Verify all required sections are present
        assert!(prompt.contains("Book a flight from SFO to LAX"));
        assert!(prompt.contains("Observed user input: \"SFO\""));
        assert!(prompt.contains("What type of variable is this?"));
        assert!(prompt.contains("variable_type"));
        assert!(prompt.contains("inferred_name"));
        assert!(prompt.contains("derivation"));
        assert!(prompt.contains("confidence"));
        assert!(prompt.contains("JSON format"));
    }

    #[test]
    fn test_cache_ttl_configuration() {
        let mut synthesizer = LlmSynthesizer::new();
        synthesizer.cache_ttl = 7200; // 2 hours

        let key = synthesizer.generate_cache_key("goal", "input", "question");
        let result = SynthesisResult {
            variable_type: "test".to_string(),
            inferred_name: "test_var".to_string(),
            derivation: "test derivation".to_string(),
            confidence: 0.8,
        };

        synthesizer.cache_result(&key, &result);

        // Should still be valid (TTL is 2 hours)
        let cached = synthesizer.get_cached(&key);
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().variable_type, "test");
    }

    #[test]
    fn test_cache_different_questions_different_keys() {
        let synthesizer = LlmSynthesizer::new();

        let key1 = synthesizer.generate_cache_key("goal", "input", "question1");
        let key2 = synthesizer.generate_cache_key("goal", "input", "question2");

        assert_ne!(key1, key2);
    }

    #[test]
    fn test_synthesis_result_json_serialization() {
        let result = SynthesisResult {
            variable_type: "email_address".to_string(),
            inferred_name: "user_email".to_string(),
            derivation: "Email pattern detected in input".to_string(),
            confidence: 0.92,
        };

        // Test serialization
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("email_address"));
        assert!(json.contains("user_email"));
        assert!(json.contains("0.92"));

        // Test deserialization
        let deserialized: SynthesisResult = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.variable_type, "email_address");
        assert_eq!(deserialized.inferred_name, "user_email");
        assert_eq!(deserialized.confidence, 0.92);
    }

    #[test]
    fn test_synthesizer_configuration_settings() {
        let synthesizer = LlmSynthesizer::with_api_key("custom-key");

        assert_eq!(synthesizer.model, "claude-sonnet-4-5-20250929");
        assert_eq!(synthesizer.temperature, 0.3);
        assert_eq!(synthesizer.cache_ttl, 3600);
        assert_eq!(
            synthesizer.api_endpoint,
            "https://api.anthropic.com/v1/messages"
        );
    }

    #[test]
    fn test_multiple_cache_entries() {
        let mut synthesizer = LlmSynthesizer::new();

        // Add multiple cache entries
        for i in 0..5 {
            let key = synthesizer.generate_cache_key(&format!("goal{}", i), "input", "question");
            let result = SynthesisResult {
                variable_type: format!("type{}", i),
                inferred_name: format!("var{}", i),
                derivation: format!("derivation{}", i),
                confidence: 0.5 + (i as f32 * 0.1),
            };
            synthesizer.cache_result(&key, &result);
        }

        // Verify all are cached
        assert_eq!(synthesizer.cache.len(), 5);

        // Verify each can be retrieved
        for i in 0..5 {
            let key = synthesizer.generate_cache_key(&format!("goal{}", i), "input", "question");
            let cached = synthesizer.get_cached(&key);
            assert!(cached.is_some());
            assert_eq!(cached.unwrap().variable_type, format!("type{}", i));
        }
    }
}
