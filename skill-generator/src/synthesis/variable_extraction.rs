//! Instance Data Detection and Variable Extraction

use crate::capture::types::EnrichedEvent;
use crate::synthesis::llm_semantic_synthesis::LlmSynthesizer;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::LazyLock;
use tracing::{debug, info, warn};

/// Regex for valid variable names: must start with letter/underscore, then alphanumeric/underscore
static VALID_VAR_NAME: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^[a-zA-Z_][a-zA-Z0-9_]*$").expect("valid regex")
});

/// Sanitize a variable name to ensure it's a valid identifier.
/// Replaces invalid characters with underscores and ensures it starts with a letter or underscore.
fn sanitize_variable_name(name: &str) -> String {
    if name.is_empty() {
        return "unnamed_variable".to_string();
    }

    let sanitized: String = name
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '_' { c } else { '_' })
        .collect();

    // Ensure starts with letter or underscore
    let sanitized = if sanitized.chars().next().map(|c| c.is_ascii_digit()).unwrap_or(false) {
        format!("var_{}", sanitized)
    } else {
        sanitized
    };

    // Collapse multiple underscores
    let mut result = String::with_capacity(sanitized.len());
    let mut prev_underscore = false;
    for c in sanitized.chars() {
        if c == '_' {
            if !prev_underscore {
                result.push(c);
            }
            prev_underscore = true;
        } else {
            result.push(c);
            prev_underscore = false;
        }
    }

    // Trim trailing underscore
    let result = result.trim_end_matches('_').to_string();

    if result.is_empty() {
        "unnamed_variable".to_string()
    } else {
        result
    }
}

/// Extracted variable information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedVariable {
    /// Variable name (mustache format)
    pub name: String,
    /// Original value detected
    pub detected_value: String,
    /// Source of detection
    pub source: VariableSource,
    /// Confidence score
    pub confidence: f32,
    /// Suggested type hint
    pub type_hint: Option<String>,
    /// Description for documentation
    pub description: Option<String>,
}

/// Source of variable detection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VariableSource {
    /// Matched from goal text
    GoalMatch,
    /// High-entropy typed string (email, date, etc.)
    TypedString,
    /// Implicit (matches current date, etc.)
    Implicit,
    /// LLM inference
    LlmInferred,
}

/// Configuration for variable extraction
#[derive(Debug, Clone)]
pub struct ExtractionConfig {
    /// Minimum string length to consider as variable
    pub min_string_length: usize,
    /// Entropy threshold for high-entropy detection
    pub entropy_threshold: f64,
    /// Whether to detect implicit variables
    pub detect_implicit: bool,
    /// Whether to use LLM for complex variable inference
    pub use_llm: bool,
    /// Confidence threshold below which to invoke LLM
    pub llm_confidence_threshold: f32,
}

impl Default for ExtractionConfig {
    fn default() -> Self {
        Self {
            min_string_length: 3,
            entropy_threshold: 2.5,
            detect_implicit: true,
            use_llm: true,
            llm_confidence_threshold: 0.7,
        }
    }
}

/// Variable extractor
pub struct VariableExtractor {
    /// Configuration
    pub config: ExtractionConfig,
    /// Known UI labels to exclude
    known_labels: HashSet<String>,
    /// LLM synthesizer for complex inference
    llm_synthesizer: Option<LlmSynthesizer>,
}

impl VariableExtractor {
    /// Create with default config
    pub fn new() -> Self {
        let mut known_labels = HashSet::new();
        // Common UI labels that shouldn't be variables
        known_labels.insert("Submit".to_string());
        known_labels.insert("Cancel".to_string());
        known_labels.insert("OK".to_string());
        known_labels.insert("Save".to_string());
        known_labels.insert("Close".to_string());
        known_labels.insert("Open".to_string());
        known_labels.insert("Delete".to_string());
        known_labels.insert("Edit".to_string());
        known_labels.insert("Add".to_string());
        known_labels.insert("Remove".to_string());

        // Initialize LLM synthesizer if API key is available
        let llm_synthesizer = LlmSynthesizer::new();
        let llm_synthesizer = if llm_synthesizer.is_configured() {
            info!("LLM synthesizer configured for variable extraction");
            Some(llm_synthesizer)
        } else {
            debug!("No ANTHROPIC_API_KEY found, LLM synthesis disabled");
            None
        };

        Self {
            config: ExtractionConfig::default(),
            known_labels,
            llm_synthesizer,
        }
    }

    /// Create with a specific API key
    pub fn with_api_key(api_key: &str) -> Self {
        let mut extractor = Self::new();
        extractor.llm_synthesizer = Some(LlmSynthesizer::with_api_key(api_key));
        extractor
    }

    /// Check if LLM is available
    pub fn has_llm(&self) -> bool {
        self.llm_synthesizer.is_some()
    }

    /// Extract variables from events given a goal
    pub fn extract(&self, events: &[EnrichedEvent], goal: &str) -> Vec<ExtractedVariable> {
        let mut variables = Vec::new();

        // Step 1: Tokenize goal and find matches
        let goal_tokens = self.tokenize_goal(goal);
        variables.extend(self.match_goal_tokens(events, &goal_tokens));

        // Step 2: Find high-entropy typed strings
        variables.extend(self.find_high_entropy_strings(events));

        // Step 3: Detect implicit variables
        if self.config.detect_implicit {
            variables.extend(self.detect_implicit_variables(events));
        }

        // Step 4: Use LLM to enhance low-confidence variables
        let variables = if self.config.use_llm && self.llm_synthesizer.is_some() {
            self.enhance_with_llm(variables, events, goal, None)
        } else {
            variables
        };

        // Deduplicate
        self.deduplicate_variables(variables)
    }

    /// Extract variables with a shared tokio runtime for LLM calls
    pub fn extract_with_runtime(&self, events: &[EnrichedEvent], goal: &str, rt: &tokio::runtime::Runtime) -> Vec<ExtractedVariable> {
        let mut variables = Vec::new();

        let goal_tokens = self.tokenize_goal(goal);
        variables.extend(self.match_goal_tokens(events, &goal_tokens));
        variables.extend(self.find_high_entropy_strings(events));

        if self.config.detect_implicit {
            variables.extend(self.detect_implicit_variables(events));
        }

        let variables = if self.config.use_llm && self.llm_synthesizer.is_some() {
            self.enhance_with_llm(variables, events, goal, Some(rt))
        } else {
            variables
        };

        self.deduplicate_variables(variables)
    }

    /// Enhance variables with LLM inference for low-confidence cases
    fn enhance_with_llm(
        &self,
        mut variables: Vec<ExtractedVariable>,
        events: &[EnrichedEvent],
        goal: &str,
        shared_runtime: Option<&tokio::runtime::Runtime>,
    ) -> Vec<ExtractedVariable> {
        // Verify LLM is available (already checked by caller, but be defensive)
        if self.llm_synthesizer.is_none() {
            return variables;
        }

        // Find typed strings that weren't matched to any variable
        let typed_text = self.collect_typed_text(events);
        let existing_values: HashSet<_> = variables.iter().map(|v| v.detected_value.clone()).collect();

        // Collect unmatched strings for LLM analysis
        let unmatched: Vec<_> = typed_text
            .iter()
            .filter(|(text, _)| !existing_values.contains(text))
            .filter(|(text, _)| !self.known_labels.contains(text))
            .collect();

        if unmatched.is_empty() {
            return variables;
        }

        debug!("Invoking LLM for {} unmatched typed strings", unmatched.len());

        // Use shared runtime if available, otherwise create one
        let owned_rt;
        let rt = if let Some(shared) = shared_runtime {
            shared
        } else {
            owned_rt = match tokio::runtime::Runtime::new() {
                Ok(rt) => rt,
                Err(e) => {
                    debug!("Failed to create tokio runtime for LLM: {}", e);
                    return variables;
                }
            };
            &owned_rt
        };

        // Process each unmatched string with LLM
        for (text, event_indices) in unmatched {
            // Get surrounding events for context
            let start_idx = event_indices.first().copied().unwrap_or(0);
            let context_start = start_idx.saturating_sub(5);
            let context_end = (start_idx + 5).min(events.len());
            let context_events: Vec<_> = events[context_start..context_end].to_vec();

            let question = format!(
                "What type of data is '{}' in the context of the goal '{}' and what should it be named as a variable?",
                text, goal
            );

            // Clone synthesizer for async block
            let mut synth_clone = LlmSynthesizer::new();
            if let Some(ref s) = self.llm_synthesizer {
                if s.is_configured() {
                    // Re-create with same config (API key from env)
                    synth_clone = LlmSynthesizer::new();
                }
            }

            let result = rt.block_on(async {
                synth_clone.synthesize(goal, &context_events, text, &question).await
            });

            if let Some(synthesis) = result {
                let var_name = sanitize_variable_name(&synthesis.inferred_name);
                if !VALID_VAR_NAME.is_match(&var_name) {
                    warn!(raw_name = synthesis.inferred_name, "LLM returned invalid variable name, skipping");
                    continue;
                }

                debug!(
                    "LLM inferred variable: {} = {} (confidence: {})",
                    var_name, text, synthesis.confidence
                );

                variables.push(ExtractedVariable {
                    name: var_name,
                    detected_value: text.clone(),
                    source: VariableSource::LlmInferred,
                    confidence: synthesis.confidence,
                    type_hint: Some(synthesis.variable_type),
                    description: Some(synthesis.derivation),
                });
            }
        }

        variables
    }

    /// Tokenize goal into meaningful tokens
    fn tokenize_goal(&self, goal: &str) -> Vec<String> {
        goal.split_whitespace()
            .filter(|s| s.len() >= 3)
            .filter(|s| !self.is_stop_word(s))
            .map(|s| s.to_lowercase())
            .collect()
    }

    /// Check if word is a stop word
    fn is_stop_word(&self, word: &str) -> bool {
        let stop_words = ["the", "a", "an", "for", "to", "in", "on", "with", "and", "or"];
        stop_words.contains(&word.to_lowercase().as_str())
    }

    /// Match goal tokens in typed text
    fn match_goal_tokens(&self, events: &[EnrichedEvent], tokens: &[String]) -> Vec<ExtractedVariable> {
        let mut variables = Vec::new();

        // Collect all typed text from events
        let typed_text = self.collect_typed_text(events);

        for (text, _event_indices) in typed_text {
            let text_lower = text.to_lowercase();

            // Check if this text relates to any goal token
            for token in tokens {
                if text_lower.contains(token) || self.is_semantic_match(&text, token) {
                    let var_name = self.generate_variable_name(token, &text);

                    variables.push(ExtractedVariable {
                        name: var_name,
                        detected_value: text.clone(),
                        source: VariableSource::GoalMatch,
                        confidence: 0.9,
                        type_hint: self.infer_type(&text),
                        description: Some(format!("Value for '{}'", token)),
                    });
                }
            }
        }

        variables
    }

    /// Collect typed text from events
    fn collect_typed_text(&self, events: &[EnrichedEvent]) -> Vec<(String, Vec<usize>)> {
        let mut result = Vec::new();
        let mut current_text = String::new();
        let mut current_indices = Vec::new();

        for (i, event) in events.iter().enumerate() {
            if event.raw.event_type.is_keyboard() {
                if let Some(ch) = event.raw.character {
                    current_text.push(ch);
                    current_indices.push(i);
                }
            } else if !current_text.is_empty() {
                // End of typing sequence
                if current_text.len() >= self.config.min_string_length {
                    result.push((current_text.clone(), current_indices.clone()));
                }
                current_text.clear();
                current_indices.clear();
            }
        }

        // Don't forget the last sequence
        if current_text.len() >= self.config.min_string_length {
            result.push((current_text, current_indices));
        }

        result
    }

    /// Check for semantic match (email, date, URL, etc.)
    fn is_semantic_match(&self, text: &str, token: &str) -> bool {
        match token {
            "email" => self.looks_like_email(text),
            "date" => self.looks_like_date(text),
            "url" | "link" => self.looks_like_url(text),
            "phone" => self.looks_like_phone(text),
            "name" => text.chars().all(|c| c.is_alphabetic() || c.is_whitespace()),
            _ => false,
        }
    }

    /// Find high-entropy strings (likely user data, not UI labels)
    fn find_high_entropy_strings(&self, events: &[EnrichedEvent]) -> Vec<ExtractedVariable> {
        let mut variables = Vec::new();
        let typed_text = self.collect_typed_text(events);

        for (text, _) in typed_text {
            // Skip known UI labels
            if self.known_labels.contains(&text) {
                continue;
            }

            let entropy = self.calculate_entropy(&text);

            if entropy > self.config.entropy_threshold {
                let var_name = self.infer_variable_name(&text);

                variables.push(ExtractedVariable {
                    name: var_name,
                    detected_value: text.clone(),
                    source: VariableSource::TypedString,
                    confidence: (entropy / 4.0).min(1.0) as f32,
                    type_hint: self.infer_type(&text),
                    description: None,
                });
            }
        }

        variables
    }

    /// Detect implicit variables (current date, etc.)
    fn detect_implicit_variables(&self, events: &[EnrichedEvent]) -> Vec<ExtractedVariable> {
        let mut variables = Vec::new();
        let typed_text = self.collect_typed_text(events);

        let today_variants = [chrono::Local::now().format("%Y-%m-%d").to_string(),
            chrono::Local::now().format("%m/%d/%Y").to_string(),
            chrono::Local::now().format("%d/%m/%Y").to_string()];

        for (text, _) in typed_text {
            // Check for current date
            if today_variants.iter().any(|d| text.contains(d)) {
                variables.push(ExtractedVariable {
                    name: "current_date".to_string(),
                    detected_value: text.clone(),
                    source: VariableSource::Implicit,
                    confidence: 0.95,
                    type_hint: Some("date".to_string()),
                    description: Some("Current date at time of recording".to_string()),
                });
            }
        }

        variables
    }

    /// Calculate Shannon entropy of a string
    fn calculate_entropy(&self, text: &str) -> f64 {
        if text.is_empty() {
            return 0.0;
        }

        let mut char_counts = std::collections::HashMap::new();
        for ch in text.chars() {
            *char_counts.entry(ch).or_insert(0) += 1;
        }

        let len = text.len() as f64;
        char_counts
            .values()
            .map(|&count| {
                let p = count as f64 / len;
                -p * p.log2()
            })
            .sum()
    }

    /// Generate variable name from token and value
    fn generate_variable_name(&self, token: &str, value: &str) -> String {
        // Try to create a meaningful name
        if self.looks_like_email(value) {
            return "email_address".to_string();
        }
        if self.looks_like_date(value) {
            return "date".to_string();
        }
        if self.looks_like_url(value) {
            return "url".to_string();
        }

        // Use token as base
        token.replace(' ', "_").to_lowercase()
    }

    /// Infer variable name from value
    fn infer_variable_name(&self, value: &str) -> String {
        if self.looks_like_email(value) {
            return "email_address".to_string();
        }
        if self.looks_like_date(value) {
            return "date_value".to_string();
        }
        if self.looks_like_url(value) {
            return "url".to_string();
        }
        if self.looks_like_phone(value) {
            return "phone_number".to_string();
        }

        "user_input".to_string()
    }

    /// Infer type from value
    fn infer_type(&self, value: &str) -> Option<String> {
        if self.looks_like_email(value) {
            Some("email".to_string())
        } else if self.looks_like_date(value) {
            Some("date".to_string())
        } else if self.looks_like_url(value) {
            Some("url".to_string())
        } else if self.looks_like_phone(value) {
            Some("phone".to_string())
        } else if value.parse::<i64>().is_ok() {
            Some("integer".to_string())
        } else if value.parse::<f64>().is_ok() {
            Some("number".to_string())
        } else {
            Some("string".to_string())
        }
    }

    /// Check if value looks like an email
    fn looks_like_email(&self, value: &str) -> bool {
        value.contains('@') && value.contains('.')
    }

    /// Check if value looks like a date
    fn looks_like_date(&self, value: &str) -> bool {
        // Date formats: YYYY-MM-DD, MM/DD/YYYY, DD/MM/YYYY, etc.
        let has_separator = value.contains('-') || value.contains('/');
        let digit_count = value.chars().filter(|c| c.is_ascii_digit()).count();
        let parts: Vec<&str> = value.split(['-', '/']).collect();

        // Must have separator and exactly 4, 6, or 8 digits (common date formats)
        // Also require 2-3 parts (e.g., YYYY-MM-DD has 3 parts)
        has_separator
            && (digit_count == 4 || digit_count == 6 || digit_count == 8)
            && (parts.len() == 2 || parts.len() == 3)
    }

    /// Check if value looks like a URL
    fn looks_like_url(&self, value: &str) -> bool {
        value.starts_with("http://") || value.starts_with("https://") || value.contains("www.")
    }

    /// Check if value looks like a phone number
    fn looks_like_phone(&self, value: &str) -> bool {
        let digit_count = value.chars().filter(|c| c.is_ascii_digit()).count();
        (7..=15).contains(&digit_count)
    }

    /// Deduplicate variables by name
    fn deduplicate_variables(&self, variables: Vec<ExtractedVariable>) -> Vec<ExtractedVariable> {
        let mut seen = HashSet::new();
        variables
            .into_iter()
            .filter(|v| seen.insert(v.name.clone()))
            .collect()
    }
}

impl Default for VariableExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capture::types::{RawEvent, EventType, CursorState, ModifierFlags, EnrichedEvent};
    use crate::time::timebase::Timestamp;

    #[test]
    fn test_entropy_calculation() {
        let extractor = VariableExtractor::new();

        // Low entropy (repeated character)
        let low = extractor.calculate_entropy("aaaaaaa");
        assert!(low < 0.1);

        // High entropy (varied characters)
        let high = extractor.calculate_entropy("abc123XYZ");
        assert!(high > 2.0);
    }

    #[test]
    fn test_entropy_empty_string() {
        let extractor = VariableExtractor::new();
        let entropy = extractor.calculate_entropy("");
        assert_eq!(entropy, 0.0);
    }

    #[test]
    fn test_entropy_special_characters() {
        let extractor = VariableExtractor::new();

        // Special characters should increase entropy
        let special = extractor.calculate_entropy("!@#$%^&*()");
        assert!(special > 2.5);

        // Mixed alphanumeric with symbols
        let mixed = extractor.calculate_entropy("P@ssw0rd!");
        assert!(mixed > 2.5);
    }

    #[test]
    fn test_type_inference() {
        let extractor = VariableExtractor::new();

        assert_eq!(extractor.infer_type("test@example.com"), Some("email".to_string()));
        assert_eq!(extractor.infer_type("2026-01-26"), Some("date".to_string()));
        assert_eq!(extractor.infer_type("https://example.com"), Some("url".to_string()));
    }

    #[test]
    fn test_type_inference_edge_cases() {
        let extractor = VariableExtractor::new();

        // Integer
        assert_eq!(extractor.infer_type("42"), Some("integer".to_string()));
        assert_eq!(extractor.infer_type("-123"), Some("integer".to_string()));

        // Float
        assert_eq!(extractor.infer_type("3.14159"), Some("number".to_string()));
        assert_eq!(extractor.infer_type("-0.5"), Some("number".to_string()));

        // Phone number
        assert_eq!(extractor.infer_type("555-123-4567"), Some("phone".to_string()));
        assert_eq!(extractor.infer_type("(555) 123-4567"), Some("phone".to_string()));

        // Regular string
        assert_eq!(extractor.infer_type("Hello World"), Some("string".to_string()));
    }

    #[test]
    fn test_known_labels() {
        let extractor = VariableExtractor::new();
        assert!(extractor.known_labels.contains("Submit"));
        assert!(extractor.known_labels.contains("Cancel"));
    }

    #[test]
    fn test_looks_like_email_edge_cases() {
        let extractor = VariableExtractor::new();

        // Valid emails
        assert!(extractor.looks_like_email("user@example.com"));
        assert!(extractor.looks_like_email("first.last@company.co.uk"));
        assert!(extractor.looks_like_email("test+tag@gmail.com"));

        // Invalid emails (but detected by simple heuristic)
        assert!(extractor.looks_like_email("no@domain."));

        // Not emails
        assert!(!extractor.looks_like_email("notanemail"));
        assert!(!extractor.looks_like_email("no@symbol"));
        assert!(!extractor.looks_like_email("nodot@"));
    }

    #[test]
    fn test_looks_like_date_various_formats() {
        let extractor = VariableExtractor::new();

        // ISO format
        assert!(extractor.looks_like_date("2026-01-26"));
        assert!(extractor.looks_like_date("2024-12-31"));

        // US format
        assert!(extractor.looks_like_date("01/26/2026"));
        assert!(extractor.looks_like_date("12/31/2024"));

        // EU format
        assert!(extractor.looks_like_date("26/01/2026"));

        // Short date
        assert!(extractor.looks_like_date("1/1/2026"));

        // Not dates
        assert!(!extractor.looks_like_date("123456789"));
        assert!(!extractor.looks_like_date("no-numbers-here"));
        assert!(!extractor.looks_like_date("12"));
    }

    #[test]
    fn test_looks_like_url_patterns() {
        let extractor = VariableExtractor::new();

        // HTTP/HTTPS URLs
        assert!(extractor.looks_like_url("http://example.com"));
        assert!(extractor.looks_like_url("https://example.com"));
        assert!(extractor.looks_like_url("https://www.example.com/path?query=1"));

        // WWW prefix
        assert!(extractor.looks_like_url("www.example.com"));

        // Not URLs
        assert!(!extractor.looks_like_url("example.com"));
        assert!(!extractor.looks_like_url("justtext"));
        assert!(!extractor.looks_like_url("ftp://old.protocol.com"));
    }

    #[test]
    fn test_looks_like_phone_number_variations() {
        let extractor = VariableExtractor::new();

        // Valid phone numbers (7-15 digits)
        assert!(extractor.looks_like_phone("555-1234"));
        assert!(extractor.looks_like_phone("(555) 123-4567"));
        assert!(extractor.looks_like_phone("+1-555-123-4567"));
        assert!(extractor.looks_like_phone("1234567"));
        assert!(extractor.looks_like_phone("+44 20 7123 4567"));

        // Invalid (too few digits)
        assert!(!extractor.looks_like_phone("123-456"));

        // Invalid (too many digits)
        assert!(!extractor.looks_like_phone("1234567890123456"));

        // Not phone numbers
        assert!(!extractor.looks_like_phone("no digits here"));
    }

    #[test]
    fn test_tokenize_goal() {
        let extractor = VariableExtractor::new();

        let tokens = extractor.tokenize_goal("Send an email to support@example.com");
        assert!(tokens.contains(&"send".to_string()));
        assert!(tokens.contains(&"email".to_string()));
        assert!(tokens.contains(&"support@example.com".to_string()));
        assert!(!tokens.contains(&"an".to_string())); // Stop word
        assert!(!tokens.contains(&"to".to_string())); // Stop word
    }

    #[test]
    fn test_tokenize_goal_short_words() {
        let extractor = VariableExtractor::new();

        // Words shorter than 3 characters should be filtered
        let tokens = extractor.tokenize_goal("Go to my home");
        assert!(!tokens.contains(&"go".to_string()));
        assert!(!tokens.contains(&"my".to_string()));
        assert!(tokens.contains(&"home".to_string()));
    }

    #[test]
    fn test_is_stop_word() {
        let extractor = VariableExtractor::new();

        assert!(extractor.is_stop_word("the"));
        assert!(extractor.is_stop_word("a"));
        assert!(extractor.is_stop_word("an"));
        assert!(extractor.is_stop_word("for"));
        assert!(extractor.is_stop_word("THE")); // Case insensitive

        assert!(!extractor.is_stop_word("email"));
        assert!(!extractor.is_stop_word("send"));
    }

    #[test]
    fn test_generate_variable_name() {
        let extractor = VariableExtractor::new();

        // Email detection
        assert_eq!(
            extractor.generate_variable_name("email", "user@example.com"),
            "email_address"
        );

        // Date detection
        assert_eq!(
            extractor.generate_variable_name("date", "2026-01-26"),
            "date"
        );

        // URL detection
        assert_eq!(
            extractor.generate_variable_name("link", "https://example.com"),
            "url"
        );

        // Generic token
        assert_eq!(
            extractor.generate_variable_name("user name", "John Doe"),
            "user_name"
        );
    }

    #[test]
    fn test_infer_variable_name() {
        let extractor = VariableExtractor::new();

        assert_eq!(extractor.infer_variable_name("test@example.com"), "email_address");
        assert_eq!(extractor.infer_variable_name("2026-01-26"), "date_value");
        assert_eq!(extractor.infer_variable_name("https://example.com"), "url");
        assert_eq!(extractor.infer_variable_name("555-1234"), "phone_number");
        assert_eq!(extractor.infer_variable_name("random text"), "user_input");
    }

    #[test]
    fn test_collect_typed_text() {
        let extractor = VariableExtractor::new();

        // Create keyboard events
        let mut events = Vec::new();
        let chars = ['h', 'e', 'l', 'l', 'o'];

        for (i, ch) in chars.iter().enumerate() {
            events.push(EnrichedEvent {
                raw: RawEvent::keyboard(
                    Timestamp::from_ticks(i as u64),
                    EventType::KeyDown,
                    0,
                    Some(*ch),
                    ModifierFlags::default(),
                    (0.0, 0.0),
                ),
                semantic: None,
                id: uuid::Uuid::new_v4(),
                sequence: i as u64,
                screenshot_filename: None,
            });
        }

        let typed_text = extractor.collect_typed_text(&events);
        assert_eq!(typed_text.len(), 1);
        assert_eq!(typed_text[0].0, "hello");
    }

    #[test]
    fn test_collect_typed_text_with_interruption() {
        let extractor = VariableExtractor::new();

        let mut events = Vec::new();

        // Type "hello"
        for (i, ch) in "hello".chars().enumerate() {
            events.push(EnrichedEvent {
                raw: RawEvent::keyboard(
                    Timestamp::from_ticks(i as u64),
                    EventType::KeyDown,
                    0,
                    Some(ch),
                    ModifierFlags::default(),
                    (0.0, 0.0),
                ),
                semantic: None,
                id: uuid::Uuid::new_v4(),
                sequence: i as u64,
                screenshot_filename: None,
            });
        }

        // Mouse click (interruption)
        events.push(EnrichedEvent {
            raw: RawEvent::mouse(
                Timestamp::from_ticks(5),
                EventType::LeftMouseDown,
                100.0,
                100.0,
                CursorState::Arrow,
                ModifierFlags::default(),
                1,
            ),
            semantic: None,
            id: uuid::Uuid::new_v4(),
            sequence: 5,
            screenshot_filename: None,
        });

        // Type "world"
        for (i, ch) in "world".chars().enumerate() {
            events.push(EnrichedEvent {
                raw: RawEvent::keyboard(
                    Timestamp::from_ticks((6 + i) as u64),
                    EventType::KeyDown,
                    0,
                    Some(ch),
                    ModifierFlags::default(),
                    (0.0, 0.0),
                ),
                semantic: None,
                id: uuid::Uuid::new_v4(),
                sequence: (6 + i) as u64,
                screenshot_filename: None,
            });
        }

        let typed_text = extractor.collect_typed_text(&events);
        assert_eq!(typed_text.len(), 2);
        assert_eq!(typed_text[0].0, "hello");
        assert_eq!(typed_text[1].0, "world");
    }

    #[test]
    fn test_collect_typed_text_minimum_length() {
        let extractor = VariableExtractor::new();

        let mut events = Vec::new();

        // Type "hi" (too short, default min is 3)
        for (i, ch) in "hi".chars().enumerate() {
            events.push(EnrichedEvent {
                raw: RawEvent::keyboard(
                    Timestamp::from_ticks(i as u64),
                    EventType::KeyDown,
                    0,
                    Some(ch),
                    ModifierFlags::default(),
                    (0.0, 0.0),
                ),
                semantic: None,
                id: uuid::Uuid::new_v4(),
                sequence: i as u64,
                screenshot_filename: None,
            });
        }

        let typed_text = extractor.collect_typed_text(&events);
        assert_eq!(typed_text.len(), 0); // Should be filtered out
    }

    #[test]
    fn test_deduplicate_variables() {
        let extractor = VariableExtractor::new();

        let variables = vec![
            ExtractedVariable {
                name: "email_address".to_string(),
                detected_value: "user1@example.com".to_string(),
                source: VariableSource::TypedString,
                confidence: 0.9,
                type_hint: Some("email".to_string()),
                description: None,
            },
            ExtractedVariable {
                name: "email_address".to_string(),
                detected_value: "user2@example.com".to_string(),
                source: VariableSource::GoalMatch,
                confidence: 0.95,
                type_hint: Some("email".to_string()),
                description: None,
            },
            ExtractedVariable {
                name: "date".to_string(),
                detected_value: "2026-01-26".to_string(),
                source: VariableSource::TypedString,
                confidence: 0.8,
                type_hint: Some("date".to_string()),
                description: None,
            },
        ];

        let deduplicated = extractor.deduplicate_variables(variables);
        assert_eq!(deduplicated.len(), 2);

        // Should keep first occurrence of each name
        assert_eq!(deduplicated[0].name, "email_address");
        assert_eq!(deduplicated[0].detected_value, "user1@example.com");
        assert_eq!(deduplicated[1].name, "date");
    }

    #[test]
    fn test_sanitize_variable_name_valid() {
        assert_eq!(sanitize_variable_name("email_address"), "email_address");
        assert_eq!(sanitize_variable_name("user_input"), "user_input");
        assert_eq!(sanitize_variable_name("_private"), "_private");
    }

    #[test]
    fn test_sanitize_variable_name_special_chars() {
        assert_eq!(sanitize_variable_name("user-name"), "user_name");
        assert_eq!(sanitize_variable_name("foo.bar"), "foo_bar");
        assert_eq!(sanitize_variable_name("hello world"), "hello_world");
    }

    #[test]
    fn test_sanitize_variable_name_starts_with_digit() {
        assert_eq!(sanitize_variable_name("123abc"), "var_123abc");
    }

    #[test]
    fn test_sanitize_variable_name_empty() {
        assert_eq!(sanitize_variable_name(""), "unnamed_variable");
    }

    #[test]
    fn test_sanitize_variable_name_collapses_underscores() {
        assert_eq!(sanitize_variable_name("foo---bar"), "foo_bar");
        assert_eq!(sanitize_variable_name("a__b__c"), "a_b_c");
    }

    #[test]
    fn test_valid_var_name_regex() {
        assert!(VALID_VAR_NAME.is_match("email_address"));
        assert!(VALID_VAR_NAME.is_match("_private"));
        assert!(VALID_VAR_NAME.is_match("X"));
        assert!(!VALID_VAR_NAME.is_match("123"));
        assert!(!VALID_VAR_NAME.is_match(""));
        assert!(!VALID_VAR_NAME.is_match("foo bar"));
    }

    #[test]
    fn test_sanitize_variable_name_emoji() {
        // Pure emoji: each emoji is_alphanumeric() but not ASCII, so they pass through
        // sanitize doesn't strip them, but VALID_VAR_NAME (ASCII-only regex) won't match.
        // This documents current behavior — emoji chars survive sanitization.
        let result = sanitize_variable_name("🎉🎊🎈");
        assert!(!result.is_empty());

        // Mixed emoji and ASCII: emoji survives, ASCII part is intact
        let result = sanitize_variable_name("😀button");
        assert!(result.contains("button"));
    }

    #[test]
    fn test_sanitize_variable_name_pure_underscores() {
        // All underscores collapse to one, then trailing trim leaves empty
        assert_eq!(sanitize_variable_name("___"), "unnamed_variable");
        assert_eq!(sanitize_variable_name("_"), "unnamed_variable");
    }

    #[test]
    fn test_sanitize_variable_name_very_long() {
        let long_name = "a".repeat(10_000);
        let result = sanitize_variable_name(&long_name);
        assert!(VALID_VAR_NAME.is_match(&result));
        assert_eq!(result.len(), 10_000);
    }

    #[test]
    fn test_sanitize_variable_name_only_digits() {
        // Starts with digit → prefixed, all valid chars kept
        let result = sanitize_variable_name("12345");
        assert!(VALID_VAR_NAME.is_match(&result));
        assert!(result.starts_with("var_"));
    }

    #[test]
    fn test_extracted_variable_serde_roundtrip() {
        let var = ExtractedVariable {
            name: "email_address".to_string(),
            detected_value: "user@example.com".to_string(),
            source: VariableSource::GoalMatch,
            confidence: 0.92,
            type_hint: Some("email".to_string()),
            description: Some("User email from goal".to_string()),
        };

        let json = serde_json::to_string(&var).unwrap();
        let roundtripped: ExtractedVariable = serde_json::from_str(&json).unwrap();

        assert_eq!(roundtripped.name, var.name);
        assert_eq!(roundtripped.detected_value, var.detected_value);
        assert_eq!(roundtripped.source, var.source);
        assert!((roundtripped.confidence - var.confidence).abs() < 0.001);
        assert_eq!(roundtripped.type_hint, var.type_hint);
        assert_eq!(roundtripped.description, var.description);
    }

    #[test]
    fn test_variable_source_serde_roundtrip() {
        for source in [VariableSource::GoalMatch, VariableSource::TypedString, VariableSource::Implicit, VariableSource::LlmInferred] {
            let json = serde_json::to_string(&source).unwrap();
            let roundtripped: VariableSource = serde_json::from_str(&json).unwrap();
            assert_eq!(roundtripped, source);
        }
    }

    #[test]
    fn test_semantic_match() {
        let extractor = VariableExtractor::new();

        // Email
        assert!(extractor.is_semantic_match("user@example.com", "email"));
        assert!(!extractor.is_semantic_match("not an email", "email"));

        // Date
        assert!(extractor.is_semantic_match("2026-01-26", "date"));
        assert!(!extractor.is_semantic_match("not a date", "date"));

        // URL
        assert!(extractor.is_semantic_match("https://example.com", "url"));
        assert!(extractor.is_semantic_match("www.example.com", "link"));
        assert!(!extractor.is_semantic_match("not a url", "url"));

        // Phone
        assert!(extractor.is_semantic_match("555-1234", "phone"));
        assert!(!extractor.is_semantic_match("123", "phone"));

        // Name (alphabetic only)
        assert!(extractor.is_semantic_match("John Doe", "name"));
        assert!(!extractor.is_semantic_match("John123", "name"));
    }
}
