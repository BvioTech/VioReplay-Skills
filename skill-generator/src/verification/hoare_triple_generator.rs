//! Formal Postcondition Logic using Hoare Triples

use super::postcondition_extractor::Postcondition;
use serde::{Deserialize, Serialize};

/// Verification condition
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Condition {
    /// Scope of the check
    pub scope: String,
    /// Attribute to check
    pub attribute: String,
    /// Comparison operator
    pub operator: String,
    /// Expected value
    pub expected_value: String,
}

/// Verification block for SKILL.md
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerificationBlock {
    /// Type of verification
    pub verification_type: String,
    /// Timeout in milliseconds
    pub timeout_ms: u64,
    /// List of conditions
    pub conditions: Vec<Condition>,
}

/// Hoare Triple representation
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HoareTriple {
    /// Precondition (P)
    pub precondition: Vec<Condition>,
    /// Command (C)
    pub command: String,
    /// Postcondition (Q)
    pub postcondition: Vec<Condition>,
}

/// Default verification timeout in milliseconds
const DEFAULT_VERIFICATION_TIMEOUT_MS: u64 = 5000;

/// Hoare triple generator
pub struct HoareTripleGenerator {
    /// Default timeout for verifications
    pub default_timeout_ms: u64,
}

impl HoareTripleGenerator {
    /// Create with default settings
    pub fn new() -> Self {
        Self {
            default_timeout_ms: DEFAULT_VERIFICATION_TIMEOUT_MS,
        }
    }

    /// Generate Hoare triple from action and postconditions
    pub fn generate(
        &self,
        action_description: &str,
        postconditions: &[Postcondition],
        _observed_latency_ms: Option<u64>,
    ) -> HoareTriple {
        let pre = self.infer_preconditions(postconditions);
        let post = self.convert_postconditions(postconditions);

        HoareTriple {
            precondition: pre,
            command: action_description.to_string(),
            postcondition: post,
        }
    }

    /// Generate verification block for SKILL.md
    pub fn generate_verification_block(
        &self,
        postconditions: &[Postcondition],
        observed_latency_ms: Option<u64>,
    ) -> VerificationBlock {
        let timeout = observed_latency_ms
            .map(|l| (l as f64 * 1.5) as u64) // Add 50% buffer
            .unwrap_or(self.default_timeout_ms)
            .max(1000); // Minimum 1 second

        let conditions = self.convert_postconditions(postconditions);

        VerificationBlock {
            verification_type: "state_assertion".to_string(),
            timeout_ms: timeout,
            conditions,
        }
    }

    /// Infer preconditions from postconditions
    fn infer_preconditions(&self, postconditions: &[Postcondition]) -> Vec<Condition> {
        postconditions
            .iter()
            .filter_map(|p| {
                p.pre_value.as_ref().map(|pre| Condition {
                    scope: "target_element".to_string(),
                    attribute: p.attribute.clone(),
                    operator: "equals".to_string(),
                    expected_value: pre.clone(),
                })
            })
            .collect()
    }

    /// Convert postconditions to verification conditions
    fn convert_postconditions(&self, postconditions: &[Postcondition]) -> Vec<Condition> {
        postconditions
            .iter()
            .map(|p| {
                let scope = match p.attribute.as_str() {
                    "WindowTitle" => "window",
                    "CursorState" => "system",
                    _ => "target_element",
                };

                let operator = if p.attribute == "WindowTitle" {
                    "contains"
                } else {
                    "equals"
                };

                Condition {
                    scope: scope.to_string(),
                    attribute: p.attribute.clone(),
                    operator: operator.to_string(),
                    expected_value: p.expected_value.clone(),
                }
            })
            .collect()
    }

    /// Format Hoare triple as string
    pub fn format_triple(&self, triple: &HoareTriple) -> String {
        let pre_str = if triple.precondition.is_empty() {
            "true".to_string()
        } else {
            triple
                .precondition
                .iter()
                .map(|c| format!("{}.{} {} '{}'", c.scope, c.attribute, c.operator, c.expected_value))
                .collect::<Vec<_>>()
                .join(" && ")
        };

        let post_str = if triple.postcondition.is_empty() {
            "true".to_string()
        } else {
            triple
                .postcondition
                .iter()
                .map(|c| format!("{}.{} {} '{}'", c.scope, c.attribute, c.operator, c.expected_value))
                .collect::<Vec<_>>()
                .join(" && ")
        };

        format!("{{ {} }} {} {{ {} }}", pre_str, triple.command, post_str)
    }
}

impl Default for HoareTripleGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::verification::postcondition_extractor::SignalType;

    #[test]
    fn test_generator_creation() {
        let generator = HoareTripleGenerator::new();
        assert!(generator.default_timeout_ms > 0);
    }

    #[test]
    fn test_verification_block_generation() {
        let generator = HoareTripleGenerator::new();

        let postconditions = vec![Postcondition {
            signal_type: SignalType::AxRole,
            attribute: "AXRole".to_string(),
            expected_value: "AXButton".to_string(),
            pre_value: None,
            stability: 1.0,
        }];

        let block = generator.generate_verification_block(&postconditions, Some(2000));

        assert_eq!(block.verification_type, "state_assertion");
        assert!(block.timeout_ms >= 2000);
        assert!(!block.conditions.is_empty());
    }

    #[test]
    fn test_triple_formatting() {
        let generator = HoareTripleGenerator::new();

        let triple = HoareTriple {
            precondition: vec![],
            command: "Click Submit".to_string(),
            postcondition: vec![Condition {
                scope: "window".to_string(),
                attribute: "title".to_string(),
                operator: "contains".to_string(),
                expected_value: "Saved".to_string(),
            }],
        };

        let formatted = generator.format_triple(&triple);
        assert!(formatted.contains("Click Submit"));
        assert!(formatted.contains("Saved"));
    }

    #[test]
    fn test_verification_block_with_no_latency() {
        let generator = HoareTripleGenerator::new();

        let postconditions = vec![Postcondition {
            signal_type: SignalType::AxRole,
            attribute: "AXValue".to_string(),
            expected_value: "Test Value".to_string(),
            pre_value: None,
            stability: 1.0,
        }];

        let block = generator.generate_verification_block(&postconditions, None);

        assert_eq!(block.verification_type, "state_assertion");
        assert_eq!(block.timeout_ms, generator.default_timeout_ms);
        assert_eq!(block.conditions.len(), 1);
    }

    #[test]
    fn test_verification_block_minimum_timeout() {
        let generator = HoareTripleGenerator::new();

        let postconditions = vec![Postcondition {
            signal_type: SignalType::AxRole,
            attribute: "AXRole".to_string(),
            expected_value: "AXButton".to_string(),
            pre_value: None,
            stability: 1.0,
        }];

        // Very low latency should still result in minimum 1000ms timeout
        let block = generator.generate_verification_block(&postconditions, Some(100));

        assert!(block.timeout_ms >= 1000);
    }

    #[test]
    fn test_verification_block_timeout_buffer() {
        let generator = HoareTripleGenerator::new();

        let postconditions = vec![Postcondition {
            signal_type: SignalType::AxRole,
            attribute: "AXRole".to_string(),
            expected_value: "AXButton".to_string(),
            pre_value: None,
            stability: 1.0,
        }];

        let latency = 2000;
        let block = generator.generate_verification_block(&postconditions, Some(latency));

        // Should add 50% buffer: 2000 * 1.5 = 3000
        assert_eq!(block.timeout_ms, 3000);
    }

    #[test]
    fn test_condition_creation_for_window_title() {
        let generator = HoareTripleGenerator::new();

        let postconditions = vec![Postcondition {
            signal_type: SignalType::WindowTitle,
            attribute: "WindowTitle".to_string(),
            expected_value: "My Window".to_string(),
            pre_value: None,
            stability: 1.0,
        }];

        let conditions = generator.convert_postconditions(&postconditions);

        assert_eq!(conditions.len(), 1);
        assert_eq!(conditions[0].scope, "window");
        assert_eq!(conditions[0].attribute, "WindowTitle");
        assert_eq!(conditions[0].operator, "contains");
        assert_eq!(conditions[0].expected_value, "My Window");
    }

    #[test]
    fn test_condition_creation_for_cursor_state() {
        let generator = HoareTripleGenerator::new();

        let postconditions = vec![Postcondition {
            signal_type: SignalType::CursorShape,
            attribute: "CursorState".to_string(),
            expected_value: "pointer".to_string(),
            pre_value: None,
            stability: 1.0,
        }];

        let conditions = generator.convert_postconditions(&postconditions);

        assert_eq!(conditions.len(), 1);
        assert_eq!(conditions[0].scope, "system");
        assert_eq!(conditions[0].attribute, "CursorState");
        assert_eq!(conditions[0].operator, "equals");
        assert_eq!(conditions[0].expected_value, "pointer");
    }

    #[test]
    fn test_condition_creation_for_target_element() {
        let generator = HoareTripleGenerator::new();

        let postconditions = vec![Postcondition {
            signal_type: SignalType::AxRole,
            attribute: "AXRole".to_string(),
            expected_value: "AXTextField".to_string(),
            pre_value: None,
            stability: 1.0,
        }];

        let conditions = generator.convert_postconditions(&postconditions);

        assert_eq!(conditions.len(), 1);
        assert_eq!(conditions[0].scope, "target_element");
        assert_eq!(conditions[0].attribute, "AXRole");
        assert_eq!(conditions[0].operator, "equals");
        assert_eq!(conditions[0].expected_value, "AXTextField");
    }

    #[test]
    fn test_empty_postconditions() {
        let generator = HoareTripleGenerator::new();

        let postconditions: Vec<Postcondition> = vec![];

        let block = generator.generate_verification_block(&postconditions, Some(2000));

        assert_eq!(block.conditions.len(), 0);
        assert_eq!(block.verification_type, "state_assertion");
    }

    #[test]
    fn test_multiple_postconditions() {
        let generator = HoareTripleGenerator::new();

        let postconditions = vec![
            Postcondition {
                signal_type: SignalType::AxRole,
                attribute: "AXRole".to_string(),
                expected_value: "AXButton".to_string(),
                pre_value: None,
                stability: 1.0,
            },
            Postcondition {
                signal_type: SignalType::WindowTitle,
                attribute: "WindowTitle".to_string(),
                expected_value: "Success".to_string(),
                pre_value: None,
                stability: 0.9,
            },
            Postcondition {
                signal_type: SignalType::CursorShape,
                attribute: "CursorState".to_string(),
                expected_value: "default".to_string(),
                pre_value: None,
                stability: 0.8,
            },
        ];

        let block = generator.generate_verification_block(&postconditions, Some(1500));

        assert_eq!(block.conditions.len(), 3);
        assert_eq!(block.conditions[0].scope, "target_element");
        assert_eq!(block.conditions[1].scope, "window");
        assert_eq!(block.conditions[2].scope, "system");
    }

    #[test]
    fn test_special_characters_in_values() {
        let generator = HoareTripleGenerator::new();

        let postconditions = vec![Postcondition {
            signal_type: SignalType::AxRole,
            attribute: "AXValue".to_string(),
            expected_value: "Test's \"quoted\" & special <chars>".to_string(),
            pre_value: None,
            stability: 1.0,
        }];

        let conditions = generator.convert_postconditions(&postconditions);

        assert_eq!(conditions.len(), 1);
        assert_eq!(conditions[0].expected_value, "Test's \"quoted\" & special <chars>");
    }

    #[test]
    fn test_long_string_values() {
        let generator = HoareTripleGenerator::new();

        let long_value = "A".repeat(1000);
        let postconditions = vec![Postcondition {
            signal_type: SignalType::ElementAppearance,
            attribute: "AXValue".to_string(),
            expected_value: long_value.clone(),
            pre_value: None,
            stability: 1.0,
        }];

        let conditions = generator.convert_postconditions(&postconditions);

        assert_eq!(conditions.len(), 1);
        assert_eq!(conditions[0].expected_value, long_value);
    }

    #[test]
    fn test_hoare_triple_with_preconditions() {
        let generator = HoareTripleGenerator::new();

        let postconditions = vec![Postcondition {
            signal_type: SignalType::AxRole,
            attribute: "AXValue".to_string(),
            expected_value: "new_value".to_string(),
            pre_value: Some("old_value".to_string()),
            stability: 1.0,
        }];

        let triple = generator.generate(
            "Type in field",
            &postconditions,
            Some(500),
        );

        assert_eq!(triple.command, "Type in field");
        assert_eq!(triple.precondition.len(), 1);
        assert_eq!(triple.precondition[0].expected_value, "old_value");
        assert_eq!(triple.postcondition.len(), 1);
        assert_eq!(triple.postcondition[0].expected_value, "new_value");
    }

    #[test]
    fn test_triple_formatting_with_preconditions() {
        let generator = HoareTripleGenerator::new();

        let triple = HoareTriple {
            precondition: vec![Condition {
                scope: "target_element".to_string(),
                attribute: "AXValue".to_string(),
                operator: "equals".to_string(),
                expected_value: "old".to_string(),
            }],
            command: "Update field".to_string(),
            postcondition: vec![Condition {
                scope: "target_element".to_string(),
                attribute: "AXValue".to_string(),
                operator: "equals".to_string(),
                expected_value: "new".to_string(),
            }],
        };

        let formatted = generator.format_triple(&triple);

        assert!(formatted.contains("target_element.AXValue equals 'old'"));
        assert!(formatted.contains("Update field"));
        assert!(formatted.contains("target_element.AXValue equals 'new'"));
    }

    #[test]
    fn test_triple_formatting_empty_postconditions() {
        let generator = HoareTripleGenerator::new();

        let triple = HoareTriple {
            precondition: vec![],
            command: "No-op action".to_string(),
            postcondition: vec![],
        };

        let formatted = generator.format_triple(&triple);

        assert_eq!(formatted, "{ true } No-op action { true }");
    }

    #[test]
    fn test_triple_formatting_multiple_conditions() {
        let generator = HoareTripleGenerator::new();

        let triple = HoareTriple {
            precondition: vec![
                Condition {
                    scope: "target_element".to_string(),
                    attribute: "AXRole".to_string(),
                    operator: "equals".to_string(),
                    expected_value: "AXButton".to_string(),
                },
                Condition {
                    scope: "target_element".to_string(),
                    attribute: "AXEnabled".to_string(),
                    operator: "equals".to_string(),
                    expected_value: "true".to_string(),
                },
            ],
            command: "Click button".to_string(),
            postcondition: vec![
                Condition {
                    scope: "window".to_string(),
                    attribute: "WindowTitle".to_string(),
                    operator: "contains".to_string(),
                    expected_value: "Success".to_string(),
                },
            ],
        };

        let formatted = generator.format_triple(&triple);

        assert!(formatted.contains("&&"));
        assert!(formatted.contains("AXButton"));
        assert!(formatted.contains("AXEnabled"));
        assert!(formatted.contains("Success"));
    }

    #[test]
    fn test_serialization_of_verification_block() {
        let block = VerificationBlock {
            verification_type: "state_assertion".to_string(),
            timeout_ms: 5000,
            conditions: vec![
                Condition {
                    scope: "window".to_string(),
                    attribute: "WindowTitle".to_string(),
                    operator: "contains".to_string(),
                    expected_value: "Test".to_string(),
                },
            ],
        };

        let json = serde_json::to_string(&block).unwrap();
        assert!(json.contains("state_assertion"));
        assert!(json.contains("5000"));
        assert!(json.contains("WindowTitle"));
    }

    #[test]
    fn test_deserialization_of_verification_block() {
        let json = r#"{
            "verification_type": "state_assertion",
            "timeout_ms": 3000,
            "conditions": [
                {
                    "scope": "target_element",
                    "attribute": "AXRole",
                    "operator": "equals",
                    "expected_value": "AXButton"
                }
            ]
        }"#;

        let block: VerificationBlock = serde_json::from_str(json).unwrap();

        assert_eq!(block.verification_type, "state_assertion");
        assert_eq!(block.timeout_ms, 3000);
        assert_eq!(block.conditions.len(), 1);
        assert_eq!(block.conditions[0].attribute, "AXRole");
    }

    #[test]
    fn test_serialization_roundtrip() {
        let original = VerificationBlock {
            verification_type: "state_assertion".to_string(),
            timeout_ms: 7500,
            conditions: vec![
                Condition {
                    scope: "system".to_string(),
                    attribute: "CursorState".to_string(),
                    operator: "equals".to_string(),
                    expected_value: "pointer".to_string(),
                },
                Condition {
                    scope: "window".to_string(),
                    attribute: "WindowTitle".to_string(),
                    operator: "contains".to_string(),
                    expected_value: "Browser".to_string(),
                },
            ],
        };

        let json = serde_json::to_string(&original).unwrap();
        let deserialized: VerificationBlock = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.verification_type, original.verification_type);
        assert_eq!(deserialized.timeout_ms, original.timeout_ms);
        assert_eq!(deserialized.conditions.len(), original.conditions.len());
        assert_eq!(deserialized.conditions[0].scope, original.conditions[0].scope);
        assert_eq!(deserialized.conditions[1].expected_value, original.conditions[1].expected_value);
    }

    #[test]
    fn test_condition_serialization() {
        let condition = Condition {
            scope: "target_element".to_string(),
            attribute: "AXValue".to_string(),
            operator: "equals".to_string(),
            expected_value: "test \"value\" with 'quotes'".to_string(),
        };

        let json = serde_json::to_string(&condition).unwrap();
        let deserialized: Condition = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.scope, condition.scope);
        assert_eq!(deserialized.attribute, condition.attribute);
        assert_eq!(deserialized.operator, condition.operator);
        assert_eq!(deserialized.expected_value, condition.expected_value);
    }

    #[test]
    fn test_default_implementation() {
        let generator = HoareTripleGenerator::default();
        assert_eq!(generator.default_timeout_ms, 5000);
    }
}
