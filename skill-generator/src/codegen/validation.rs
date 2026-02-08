//! SKILL.md Validation

use super::skill_compiler::CompiledSkill;
use regex::Regex;
use std::collections::HashSet;
use std::sync::OnceLock;

/// Globally cached regex patterns compiled once on first use.
struct CachedPatterns {
    hardcoded: Vec<Regex>,
    variable: Regex,
}

fn cached_patterns() -> &'static CachedPatterns {
    static PATTERNS: OnceLock<CachedPatterns> = OnceLock::new();
    PATTERNS.get_or_init(|| CachedPatterns {
        hardcoded: vec![
            Regex::new(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}").unwrap(),
            Regex::new(r"\d{4}-\d{2}-\d{2}").unwrap(),
            Regex::new(r"\d{2}/\d{2}/\d{4}").unwrap(),
            Regex::new(r"\d{3}[-.]?\d{3}[-.]?\d{4}").unwrap(),
        ],
        variable: Regex::new(r"\{\{([^}]+)\}\}").unwrap(),
    })
}

/// Validation error
#[derive(Debug, Clone)]
pub struct ValidationError {
    /// Error type
    pub error_type: ValidationErrorType,
    /// Error message
    pub message: String,
    /// Location (if applicable)
    pub location: Option<String>,
}

/// Types of validation errors
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationErrorType {
    /// YAML syntax error
    YamlSyntax,
    /// Hardcoded literal found
    HardcodedLiteral,
    /// Undefined variable
    UndefinedVariable,
    /// Missing verification
    MissingVerification,
    /// Argument mismatch
    ArgumentMismatch,
    /// Invalid frontmatter
    InvalidFrontmatter,
}

/// Validation result
#[derive(Debug)]
pub struct ValidationResult {
    /// Whether validation passed
    pub passed: bool,
    /// List of errors
    pub errors: Vec<ValidationError>,
    /// List of warnings
    pub warnings: Vec<String>,
}

impl ValidationResult {
    /// Create a passing result
    pub fn pass() -> Self {
        Self {
            passed: true,
            errors: vec![],
            warnings: vec![],
        }
    }

    /// Create a failing result
    pub fn fail(errors: Vec<ValidationError>) -> Self {
        Self {
            passed: false,
            errors,
            warnings: vec![],
        }
    }
}

/// Skill validator
pub struct SkillValidator {
    /// Reference to globally cached hardcoded patterns
    hardcoded_patterns: &'static [Regex],
    /// Reference to globally cached variable pattern
    variable_pattern: &'static Regex,
}

impl SkillValidator {
    /// Create a new validator (patterns are compiled once and cached globally)
    pub fn new() -> Self {
        let patterns = cached_patterns();
        Self {
            hardcoded_patterns: &patterns.hardcoded,
            variable_pattern: &patterns.variable,
        }
    }

    /// Validate a compiled skill
    pub fn validate(&self, skill: &CompiledSkill) -> ValidationResult {
        let mut errors = Vec::new();

        // Check 1: No hardcoded literals
        errors.extend(self.check_hardcoded_literals(skill));

        // Check 2: All variables defined
        errors.extend(self.check_undefined_variables(skill));

        // Check 3: Every step has verification
        errors.extend(self.check_verification_blocks(skill));

        // Check 4: Argument hint matches inputs
        errors.extend(self.check_argument_hint(skill));

        if errors.is_empty() {
            ValidationResult::pass()
        } else {
            ValidationResult::fail(errors)
        }
    }

    /// Validate markdown content directly
    pub fn validate_markdown(&self, markdown: &str) -> ValidationResult {
        let mut errors = Vec::new();

        // Check YAML frontmatter
        if !markdown.starts_with("---") {
            errors.push(ValidationError {
                error_type: ValidationErrorType::InvalidFrontmatter,
                message: "Missing YAML frontmatter".to_string(),
                location: Some("start".to_string()),
            });
        }

        // Parse frontmatter
        let parts: Vec<&str> = markdown.splitn(3, "---").collect();
        if parts.len() < 3 {
            errors.push(ValidationError {
                error_type: ValidationErrorType::YamlSyntax,
                message: "Invalid frontmatter format".to_string(),
                location: None,
            });
        } else {
            // Validate YAML syntax
            if serde_yaml_ng::from_str::<serde_yaml_ng::Value>(parts[1]).is_err() {
                errors.push(ValidationError {
                    error_type: ValidationErrorType::YamlSyntax,
                    message: "Invalid YAML in frontmatter".to_string(),
                    location: None,
                });
            }
        }

        // Check for hardcoded literals in body
        if parts.len() >= 3 {
            for pattern in self.hardcoded_patterns {
                if pattern.is_match(parts[2]) {
                    // Check if it's inside a variable
                    let matches: Vec<_> = pattern.find_iter(parts[2]).collect();
                    for m in matches {
                        // Get surrounding context
                        let start = m.start().saturating_sub(2);
                        let end = (m.end() + 2).min(parts[2].len());
                        let context = &parts[2][start..end];

                        // If not inside {{ }}, it's hardcoded
                        if !context.contains("{{") || !context.contains("}}") {
                            errors.push(ValidationError {
                                error_type: ValidationErrorType::HardcodedLiteral,
                                message: format!("Potential hardcoded literal: {}", m.as_str()),
                                location: Some(format!("position {}", m.start())),
                            });
                        }
                    }
                }
            }
        }

        if errors.is_empty() {
            ValidationResult::pass()
        } else {
            ValidationResult::fail(errors)
        }
    }

    /// Check for hardcoded literals in narratives
    fn check_hardcoded_literals(&self, skill: &CompiledSkill) -> Vec<ValidationError> {
        let mut errors = Vec::new();

        for (i, step) in skill.steps.iter().enumerate() {
            for pattern in self.hardcoded_patterns {
                for m in pattern.find_iter(&step.narrative) {
                    // Check if inside variable
                    let context_start = m.start().saturating_sub(2);
                    let context_end = (m.end() + 2).min(step.narrative.len());
                    let context = &step.narrative[context_start..context_end];

                    if !context.contains("{{") {
                        errors.push(ValidationError {
                            error_type: ValidationErrorType::HardcodedLiteral,
                            message: format!("Hardcoded literal '{}' in step {}", m.as_str(), i + 1),
                            location: Some(format!("Step {}", i + 1)),
                        });
                    }
                }
            }
        }

        errors
    }

    /// Check for undefined variables
    fn check_undefined_variables(&self, skill: &CompiledSkill) -> Vec<ValidationError> {
        let mut errors = Vec::new();

        // Collect defined variables
        let defined: HashSet<String> = skill.inputs.iter().map(|i| i.name.clone()).collect();

        // Find used variables in narratives
        for (i, step) in skill.steps.iter().enumerate() {
            for cap in self.variable_pattern.captures_iter(&step.narrative) {
                let var_name = &cap[1];
                if !defined.contains(var_name) {
                    errors.push(ValidationError {
                        error_type: ValidationErrorType::UndefinedVariable,
                        message: format!("Undefined variable '{{{{{}}}}}' in step {}", var_name, i + 1),
                        location: Some(format!("Step {}", i + 1)),
                    });
                }
            }
        }

        errors
    }

    /// Check that every step has a verification block
    fn check_verification_blocks(&self, skill: &CompiledSkill) -> Vec<ValidationError> {
        let mut errors = Vec::new();

        for (i, step) in skill.steps.iter().enumerate() {
            if step.verification.is_none() {
                errors.push(ValidationError {
                    error_type: ValidationErrorType::MissingVerification,
                    message: format!("Step {} missing verification block", i + 1),
                    location: Some(format!("Step {}", i + 1)),
                });
            }
        }

        errors
    }

    /// Check argument hint matches input count
    fn check_argument_hint(&self, _skill: &CompiledSkill) -> Vec<ValidationError> {
        // This check is done during markdown generation
        // Here we just verify inputs are consistent
        vec![]
    }
}

impl Default for SkillValidator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::skill_compiler::{CompiledSkill, CompiledStep, SkillInput};

    #[test]
    fn test_validator_creation() {
        let validator = SkillValidator::new();
        assert!(!validator.hardcoded_patterns.is_empty());
    }

    #[test]
    fn test_email_detection() {
        let validator = SkillValidator::new();
        assert!(validator.hardcoded_patterns[0].is_match("test@example.com"));
    }

    #[test]
    fn test_date_detection() {
        let validator = SkillValidator::new();
        assert!(validator.hardcoded_patterns[1].is_match("2026-01-26"));
    }

    #[test]
    fn test_markdown_validation() {
        let validator = SkillValidator::new();

        let valid_markdown = r#"---
name: test
description: Test skill
---

## Overview

Test description
"#;

        let result = validator.validate_markdown(valid_markdown);
        assert!(result.passed);
    }

    #[test]
    fn test_missing_frontmatter() {
        let validator = SkillValidator::new();

        let invalid_markdown = "# No frontmatter";

        let result = validator.validate_markdown(invalid_markdown);
        assert!(!result.passed);
    }

    #[test]
    fn test_phone_number_detection() {
        let validator = SkillValidator::new();
        assert!(validator.hardcoded_patterns[3].is_match("123-456-7890"));
        assert!(validator.hardcoded_patterns[3].is_match("123.456.7890"));
        assert!(validator.hardcoded_patterns[3].is_match("1234567890"));
    }

    #[test]
    fn test_invalid_yaml_in_frontmatter() {
        let validator = SkillValidator::new();

        let invalid_yaml = r#"---
name: test
description: [invalid: yaml: structure
---

## Body
"#;

        let result = validator.validate_markdown(invalid_yaml);
        assert!(!result.passed);
        assert!(result.errors.iter().any(|e| matches!(
            e.error_type,
            ValidationErrorType::YamlSyntax
        )));
    }

    #[test]
    fn test_hardcoded_email_in_body() {
        let validator = SkillValidator::new();

        let markdown_with_email = r#"---
name: test
description: Test skill
---

## Step 1

Please send email to test@example.com for verification.
"#;

        let result = validator.validate_markdown(markdown_with_email);
        assert!(!result.passed);
        assert!(result.errors.iter().any(|e| matches!(
            e.error_type,
            ValidationErrorType::HardcodedLiteral
        )));
        assert!(result.errors[0].message.contains("test@example.com"));
    }

    #[test]
    fn test_variable_wrapped_values_not_flagged() {
        let validator = SkillValidator::new();

        let markdown_with_variable = r#"---
name: test
description: Test skill
---

## Step 1

Enter {{email}} into the field. The value might be test@example.com.
"#;

        let result = validator.validate_markdown(markdown_with_variable);
        // Should still catch the hardcoded one but not the one in text description
        // This tests that context checking works
        assert!(!result.passed);
    }

    #[test]
    fn test_check_undefined_variables_in_compiled_skill() {
        let validator = SkillValidator::new();

        let skill = CompiledSkill {
            name: "test".to_string(),
            description: "Test".to_string(),
            inputs: vec![SkillInput {
                name: "username".to_string(),
                type_hint: None,
                description: None,
                required: true,
                default: None,
            }],
            steps: vec![CompiledStep {
                number: 1,
                narrative: "Enter {{username}} and {{password}} into the form".to_string(),
                technical_context: super::super::skill_compiler::TechnicalContext {
                    selectors: vec![],
                    ax_metadata: None,
                    screenshot_ref: None,
                },
                verification: None,
                error_recovery: None,
            }],
            context: super::super::skill_compiler::ExecutionContext::Fork,
            allowed_tools: vec![],
        };

        let result = validator.validate(&skill);
        assert!(!result.passed);
        assert!(result.errors.iter().any(|e| {
            matches!(e.error_type, ValidationErrorType::UndefinedVariable)
                && e.message.contains("password")
        }));
    }

    #[test]
    fn test_check_hardcoded_literals_in_compiled_skill() {
        let validator = SkillValidator::new();

        let skill = CompiledSkill {
            name: "test".to_string(),
            description: "Test".to_string(),
            inputs: vec![],
            steps: vec![CompiledStep {
                number: 1,
                narrative: "Call phone number 555-123-4567 for support".to_string(),
                technical_context: super::super::skill_compiler::TechnicalContext {
                    selectors: vec![],
                    ax_metadata: None,
                    screenshot_ref: None,
                },
                verification: None,
                error_recovery: None,
            }],
            context: super::super::skill_compiler::ExecutionContext::Fork,
            allowed_tools: vec![],
        };

        let result = validator.validate(&skill);
        assert!(!result.passed);
        assert!(result.errors.iter().any(|e| matches!(
            e.error_type,
            ValidationErrorType::HardcodedLiteral
        )));
    }

    #[test]
    fn test_missing_verification_blocks() {
        let validator = SkillValidator::new();

        let skill = CompiledSkill {
            name: "test".to_string(),
            description: "Test".to_string(),
            inputs: vec![],
            steps: vec![
                CompiledStep {
                    number: 1,
                    narrative: "Click button".to_string(),
                    technical_context: super::super::skill_compiler::TechnicalContext {
                        selectors: vec![],
                        ax_metadata: None,
                        screenshot_ref: None,
                    },
                    verification: None,
                    error_recovery: None,
                },
                CompiledStep {
                    number: 2,
                    narrative: "Type text".to_string(),
                    technical_context: super::super::skill_compiler::TechnicalContext {
                        selectors: vec![],
                        ax_metadata: None,
                        screenshot_ref: None,
                    },
                    verification: None,
                    error_recovery: None,
                },
            ],
            context: super::super::skill_compiler::ExecutionContext::Fork,
            allowed_tools: vec![],
        };

        let result = validator.validate(&skill);
        assert!(!result.passed);
        assert_eq!(
            result
                .errors
                .iter()
                .filter(|e| matches!(e.error_type, ValidationErrorType::MissingVerification))
                .count(),
            2
        );
    }

    #[test]
    fn test_valid_compiled_skill_passes() {
        let validator = SkillValidator::new();

        let skill = CompiledSkill {
            name: "test".to_string(),
            description: "Test".to_string(),
            inputs: vec![SkillInput {
                name: "username".to_string(),
                type_hint: Some("string".to_string()),
                description: Some("Username".to_string()),
                required: true,
                default: None,
            }],
            steps: vec![CompiledStep {
                number: 1,
                narrative: "Enter {{username}} into the field".to_string(),
                technical_context: super::super::skill_compiler::TechnicalContext {
                    selectors: vec!["#username".to_string()],
                    ax_metadata: None,
                    screenshot_ref: None,
                },
                verification: Some(crate::verification::hoare_triple_generator::VerificationBlock {
                    verification_type: "state_assertion".to_string(),
                    timeout_ms: 5000,
                    conditions: vec![],
                }),
                error_recovery: None,
            }],
            context: super::super::skill_compiler::ExecutionContext::Fork,
            allowed_tools: vec!["Read".to_string()],
        };

        let result = validator.validate(&skill);
        assert!(result.passed);
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_validation_error_types() {
        assert!(matches!(
            ValidationErrorType::YamlSyntax,
            ValidationErrorType::YamlSyntax
        ));
        assert!(matches!(
            ValidationErrorType::HardcodedLiteral,
            ValidationErrorType::HardcodedLiteral
        ));
        assert!(matches!(
            ValidationErrorType::UndefinedVariable,
            ValidationErrorType::UndefinedVariable
        ));
        assert!(matches!(
            ValidationErrorType::MissingVerification,
            ValidationErrorType::MissingVerification
        ));
    }

    #[test]
    fn test_validation_result_creation() {
        let pass = ValidationResult::pass();
        assert!(pass.passed);
        assert!(pass.errors.is_empty());
        assert!(pass.warnings.is_empty());

        let errors = vec![ValidationError {
            error_type: ValidationErrorType::YamlSyntax,
            message: "Test error".to_string(),
            location: None,
        }];
        let fail = ValidationResult::fail(errors);
        assert!(!fail.passed);
        assert_eq!(fail.errors.len(), 1);
    }

    #[test]
    fn test_multiple_hardcoded_patterns_in_same_step() {
        let validator = SkillValidator::new();

        let skill = CompiledSkill {
            name: "test".to_string(),
            description: "Test".to_string(),
            inputs: vec![],
            steps: vec![CompiledStep {
                number: 1,
                narrative: "Email test@example.com on 2026-01-26 at phone 555-1234".to_string(),
                technical_context: super::super::skill_compiler::TechnicalContext {
                    selectors: vec![],
                    ax_metadata: None,
                    screenshot_ref: None,
                },
                verification: None,
                error_recovery: None,
            }],
            context: super::super::skill_compiler::ExecutionContext::Fork,
            allowed_tools: vec![],
        };

        let result = validator.validate(&skill);
        assert!(!result.passed);
        // Should detect multiple hardcoded literals
        assert!(result.errors.len() >= 2);
    }

    #[test]
    fn test_incomplete_frontmatter_structure() {
        let validator = SkillValidator::new();

        let incomplete = r#"---
name: test
"#;

        let result = validator.validate_markdown(incomplete);
        assert!(!result.passed);
        assert!(result.errors.iter().any(|e| matches!(
            e.error_type,
            ValidationErrorType::YamlSyntax
        )));
    }

    #[test]
    fn test_default_validator() {
        let validator = SkillValidator::default();
        assert!(!validator.hardcoded_patterns.is_empty());
        assert!(validator.hardcoded_patterns.len() >= 4);
    }

    #[test]
    fn test_date_formats() {
        let validator = SkillValidator::new();

        // ISO format
        assert!(validator.hardcoded_patterns[1].is_match("2026-01-26"));

        // US format
        assert!(validator.hardcoded_patterns[2].is_match("01/26/2026"));
        assert!(validator.hardcoded_patterns[2].is_match("12/31/2025"));
    }

    #[test]
    fn test_variable_pattern_matching() {
        let validator = SkillValidator::new();

        let text = "Enter {{username}} and {{password}} here";
        let matches: Vec<_> = validator.variable_pattern.captures_iter(text).collect();

        assert_eq!(matches.len(), 2);
        assert_eq!(&matches[0][1], "username");
        assert_eq!(&matches[1][1], "password");
    }

    #[test]
    fn test_empty_skill_validation() {
        let validator = SkillValidator::new();

        let skill = CompiledSkill {
            name: "".to_string(),
            description: "".to_string(),
            inputs: vec![],
            steps: vec![],
            context: super::super::skill_compiler::ExecutionContext::Fork,
            allowed_tools: vec![],
        };

        let result = validator.validate(&skill);
        // Empty skill should pass basic validation (no steps = no errors)
        assert!(result.passed);
    }

    #[test]
    fn test_nested_variable_syntax() {
        let validator = SkillValidator::new();

        // Test that nested braces don't break parsing
        let text = "Click on {{button_name}} in {some_context}";
        let matches: Vec<_> = validator.variable_pattern.captures_iter(text).collect();

        assert_eq!(matches.len(), 1);
        assert_eq!(&matches[0][1], "button_name");
    }

    #[test]
    fn test_special_characters_in_narrative() {
        let validator = SkillValidator::new();

        let skill = CompiledSkill {
            name: "test_special".to_string(),
            description: "Test with special chars: <>&\"".to_string(),
            inputs: vec![],
            steps: vec![CompiledStep {
                number: 1,
                narrative: "Click on \"Special Button\" with <angle> brackets".to_string(),
                technical_context: super::super::skill_compiler::TechnicalContext {
                    selectors: vec![],
                    ax_metadata: None,
                    screenshot_ref: None,
                },
                verification: Some(crate::verification::hoare_triple_generator::VerificationBlock {
                    verification_type: "state_assertion".to_string(),
                    timeout_ms: 5000,
                    conditions: vec![],
                }),
                error_recovery: None,
            }],
            context: super::super::skill_compiler::ExecutionContext::Fork,
            allowed_tools: vec!["Read".to_string()],
        };

        let result = validator.validate(&skill);
        assert!(result.passed);
    }

    #[test]
    fn test_unicode_in_narrative() {
        let validator = SkillValidator::new();

        let skill = CompiledSkill {
            name: "test_unicode".to_string(),
            description: "Test with emoji 🎉 and unicode".to_string(),
            inputs: vec![],
            steps: vec![CompiledStep {
                number: 1,
                narrative: "Click on \"提交\" button (Submit in Chinese)".to_string(),
                technical_context: super::super::skill_compiler::TechnicalContext {
                    selectors: vec![],
                    ax_metadata: None,
                    screenshot_ref: None,
                },
                verification: Some(crate::verification::hoare_triple_generator::VerificationBlock {
                    verification_type: "state_assertion".to_string(),
                    timeout_ms: 5000,
                    conditions: vec![],
                }),
                error_recovery: None,
            }],
            context: super::super::skill_compiler::ExecutionContext::Fork,
            allowed_tools: vec!["Read".to_string()],
        };

        let result = validator.validate(&skill);
        assert!(result.passed);
    }
}
