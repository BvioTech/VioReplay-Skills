//! Skill Markdown Assembly

use super::skill_compiler::{CompiledSkill, CompiledStep, ExecutionContext, SkillInput};
use std::fmt::Write;

/// Markdown builder for SKILL.md files
pub struct MarkdownBuilder {
    /// Buffer for building markdown
    buffer: String,
}

impl MarkdownBuilder {
    /// Create a new markdown builder
    pub fn new() -> Self {
        Self {
            buffer: String::with_capacity(4096),
        }
    }

    /// Build SKILL.md content from compiled skill
    pub fn build(&mut self, skill: &CompiledSkill) -> String {
        self.buffer.clear();

        // Writing to a String is infallible, so these cannot fail
        self.write_frontmatter(skill).expect("write to String");
        self.write_body(skill).expect("write to String");

        std::mem::take(&mut self.buffer)
    }

    /// Write YAML frontmatter
    fn write_frontmatter(&mut self, skill: &CompiledSkill) -> std::fmt::Result {
        writeln!(self.buffer, "---")?;
        writeln!(self.buffer, "name: {}", skill.name)?;
        writeln!(self.buffer, "description: {}", skill.description)?;
        writeln!(
            self.buffer,
            "context: {}",
            match skill.context {
                ExecutionContext::Fork => "fork",
                ExecutionContext::Inline => "inline",
            }
        )?;

        // Allowed tools
        if !skill.allowed_tools.is_empty() {
            writeln!(
                self.buffer,
                "allowed-tools: {}",
                skill.allowed_tools.join(", ")
            )?;
        }

        // Argument hint
        if !skill.inputs.is_empty() {
            let hints: Vec<String> = skill
                .inputs
                .iter()
                .map(|i| format!("[{}]", i.name))
                .collect();
            writeln!(self.buffer, "argument-hint: {}", hints.join(" "))?;
        }

        writeln!(self.buffer, "disable-model-invocation: false")?;
        writeln!(self.buffer, "---")?;
        writeln!(self.buffer)?;
        Ok(())
    }

    /// Write markdown body
    fn write_body(&mut self, skill: &CompiledSkill) -> std::fmt::Result {
        // Overview section
        self.write_overview(skill)?;

        // Input variables section
        if !skill.inputs.is_empty() {
            self.write_inputs(&skill.inputs)?;
        }

        // Steps
        for step in &skill.steps {
            self.write_step(step)?;
        }
        Ok(())
    }

    /// Write overview section
    fn write_overview(&mut self, skill: &CompiledSkill) -> std::fmt::Result {
        writeln!(self.buffer, "## Overview")?;
        writeln!(self.buffer)?;
        writeln!(self.buffer, "{}", skill.description)?;
        writeln!(self.buffer)?;
        Ok(())
    }

    /// Write inputs section
    fn write_inputs(&mut self, inputs: &[SkillInput]) -> std::fmt::Result {
        writeln!(self.buffer, "## Inputs")?;
        writeln!(self.buffer)?;

        for input in inputs {
            write!(self.buffer, "- `{{{{{}}}}}`", input.name)?;

            if let Some(type_hint) = &input.type_hint {
                write!(self.buffer, " ({})", type_hint)?;
            }

            if !input.required {
                write!(self.buffer, " [optional]")?;
            }

            if let Some(desc) = &input.description {
                write!(self.buffer, ": {}", desc)?;
            }

            writeln!(self.buffer)?;
        }

        writeln!(self.buffer)?;
        Ok(())
    }

    /// Write a step section
    fn write_step(&mut self, step: &CompiledStep) -> std::fmt::Result {
        writeln!(self.buffer, "## Step {}", step.number)?;
        writeln!(self.buffer)?;

        // Narrative
        writeln!(self.buffer, "{}", step.narrative)?;
        writeln!(self.buffer)?;

        // Technical context
        self.write_technical_context(step)?;

        // Verification
        if let Some(verification) = &step.verification {
            self.write_verification(verification)?;
        }

        // Error recovery
        if let Some(recovery) = &step.error_recovery {
            self.write_error_recovery(recovery)?;
        }

        writeln!(self.buffer)?;
        Ok(())
    }

    /// Write technical context block
    fn write_technical_context(&mut self, step: &CompiledStep) -> std::fmt::Result {
        let ctx = &step.technical_context;

        writeln!(self.buffer, "<details>")?;
        writeln!(self.buffer, "<summary>Technical Context</summary>")?;
        writeln!(self.buffer)?;
        writeln!(self.buffer, "```json")?;
        writeln!(self.buffer, "{{")?;

        // Selectors
        if !ctx.selectors.is_empty() {
            writeln!(
                self.buffer,
                "  \"selectors\": {},",
                serde_json::to_string(&ctx.selectors).unwrap_or_default()
            )?;
        }

        // AX metadata
        if let Some(ax) = &ctx.ax_metadata {
            writeln!(self.buffer, "  \"ax_metadata\": {{")?;
            if let Some(role) = &ax.role {
                writeln!(self.buffer, "    \"role\": \"{}\",", role)?;
            }
            if let Some(title) = &ax.title {
                writeln!(self.buffer, "    \"title\": \"{}\",", title)?;
            }
            if let Some(id) = &ax.identifier {
                writeln!(self.buffer, "    \"identifier\": \"{}\",", id)?;
            }
            if let Some(window) = &ax.window_title {
                writeln!(self.buffer, "    \"window_title\": \"{}\"", window)?;
            }
            writeln!(self.buffer, "  }}")?;
        }

        writeln!(self.buffer, "}}")?;
        writeln!(self.buffer, "```")?;
        writeln!(self.buffer)?;
        writeln!(self.buffer, "</details>")?;
        writeln!(self.buffer)?;
        Ok(())
    }

    /// Write verification block
    fn write_verification(&mut self, verification: &crate::verification::hoare_triple_generator::VerificationBlock) -> std::fmt::Result {
        writeln!(self.buffer, "### Verification")?;
        writeln!(self.buffer)?;
        writeln!(self.buffer, "```yaml")?;
        writeln!(self.buffer, "verification:")?;
        writeln!(self.buffer, "  type: \"{}\"", verification.verification_type)?;
        writeln!(self.buffer, "  timeout_ms: {}", verification.timeout_ms)?;
        writeln!(self.buffer, "  conditions:")?;

        for condition in &verification.conditions {
            writeln!(self.buffer, "    - scope: \"{}\"", condition.scope)?;
            writeln!(self.buffer, "      attribute: \"{}\"", condition.attribute)?;
            writeln!(self.buffer, "      operator: \"{}\"", condition.operator)?;
            writeln!(self.buffer, "      expected_value: \"{}\"", condition.expected_value)?;
        }

        writeln!(self.buffer, "```")?;
        writeln!(self.buffer)?;
        Ok(())
    }

    /// Write error recovery block
    fn write_error_recovery(&mut self, recovery: &super::skill_compiler::ErrorRecovery) -> std::fmt::Result {
        writeln!(self.buffer, "### Error Recovery")?;
        writeln!(self.buffer)?;
        writeln!(self.buffer, "**Trigger:** {}", recovery.trigger_condition)?;
        writeln!(self.buffer)?;
        writeln!(self.buffer, "**Action:** {}", recovery.recovery_action)?;
        writeln!(self.buffer)?;
        writeln!(self.buffer, "**Max Retries:** {}", recovery.max_retries)?;
        writeln!(self.buffer)?;
        Ok(())
    }
}

impl Default for MarkdownBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::skill_compiler::{TechnicalContext, AxMetadata};

    #[test]
    fn test_markdown_builder() {
        let mut builder = MarkdownBuilder::new();

        let skill = CompiledSkill {
            name: "test_skill".to_string(),
            description: "Test skill description".to_string(),
            inputs: vec![SkillInput {
                name: "test_input".to_string(),
                type_hint: Some("string".to_string()),
                description: Some("Test input".to_string()),
                required: true,
                default: None,
            }],
            steps: vec![],
            context: ExecutionContext::Fork,
            allowed_tools: vec!["Read".to_string()],
        };

        let markdown = builder.build(&skill);

        assert!(markdown.contains("---"));
        assert!(markdown.contains("name: test_skill"));
        assert!(markdown.contains("## Overview"));
    }

    #[test]
    fn test_frontmatter_generation() {
        let mut builder = MarkdownBuilder::new();

        let skill = CompiledSkill {
            name: "my_skill".to_string(),
            description: "A test skill".to_string(),
            inputs: vec![],
            steps: vec![],
            context: ExecutionContext::Inline,
            allowed_tools: vec!["Bash".to_string(), "Read".to_string()],
        };

        let markdown = builder.build(&skill);

        assert!(markdown.contains("name: my_skill"));
        assert!(markdown.contains("context: inline"));
        assert!(markdown.contains("allowed-tools: Bash, Read"));
        assert!(markdown.contains("disable-model-invocation: false"));
    }

    #[test]
    fn test_inputs_section() {
        let mut builder = MarkdownBuilder::new();

        let skill = CompiledSkill {
            name: "input_test".to_string(),
            description: "Test inputs".to_string(),
            inputs: vec![
                SkillInput {
                    name: "required_input".to_string(),
                    type_hint: Some("string".to_string()),
                    description: Some("A required input".to_string()),
                    required: true,
                    default: None,
                },
                SkillInput {
                    name: "optional_input".to_string(),
                    type_hint: None,
                    description: None,
                    required: false,
                    default: Some("default_value".to_string()),
                },
            ],
            steps: vec![],
            context: ExecutionContext::Fork,
            allowed_tools: vec![],
        };

        let markdown = builder.build(&skill);

        assert!(markdown.contains("## Inputs"));
        assert!(markdown.contains("`{{required_input}}`"));
        assert!(markdown.contains("(string)"));
        assert!(markdown.contains("[optional]"));
        assert!(markdown.contains("argument-hint: [required_input] [optional_input]"));
    }

    #[test]
    fn test_step_generation() {
        let mut builder = MarkdownBuilder::new();

        let skill = CompiledSkill {
            name: "step_test".to_string(),
            description: "Test steps".to_string(),
            inputs: vec![],
            steps: vec![CompiledStep {
                number: 1,
                narrative: "Click the submit button".to_string(),
                technical_context: TechnicalContext {
                    selectors: vec!["#submit".to_string()],
                    ax_metadata: Some(AxMetadata {
                        role: Some("AXButton".to_string()),
                        title: Some("Submit".to_string()),
                        identifier: Some("btn_submit".to_string()),
                        window_title: Some("My App".to_string()),
                    }),
                    screenshot_ref: None,
                },
                verification: None,
                error_recovery: None,
            }],
            context: ExecutionContext::Fork,
            allowed_tools: vec![],
        };

        let markdown = builder.build(&skill);

        assert!(markdown.contains("## Step 1"));
        assert!(markdown.contains("Click the submit button"));
        assert!(markdown.contains("Technical Context"));
        assert!(markdown.contains("AXButton"));
    }

    #[test]
    fn test_default_builder() {
        let builder = MarkdownBuilder::default();
        assert!(builder.buffer.capacity() >= 4096);
    }

    #[test]
    fn test_empty_skill_generation() {
        let mut builder = MarkdownBuilder::new();

        let skill = CompiledSkill {
            name: "empty_skill".to_string(),
            description: "An empty skill".to_string(),
            inputs: vec![],
            steps: vec![],
            context: ExecutionContext::Fork,
            allowed_tools: vec![],
        };

        let markdown = builder.build(&skill);

        assert!(markdown.contains("---"));
        assert!(markdown.contains("name: empty_skill"));
        assert!(markdown.contains("description: An empty skill"));
        assert!(markdown.contains("## Overview"));
        // Should not contain inputs or steps sections
        assert!(!markdown.contains("## Inputs"));
        assert!(!markdown.contains("## Step"));
    }

    #[test]
    fn test_inline_context_generation() {
        let mut builder = MarkdownBuilder::new();

        let skill = CompiledSkill {
            name: "inline_test".to_string(),
            description: "Test inline context".to_string(),
            inputs: vec![],
            steps: vec![],
            context: ExecutionContext::Inline,
            allowed_tools: vec![],
        };

        let markdown = builder.build(&skill);

        assert!(markdown.contains("context: inline"));
    }

    #[test]
    fn test_multiple_steps_generation() {
        let mut builder = MarkdownBuilder::new();

        let skill = CompiledSkill {
            name: "multi_step".to_string(),
            description: "Multiple steps test".to_string(),
            inputs: vec![],
            steps: vec![
                CompiledStep {
                    number: 1,
                    narrative: "First step".to_string(),
                    technical_context: TechnicalContext {
                        selectors: vec!["#first".to_string()],
                        ax_metadata: None,
                        screenshot_ref: None,
                    },
                    verification: None,
                    error_recovery: None,
                },
                CompiledStep {
                    number: 2,
                    narrative: "Second step".to_string(),
                    technical_context: TechnicalContext {
                        selectors: vec!["#second".to_string()],
                        ax_metadata: None,
                        screenshot_ref: None,
                    },
                    verification: None,
                    error_recovery: None,
                },
                CompiledStep {
                    number: 3,
                    narrative: "Third step".to_string(),
                    technical_context: TechnicalContext {
                        selectors: vec!["#third".to_string()],
                        ax_metadata: None,
                        screenshot_ref: None,
                    },
                    verification: None,
                    error_recovery: None,
                },
            ],
            context: ExecutionContext::Fork,
            allowed_tools: vec![],
        };

        let markdown = builder.build(&skill);

        assert!(markdown.contains("## Step 1"));
        assert!(markdown.contains("## Step 2"));
        assert!(markdown.contains("## Step 3"));
        assert!(markdown.contains("First step"));
        assert!(markdown.contains("Second step"));
        assert!(markdown.contains("Third step"));
    }

    #[test]
    fn test_verification_block_generation() {
        let mut builder = MarkdownBuilder::new();

        let skill = CompiledSkill {
            name: "verification_test".to_string(),
            description: "Test verification".to_string(),
            inputs: vec![],
            steps: vec![CompiledStep {
                number: 1,
                narrative: "Click button".to_string(),
                technical_context: TechnicalContext {
                    selectors: vec![],
                    ax_metadata: None,
                    screenshot_ref: None,
                },
                verification: Some(crate::verification::hoare_triple_generator::VerificationBlock {
                    verification_type: "state_assertion".to_string(),
                    timeout_ms: 3000,
                    conditions: vec![
                        crate::verification::hoare_triple_generator::Condition {
                            scope: "button".to_string(),
                            attribute: "enabled".to_string(),
                            operator: "equals".to_string(),
                            expected_value: "true".to_string(),
                        },
                    ],
                }),
                error_recovery: None,
            }],
            context: ExecutionContext::Fork,
            allowed_tools: vec![],
        };

        let markdown = builder.build(&skill);

        assert!(markdown.contains("### Verification"));
        assert!(markdown.contains("```yaml"));
        assert!(markdown.contains("verification:"));
        assert!(markdown.contains("type: \"state_assertion\""));
        assert!(markdown.contains("timeout_ms: 3000"));
        assert!(markdown.contains("conditions:"));
        assert!(markdown.contains("scope: \"button\""));
        assert!(markdown.contains("attribute: \"enabled\""));
        assert!(markdown.contains("operator: \"equals\""));
        assert!(markdown.contains("expected_value: \"true\""));
    }

    #[test]
    fn test_error_recovery_generation() {
        let mut builder = MarkdownBuilder::new();

        let skill = CompiledSkill {
            name: "recovery_test".to_string(),
            description: "Test error recovery".to_string(),
            inputs: vec![],
            steps: vec![CompiledStep {
                number: 1,
                narrative: "Click button".to_string(),
                technical_context: TechnicalContext {
                    selectors: vec![],
                    ax_metadata: None,
                    screenshot_ref: None,
                },
                verification: None,
                error_recovery: Some(super::super::skill_compiler::ErrorRecovery {
                    trigger_condition: "Button not found".to_string(),
                    recovery_action: "Wait and retry".to_string(),
                    max_retries: 5,
                }),
            }],
            context: ExecutionContext::Fork,
            allowed_tools: vec![],
        };

        let markdown = builder.build(&skill);

        assert!(markdown.contains("### Error Recovery"));
        assert!(markdown.contains("**Trigger:** Button not found"));
        assert!(markdown.contains("**Action:** Wait and retry"));
        assert!(markdown.contains("**Max Retries:** 5"));
    }

    #[test]
    fn test_complex_ax_metadata_generation() {
        let mut builder = MarkdownBuilder::new();

        let skill = CompiledSkill {
            name: "ax_test".to_string(),
            description: "Test AX metadata".to_string(),
            inputs: vec![],
            steps: vec![CompiledStep {
                number: 1,
                narrative: "Click login".to_string(),
                technical_context: TechnicalContext {
                    selectors: vec!["#login".to_string()],
                    ax_metadata: Some(AxMetadata {
                        role: Some("AXButton".to_string()),
                        title: Some("Login".to_string()),
                        identifier: Some("login-button".to_string()),
                        window_title: Some("Authentication Window".to_string()),
                    }),
                    screenshot_ref: None,
                },
                verification: None,
                error_recovery: None,
            }],
            context: ExecutionContext::Fork,
            allowed_tools: vec![],
        };

        let markdown = builder.build(&skill);

        assert!(markdown.contains("\"ax_metadata\":"));
        assert!(markdown.contains("\"role\": \"AXButton\""));
        assert!(markdown.contains("\"title\": \"Login\""));
        assert!(markdown.contains("\"identifier\": \"login-button\""));
        assert!(markdown.contains("\"window_title\": \"Authentication Window\""));
    }

    #[test]
    fn test_partial_ax_metadata() {
        let mut builder = MarkdownBuilder::new();

        let skill = CompiledSkill {
            name: "partial_ax".to_string(),
            description: "Partial AX metadata".to_string(),
            inputs: vec![],
            steps: vec![CompiledStep {
                number: 1,
                narrative: "Click something".to_string(),
                technical_context: TechnicalContext {
                    selectors: vec![],
                    ax_metadata: Some(AxMetadata {
                        role: Some("AXButton".to_string()),
                        title: None,
                        identifier: None,
                        window_title: None,
                    }),
                    screenshot_ref: None,
                },
                verification: None,
                error_recovery: None,
            }],
            context: ExecutionContext::Fork,
            allowed_tools: vec![],
        };

        let markdown = builder.build(&skill);

        assert!(markdown.contains("\"ax_metadata\":"));
        assert!(markdown.contains("\"role\": \"AXButton\""));
        // Should not contain fields that are None
    }

    #[test]
    fn test_multiple_selectors_in_technical_context() {
        let mut builder = MarkdownBuilder::new();

        let skill = CompiledSkill {
            name: "selectors_test".to_string(),
            description: "Multiple selectors".to_string(),
            inputs: vec![],
            steps: vec![CompiledStep {
                number: 1,
                narrative: "Find element".to_string(),
                technical_context: TechnicalContext {
                    selectors: vec![
                        "#primary-selector".to_string(),
                        ".fallback-class".to_string(),
                        "[data-testid='element']".to_string(),
                    ],
                    ax_metadata: None,
                    screenshot_ref: None,
                },
                verification: None,
                error_recovery: None,
            }],
            context: ExecutionContext::Fork,
            allowed_tools: vec![],
        };

        let markdown = builder.build(&skill);

        assert!(markdown.contains("\"selectors\":"));
        assert!(markdown.contains("#primary-selector"));
        assert!(markdown.contains(".fallback-class"));
        assert!(markdown.contains("[data-testid='element']"));
    }

    #[test]
    fn test_input_with_all_fields() {
        let mut builder = MarkdownBuilder::new();

        let skill = CompiledSkill {
            name: "full_input_test".to_string(),
            description: "Full input fields".to_string(),
            inputs: vec![SkillInput {
                name: "email".to_string(),
                type_hint: Some("email".to_string()),
                description: Some("User email address".to_string()),
                required: true,
                default: None,
            }],
            steps: vec![],
            context: ExecutionContext::Fork,
            allowed_tools: vec![],
        };

        let markdown = builder.build(&skill);

        assert!(markdown.contains("## Inputs"));
        assert!(markdown.contains("`{{email}}`"));
        assert!(markdown.contains("(email)"));
        assert!(markdown.contains(": User email address"));
        assert!(!markdown.contains("[optional]"));
    }

    #[test]
    fn test_optional_input_with_default() {
        let mut builder = MarkdownBuilder::new();

        let skill = CompiledSkill {
            name: "optional_input".to_string(),
            description: "Optional input test".to_string(),
            inputs: vec![SkillInput {
                name: "timeout".to_string(),
                type_hint: Some("number".to_string()),
                description: Some("Timeout in seconds".to_string()),
                required: false,
                default: Some("30".to_string()),
            }],
            steps: vec![],
            context: ExecutionContext::Fork,
            allowed_tools: vec![],
        };

        let markdown = builder.build(&skill);

        assert!(markdown.contains("`{{timeout}}`"));
        assert!(markdown.contains("(number)"));
        assert!(markdown.contains("[optional]"));
        assert!(markdown.contains(": Timeout in seconds"));
    }

    #[test]
    fn test_mixed_required_optional_inputs() {
        let mut builder = MarkdownBuilder::new();

        let skill = CompiledSkill {
            name: "mixed_inputs".to_string(),
            description: "Mixed inputs".to_string(),
            inputs: vec![
                SkillInput {
                    name: "username".to_string(),
                    type_hint: Some("string".to_string()),
                    description: Some("Required username".to_string()),
                    required: true,
                    default: None,
                },
                SkillInput {
                    name: "port".to_string(),
                    type_hint: Some("number".to_string()),
                    description: None,
                    required: false,
                    default: Some("8080".to_string()),
                },
            ],
            steps: vec![],
            context: ExecutionContext::Fork,
            allowed_tools: vec![],
        };

        let markdown = builder.build(&skill);

        assert!(markdown.contains("argument-hint: [username] [port]"));
        assert!(markdown.contains("`{{username}}`"));
        assert!(markdown.contains("`{{port}}`"));
        let username_pos = markdown.find("`{{username}}`").unwrap();
        let optional_marker = markdown.find("[optional]").unwrap();
        // username should appear before the optional marker
        assert!(username_pos < optional_marker);
    }

    #[test]
    fn test_multiple_allowed_tools() {
        let mut builder = MarkdownBuilder::new();

        let skill = CompiledSkill {
            name: "tools_test".to_string(),
            description: "Test allowed tools".to_string(),
            inputs: vec![],
            steps: vec![],
            context: ExecutionContext::Fork,
            allowed_tools: vec![
                "Read".to_string(),
                "Write".to_string(),
                "Bash".to_string(),
                "WebSearch".to_string(),
            ],
        };

        let markdown = builder.build(&skill);

        assert!(markdown.contains("allowed-tools: Read, Write, Bash, WebSearch"));
    }

    #[test]
    fn test_buffer_reuse() {
        let mut builder = MarkdownBuilder::new();

        let skill1 = CompiledSkill {
            name: "first".to_string(),
            description: "First skill".to_string(),
            inputs: vec![],
            steps: vec![],
            context: ExecutionContext::Fork,
            allowed_tools: vec![],
        };

        let markdown1 = builder.build(&skill1);
        assert!(markdown1.contains("name: first"));

        let skill2 = CompiledSkill {
            name: "second".to_string(),
            description: "Second skill".to_string(),
            inputs: vec![],
            steps: vec![],
            context: ExecutionContext::Inline,
            allowed_tools: vec![],
        };

        let markdown2 = builder.build(&skill2);
        assert!(markdown2.contains("name: second"));
        assert!(!markdown2.contains("name: first")); // Buffer should be cleared
    }

    #[test]
    fn test_technical_context_details() {
        let mut builder = MarkdownBuilder::new();

        let skill = CompiledSkill {
            name: "details_test".to_string(),
            description: "Test details tag".to_string(),
            inputs: vec![],
            steps: vec![CompiledStep {
                number: 1,
                narrative: "Test".to_string(),
                technical_context: TechnicalContext {
                    selectors: vec![],
                    ax_metadata: None,
                    screenshot_ref: None,
                },
                verification: None,
                error_recovery: None,
            }],
            context: ExecutionContext::Fork,
            allowed_tools: vec![],
        };

        let markdown = builder.build(&skill);

        assert!(markdown.contains("<details>"));
        assert!(markdown.contains("<summary>Technical Context</summary>"));
        assert!(markdown.contains("</details>"));
        assert!(markdown.contains("```json"));
    }

    #[test]
    fn test_disable_model_invocation_flag() {
        let mut builder = MarkdownBuilder::new();

        let skill = CompiledSkill {
            name: "invocation_test".to_string(),
            description: "Test model invocation".to_string(),
            inputs: vec![],
            steps: vec![],
            context: ExecutionContext::Fork,
            allowed_tools: vec![],
        };

        let markdown = builder.build(&skill);

        assert!(markdown.contains("disable-model-invocation: false"));
    }

    #[test]
    fn test_step_with_all_components() {
        let mut builder = MarkdownBuilder::new();

        let skill = CompiledSkill {
            name: "complete_step".to_string(),
            description: "Complete step test".to_string(),
            inputs: vec![],
            steps: vec![CompiledStep {
                number: 1,
                narrative: "Complete action".to_string(),
                technical_context: TechnicalContext {
                    selectors: vec!["#element".to_string()],
                    ax_metadata: Some(AxMetadata {
                        role: Some("AXButton".to_string()),
                        title: Some("Action".to_string()),
                        identifier: Some("action-btn".to_string()),
                        window_title: Some("App".to_string()),
                    }),
                    screenshot_ref: None,
                },
                verification: Some(crate::verification::hoare_triple_generator::VerificationBlock {
                    verification_type: "state_check".to_string(),
                    timeout_ms: 2000,
                    conditions: vec![],
                }),
                error_recovery: Some(super::super::skill_compiler::ErrorRecovery {
                    trigger_condition: "Failure".to_string(),
                    recovery_action: "Retry".to_string(),
                    max_retries: 3,
                }),
            }],
            context: ExecutionContext::Fork,
            allowed_tools: vec![],
        };

        let markdown = builder.build(&skill);

        // All components should be present
        assert!(markdown.contains("## Step 1"));
        assert!(markdown.contains("Complete action"));
        assert!(markdown.contains("Technical Context"));
        assert!(markdown.contains("### Verification"));
        assert!(markdown.contains("### Error Recovery"));
    }
}
