//! Multi-Pass Skill Compiler

use crate::chunking::action_clustering::UnitTask;
use crate::synthesis::variable_extraction::ExtractedVariable;
use crate::verification::hoare_triple_generator::VerificationBlock;
use serde::{Deserialize, Serialize};

/// Skill compilation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompiledSkill {
    /// Skill name
    pub name: String,
    /// Description
    pub description: String,
    /// Input variables
    pub inputs: Vec<SkillInput>,
    /// Compiled steps
    pub steps: Vec<CompiledStep>,
    /// Execution context
    pub context: ExecutionContext,
    /// Allowed tools
    pub allowed_tools: Vec<String>,
}

/// Skill input variable
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillInput {
    /// Variable name
    pub name: String,
    /// Type hint
    pub type_hint: Option<String>,
    /// Description
    pub description: Option<String>,
    /// Whether required
    pub required: bool,
    /// Default value
    pub default: Option<String>,
}

/// Compiled step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompiledStep {
    /// Step number
    pub number: usize,
    /// Human-readable narrative
    pub narrative: String,
    /// Technical context
    pub technical_context: TechnicalContext,
    /// Verification block
    pub verification: Option<VerificationBlock>,
    /// Error recovery (if applicable)
    pub error_recovery: Option<ErrorRecovery>,
}

/// Technical context for a step
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TechnicalContext {
    /// Element selectors
    pub selectors: Vec<String>,
    /// AX metadata
    pub ax_metadata: Option<AxMetadata>,
    /// Screenshot reference (if available)
    pub screenshot_ref: Option<String>,
}

/// AX metadata for a step
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AxMetadata {
    pub role: Option<String>,
    pub title: Option<String>,
    pub identifier: Option<String>,
    pub window_title: Option<String>,
}

/// Error recovery block
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErrorRecovery {
    /// Condition that triggers recovery
    pub trigger_condition: String,
    /// Recovery action
    pub recovery_action: String,
    /// Max retries
    pub max_retries: u8,
}

/// Execution context
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ExecutionContext {
    /// Fork a new process
    Fork,
    /// Run inline
    Inline,
}

impl Default for ExecutionContext {
    fn default() -> Self {
        Self::Fork
    }
}

/// Compiler configuration
#[derive(Debug, Clone)]
pub struct CompilerConfig {
    /// Model for semantic pass
    pub semantic_model: String,
    /// Temperature for generation
    pub temperature: f32,
    /// Whether to include screenshots
    pub include_screenshots: bool,
}

impl Default for CompilerConfig {
    fn default() -> Self {
        Self {
            semantic_model: "claude-sonnet-4-5-20250929".to_string(),
            temperature: 0.3,
            include_screenshots: false,
        }
    }
}

/// Multi-pass skill compiler
pub struct SkillCompiler {
    /// Configuration
    pub config: CompilerConfig,
}

impl SkillCompiler {
    /// Create with default config
    pub fn new() -> Self {
        Self {
            config: CompilerConfig::default(),
        }
    }

    /// Create with custom config
    pub fn with_config(config: CompilerConfig) -> Self {
        Self { config }
    }

    /// Compile a skill from unit tasks
    pub fn compile(
        &self,
        name: &str,
        tasks: &[UnitTask],
        variables: &[ExtractedVariable],
    ) -> CompiledSkill {
        // Pass 1: Generate semantic narratives
        let narratives = self.pass_semantic(tasks);

        // Pass 2: Extract verification conditions
        let verifications = self.pass_verification(tasks);

        // Pass 3: Detect and synthesize error recovery
        let recoveries = self.pass_correction(tasks);

        // Pass 4: Bind variables
        let bound_narratives = self.pass_variable_binding(&narratives, variables);

        // Assemble steps
        let steps = self.assemble_steps(tasks, bound_narratives, verifications, recoveries);

        // Generate inputs from variables
        let inputs = self.generate_inputs(variables);

        // Infer allowed tools
        let allowed_tools = self.infer_allowed_tools(tasks);

        // Generate description
        let description = self.generate_description(name, tasks);

        CompiledSkill {
            name: name.to_string(),
            description,
            inputs,
            steps,
            context: ExecutionContext::Fork,
            allowed_tools,
        }
    }

    /// Pass 1: Generate semantic narratives
    fn pass_semantic(&self, tasks: &[UnitTask]) -> Vec<String> {
        tasks
            .iter()
            .map(|task| self.generate_narrative(task))
            .collect()
    }

    /// Generate a context-aware narrative for a task
    fn generate_narrative(&self, task: &UnitTask) -> String {
        use crate::analysis::intent_binding::Action;

        // Extract target information from semantic data
        let target = self.extract_target_description(task);
        let window_context = self.extract_window_context(task);

        let narrative = match &task.primary_action {
            Action::Click { element_name, .. } => {
                if !element_name.is_empty() {
                    format!("Click on \"{}\"", element_name)
                } else {
                    format!("Click on {}", target)
                }
            }
            Action::DoubleClick { element_name, .. } => {
                if !element_name.is_empty() {
                    format!("Double-click on \"{}\" to open or select", element_name)
                } else {
                    format!("Double-click on {} to open or select", target)
                }
            }
            Action::RightClick { element_name, .. } => {
                if !element_name.is_empty() {
                    format!("Right-click on \"{}\" to open context menu", element_name)
                } else {
                    format!("Right-click on {} to open context menu", target)
                }
            }
            Action::Fill { field_name, value, .. } => {
                if value.len() > 30 {
                    format!("Enter text into \"{}\" field", field_name)
                } else {
                    format!("Enter \"{}\" into \"{}\"", value, field_name)
                }
            }
            Action::Select { menu_name, option, .. } => {
                format!("Select \"{}\" from \"{}\"", option, menu_name)
            }
            Action::Type { text, .. } => {
                if text.len() > 30 {
                    format!("Type text into {}", target)
                } else {
                    format!("Type \"{}\"", text)
                }
            }
            Action::Scroll { direction, .. } => {
                format!("Scroll {:?} in {}", direction, target)
            }
            Action::Shortcut { keys, .. } => {
                let (modifiers, key) = self.parse_shortcut_keys(keys);
                let purpose = self.infer_shortcut_purpose(&modifiers, &key);
                // Format modifiers for display
                let formatted_keys: Vec<String> = modifiers
                    .iter()
                    .map(|m| self.format_modifier(m))
                    .chain(std::iter::once(key.to_uppercase()))
                    .collect();
                format!("{} {} to {}", self.action_to_verb(&task.primary_action), formatted_keys.join("+"), purpose)
            }
            Action::DragDrop { source, target: drop_target, .. } => {
                format!("Drag \"{}\" to \"{}\"", source, drop_target)
            }
            Action::Search { candidate_elements, .. } => {
                if candidate_elements.is_empty() {
                    "Search for element".to_string()
                } else {
                    format!("Search among: {}", candidate_elements.join(", "))
                }
            }
            Action::Unknown { description, .. } => {
                description.clone()
            }
        };

        // Add window context if available and relevant
        if let Some(window) = window_context {
            if !narrative.contains(&window) {
                format!("{} in {}", narrative, window)
            } else {
                narrative
            }
        } else {
            narrative
        }
    }

    /// Extract a human-readable target description from task semantic data
    fn extract_target_description(&self, task: &UnitTask) -> String {
        // Try to get the most descriptive information from semantic data
        for event in &task.events {
            if let Some(ref semantic) = event.semantic {
                // Prefer title, then identifier, then role
                if let Some(ref title) = semantic.title {
                    if !title.is_empty() {
                        return format!("\"{}\"", title);
                    }
                }
                if let Some(ref identifier) = semantic.identifier {
                    if !identifier.is_empty() {
                        return format!("element \"{}\"", identifier);
                    }
                }
                if let Some(ref role) = semantic.ax_role {
                    return self.role_to_description(role);
                }
            }
        }

        // Fallback to the task's own description
        if !task.description.is_empty() {
            task.description.clone()
        } else {
            "the element".to_string()
        }
    }

    /// Convert AX role to human-readable description
    fn role_to_description(&self, role: &str) -> String {
        let role_lower = role.to_lowercase();
        match role_lower.as_str() {
            "axbutton" => "the button".to_string(),
            "axtextfield" | "axtextarea" => "the text field".to_string(),
            "axcheckbox" => "the checkbox".to_string(),
            "axradiobutton" => "the radio button".to_string(),
            "axcombobox" | "axpopupbutton" => "the dropdown".to_string(),
            "axmenuitem" => "the menu item".to_string(),
            "axmenu" => "the menu".to_string(),
            "axlink" => "the link".to_string(),
            "aximage" => "the image".to_string(),
            "axtab" | "axtabgroup" => "the tab".to_string(),
            "axlist" | "axtable" => "the list".to_string(),
            "axrow" => "the row".to_string(),
            "axcell" => "the cell".to_string(),
            "axslider" => "the slider".to_string(),
            "axscrollbar" => "the scrollbar".to_string(),
            "axstatictext" => "the text".to_string(),
            "axwindow" => "the window".to_string(),
            "axtoolbar" => "the toolbar".to_string(),
            _ => format!("the {}", role.trim_start_matches("AX").to_lowercase()),
        }
    }

    /// Extract window context from task
    fn extract_window_context(&self, task: &UnitTask) -> Option<String> {
        for event in &task.events {
            if let Some(ref semantic) = event.semantic {
                if let Some(ref window_title) = semantic.window_title {
                    if !window_title.is_empty() {
                        return Some(window_title.clone());
                    }
                }
            }
        }
        None
    }

    /// Parse shortcut keys into modifiers and main key
    fn parse_shortcut_keys(&self, keys: &[String]) -> (Vec<String>, String) {
        let modifiers: Vec<String> = keys
            .iter()
            .filter(|k| {
                let lower = k.to_lowercase();
                lower == "command" || lower == "cmd" || lower == "control" || lower == "ctrl"
                    || lower == "option" || lower == "alt" || lower == "shift"
            })
            .cloned()
            .collect();

        let main_key = keys
            .iter()
            .find(|k| {
                let lower = k.to_lowercase();
                !(lower == "command" || lower == "cmd" || lower == "control" || lower == "ctrl"
                    || lower == "option" || lower == "alt" || lower == "shift")
            })
            .cloned()
            .unwrap_or_default();

        (modifiers, main_key)
    }

    /// Format modifier key for display
    fn format_modifier(&self, modifier: &str) -> String {
        match modifier.to_lowercase().as_str() {
            "command" | "cmd" => "Cmd".to_string(),
            "control" | "ctrl" => "Ctrl".to_string(),
            "option" | "alt" => "Option".to_string(),
            "shift" => "Shift".to_string(),
            _ => modifier.to_string(),
        }
    }

    /// Infer the purpose of a keyboard shortcut
    fn infer_shortcut_purpose(&self, modifiers: &[String], key: &str) -> String {
        let has_cmd = modifiers.iter().any(|m| m.to_lowercase() == "command");
        let has_shift = modifiers.iter().any(|m| m.to_lowercase() == "shift");
        let has_option = modifiers.iter().any(|m| m.to_lowercase() == "option");

        let key_lower = key.to_lowercase();

        if has_cmd {
            match key_lower.as_str() {
                "c" => return "copy".to_string(),
                "v" => return "paste".to_string(),
                "x" => return "cut".to_string(),
                "z" if !has_shift => return "undo".to_string(),
                "z" if has_shift => return "redo".to_string(),
                "a" => return "select all".to_string(),
                "s" => return "save".to_string(),
                "o" => return "open".to_string(),
                "n" => return "create new".to_string(),
                "w" => return "close window".to_string(),
                "q" => return "quit application".to_string(),
                "f" => return "find".to_string(),
                "p" => return "print".to_string(),
                "t" if !has_shift => return "open new tab".to_string(),
                "t" if has_shift => return "reopen closed tab".to_string(),
                "r" => return "refresh".to_string(),
                "," => return "open preferences".to_string(),
                "tab" => return "switch application".to_string(),
                "`" => return "switch window".to_string(),
                _ => {}
            }
        }

        if has_option && has_cmd {
            match key_lower.as_str() {
                "escape" | "esc" => return "force quit".to_string(),
                "h" => return "hide other windows".to_string(),
                _ => {}
            }
        }

        match key_lower.as_str() {
            "enter" | "return" => "confirm".to_string(),
            "escape" | "esc" => "cancel or close".to_string(),
            "tab" => "move to next field".to_string(),
            "space" => "activate or toggle".to_string(),
            "delete" | "backspace" => "delete".to_string(),
            _ => "execute shortcut".to_string(),
        }
    }

    /// Convert action to verb phrase for narrative generation
    fn action_to_verb(&self, action: &crate::analysis::intent_binding::Action) -> &str {
        use crate::analysis::intent_binding::Action;
        match action {
            Action::Click { .. } => "Click",
            Action::DoubleClick { .. } => "Double-click",
            Action::RightClick { .. } => "Right-click",
            Action::Fill { .. } => "Enter",
            Action::Select { .. } => "Select",
            Action::Type { .. } => "Type",
            Action::Scroll { .. } => "Scroll",
            Action::Shortcut { .. } => "Press",
            Action::DragDrop { .. } => "Drag",
            Action::Search { .. } => "Find",
            Action::Unknown { .. } => "Perform",
        }
    }

    /// Pass 2: Extract verification conditions
    fn pass_verification(&self, tasks: &[UnitTask]) -> Vec<Option<VerificationBlock>> {
        tasks
            .iter()
            .map(|task| {
                // Extract from semantic data in events
                task.events.last().and_then(|e| {
                    e.semantic.as_ref().map(|sem| VerificationBlock {
                        verification_type: "state_assertion".to_string(),
                        timeout_ms: 5000,
                        conditions: vec![crate::verification::hoare_triple_generator::Condition {
                            scope: "target_element".to_string(),
                            attribute: "AXRole".to_string(),
                            operator: "exists".to_string(),
                            expected_value: sem.ax_role.clone().unwrap_or_default(),
                        }],
                    })
                })
            })
            .collect()
    }

    /// Pass 3: Detect correction loops and generate recovery
    fn pass_correction(&self, tasks: &[UnitTask]) -> Vec<Option<ErrorRecovery>> {
        tasks
            .iter()
            .enumerate()
            .map(|(i, task)| self.detect_error_recovery(tasks, i, task))
            .collect()
    }

    /// Detect error recovery patterns for a task
    fn detect_error_recovery(
        &self,
        tasks: &[UnitTask],
        task_index: usize,
        task: &UnitTask,
    ) -> Option<ErrorRecovery> {
        // Pattern 1: Detect Cmd+Z (Undo) after this task
        if task_index + 1 < tasks.len()
            && self.is_undo_action(&tasks[task_index + 1])
        {
            return Some(ErrorRecovery {
                trigger_condition: "Action failed or unintended result".to_string(),
                recovery_action: "Undo and retry with verification".to_string(),
                max_retries: 2,
            });
        }

        // Pattern 2: Detect repeated similar actions (retry pattern)
        let similar_count = self.count_similar_tasks(tasks, task_index, task);
        if similar_count >= 2 {
            return Some(ErrorRecovery {
                trigger_condition: "Element not found or action did not complete".to_string(),
                recovery_action: format!(
                    "Retry up to {} times with exponential backoff",
                    similar_count
                ),
                max_retries: similar_count as u8,
            });
        }

        // Pattern 3: Detect escape key after task (cancel pattern)
        if self.has_cancel_after(tasks, task_index) {
            return Some(ErrorRecovery {
                trigger_condition: "Dialog or modal appeared unexpectedly".to_string(),
                recovery_action: "Press Escape to dismiss and retry".to_string(),
                max_retries: 1,
            });
        }

        // Pattern 4: Detect text correction (backspace followed by retype)
        if self.is_text_correction(task) {
            return Some(ErrorRecovery {
                trigger_condition: "Text input error detected".to_string(),
                recovery_action: "Clear field and retype".to_string(),
                max_retries: 3,
            });
        }

        None
    }

    /// Check if a task is an undo action
    fn is_undo_action(&self, task: &UnitTask) -> bool {
        use crate::analysis::intent_binding::Action;
        if let Action::Shortcut { keys, .. } = &task.primary_action {
            let has_cmd = keys.iter().any(|k| {
                let lower = k.to_lowercase();
                lower == "command" || lower == "cmd"
            });
            let has_z = keys.iter().any(|k| k.to_lowercase() == "z");
            has_cmd && has_z
        } else {
            false
        }
    }

    /// Count similar tasks (for retry pattern detection)
    fn count_similar_tasks(&self, tasks: &[UnitTask], index: usize, task: &UnitTask) -> usize {
        let window_start = index.saturating_sub(3);
        let window_end = (index + 3).min(tasks.len());

        tasks[window_start..window_end]
            .iter()
            .filter(|t| self.tasks_are_similar(t, task))
            .count()
    }

    /// Check if two tasks are similar (same action type and target)
    fn tasks_are_similar(&self, a: &UnitTask, b: &UnitTask) -> bool {
        // Same action variant
        std::mem::discriminant(&a.primary_action) == std::mem::discriminant(&b.primary_action)
            // And similar target (if both have semantic data)
            && a.events.first().and_then(|e| e.semantic.as_ref().and_then(|s| s.ax_role.as_ref()))
                == b.events.first().and_then(|e| e.semantic.as_ref().and_then(|s| s.ax_role.as_ref()))
    }

    /// Check if there's a cancel action (Escape) after this task
    fn has_cancel_after(&self, tasks: &[UnitTask], index: usize) -> bool {
        if index + 1 >= tasks.len() {
            return false;
        }

        use crate::analysis::intent_binding::Action;
        if let Action::Shortcut { keys, .. } = &tasks[index + 1].primary_action {
            // Check if it's just Escape (no modifiers)
            keys.len() == 1 && keys.iter().any(|k| {
                let lower = k.to_lowercase();
                lower == "escape" || lower == "esc"
            })
        } else {
            false
        }
    }

    /// Check if task contains text correction (backspace sequence)
    fn is_text_correction(&self, task: &UnitTask) -> bool {
        let mut backspace_count = 0;
        let mut typing_after_backspace = false;

        for event in &task.events {
            if event.raw.event_type.is_keyboard() {
                // Check for backspace in key events
                if let Some(ref semantic) = event.semantic {
                    if let Some(ref value) = semantic.value {
                        if value.contains('\u{8}') || value.to_lowercase() == "delete" {
                            backspace_count += 1;
                        } else if backspace_count > 0 {
                            typing_after_backspace = true;
                        }
                    }
                }
            }
        }

        backspace_count >= 2 && typing_after_backspace
    }

    /// Pass 4: Bind variables to narratives
    fn pass_variable_binding(
        &self,
        narratives: &[String],
        variables: &[ExtractedVariable],
    ) -> Vec<String> {
        narratives
            .iter()
            .map(|narrative| {
                let mut bound = narrative.clone();
                for var in variables {
                    bound = bound.replace(&var.detected_value, &format!("{{{{{}}}}}", var.name));
                }
                bound
            })
            .collect()
    }

    /// Assemble final steps
    fn assemble_steps(
        &self,
        tasks: &[UnitTask],
        narratives: Vec<String>,
        verifications: Vec<Option<VerificationBlock>>,
        recoveries: Vec<Option<ErrorRecovery>>,
    ) -> Vec<CompiledStep> {
        tasks
            .iter()
            .enumerate()
            .map(|(i, task)| {
                let technical_context = TechnicalContext {
                    selectors: self.extract_selectors(task),
                    ax_metadata: self.extract_ax_metadata(task),
                    screenshot_ref: None,
                };

                CompiledStep {
                    number: i + 1,
                    narrative: narratives.get(i).cloned().unwrap_or_default(),
                    technical_context,
                    verification: verifications.get(i).cloned().flatten(),
                    error_recovery: recoveries.get(i).cloned().flatten(),
                }
            })
            .collect()
    }

    /// Extract selectors from task
    fn extract_selectors(&self, task: &UnitTask) -> Vec<String> {
        task.events
            .iter()
            .filter_map(|e| {
                e.semantic.as_ref().and_then(|sem| {
                    sem.identifier
                        .clone()
                        .or_else(|| sem.title.clone())
                })
            })
            .collect()
    }

    /// Extract AX metadata from task
    fn extract_ax_metadata(&self, task: &UnitTask) -> Option<AxMetadata> {
        task.events.last().and_then(|e| {
            e.semantic.as_ref().map(|sem| AxMetadata {
                role: sem.ax_role.clone(),
                title: sem.title.clone(),
                identifier: sem.identifier.clone(),
                window_title: sem.window_title.clone(),
            })
        })
    }

    /// Generate inputs from variables
    fn generate_inputs(&self, variables: &[ExtractedVariable]) -> Vec<SkillInput> {
        variables
            .iter()
            .map(|var| SkillInput {
                name: var.name.clone(),
                type_hint: var.type_hint.clone(),
                description: var.description.clone(),
                required: true,
                default: None,
            })
            .collect()
    }

    /// Infer allowed tools from task types
    fn infer_allowed_tools(&self, _tasks: &[UnitTask]) -> Vec<String> {
        // Default minimal set
        vec!["Read".to_string(), "Bash".to_string()]
    }

    /// Generate skill description
    fn generate_description(&self, name: &str, tasks: &[UnitTask]) -> String {
        let step_count = tasks.len();
        format!(
            "AI-generated skill '{}'. Performs {} steps to complete the workflow.",
            name, step_count
        )
    }
}

impl Default for SkillCompiler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::Action;
    use crate::capture::types::EnrichedEvent;

    #[test]
    fn test_compiler_creation() {
        let compiler = SkillCompiler::new();
        assert!(!compiler.config.semantic_model.is_empty());
    }

    #[test]
    fn test_execution_context_default() {
        let context = ExecutionContext::default();
        assert!(matches!(context, ExecutionContext::Fork));
    }

    #[test]
    fn test_shortcut_purpose_inference() {
        let compiler = SkillCompiler::new();

        // Test common shortcuts
        let mods_cmd = vec!["command".to_string()];
        assert_eq!(compiler.infer_shortcut_purpose(&mods_cmd, "c"), "copy");
        assert_eq!(compiler.infer_shortcut_purpose(&mods_cmd, "v"), "paste");
        assert_eq!(compiler.infer_shortcut_purpose(&mods_cmd, "x"), "cut");
        assert_eq!(compiler.infer_shortcut_purpose(&mods_cmd, "z"), "undo");
        assert_eq!(compiler.infer_shortcut_purpose(&mods_cmd, "s"), "save");
        assert_eq!(compiler.infer_shortcut_purpose(&mods_cmd, "a"), "select all");

        // Test with shift modifier
        let mods_cmd_shift = vec!["command".to_string(), "shift".to_string()];
        assert_eq!(compiler.infer_shortcut_purpose(&mods_cmd_shift, "z"), "redo");

        // Test standalone keys
        let empty_mods: Vec<String> = vec![];
        assert_eq!(compiler.infer_shortcut_purpose(&empty_mods, "escape"), "cancel or close");
        assert_eq!(compiler.infer_shortcut_purpose(&empty_mods, "enter"), "confirm");
        assert_eq!(compiler.infer_shortcut_purpose(&empty_mods, "tab"), "move to next field");
    }

    #[test]
    fn test_role_to_description() {
        let compiler = SkillCompiler::new();

        assert_eq!(compiler.role_to_description("AXButton"), "the button");
        assert_eq!(compiler.role_to_description("AXTextField"), "the text field");
        assert_eq!(compiler.role_to_description("AXCheckBox"), "the checkbox");
        assert_eq!(compiler.role_to_description("AXMenuItem"), "the menu item");
        assert_eq!(compiler.role_to_description("AXLink"), "the link");
        assert_eq!(compiler.role_to_description("AXCustomWidget"), "the customwidget");
    }

    #[test]
    fn test_parse_shortcut_keys() {
        let compiler = SkillCompiler::new();

        let keys = vec!["command".to_string(), "shift".to_string(), "s".to_string()];
        let (modifiers, main_key) = compiler.parse_shortcut_keys(&keys);
        assert_eq!(modifiers.len(), 2);
        assert!(modifiers.contains(&"command".to_string()));
        assert!(modifiers.contains(&"shift".to_string()));
        assert_eq!(main_key, "s");

        // Single key shortcut
        let single = vec!["escape".to_string()];
        let (mods, key) = compiler.parse_shortcut_keys(&single);
        assert!(mods.is_empty());
        assert_eq!(key, "escape");
    }

    #[test]
    fn test_format_modifier() {
        let compiler = SkillCompiler::new();

        assert_eq!(compiler.format_modifier("command"), "Cmd");
        assert_eq!(compiler.format_modifier("cmd"), "Cmd");
        assert_eq!(compiler.format_modifier("control"), "Ctrl");
        assert_eq!(compiler.format_modifier("option"), "Option");
        assert_eq!(compiler.format_modifier("shift"), "Shift");
        assert_eq!(compiler.format_modifier("unknown"), "unknown");
    }

    #[test]
    fn test_error_recovery_fields() {
        let recovery = ErrorRecovery {
            trigger_condition: "Element not found".to_string(),
            recovery_action: "Retry with wait".to_string(),
            max_retries: 3,
        };

        assert!(!recovery.trigger_condition.is_empty());
        assert!(!recovery.recovery_action.is_empty());
        assert_eq!(recovery.max_retries, 3);
    }

    #[test]
    fn test_compiled_skill_structure() {
        let skill = CompiledSkill {
            name: "test_skill".to_string(),
            description: "Test description".to_string(),
            inputs: vec![SkillInput {
                name: "username".to_string(),
                type_hint: Some("string".to_string()),
                description: Some("User's name".to_string()),
                required: true,
                default: None,
            }],
            steps: vec![CompiledStep {
                number: 1,
                narrative: "Click on login button".to_string(),
                technical_context: TechnicalContext {
                    selectors: vec!["#login-btn".to_string()],
                    ax_metadata: Some(AxMetadata {
                        role: Some("AXButton".to_string()),
                        title: Some("Login".to_string()),
                        identifier: Some("login-btn".to_string()),
                        window_title: Some("Login Window".to_string()),
                    }),
                    screenshot_ref: None,
                },
                verification: None,
                error_recovery: Some(ErrorRecovery {
                    trigger_condition: "Button not visible".to_string(),
                    recovery_action: "Scroll down".to_string(),
                    max_retries: 2,
                }),
            }],
            context: ExecutionContext::Fork,
            allowed_tools: vec!["Read".to_string()],
        };

        assert_eq!(skill.name, "test_skill");
        assert_eq!(skill.steps.len(), 1);
        assert!(skill.steps[0].error_recovery.is_some());
    }

    #[test]
    fn test_compiler_config() {
        let custom_config = CompilerConfig {
            semantic_model: "custom-model".to_string(),
            temperature: 0.5,
            include_screenshots: true,
        };

        let compiler = SkillCompiler::with_config(custom_config);
        assert_eq!(compiler.config.semantic_model, "custom-model");
        assert!(compiler.config.include_screenshots);
    }

    #[test]
    fn test_default_compiler() {
        let compiler = SkillCompiler::default();
        assert!(!compiler.config.semantic_model.is_empty());
        assert!(compiler.config.temperature > 0.0);
    }

    #[test]
    fn test_compile_with_empty_tasks() {
        let compiler = SkillCompiler::new();
        let tasks: Vec<UnitTask> = vec![];
        let variables: Vec<ExtractedVariable> = vec![];

        let skill = compiler.compile("empty_skill", &tasks, &variables);

        assert_eq!(skill.name, "empty_skill");
        assert_eq!(skill.steps.len(), 0);
        assert_eq!(skill.inputs.len(), 0);
        assert!(skill.description.contains("empty_skill"));
    }

    #[test]
    fn test_compile_single_click_task() {
        use crate::analysis::intent_binding::Action;
        use crate::capture::types::{
            CursorState, EventType, ModifierFlags, RawEvent, SemanticContext,
        };
        use crate::time::timebase::Timestamp;

        let compiler = SkillCompiler::new();

        // Create a simple click task
        let raw = RawEvent::mouse(
            Timestamp::from_ticks(1000),
            EventType::LeftMouseDown,
            100.0,
            200.0,
            CursorState::Arrow,
            ModifierFlags::default(),
            1,
        );

        let semantic = SemanticContext {
            ax_role: Some("AXButton".to_string()),
            title: Some("Submit".to_string()),
            ..Default::default()
        };

        let event = EnrichedEvent::new(raw, 0).with_semantic(semantic);

        let task = UnitTask {
            id: uuid::Uuid::new_v4(),
            name: "Click Submit".to_string(),
            description: "Click the submit button".to_string(),
            primary_action: Action::Click {
                element_name: "Submit".to_string(),
                element_role: "AXButton".to_string(),
                confidence: 0.9,
            },
            events: vec![event],
            start_time: 1000,
            end_time: 1100,
            nesting_level: 0,
            parent_id: None,
        };

        let skill = compiler.compile("click_skill", &[task], &[]);

        assert_eq!(skill.name, "click_skill");
        assert_eq!(skill.steps.len(), 1);
        assert!(skill.steps[0].narrative.contains("Submit"));
    }

    #[test]
    fn test_narrative_generation_for_fill_action() {
        use crate::analysis::intent_binding::Action;
        use crate::capture::types::{EventType, ModifierFlags, RawEvent, SemanticContext};
        use crate::time::timebase::Timestamp;

        let compiler = SkillCompiler::new();

        let raw = RawEvent::keyboard(
            Timestamp::from_ticks(1000),
            EventType::KeyDown,
            42,
            Some('a'),
            ModifierFlags::default(),
            (100.0, 200.0),
        );

        let semantic = SemanticContext {
            ax_role: Some("AXTextField".to_string()),
            title: Some("Email".to_string()),
            ..Default::default()
        };

        let event = EnrichedEvent::new(raw, 0).with_semantic(semantic);

        let task = UnitTask {
            id: uuid::Uuid::new_v4(),
            name: "Fill Email".to_string(),
            description: "Enter email address".to_string(),
            primary_action: Action::Fill {
                field_name: "Email".to_string(),
                value: "user@example.com".to_string(),
                confidence: 0.95,
            },
            events: vec![event],
            start_time: 1000,
            end_time: 1500,
            nesting_level: 0,
            parent_id: None,
        };

        let narrative = compiler.generate_narrative(&task);

        assert!(narrative.contains("Email"));
        assert!(narrative.contains("user@example.com"));
    }

    #[test]
    fn test_narrative_generation_for_shortcut_action() {
        use crate::analysis::intent_binding::Action;

        let compiler = SkillCompiler::new();

        let task = UnitTask {
            id: uuid::Uuid::new_v4(),
            name: "Save".to_string(),
            description: "Save document".to_string(),
            primary_action: Action::Shortcut {
                keys: vec!["command".to_string(), "s".to_string()],
                confidence: 1.0,
            },
            events: vec![],
            start_time: 1000,
            end_time: 1100,
            nesting_level: 0,
            parent_id: None,
        };

        let narrative = compiler.generate_narrative(&task);

        assert!(narrative.contains("save"));
        // Now uses formatted modifiers: "Cmd+S" instead of "command+s"
        assert!(narrative.contains("Cmd+S"), "Expected 'Cmd+S' in narrative: {}", narrative);
    }

    #[test]
    fn test_error_recovery_undo_detection() {
        use crate::analysis::intent_binding::Action;

        let compiler = SkillCompiler::new();

        // Create a task followed by an undo
        let task1 = UnitTask {
            id: uuid::Uuid::new_v4(),
            name: "Click".to_string(),
            description: "Click button".to_string(),
            primary_action: Action::Click {
                element_name: "Button".to_string(),
                element_role: "AXButton".to_string(),
                confidence: 0.8,
            },
            events: vec![],
            start_time: 1000,
            end_time: 1100,
            nesting_level: 0,
            parent_id: None,
        };

        let task2_undo = UnitTask {
            id: uuid::Uuid::new_v4(),
            name: "Undo".to_string(),
            description: "Undo action".to_string(),
            primary_action: Action::Shortcut {
                keys: vec!["command".to_string(), "z".to_string()],
                confidence: 1.0,
            },
            events: vec![],
            start_time: 1200,
            end_time: 1300,
            nesting_level: 0,
            parent_id: None,
        };

        let tasks = vec![task1, task2_undo];
        let recovery = compiler.detect_error_recovery(&tasks, 0, &tasks[0]);

        assert!(recovery.is_some());
        let recovery = recovery.unwrap();
        assert!(recovery.recovery_action.contains("Undo"));
        assert!(recovery.max_retries > 0);
    }

    #[test]
    fn test_error_recovery_retry_pattern_detection() {
        use crate::analysis::intent_binding::Action;
        use crate::capture::types::{
            CursorState, EventType, ModifierFlags, RawEvent, SemanticContext,
        };
        use crate::time::timebase::Timestamp;

        let compiler = SkillCompiler::new();

        // Create multiple similar tasks (retry pattern)
        let mut tasks = vec![];
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

            let semantic = SemanticContext {
                ax_role: Some("AXButton".to_string()),
                ..Default::default()
            };

            let event = EnrichedEvent::new(raw, i).with_semantic(semantic);

            tasks.push(UnitTask {
                id: uuid::Uuid::new_v4(),
                name: "Click".to_string(),
                description: "Click button".to_string(),
                primary_action: Action::Click {
                    element_name: "Button".to_string(),
                    element_role: "AXButton".to_string(),
                    confidence: 0.8,
                },
                events: vec![event],
                start_time: 1000 * (i + 1),
                end_time: 1000 * (i + 1) + 100,
                nesting_level: 0,
                parent_id: None,
            });
        }

        let recovery = compiler.detect_error_recovery(&tasks, 1, &tasks[1]);

        assert!(recovery.is_some());
        let recovery = recovery.unwrap();
        assert!(recovery.max_retries >= 2);
    }

    #[test]
    fn test_error_recovery_escape_detection() {
        use crate::analysis::intent_binding::Action;

        let compiler = SkillCompiler::new();

        let task1 = UnitTask {
            id: uuid::Uuid::new_v4(),
            name: "Click".to_string(),
            description: "Click button".to_string(),
            primary_action: Action::Click {
                element_name: "Button".to_string(),
                element_role: "AXButton".to_string(),
                confidence: 0.8,
            },
            events: vec![],
            start_time: 1000,
            end_time: 1100,
            nesting_level: 0,
            parent_id: None,
        };

        let task2_escape = UnitTask {
            id: uuid::Uuid::new_v4(),
            name: "Cancel".to_string(),
            description: "Press escape".to_string(),
            primary_action: Action::Shortcut {
                keys: vec!["escape".to_string()],
                confidence: 1.0,
            },
            events: vec![],
            start_time: 1200,
            end_time: 1300,
            nesting_level: 0,
            parent_id: None,
        };

        let tasks = vec![task1, task2_escape];
        let recovery = compiler.detect_error_recovery(&tasks, 0, &tasks[0]);

        assert!(recovery.is_some());
        let recovery = recovery.unwrap();
        assert!(recovery.trigger_condition.contains("Dialog") || recovery.trigger_condition.contains("modal"));
        assert_eq!(recovery.max_retries, 1);
    }

    #[test]
    fn test_variable_binding_in_narratives() {
        use crate::synthesis::variable_extraction::{ExtractedVariable, VariableSource};

        let compiler = SkillCompiler::new();

        let narratives = vec![
            "Enter user@example.com into Email field".to_string(),
            "Click on Submit button".to_string(),
        ];

        let variables = vec![ExtractedVariable {
            name: "email".to_string(),
            detected_value: "user@example.com".to_string(),
            source: VariableSource::TypedString,
            confidence: 0.9,
            type_hint: Some("email".to_string()),
            description: Some("User's email address".to_string()),
        }];

        let bound = compiler.pass_variable_binding(&narratives, &variables);

        assert_eq!(bound.len(), 2);
        assert!(bound[0].contains("{{email}}"));
        assert!(!bound[0].contains("user@example.com"));
        assert_eq!(bound[1], "Click on Submit button"); // Unchanged
    }

    #[test]
    fn test_generate_inputs_from_variables() {
        use crate::synthesis::variable_extraction::{ExtractedVariable, VariableSource};

        let compiler = SkillCompiler::new();

        let variables = vec![
            ExtractedVariable {
                name: "username".to_string(),
                detected_value: "john_doe".to_string(),
                source: VariableSource::TypedString,
                confidence: 0.95,
                type_hint: Some("string".to_string()),
                description: Some("Username for login".to_string()),
            },
            ExtractedVariable {
                name: "password".to_string(),
                detected_value: "secret123".to_string(),
                source: VariableSource::TypedString,
                confidence: 0.9,
                type_hint: Some("password".to_string()),
                description: Some("User password".to_string()),
            },
        ];

        let inputs = compiler.generate_inputs(&variables);

        assert_eq!(inputs.len(), 2);
        assert_eq!(inputs[0].name, "username");
        assert_eq!(inputs[0].type_hint, Some("string".to_string()));
        assert!(inputs[0].required);
        assert_eq!(inputs[1].name, "password");
        assert_eq!(inputs[1].type_hint, Some("password".to_string()));
    }

    #[test]
    fn test_infer_allowed_tools() {
        let compiler = SkillCompiler::new();

        let tasks: Vec<UnitTask> = vec![];
        let tools = compiler.infer_allowed_tools(&tasks);

        // Should always include basic tools
        assert!(tools.contains(&"Read".to_string()));
        assert!(tools.contains(&"Bash".to_string()));
    }

    #[test]
    fn test_generate_description() {
        let compiler = SkillCompiler::new();

        let tasks: Vec<UnitTask> = vec![
            UnitTask {
                id: uuid::Uuid::new_v4(),
                name: "Task1".to_string(),
                description: "".to_string(),
                primary_action: Action::Unknown {
                    description: "test".to_string(),
                    confidence: 0.5,
                },
                events: vec![],
                start_time: 0,
                end_time: 0,
                nesting_level: 0,
                parent_id: None,
            },
            UnitTask {
                id: uuid::Uuid::new_v4(),
                name: "Task2".to_string(),
                description: "".to_string(),
                primary_action: Action::Unknown {
                    description: "test".to_string(),
                    confidence: 0.5,
                },
                events: vec![],
                start_time: 0,
                end_time: 0,
                nesting_level: 0,
                parent_id: None,
            },
        ];

        let description = compiler.generate_description("test_workflow", &tasks);

        assert!(description.contains("test_workflow"));
        assert!(description.contains("2"));
    }

    #[test]
    fn test_is_undo_action_detection() {
        use crate::analysis::intent_binding::Action;

        let compiler = SkillCompiler::new();

        // Test undo shortcut
        let undo_task = UnitTask {
            id: uuid::Uuid::new_v4(),
            name: "Undo".to_string(),
            description: "".to_string(),
            primary_action: Action::Shortcut {
                keys: vec!["command".to_string(), "z".to_string()],
                confidence: 1.0,
            },
            events: vec![],
            start_time: 0,
            end_time: 0,
            nesting_level: 0,
            parent_id: None,
        };

        assert!(compiler.is_undo_action(&undo_task));

        // Test non-undo action
        let click_task = UnitTask {
            id: uuid::Uuid::new_v4(),
            name: "Click".to_string(),
            description: "".to_string(),
            primary_action: Action::Click {
                element_name: "Button".to_string(),
                element_role: "AXButton".to_string(),
                confidence: 0.8,
            },
            events: vec![],
            start_time: 0,
            end_time: 0,
            nesting_level: 0,
            parent_id: None,
        };

        assert!(!compiler.is_undo_action(&click_task));
    }

    #[test]
    fn test_tasks_are_similar() {
        use crate::analysis::intent_binding::Action;
        use crate::capture::types::{
            CursorState, EventType, ModifierFlags, RawEvent, SemanticContext,
        };
        use crate::time::timebase::Timestamp;

        let compiler = SkillCompiler::new();

        // Create two similar click tasks
        let raw1 = RawEvent::mouse(
            Timestamp::from_ticks(1000),
            EventType::LeftMouseDown,
            100.0,
            200.0,
            CursorState::Arrow,
            ModifierFlags::default(),
            1,
        );

        let semantic1 = SemanticContext {
            ax_role: Some("AXButton".to_string()),
            ..Default::default()
        };

        let event1 = EnrichedEvent::new(raw1, 0).with_semantic(semantic1);

        let task1 = UnitTask {
            id: uuid::Uuid::new_v4(),
            name: "Click".to_string(),
            description: "".to_string(),
            primary_action: Action::Click {
                element_name: "Button1".to_string(),
                element_role: "AXButton".to_string(),
                confidence: 0.8,
            },
            events: vec![event1],
            start_time: 1000,
            end_time: 1100,
            nesting_level: 0,
            parent_id: None,
        };

        let raw2 = RawEvent::mouse(
            Timestamp::from_ticks(2000),
            EventType::LeftMouseDown,
            150.0,
            250.0,
            CursorState::Arrow,
            ModifierFlags::default(),
            1,
        );

        let semantic2 = SemanticContext {
            ax_role: Some("AXButton".to_string()),
            ..Default::default()
        };

        let event2 = EnrichedEvent::new(raw2, 1).with_semantic(semantic2);

        let task2 = UnitTask {
            id: uuid::Uuid::new_v4(),
            name: "Click".to_string(),
            description: "".to_string(),
            primary_action: Action::Click {
                element_name: "Button2".to_string(),
                element_role: "AXButton".to_string(),
                confidence: 0.8,
            },
            events: vec![event2],
            start_time: 2000,
            end_time: 2100,
            nesting_level: 0,
            parent_id: None,
        };

        assert!(compiler.tasks_are_similar(&task1, &task2));

        // Create a different type of task
        let task3 = UnitTask {
            id: uuid::Uuid::new_v4(),
            name: "Type".to_string(),
            description: "".to_string(),
            primary_action: Action::Type {
                text: "hello".to_string(),
                confidence: 0.9,
            },
            events: vec![],
            start_time: 3000,
            end_time: 3100,
            nesting_level: 0,
            parent_id: None,
        };

        assert!(!compiler.tasks_are_similar(&task1, &task3));
    }
}
