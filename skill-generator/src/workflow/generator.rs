//! Skill Generator
//!
//! Transforms recordings into SKILL.md files with proper variable extraction,
//! selector ranking, and verification blocks.

use super::recording::Recording;
use crate::analysis::intent_binding::{Action, ScrollDirection};
use crate::capture::types::{EnrichedEvent, EventType};
use crate::codegen::markdown_builder::MarkdownBuilder;
use crate::codegen::skill_compiler::{AxMetadata, CompiledSkill, CompiledStep, ExecutionContext as CompiledContext, SkillInput, TechnicalContext};
use crate::codegen::validation::{SkillValidator, ValidationResult};
use crate::semantic::context_reconstruction::ContextReconstructor;
use crate::synthesis::selector_ranking::{RankedSelector, SelectorRanker, SelectorType};
use crate::synthesis::variable_extraction::{ExtractedVariable, VariableExtractor};
use crate::verification::hoare_triple_generator::{HoareTripleGenerator, VerificationBlock};
use crate::verification::postcondition_extractor::PostconditionExtractor;

use std::path::Path;
use tracing::{debug, info};

/// Skill generation configuration
#[derive(Debug, Clone)]
pub struct GeneratorConfig {
    /// Minimum confidence for including a step
    pub min_step_confidence: f32,
    /// Include verification blocks
    pub include_verification: bool,
    /// Variable extraction settings
    pub extract_variables: bool,
    /// Selector fallback chain depth
    pub selector_chain_depth: usize,
    /// Use LLM to fill missing semantic data
    pub use_llm_semantic: bool,
    /// API key for LLM calls (overrides ANTHROPIC_API_KEY env var)
    pub api_key: Option<String>,
}

impl Default for GeneratorConfig {
    fn default() -> Self {
        Self {
            min_step_confidence: 0.7,
            include_verification: true,
            extract_variables: true,
            selector_chain_depth: 3,
            use_llm_semantic: true,
            api_key: None,
        }
    }
}

/// Generated step with all enriched data
#[derive(Debug, Clone)]
pub struct GeneratedStep {
    /// Step number
    pub number: usize,
    /// Human-readable description
    pub description: String,
    /// Primary action
    pub action: Action,
    /// Target selector (primary)
    pub selector: Option<RankedSelector>,
    /// Fallback selectors
    pub fallback_selectors: Vec<RankedSelector>,
    /// Variables used in this step
    pub variables: Vec<ExtractedVariable>,
    /// Verification block
    pub verification: Option<VerificationBlock>,
    /// Source events
    pub source_events: Vec<usize>,
    /// Confidence score
    pub confidence: f32,
}

/// Skill Generator orchestrates the complete generation pipeline
pub struct SkillGenerator {
    /// Configuration
    config: GeneratorConfig,
    /// Variable extractor
    variable_extractor: VariableExtractor,
    /// Selector ranker for generating robust UI selectors
    selector_ranker: SelectorRanker,
    /// Postcondition extractor
    postcondition_extractor: PostconditionExtractor,
    /// Hoare triple generator
    hoare_generator: HoareTripleGenerator,
    /// Validator
    validator: SkillValidator,
    /// Context reconstructor for LLM semantic inference
    context_reconstructor: ContextReconstructor,
}

impl SkillGenerator {
    /// Create a new skill generator with default config
    pub fn new() -> Self {
        Self::with_config(GeneratorConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: GeneratorConfig) -> Self {
        Self {
            config,
            variable_extractor: VariableExtractor::new(),
            selector_ranker: SelectorRanker::new(),
            postcondition_extractor: PostconditionExtractor::new(),
            hoare_generator: HoareTripleGenerator::new(),
            validator: SkillValidator::new(),
            context_reconstructor: ContextReconstructor::new(),
        }
    }

    /// Generate a skill from a recording
    pub fn generate(&self, recording: &Recording) -> Result<GeneratedSkill, GeneratorError> {
        // Step 0: Enrich events with LLM semantic inference if enabled
        let enriched_recording = if self.config.use_llm_semantic {
            self.enrich_with_llm_semantic(recording)
        } else {
            recording.clone()
        };

        // Step 1: Extract significant events (clicks, keystrokes)
        let significant_events = self.extract_significant_events(&enriched_recording);

        if significant_events.is_empty() {
            return Err(GeneratorError::NoSignificantEvents);
        }

        // Step 2: Bind intents to events
        let actions = self.bind_intents(&significant_events, &enriched_recording);

        // Step 3: Extract variables
        let variables = if self.config.extract_variables {
            let goal = enriched_recording.metadata.goal.as_deref().unwrap_or("");
            self.variable_extractor.extract(&enriched_recording.events, goal)
        } else {
            Vec::new()
        };

        // Step 4: Generate steps
        let steps = self.generate_steps(&actions, &variables, &significant_events);

        // Step 5: Generate selectors
        let steps_with_selectors = self.add_selectors(steps, &significant_events);

        // Step 6: Add verification blocks
        let steps_with_verification = if self.config.include_verification {
            self.add_verification(steps_with_selectors, &significant_events)
        } else {
            steps_with_selectors
        };

        // Step 7: Build the skill
        let skill = GeneratedSkill {
            name: recording.metadata.name.clone(),
            description: recording.metadata.goal.clone().unwrap_or_else(|| {
                format!("Skill generated from recording: {}", recording.metadata.name)
            }),
            context: "fork".to_string(),
            allowed_tools: vec!["Read".to_string(), "Bash".to_string()],
            variables: variables.clone(),
            steps: steps_with_verification,
            source_recording_id: recording.metadata.id,
        };

        Ok(skill)
    }

    /// Enrich events with LLM semantic inference for those missing semantic data
    fn enrich_with_llm_semantic(&self, recording: &Recording) -> Recording {
        let goal = recording.metadata.goal.as_deref().unwrap_or("");

        // Resolve API key: config takes priority, then env var
        let api_key = self.config.api_key.clone()
            .or_else(|| std::env::var("ANTHROPIC_API_KEY").ok());

        if api_key.is_none() {
            debug!("No API key found (config or ANTHROPIC_API_KEY env), skipping LLM semantic enrichment");
            return recording.clone();
        }

        // Count events needing semantic data
        let events_needing_semantic: Vec<_> = recording
            .events
            .iter()
            .enumerate()
            .filter(|(_, e)| {
                e.raw.event_type.is_click() && e.semantic.is_none()
            })
            .collect();

        if events_needing_semantic.is_empty() {
            debug!("All click events have semantic data, no LLM enrichment needed");
            return recording.clone();
        }

        info!(
            "Enriching {} events with LLM semantic inference",
            events_needing_semantic.len()
        );

        // Create a mutable copy of the recording
        let mut enriched = recording.clone();

        // Create tokio runtime for async LLM calls
        let rt = match tokio::runtime::Runtime::new() {
            Ok(rt) => rt,
            Err(e) => {
                debug!("Failed to create tokio runtime for LLM: {}", e);
                return recording.clone();
            }
        };

        // Process each event that needs semantic data
        for (idx, _event) in events_needing_semantic {
            // Build context from surrounding events
            let context_start = idx.saturating_sub(3);
            let context_end = (idx + 3).min(recording.events.len());
            let context_summary = self.build_context_summary(&recording.events[context_start..context_end]);

            // Use ContextReconstructor for LLM inference
            let result = rt.block_on(async {
                self.context_reconstructor
                    .llm_infer(&recording.events[idx], goal, &context_summary, api_key.as_deref())
                    .await
            });

            if let Some(semantic) = result {
                debug!(
                    "LLM inferred semantic for event {}: role={:?}, title={:?}",
                    idx, semantic.ax_role, semantic.title
                );
                enriched.events[idx].semantic = Some(semantic);
            }
        }

        enriched
    }

    /// Build a text summary of context events for LLM inference
    fn build_context_summary(&self, events: &[EnrichedEvent]) -> String {
        events
            .iter()
            .map(|e| {
                let action_type = if e.raw.event_type.is_click() {
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
                    .unwrap_or_else(|| format!("({:.0},{:.0})", e.raw.coordinates.0, e.raw.coordinates.1));
                format!("- {} on {}", action_type, target)
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Extract significant events from recording
    ///
    /// Filters out mouse-up events (implicit after mouse-down), key-up events,
    /// and mouse-move events. Only retains actionable events that should
    /// produce steps in the generated skill.
    fn extract_significant_events<'a>(&self, recording: &'a Recording) -> Vec<&'a EnrichedEvent> {
        recording
            .events
            .iter()
            .filter(|e| {
                match e.raw.event_type {
                    // Include mouse-down events only (up is implicit)
                    EventType::LeftMouseDown | EventType::RightMouseDown => true,
                    // Include key-down events only
                    EventType::KeyDown => true,
                    // Include scroll events
                    EventType::ScrollWheel => true,
                    // Exclude everything else (mouse-up, mouse-move, key-up)
                    _ => false,
                }
            })
            .collect()
    }

    /// Bind intents to events
    fn bind_intents(
        &self,
        events: &[&EnrichedEvent],
        _recording: &Recording,
    ) -> Vec<(usize, Action)> {
        let mut actions = Vec::new();

        for (i, event) in events.iter().enumerate() {
            let action = self.infer_action(event);
            actions.push((i, action));
        }

        actions
    }

    /// Infer action from event
    fn infer_action(&self, event: &EnrichedEvent) -> Action {
        let element_name = self.get_element_name(event);
        let element_role = self.get_element_role(event);

        match event.raw.event_type {
            EventType::LeftMouseDown | EventType::LeftMouseUp => {
                // Check for double-click
                if event.raw.click_count >= 2 {
                    return Action::DoubleClick {
                        element_name,
                        element_role,
                        confidence: 0.9,
                    };
                }

                // Check semantic context for more specific action
                if let Some(ref semantic) = event.semantic {
                    if let Some(ref role) = semantic.ax_role {
                        match role.as_str() {
                            "AXTextField" | "AXTextArea" => {
                                return Action::Fill {
                                    field_name: element_name,
                                    value: "{{input}}".to_string(),
                                    confidence: 0.85,
                                }
                            }
                            "AXPopUpButton" | "AXComboBox" => {
                                return Action::Select {
                                    menu_name: element_name,
                                    option: "{{option}}".to_string(),
                                    confidence: 0.85,
                                }
                            }
                            _ => {}
                        }
                    }
                }

                // Default to click
                Action::Click {
                    element_name,
                    element_role,
                    confidence: 0.9,
                }
            }
            EventType::RightMouseDown | EventType::RightMouseUp => Action::RightClick {
                element_name,
                element_role,
                confidence: 0.9,
            },
            EventType::KeyDown => {
                if let Some(char) = event.raw.character {
                    Action::Type {
                        text: char.to_string(),
                        confidence: 0.95,
                    }
                } else if let Some(key_code) = event.raw.key_code {
                    // Check for modifiers (keyboard shortcut)
                    if event.raw.modifiers.command || event.raw.modifiers.control {
                        Action::Shortcut {
                            keys: self.build_key_combo(event),
                            confidence: 0.9,
                        }
                    } else {
                        Action::Type {
                            text: format!("[keycode:{}]", key_code),
                            confidence: 0.7,
                        }
                    }
                } else {
                    Action::Unknown {
                        description: "Key press".to_string(),
                        confidence: 0.5,
                    }
                }
            }
            EventType::ScrollWheel => {
                let delta = event.raw.scroll_delta.unwrap_or((0.0, 0.0));
                let (direction, magnitude) = if delta.1.abs() > delta.0.abs() {
                    if delta.1 > 0.0 {
                        (ScrollDirection::Up, delta.1)
                    } else {
                        (ScrollDirection::Down, -delta.1)
                    }
                } else if delta.0 > 0.0 {
                    (ScrollDirection::Right, delta.0)
                } else {
                    (ScrollDirection::Left, -delta.0)
                };

                Action::Scroll {
                    direction,
                    magnitude,
                    confidence: 0.9,
                }
            }
            _ => Action::Unknown {
                description: format!("{:?}", event.raw.event_type),
                confidence: 0.5,
            },
        }
    }

    /// Get element name from event
    fn get_element_name(&self, event: &EnrichedEvent) -> String {
        if let Some(ref semantic) = event.semantic {
            if let Some(ref title) = semantic.title {
                if !title.is_empty() {
                    return title.clone();
                }
            }
            if let Some(ref id) = semantic.identifier {
                return id.clone();
            }
        }
        format!("element_at_({:.0},{:.0})", event.raw.coordinates.0, event.raw.coordinates.1)
    }

    /// Get element role from event
    fn get_element_role(&self, event: &EnrichedEvent) -> String {
        if let Some(ref semantic) = event.semantic {
            if let Some(ref role) = semantic.ax_role {
                return role.clone();
            }
        }
        "unknown".to_string()
    }

    /// Build key combination string
    fn build_key_combo(&self, event: &EnrichedEvent) -> Vec<String> {
        let mut keys = Vec::new();
        if event.raw.modifiers.command {
            keys.push("Cmd".to_string());
        }
        if event.raw.modifiers.control {
            keys.push("Ctrl".to_string());
        }
        if event.raw.modifiers.option {
            keys.push("Option".to_string());
        }
        if event.raw.modifiers.shift {
            keys.push("Shift".to_string());
        }
        if let Some(char) = event.raw.character {
            keys.push(char.to_string().to_uppercase());
        } else if let Some(code) = event.raw.key_code {
            keys.push(format!("Key{}", code));
        }
        keys
    }

    /// Generate steps from actions, consolidating consecutive typing and scrolling
    fn generate_steps(
        &self,
        actions: &[(usize, Action)],
        variables: &[ExtractedVariable],
        events: &[&EnrichedEvent],
    ) -> Vec<GeneratedStep> {
        let mut steps = Vec::new();
        let mut i = 0;

        while i < actions.len() {
            let (event_idx, action) = &actions[i];

            // Consolidate consecutive Type actions (has actual text, not a keycode)
            if let Action::Type { ref text, .. } = action {
                if !text.starts_with("[keycode:") {
                    let mut consolidated_text = text.clone();
                    let mut source_events = vec![*event_idx];
                    let mut min_confidence = action.confidence();

                    while i + 1 < actions.len() {
                        if let Action::Type { ref text, confidence } = actions[i + 1].1 {
                            if !text.starts_with("[keycode:") {
                                consolidated_text.push_str(text);
                                source_events.push(actions[i + 1].0);
                                if confidence < min_confidence {
                                    min_confidence = confidence;
                                }
                                i += 1;
                                continue;
                            }
                        }
                        break;
                    }

                    let description = format!("Type \"{}\"", consolidated_text);

                    let step_variables: Vec<ExtractedVariable> = variables
                        .iter()
                        .filter(|v| description.contains(&format!("{{{{{}}}}}", v.name)))
                        .cloned()
                        .collect();

                    steps.push(GeneratedStep {
                        number: steps.len() + 1,
                        description,
                        action: Action::Type {
                            text: consolidated_text,
                            confidence: min_confidence,
                        },
                        selector: None,
                        fallback_selectors: vec![],
                        variables: step_variables,
                        verification: None,
                        source_events,
                        confidence: min_confidence,
                    });

                    i += 1;
                    continue;
                }
            }

            // Consolidate consecutive Scroll actions in the same direction
            if let Action::Scroll { direction, magnitude, .. } = action {
                let mut total_magnitude = *magnitude;
                let mut source_events = vec![*event_idx];
                let mut min_confidence = action.confidence();
                let scroll_direction = *direction;

                while i + 1 < actions.len() {
                    if let Action::Scroll { direction: next_dir, magnitude: next_mag, confidence } = actions[i + 1].1 {
                        if next_dir == scroll_direction {
                            total_magnitude += next_mag;
                            source_events.push(actions[i + 1].0);
                            if confidence < min_confidence {
                                min_confidence = confidence;
                            }
                            i += 1;
                            continue;
                        }
                    }
                    break;
                }

                let description = format!("Scroll {:?} {:.0} units", scroll_direction, total_magnitude);

                steps.push(GeneratedStep {
                    number: steps.len() + 1,
                    description,
                    action: Action::Scroll {
                        direction: scroll_direction,
                        magnitude: total_magnitude,
                        confidence: min_confidence,
                    },
                    selector: None,
                    fallback_selectors: vec![],
                    variables: vec![],
                    verification: None,
                    source_events,
                    confidence: min_confidence,
                });

                i += 1;
                continue;
            }

            // Non-consolidatable actions: generate normally
            let event = events[*event_idx];
            let description = self.generate_step_description(action, event);

            let step_variables: Vec<ExtractedVariable> = variables
                .iter()
                .filter(|v| description.contains(&format!("{{{{{}}}}}", v.name)))
                .cloned()
                .collect();

            steps.push(GeneratedStep {
                number: steps.len() + 1,
                description,
                action: action.clone(),
                selector: None,
                fallback_selectors: vec![],
                variables: step_variables,
                verification: None,
                source_events: vec![*event_idx],
                confidence: action.confidence(),
            });

            i += 1;
        }

        steps
    }

    /// Generate human-readable step description
    fn generate_step_description(&self, action: &Action, event: &EnrichedEvent) -> String {
        let target = self.get_target_description(event);
        
        match action {
            Action::Click { element_name, .. } => {
                if element_name.starts_with("element_at_") {
                    format!("Click on {}", target)
                } else {
                    format!("Click on \"{}\"", element_name)
                }
            }
            Action::DoubleClick { element_name, .. } => {
                if element_name.starts_with("element_at_") {
                    format!("Double-click on {}", target)
                } else {
                    format!("Double-click on \"{}\"", element_name)
                }
            }
            Action::RightClick { element_name, .. } => {
                if element_name.starts_with("element_at_") {
                    format!("Right-click on {}", target)
                } else {
                    format!("Right-click on \"{}\"", element_name)
                }
            }
            Action::Fill { field_name, value, .. } => {
                format!("Enter {} in \"{}\"", value, field_name)
            }
            Action::Select { menu_name, option, .. } => {
                format!("Select {} from \"{}\"", option, menu_name)
            }
            Action::Type { text, .. } => format!("Type \"{}\"", text),
            Action::Shortcut { keys, .. } => {
                format!("Press {}", keys.join("+"))
            }
            Action::Scroll { direction, magnitude, .. } => {
                format!("Scroll {:?} {:.0} units", direction, magnitude)
            }
            Action::DragDrop { source, target, .. } => {
                format!("Drag \"{}\" to \"{}\"", source, target)
            }
            Action::Search { candidate_elements, .. } => {
                format!("Search for: {}", candidate_elements.join(", "))
            }
            Action::Unknown { description, .. } => description.clone(),
        }
    }

    /// Get target description from event for use in step descriptions
    fn get_target_description(&self, event: &EnrichedEvent) -> String {
        if let Some(ref semantic) = event.semantic {
            if let Some(ref title) = semantic.title {
                if !title.is_empty() {
                    return format!("\"{}\"", title);
                }
            }
            if let Some(ref role) = semantic.ax_role {
                if let Some(ref id) = semantic.identifier {
                    return format!("{} ({})", role, id);
                }
                return role.clone();
            }
        }
        format!("({:.0}, {:.0})", event.raw.coordinates.0, event.raw.coordinates.1)
    }

    /// Add selectors to steps
    fn add_selectors(
        &self,
        mut steps: Vec<GeneratedStep>,
        events: &[&EnrichedEvent],
    ) -> Vec<GeneratedStep> {
        for step in &mut steps {
            if let Some(&event_idx) = step.source_events.first() {
                let event = events[event_idx];
                let selectors = self.generate_selectors(event);

                if !selectors.is_empty() {
                    step.selector = Some(selectors[0].clone());
                    step.fallback_selectors = selectors
                        .into_iter()
                        .skip(1)
                        .take(self.config.selector_chain_depth)
                        .collect();
                }
            }
        }

        steps
    }

    /// Generate selectors for an event using the selector ranker
    fn generate_selectors(&self, event: &EnrichedEvent) -> Vec<RankedSelector> {
        if let Some(ref semantic) = event.semantic {
            // Use the selector ranker to generate properly ranked selectors
            let chain = self.selector_ranker.generate_selectors(semantic);
            
            // Collect all selectors from the chain
            let mut selectors: Vec<RankedSelector> = chain.all_selectors().cloned().collect();
            
            // Add coordinate fallback if not already present
            let has_position = selectors.iter().any(|s| matches!(s.selector_type, SelectorType::RelativePosition));
            if !has_position {
                selectors.push(RankedSelector {
                    selector_type: SelectorType::RelativePosition,
                    value: format!("{:.0},{:.0}", event.raw.coordinates.0, event.raw.coordinates.1),
                    stability: 0.3,
                    specificity: 0.5,
                    rank_score: 0.35, // Pre-calculated rank
                });
            }
            
            selectors
        } else {
            // No semantic context, use coordinate-only fallback
            vec![RankedSelector {
                selector_type: SelectorType::RelativePosition,
                value: format!("{:.0},{:.0}", event.raw.coordinates.0, event.raw.coordinates.1),
                stability: 0.3,
                specificity: 0.5,
                rank_score: 0.35,
            }]
        }
    }

    /// Add verification blocks to steps
    fn add_verification(
        &self,
        mut steps: Vec<GeneratedStep>,
        events: &[&EnrichedEvent],
    ) -> Vec<GeneratedStep> {
        for (i, step) in steps.iter_mut().enumerate() {
            let pre_event = step.source_events.first().and_then(|&idx| events.get(idx)).copied();
            let post_event = if i + 1 < events.len() {
                Some(events[i + 1])
            } else {
                pre_event
            };

            if let Some(post) = post_event {
                let postconditions = self.postcondition_extractor.extract(pre_event, post);

                if !postconditions.is_empty() {
                    let verification = self.hoare_generator.generate_verification_block(
                        &postconditions,
                        None,
                    );
                    step.verification = Some(verification);
                }
            }
        }

        steps
    }

    /// Render the skill to SKILL.md format
    pub fn render_to_markdown(&self, skill: &GeneratedSkill) -> String {
        let mut output = String::new();

        // YAML frontmatter
        output.push_str("---\n");
        output.push_str(&format!("name: {}\n", skill.name));
        output.push_str(&format!("description: {}\n", skill.description));
        output.push_str(&format!("context: {}\n", skill.context));
        output.push_str(&format!(
            "allowed-tools: {}\n",
            skill.allowed_tools.join(", ")
        ));
        output.push_str("---\n\n");

        // Variables section
        if !skill.variables.is_empty() {
            output.push_str("## Variables\n\n");
            for var in &skill.variables {
                let desc = var.description.as_deref().unwrap_or("No description");
                let type_hint = var.type_hint.as_deref().unwrap_or("string");
                output.push_str(&format!(
                    "- `{{{{{}}}}}`: {} (type: {})\n",
                    var.name, desc, type_hint
                ));
            }
            output.push('\n');
        }

        // Steps section
        output.push_str("## Steps\n\n");
        for step in &skill.steps {
            output.push_str(&format!("### Step {}: {}\n\n", step.number, step.description));

            // Selector info
            if let Some(ref selector) = step.selector {
                output.push_str(&format!(
                    "**Target**: `{}` (type: {:?})\n",
                    selector.value, selector.selector_type
                ));

                if !step.fallback_selectors.is_empty() {
                    output.push_str("**Fallback selectors**:\n");
                    for fb in &step.fallback_selectors {
                        output.push_str(&format!("  - `{}` ({:?})\n", fb.value, fb.selector_type));
                    }
                }
                output.push('\n');
            }

            // Verification block
            if let Some(ref verification) = step.verification {
                output.push_str("**Verification**:\n");
                output.push_str("```\n");
                output.push_str(&format!("Type: {}\n", verification.verification_type));
                output.push_str(&format!("Timeout: {}ms\n", verification.timeout_ms));
                if !verification.conditions.is_empty() {
                    output.push_str("Conditions:\n");
                    for cond in &verification.conditions {
                        output.push_str(&format!(
                            "  - {}.{} {} '{}'\n",
                            cond.scope, cond.attribute, cond.operator, cond.expected_value
                        ));
                    }
                }
                output.push_str("```\n\n");
            }
        }

        output
    }

    /// Validate a generated skill
    pub fn validate(&self, skill: &GeneratedSkill) -> ValidationResult {
        let markdown = self.render_to_markdown(skill);
        self.validator.validate_markdown(&markdown)
    }

    /// Save skill to file
    pub fn save_skill(
        &self,
        skill: &GeneratedSkill,
        path: &Path,
    ) -> Result<(), std::io::Error> {
        let markdown = self.render_to_markdown(skill);
        std::fs::write(path, markdown)
    }

    /// Render skill to markdown using the advanced MarkdownBuilder
    /// This produces a more detailed output with technical context blocks
    pub fn render_to_compiled_markdown(&self, skill: &GeneratedSkill) -> String {
        let compiled = self.to_compiled_skill(skill);
        let mut builder = MarkdownBuilder::new();
        builder.build(&compiled)
    }

    /// Convert GeneratedSkill to CompiledSkill for use with MarkdownBuilder
    fn to_compiled_skill(&self, skill: &GeneratedSkill) -> CompiledSkill {
        // Convert variables to inputs
        let inputs: Vec<SkillInput> = skill.variables.iter().map(|v| {
            SkillInput {
                name: v.name.clone(),
                type_hint: v.type_hint.clone(),
                description: v.description.clone(),
                required: true,
                default: None,
            }
        }).collect();

        // Convert steps
        let steps: Vec<CompiledStep> = skill.steps.iter().map(|step| {
            // Build selectors list for technical context
            let mut selectors = Vec::new();
            if let Some(ref selector) = step.selector {
                selectors.push(selector.value.clone());
            }
            for fb in &step.fallback_selectors {
                selectors.push(fb.value.clone());
            }

            // Extract AX metadata from the action or selector
            let ax_metadata = self.extract_ax_metadata_from_step(step);

            CompiledStep {
                number: step.number,
                narrative: step.description.clone(),
                technical_context: TechnicalContext {
                    selectors,
                    ax_metadata,
                    screenshot_ref: None,
                },
                verification: step.verification.clone(),
                error_recovery: None,
            }
        }).collect();

        // Determine context
        let context = if skill.context == "inline" {
            CompiledContext::Inline
        } else {
            CompiledContext::Fork
        };

        CompiledSkill {
            name: skill.name.clone(),
            description: skill.description.clone(),
            inputs,
            steps,
            context,
            allowed_tools: skill.allowed_tools.clone(),
        }
    }

    /// Extract AX metadata from a step
    fn extract_ax_metadata_from_step(&self, step: &GeneratedStep) -> Option<AxMetadata> {
        // Try to extract from the action
        let (role, title) = match &step.action {
            Action::Click { element_role, element_name, .. } |
            Action::DoubleClick { element_role, element_name, .. } |
            Action::RightClick { element_role, element_name, .. } => {
                (Some(element_role.clone()), Some(element_name.clone()))
            }
            Action::Fill { field_name, .. } => {
                (Some("AXTextField".to_string()), Some(field_name.clone()))
            }
            Action::Select { menu_name, .. } => {
                (Some("AXPopUpButton".to_string()), Some(menu_name.clone()))
            }
            _ => (None, None),
        };

        // Get identifier from primary selector if AxIdentifier type
        let identifier = step.selector.as_ref().and_then(|s| {
            if matches!(s.selector_type, SelectorType::AxIdentifier) {
                Some(s.value.clone())
            } else {
                None
            }
        });

        if role.is_some() || title.is_some() || identifier.is_some() {
            Some(AxMetadata {
                role,
                title,
                identifier,
                window_title: None,
            })
        } else {
            None
        }
    }
}

impl Default for SkillGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// A generated skill ready for output
#[derive(Debug, Clone)]
pub struct GeneratedSkill {
    /// Skill name
    pub name: String,
    /// Description
    pub description: String,
    /// Execution context (fork, inline)
    pub context: String,
    /// Allowed tools
    pub allowed_tools: Vec<String>,
    /// Variables used in the skill
    pub variables: Vec<ExtractedVariable>,
    /// Generated steps
    pub steps: Vec<GeneratedStep>,
    /// Source recording ID
    pub source_recording_id: uuid::Uuid,
}

/// Error type for skill generation
#[derive(Debug, Clone)]
pub enum GeneratorError {
    /// No significant events in recording
    NoSignificantEvents,
    /// Validation failed
    ValidationFailed(Vec<String>),
    /// IO error
    IoError(String),
}

impl std::fmt::Display for GeneratorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GeneratorError::NoSignificantEvents => {
                write!(f, "No significant events found in recording")
            }
            GeneratorError::ValidationFailed(errors) => {
                write!(f, "Validation failed: {}", errors.join(", "))
            }
            GeneratorError::IoError(msg) => write!(f, "IO error: {}", msg),
        }
    }
}

impl std::error::Error for GeneratorError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capture::types::{CursorState, ModifierFlags, SemanticContext};
    use crate::time::timebase::{MachTimebase, Timestamp};

    fn make_test_event(event_type: EventType, x: f64, y: f64) -> EnrichedEvent {
        MachTimebase::init();
        let raw = crate::capture::types::RawEvent {
            timestamp: Timestamp::now(),
            event_type,
            coordinates: (x, y),
            cursor_state: CursorState::Arrow,
            key_code: None,
            character: None,
            modifiers: ModifierFlags::default(),
            scroll_delta: None,
            click_count: 1,
        };
        EnrichedEvent::new(raw, 0)
    }

    fn make_test_event_with_modifiers(
        event_type: EventType,
        x: f64,
        y: f64,
        modifiers: ModifierFlags,
        key_code: Option<u16>,
        character: Option<char>,
    ) -> EnrichedEvent {
        MachTimebase::init();
        let raw = crate::capture::types::RawEvent {
            timestamp: Timestamp::now(),
            event_type,
            coordinates: (x, y),
            cursor_state: CursorState::Arrow,
            key_code,
            character,
            modifiers,
            scroll_delta: None,
            click_count: 1,
        };
        EnrichedEvent::new(raw, 0)
    }

    #[test]
    fn test_generator_creation() {
        let generator = SkillGenerator::new();
        assert!(generator.config.include_verification);
    }

    #[test]
    fn test_generate_from_empty_recording() {
        let generator = SkillGenerator::new();
        let recording = Recording::new("test".to_string(), None);

        let result = generator.generate(&recording);
        assert!(matches!(result, Err(GeneratorError::NoSignificantEvents)));
    }

    #[test]
    fn test_generate_from_recording_with_clicks() {
        MachTimebase::init();
        let generator = SkillGenerator::new();
        let mut recording = Recording::new("test".to_string(), Some("Click a button".to_string()));

        // Add click event with semantic context
        let mut event = make_test_event(EventType::LeftMouseDown, 100.0, 200.0);
        event.semantic = Some(SemanticContext {
            ax_role: Some("AXButton".to_string()),
            title: Some("Submit".to_string()),
            ..Default::default()
        });
        recording.add_event(event);

        let result = generator.generate(&recording);
        assert!(result.is_ok());

        let skill = result.unwrap();
        assert_eq!(skill.name, "test");
        assert!(!skill.steps.is_empty());
    }

    #[test]
    fn test_render_to_markdown() {
        MachTimebase::init();
        let generator = SkillGenerator::new();
        let mut recording = Recording::new("test_skill".to_string(), Some("Test the form".to_string()));

        let mut event = make_test_event(EventType::LeftMouseDown, 100.0, 200.0);
        event.semantic = Some(SemanticContext {
            ax_role: Some("AXButton".to_string()),
            title: Some("Submit".to_string()),
            ..Default::default()
        });
        recording.add_event(event);

        let skill = generator.generate(&recording).unwrap();
        let markdown = generator.render_to_markdown(&skill);

        assert!(markdown.contains("---"));
        assert!(markdown.contains("name: test_skill"));
        assert!(markdown.contains("## Steps"));
    }

    #[test]
    fn test_custom_generator_config() {
        let config = GeneratorConfig {
            min_step_confidence: 0.5,
            include_verification: false,
            extract_variables: false,
            selector_chain_depth: 2,
            use_llm_semantic: false,
            api_key: None,
        };
        let generator = SkillGenerator::with_config(config);
        assert!(!generator.config.include_verification);
        assert_eq!(generator.config.selector_chain_depth, 2);
        assert!(!generator.config.use_llm_semantic);
    }

    #[test]
    fn test_extract_significant_events_filters_correctly() {
        MachTimebase::init();
        let generator = SkillGenerator::new();
        let mut recording = Recording::new("test".to_string(), None);

        // Add various event types
        recording.add_event(make_test_event(EventType::MouseMoved, 10.0, 10.0));
        recording.add_event(make_test_event(EventType::LeftMouseDown, 20.0, 20.0));
        recording.add_event(make_test_event(EventType::KeyDown, 0.0, 0.0));
        recording.add_event(make_test_event(EventType::ScrollWheel, 30.0, 30.0));
        recording.add_event(make_test_event(EventType::MouseMoved, 40.0, 40.0));

        let significant = generator.extract_significant_events(&recording);

        // Should only include click, keyboard, and scroll events (not mouse moves)
        assert_eq!(significant.len(), 3);
        assert!(significant.iter().any(|e| e.raw.event_type == EventType::LeftMouseDown));
        assert!(significant.iter().any(|e| e.raw.event_type == EventType::KeyDown));
        assert!(significant.iter().any(|e| e.raw.event_type == EventType::ScrollWheel));
    }

    #[test]
    fn test_infer_action_double_click() {
        MachTimebase::init();
        let generator = SkillGenerator::new();
        let mut event = make_test_event(EventType::LeftMouseDown, 100.0, 200.0);
        event.raw.click_count = 2;
        event.semantic = Some(SemanticContext {
            title: Some("File".to_string()),
            ..Default::default()
        });

        let action = generator.infer_action(&event);
        match action {
            Action::DoubleClick { element_name, confidence, .. } => {
                assert_eq!(element_name, "File");
                assert_eq!(confidence, 0.9);
            }
            _ => panic!("Expected DoubleClick action"),
        }
    }

    #[test]
    fn test_infer_action_text_field_fill() {
        MachTimebase::init();
        let generator = SkillGenerator::new();
        let mut event = make_test_event(EventType::LeftMouseDown, 100.0, 200.0);
        event.semantic = Some(SemanticContext {
            ax_role: Some("AXTextField".to_string()),
            title: Some("Username".to_string()),
            ..Default::default()
        });

        let action = generator.infer_action(&event);
        match action {
            Action::Fill { field_name, value, confidence } => {
                assert_eq!(field_name, "Username");
                assert_eq!(value, "{{input}}");
                assert_eq!(confidence, 0.85);
            }
            _ => panic!("Expected Fill action"),
        }
    }

    #[test]
    fn test_infer_action_dropdown_select() {
        MachTimebase::init();
        let generator = SkillGenerator::new();
        let mut event = make_test_event(EventType::LeftMouseDown, 100.0, 200.0);
        event.semantic = Some(SemanticContext {
            ax_role: Some("AXPopUpButton".to_string()),
            title: Some("Country".to_string()),
            ..Default::default()
        });

        let action = generator.infer_action(&event);
        match action {
            Action::Select { menu_name, option, confidence } => {
                assert_eq!(menu_name, "Country");
                assert_eq!(option, "{{option}}");
                assert_eq!(confidence, 0.85);
            }
            _ => panic!("Expected Select action"),
        }
    }

    #[test]
    fn test_infer_action_keyboard_shortcut() {
        MachTimebase::init();
        let generator = SkillGenerator::new();
        let modifiers = ModifierFlags {
            command: true,
            control: false,
            option: false,
            shift: false,
            caps_lock: false,
            function: false,
        };
        // Don't provide character, only key_code to trigger shortcut detection
        let event = make_test_event_with_modifiers(
            EventType::KeyDown,
            0.0,
            0.0,
            modifiers,
            Some(1),
            None,
        );

        let action = generator.infer_action(&event);
        match action {
            Action::Shortcut { keys, confidence } => {
                assert!(keys.contains(&"Cmd".to_string()));
                assert_eq!(confidence, 0.9);
            }
            _ => panic!("Expected Shortcut action, got {:?}", action),
        }
    }

    #[test]
    fn test_infer_action_scroll() {
        MachTimebase::init();
        let generator = SkillGenerator::new();
        let mut event = make_test_event(EventType::ScrollWheel, 100.0, 200.0);
        event.raw.scroll_delta = Some((0.0, -50.0)); // Negative Y means scroll down

        let action = generator.infer_action(&event);
        match action {
            Action::Scroll { direction, magnitude, confidence } => {
                assert!(matches!(direction, ScrollDirection::Down));
                assert_eq!(magnitude, 50.0);
                assert_eq!(confidence, 0.9);
            }
            _ => panic!("Expected Scroll action"),
        }
    }

    #[test]
    fn test_generate_selectors_with_hierarchy() {
        MachTimebase::init();
        let generator = SkillGenerator::new();
        let mut event = make_test_event(EventType::LeftMouseDown, 150.0, 250.0);
        event.semantic = Some(SemanticContext {
            ax_role: Some("AXButton".to_string()),
            title: Some("Save".to_string()),
            identifier: Some("save-btn".to_string()),
            ..Default::default()
        });

        let selectors = generator.generate_selectors(&event);

        // Should have identifier, text, and coordinate selectors
        assert!(selectors.len() >= 3);

        // First should be identifier (highest priority)
        assert!(matches!(selectors[0].selector_type, SelectorType::AxIdentifier));
        assert_eq!(selectors[0].value, "save-btn");

        // Should include text-based selector
        assert!(selectors.iter().any(|s| matches!(s.selector_type, SelectorType::TextContent)));

        // Should include coordinate fallback
        assert!(selectors.iter().any(|s| matches!(s.selector_type, SelectorType::RelativePosition)));
    }

    #[test]
    fn test_generate_steps_with_multiple_actions() {
        MachTimebase::init();
        let generator = SkillGenerator::new();
        let mut recording = Recording::new("multi_step".to_string(), Some("Fill form".to_string()));

        // Add multiple events
        let mut click_event = make_test_event(EventType::LeftMouseDown, 100.0, 100.0);
        click_event.semantic = Some(SemanticContext {
            ax_role: Some("AXButton".to_string()),
            title: Some("Start".to_string()),
            ..Default::default()
        });
        recording.add_event(click_event);

        let modifiers = ModifierFlags::default();
        let type_event = make_test_event_with_modifiers(
            EventType::KeyDown,
            0.0,
            0.0,
            modifiers,
            None,
            Some('a'),
        );
        recording.add_event(type_event);

        let result = generator.generate(&recording);
        assert!(result.is_ok());

        let skill = result.unwrap();
        assert_eq!(skill.steps.len(), 2);
        assert!(skill.steps[0].description.contains("Start"));
    }

    #[test]
    fn test_markdown_includes_fallback_selectors() {
        MachTimebase::init();
        let config = GeneratorConfig {
            selector_chain_depth: 2,
            ..Default::default()
        };
        let generator = SkillGenerator::with_config(config);
        let mut recording = Recording::new("fallback_test".to_string(), None);

        let mut event = make_test_event(EventType::LeftMouseDown, 100.0, 200.0);
        event.semantic = Some(SemanticContext {
            ax_role: Some("AXButton".to_string()),
            title: Some("Submit".to_string()),
            identifier: Some("submit-btn".to_string()),
            ..Default::default()
        });
        recording.add_event(event);

        let skill = generator.generate(&recording).unwrap();
        let markdown = generator.render_to_markdown(&skill);

        assert!(markdown.contains("Fallback selectors"));
    }

    #[test]
    fn test_validation_integration() {
        MachTimebase::init();
        let generator = SkillGenerator::new();
        let mut recording = Recording::new("valid_skill".to_string(), Some("Test validation".to_string()));

        let mut event = make_test_event(EventType::LeftMouseDown, 100.0, 200.0);
        event.semantic = Some(SemanticContext {
            ax_role: Some("AXButton".to_string()),
            title: Some("OK".to_string()),
            ..Default::default()
        });
        recording.add_event(event);

        let skill = generator.generate(&recording).unwrap();
        let validation_result = generator.validate(&skill);

        assert!(validation_result.passed);
    }

    #[test]
    fn test_generator_error_display() {
        let error = GeneratorError::NoSignificantEvents;
        assert_eq!(
            error.to_string(),
            "No significant events found in recording"
        );

        let error = GeneratorError::ValidationFailed(vec!["Error 1".to_string(), "Error 2".to_string()]);
        assert!(error.to_string().contains("Error 1"));
        assert!(error.to_string().contains("Error 2"));

        let error = GeneratorError::IoError("File not found".to_string());
        assert!(error.to_string().contains("File not found"));
    }

    #[test]
    fn test_generate_step_description_formats() {
        let generator = SkillGenerator::new();
        let event = make_test_event(EventType::LeftMouseDown, 0.0, 0.0);

        // Test various action types
        let click = Action::Click {
            element_name: "Button".to_string(),
            element_role: "AXButton".to_string(),
            confidence: 0.9,
        };
        assert_eq!(generator.generate_step_description(&click, &event), "Click on \"Button\"");

        let fill = Action::Fill {
            field_name: "Email".to_string(),
            value: "{{email}}".to_string(),
            confidence: 0.85,
        };
        assert_eq!(generator.generate_step_description(&fill, &event), "Enter {{email}} in \"Email\"");

        let shortcut = Action::Shortcut {
            keys: vec!["Cmd".to_string(), "C".to_string()],
            confidence: 0.9,
        };
        assert_eq!(generator.generate_step_description(&shortcut, &event), "Press Cmd+C");
    }

    #[test]
    fn test_right_click_action() {
        MachTimebase::init();
        let generator = SkillGenerator::new();
        let mut event = make_test_event(EventType::RightMouseDown, 100.0, 200.0);
        event.semantic = Some(SemanticContext {
            title: Some("Context Menu".to_string()),
            ..Default::default()
        });

        let action = generator.infer_action(&event);
        match action {
            Action::RightClick { element_name, confidence, .. } => {
                assert_eq!(element_name, "Context Menu");
                assert_eq!(confidence, 0.9);
            }
            _ => panic!("Expected RightClick action"),
        }
    }

    #[test]
    fn test_consecutive_typing_consolidated_into_single_step() {
        MachTimebase::init();
        let config = GeneratorConfig {
            include_verification: false,
            extract_variables: false,
            use_llm_semantic: false,
            ..Default::default()
        };
        let generator = SkillGenerator::with_config(config);
        let mut recording = Recording::new("type_test".to_string(), Some("Type hello".to_string()));

        // Simulate typing "hello" - 5 consecutive key events
        for ch in ['h', 'e', 'l', 'l', 'o'] {
            let modifiers = ModifierFlags::default();
            let event = make_test_event_with_modifiers(
                EventType::KeyDown,
                0.0,
                0.0,
                modifiers,
                None,
                Some(ch),
            );
            recording.add_event(event);
        }

        let skill = generator.generate(&recording).unwrap();

        // Should be consolidated into 1 step instead of 5
        assert_eq!(skill.steps.len(), 1);
        assert_eq!(skill.steps[0].description, "Type \"hello\"");
        assert_eq!(skill.steps[0].source_events.len(), 5);
    }

    #[test]
    fn test_typing_interrupted_by_click_creates_separate_steps() {
        MachTimebase::init();
        let config = GeneratorConfig {
            include_verification: false,
            extract_variables: false,
            use_llm_semantic: false,
            ..Default::default()
        };
        let generator = SkillGenerator::with_config(config);
        let mut recording = Recording::new("mixed_test".to_string(), None);

        // Type "ab"
        for ch in ['a', 'b'] {
            let event = make_test_event_with_modifiers(
                EventType::KeyDown, 0.0, 0.0,
                ModifierFlags::default(), None, Some(ch),
            );
            recording.add_event(event);
        }

        // Click
        let mut click = make_test_event(EventType::LeftMouseDown, 100.0, 200.0);
        click.semantic = Some(SemanticContext {
            title: Some("Button".to_string()),
            ..Default::default()
        });
        recording.add_event(click);

        // Type "cd"
        for ch in ['c', 'd'] {
            let event = make_test_event_with_modifiers(
                EventType::KeyDown, 0.0, 0.0,
                ModifierFlags::default(), None, Some(ch),
            );
            recording.add_event(event);
        }

        let skill = generator.generate(&recording).unwrap();

        // Should be 3 steps: Type "ab", Click, Type "cd"
        assert_eq!(skill.steps.len(), 3);
        assert_eq!(skill.steps[0].description, "Type \"ab\"");
        assert!(skill.steps[1].description.contains("Button"));
        assert_eq!(skill.steps[2].description, "Type \"cd\"");
    }

    #[test]
    fn test_keyboard_shortcuts_not_consolidated_with_typing() {
        MachTimebase::init();
        let config = GeneratorConfig {
            include_verification: false,
            extract_variables: false,
            use_llm_semantic: false,
            ..Default::default()
        };
        let generator = SkillGenerator::with_config(config);
        let mut recording = Recording::new("shortcut_test".to_string(), None);

        // Type "a"
        recording.add_event(make_test_event_with_modifiers(
            EventType::KeyDown, 0.0, 0.0,
            ModifierFlags::default(), None, Some('a'),
        ));

        // Cmd+S (shortcut - no character, only key_code + modifier)
        let shortcut_mods = ModifierFlags {
            command: true,
            ..Default::default()
        };
        recording.add_event(make_test_event_with_modifiers(
            EventType::KeyDown, 0.0, 0.0,
            shortcut_mods, Some(1), None,
        ));

        // Type "b"
        recording.add_event(make_test_event_with_modifiers(
            EventType::KeyDown, 0.0, 0.0,
            ModifierFlags::default(), None, Some('b'),
        ));

        let skill = generator.generate(&recording).unwrap();

        // Should be 3 steps: Type "a", Press Cmd+Key1, Type "b"
        assert_eq!(skill.steps.len(), 3);
        assert_eq!(skill.steps[0].description, "Type \"a\"");
        assert!(skill.steps[1].description.contains("Cmd"));
        assert_eq!(skill.steps[2].description, "Type \"b\"");
    }

    #[test]
    fn test_mouse_up_events_excluded_from_steps() {
        MachTimebase::init();
        let config = GeneratorConfig {
            include_verification: false,
            extract_variables: false,
            use_llm_semantic: false,
            ..Default::default()
        };
        let generator = SkillGenerator::with_config(config);
        let mut recording = Recording::new("mouseup_test".to_string(), None);

        // Add a complete click cycle: down + up
        let mut down = make_test_event(EventType::LeftMouseDown, 100.0, 200.0);
        down.semantic = Some(SemanticContext {
            title: Some("Button".to_string()),
            ..Default::default()
        });
        recording.add_event(down);

        let mut up = make_test_event(EventType::LeftMouseUp, 100.0, 200.0);
        up.semantic = Some(SemanticContext {
            title: Some("Button".to_string()),
            ..Default::default()
        });
        recording.add_event(up);

        let skill = generator.generate(&recording).unwrap();

        // Should only have 1 step (mouse-up filtered out)
        assert_eq!(skill.steps.len(), 1);
    }

    #[test]
    fn test_consecutive_scrolls_same_direction_consolidated() {
        MachTimebase::init();
        let config = GeneratorConfig {
            include_verification: false,
            extract_variables: false,
            use_llm_semantic: false,
            ..Default::default()
        };
        let generator = SkillGenerator::with_config(config);
        let mut recording = Recording::new("scroll_test".to_string(), None);

        // 5 consecutive scroll-down events
        for _ in 0..5 {
            let mut event = make_test_event(EventType::ScrollWheel, 200.0, 300.0);
            event.raw.scroll_delta = Some((0.0, -10.0)); // Scroll down
            recording.add_event(event);
        }

        let skill = generator.generate(&recording).unwrap();

        // Should be consolidated into 1 step
        assert_eq!(skill.steps.len(), 1);
        assert!(skill.steps[0].description.contains("Down"));
        assert!(skill.steps[0].description.contains("50")); // 5 * 10
        assert_eq!(skill.steps[0].source_events.len(), 5);
    }

    #[test]
    fn test_scrolls_different_directions_not_consolidated() {
        MachTimebase::init();
        let config = GeneratorConfig {
            include_verification: false,
            extract_variables: false,
            use_llm_semantic: false,
            ..Default::default()
        };
        let generator = SkillGenerator::with_config(config);
        let mut recording = Recording::new("scroll_dir_test".to_string(), None);

        // Scroll down
        let mut event = make_test_event(EventType::ScrollWheel, 200.0, 300.0);
        event.raw.scroll_delta = Some((0.0, -20.0));
        recording.add_event(event);

        // Scroll up (different direction)
        let mut event = make_test_event(EventType::ScrollWheel, 200.0, 300.0);
        event.raw.scroll_delta = Some((0.0, 15.0));
        recording.add_event(event);

        let skill = generator.generate(&recording).unwrap();

        // Should be 2 separate steps
        assert_eq!(skill.steps.len(), 2);
        assert!(skill.steps[0].description.contains("Down"));
        assert!(skill.steps[1].description.contains("Up"));
    }

    #[test]
    fn test_key_up_events_excluded_from_steps() {
        MachTimebase::init();
        let config = GeneratorConfig {
            include_verification: false,
            extract_variables: false,
            use_llm_semantic: false,
            ..Default::default()
        };
        let generator = SkillGenerator::with_config(config);
        let mut recording = Recording::new("keyup_test".to_string(), None);

        // KeyDown event (should produce step)
        recording.add_event(make_test_event_with_modifiers(
            EventType::KeyDown, 0.0, 0.0,
            ModifierFlags::default(), None, Some('a'),
        ));

        // KeyUp event (should be filtered out)
        recording.add_event(make_test_event_with_modifiers(
            EventType::KeyUp, 0.0, 0.0,
            ModifierFlags::default(), None, Some('a'),
        ));

        let skill = generator.generate(&recording).unwrap();

        // Should only have 1 step (KeyUp filtered)
        assert_eq!(skill.steps.len(), 1);
        assert_eq!(skill.steps[0].description, "Type \"a\"");
    }
}
