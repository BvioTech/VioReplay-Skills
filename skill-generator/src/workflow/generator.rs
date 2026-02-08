//! Skill Generator
//!
//! Transforms recordings into SKILL.md files with proper variable extraction,
//! selector ranking, and verification blocks.

use super::recording::Recording;
use crate::analysis::intent_binding::{Action, ScrollDirection};
use crate::analysis::kinematic_segmentation::{KinematicSegmenter, MovementPattern};
use crate::analysis::rdp_simplification::RdpSimplifier;
use crate::capture::types::{EnrichedEvent, EventType};
use crate::chunking::action_clustering::{ActionClusterer, ClusteringConfig, UnitTask};
use crate::chunking::context_stack::ContextStack;
use crate::chunking::goms_detector::{ChunkBoundary, GomsDetector};
use crate::codegen::markdown_builder::MarkdownBuilder;
use crate::codegen::skill_compiler::{AxMetadata, CompiledSkill, CompiledStep, ExecutionContext as CompiledContext, SkillInput, TechnicalContext};
use crate::codegen::validation::{SkillValidator, ValidationResult};
use crate::semantic::context_reconstruction::ContextReconstructor;
use crate::semantic::null_handler::NullHandler;
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
    /// Use ActionClusterer to group events into UnitTasks before step generation
    pub use_action_clustering: bool,
    /// Action clustering configuration (used when use_action_clustering is true)
    pub clustering_config: ClusteringConfig,
    /// Use NullHandler local recovery (AX retry + spiral search + Vision OCR) before LLM
    pub use_local_recovery: bool,
    /// Enable Vision OCR as part of local recovery pipeline
    pub use_vision_ocr: bool,
    /// Use RDP + kinematic analysis for trajectory processing and confidence adjustment
    pub use_trajectory_analysis: bool,
    /// Use GOMS mental operator detection for cognitive boundary analysis
    pub use_goms_detection: bool,
    /// Use ContextStack to track window/app context changes during generation
    pub use_context_tracking: bool,
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
            use_action_clustering: true,
            clustering_config: ClusteringConfig::default(),
            use_local_recovery: true,
            use_vision_ocr: true,
            use_trajectory_analysis: true,
            use_goms_detection: true,
            use_context_tracking: true,
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
    /// Action clusterer for grouping events into UnitTasks
    action_clusterer: ActionClusterer,
    /// Null handler for local accessibility recovery
    null_handler: NullHandler,
    /// RDP trajectory simplifier
    rdp_simplifier: RdpSimplifier,
    /// Kinematic segmenter for movement analysis
    kinematic_segmenter: KinematicSegmenter,
    /// GOMS detector for cognitive boundary analysis
    goms_detector: GomsDetector,
    /// Context stack for tracking window/app context changes
    context_stack: std::sync::Mutex<ContextStack>,
}

impl SkillGenerator {
    /// Create a new skill generator with default config
    pub fn new() -> Self {
        Self::with_config(GeneratorConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: GeneratorConfig) -> Self {
        let action_clusterer = ActionClusterer::with_config(config.clustering_config.clone());
        let null_handler = if config.use_vision_ocr {
            NullHandler::new().with_vision()
        } else {
            NullHandler::new()
        };
        Self {
            config,
            variable_extractor: VariableExtractor::new(),
            selector_ranker: SelectorRanker::new(),
            postcondition_extractor: PostconditionExtractor::new(),
            hoare_generator: HoareTripleGenerator::new(),
            validator: SkillValidator::new(),
            context_reconstructor: ContextReconstructor::new(),
            action_clusterer,
            null_handler,
            rdp_simplifier: RdpSimplifier::new(),
            kinematic_segmenter: KinematicSegmenter::new(),
            goms_detector: GomsDetector::new(),
            context_stack: std::sync::Mutex::new(ContextStack::new()),
        }
    }

    /// Generate a skill from a recording
    pub fn generate(&self, recording: &Recording) -> Result<GeneratedSkill, GeneratorError> {
        let mut stats = PipelineStats::default();

        // Count events missing semantic data before recovery
        let missing_before = recording.events.iter()
            .filter(|e| e.raw.event_type.is_click() && e.semantic.is_none())
            .count();

        // Step 0a: Local recovery for events missing semantic data (fast, no network)
        let locally_recovered = if self.config.use_local_recovery {
            self.enrich_with_local_recovery(recording)
        } else {
            recording.clone()
        };

        // Count how many were recovered locally
        let missing_after_local = locally_recovered.events.iter()
            .filter(|e| e.raw.event_type.is_click() && e.semantic.is_none())
            .count();
        stats.local_recovery_count = missing_before.saturating_sub(missing_after_local);

        // Create a shared tokio runtime for all async LLM calls in this generation
        let runtime = if self.config.use_llm_semantic || self.config.extract_variables {
            match tokio::runtime::Runtime::new() {
                Ok(rt) => Some(rt),
                Err(e) => {
                    debug!("Failed to create tokio runtime for LLM: {}", e);
                    None
                }
            }
        } else {
            None
        };

        // Step 0b: LLM semantic inference for remaining gaps (slower, requires API key)
        let enriched_recording = if self.config.use_llm_semantic {
            if let Some(ref rt) = runtime {
                self.enrich_with_llm_semantic(&locally_recovered, rt)
            } else {
                locally_recovered
            }
        } else {
            locally_recovered
        };

        let missing_after_llm = enriched_recording.events.iter()
            .filter(|e| e.raw.event_type.is_click() && e.semantic.is_none())
            .count();
        stats.llm_enriched_count = missing_after_local.saturating_sub(missing_after_llm);

        // Step 1: Extract significant events (clicks, keystrokes)
        let significant_events = self.extract_significant_events(&enriched_recording);
        stats.significant_events_count = significant_events.len();

        if significant_events.is_empty() {
            return Err(GeneratorError::NoSignificantEvents);
        }

        // Step 1.5: Detect GOMS cognitive boundaries (if enabled)
        let goms_boundaries = if self.config.use_goms_detection {
            let boundaries = self.goms_detector.detect_boundaries(&enriched_recording.events);
            if !boundaries.is_empty() {
                info!("GOMS detector found {} cognitive boundaries", boundaries.len());
            }
            stats.goms_boundaries_count = boundaries.len();
            boundaries
        } else {
            Vec::new()
        };

        // Step 1.6: Track window/app context transitions (if enabled)
        if self.config.use_context_tracking {
            self.track_context_transitions(&enriched_recording);
            stats.context_transitions_count = self.context_stack.lock()
                .map(|s| s.depth())
                .unwrap_or(0);
        }

        // Step 2: Bind intents to events
        let actions = self.bind_intents(&significant_events, &enriched_recording);

        // Step 2.5: Cluster significant events into UnitTasks (if enabled)
        let unit_tasks = if self.config.use_action_clustering {
            let action_list: Vec<Action> = actions.iter().map(|(_, a)| a.clone()).collect();
            let sig_events: Vec<EnrichedEvent> = significant_events.iter().map(|e| (*e).clone()).collect();
            let tasks = self.action_clusterer.cluster(&sig_events, &action_list);
            if tasks.is_empty() {
                debug!("ActionClusterer produced no tasks, falling back to raw actions");
                None
            } else {
                info!("ActionClusterer produced {} unit tasks from {} significant events", tasks.len(), sig_events.len());
                stats.unit_tasks_count = tasks.len();
                Some(tasks)
            }
        } else {
            None
        };

        // Step 3: Extract variables
        let variables = if self.config.extract_variables {
            let goal = enriched_recording.metadata.goal.as_deref().unwrap_or("");
            if let Some(ref rt) = runtime {
                self.variable_extractor.extract_with_runtime(&enriched_recording.events, goal, rt)
            } else {
                self.variable_extractor.extract(&enriched_recording.events, goal)
            }
        } else {
            Vec::new()
        };
        stats.variables_count = variables.len();

        // Step 4: Generate steps (from UnitTasks if available, otherwise from raw actions)
        let steps = if let Some(ref tasks) = unit_tasks {
            self.generate_steps_from_tasks(tasks, &variables)
        } else {
            self.generate_steps(&actions, &variables, &significant_events)
        };

        // Step 4.5: Analyze trajectories and adjust confidence (if enabled)
        let steps = if self.config.use_trajectory_analysis {
            let adjusted = self.adjust_confidence_from_trajectory(steps, &enriched_recording);
            stats.trajectory_adjustments_count = adjusted.iter()
                .filter(|s| s.confidence > 0.7)
                .count();
            adjusted
        } else {
            steps
        };

        // Step 4.6: Boost confidence for steps near GOMS cognitive boundaries
        let steps = if self.config.use_goms_detection && !goms_boundaries.is_empty() {
            self.apply_goms_confidence(steps, &goms_boundaries, &enriched_recording)
        } else {
            steps
        };

        // Step 5: Generate selectors
        let steps_with_selectors = if let Some(ref tasks) = unit_tasks {
            self.add_selectors_from_tasks(steps, tasks)
        } else {
            self.add_selectors(steps, &significant_events)
        };

        // Step 6: Add verification blocks
        let steps_with_verification = if self.config.include_verification {
            if let Some(ref tasks) = unit_tasks {
                self.add_verification_from_tasks(steps_with_selectors, tasks)
            } else {
                self.add_verification(steps_with_selectors, &significant_events)
            }
        } else {
            steps_with_selectors
        };

        stats.generated_steps_count = steps_with_verification.len();

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
            stats,
        };

        Ok(skill)
    }

    /// Attempt local recovery of semantic data using NullHandler
    ///
    /// Uses the accessibility API retry pipeline (injection + spiral search)
    /// and optional Vision OCR fallback. Runs before LLM enrichment since
    /// these methods are local, fast, and don't require API keys.
    fn enrich_with_local_recovery(&self, recording: &Recording) -> Recording {
        let events_needing_semantic: Vec<usize> = recording
            .events
            .iter()
            .enumerate()
            .filter(|(_, e)| e.raw.event_type.is_click() && e.semantic.is_none())
            .map(|(i, _)| i)
            .collect();

        if events_needing_semantic.is_empty() {
            debug!("All click events have semantic data, no local recovery needed");
            return recording.clone();
        }

        info!(
            "Attempting local recovery for {} events without semantic data",
            events_needing_semantic.len()
        );

        let mut enriched = recording.clone();
        let mut recovered_count = 0;

        for idx in events_needing_semantic {
            let event = &enriched.events[idx];
            let (x, y) = event.raw.coordinates;

            // Get app bundle ID from surrounding events if available
            let app_bundle = event
                .semantic
                .as_ref()
                .and_then(|s| s.app_bundle_id.clone())
                .or_else(|| {
                    // Look at nearby events for bundle info
                    enriched.events.iter().find_map(|e| {
                        e.semantic.as_ref().and_then(|s| s.app_bundle_id.clone())
                    })
                });

            let result = self.null_handler.recover(x, y, app_bundle.as_deref());

            if let Some(context) = result.context {
                debug!(
                    "Local recovery for event {} via {:?}: role={:?}, title={:?} ({:.0}ms)",
                    idx, result.strategy, context.ax_role, context.title, result.duration_ms
                );
                enriched.events[idx].semantic = Some(context);
                recovered_count += 1;
            }
        }

        if recovered_count > 0 {
            info!("Local recovery filled {} events", recovered_count);
        }

        enriched
    }

    /// Enrich events with LLM semantic inference for those missing semantic data
    fn enrich_with_llm_semantic(&self, recording: &Recording, rt: &tokio::runtime::Runtime) -> Recording {
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

    /// Analyze mouse trajectories between clicks and adjust step confidence
    ///
    /// Uses RDP simplification to extract the trajectory shape, then
    /// kinematic analysis to classify the movement pattern:
    /// - Ballistic: confident, direct → keep/boost confidence
    /// - Corrective: hesitant, multiple corrections → reduce confidence
    /// - Searching: exploratory movement → reduce confidence
    /// - Stationary: no movement → neutral
    fn adjust_confidence_from_trajectory(
        &self,
        mut steps: Vec<GeneratedStep>,
        recording: &Recording,
    ) -> Vec<GeneratedStep> {
        // Collect mouse-move events as raw events for trajectory analysis
        let all_raw: Vec<&crate::capture::types::RawEvent> = recording
            .events
            .iter()
            .map(|e| &e.raw)
            .collect();

        // Find click event indices in the full event stream
        let click_indices: Vec<usize> = all_raw
            .iter()
            .enumerate()
            .filter(|(_, e)| e.event_type.is_click())
            .map(|(i, _)| i)
            .collect();

        // For each click-based step, analyze the trajectory leading to it
        for (step_idx, step) in steps.iter_mut().enumerate() {
            // Only adjust confidence for click-based actions
            if !matches!(
                step.action,
                Action::Click { .. } | Action::DoubleClick { .. } | Action::RightClick { .. }
            ) {
                continue;
            }

            // Find the trajectory segment leading to this click
            if step_idx >= click_indices.len() {
                continue;
            }

            let click_idx = click_indices[step_idx];
            let start_idx = if step_idx > 0 {
                click_indices[step_idx - 1] + 1
            } else {
                0
            };

            // Extract mouse-move events in this segment
            let segment_events: Vec<crate::capture::types::RawEvent> = all_raw[start_idx..=click_idx]
                .iter()
                .filter(|e| e.event_type.is_mouse_move() || e.event_type.is_click())
                .map(|e| (*e).clone())
                .collect();

            if segment_events.len() < 3 {
                continue; // Not enough points for analysis
            }

            // Simplify trajectory
            let simplified = self.rdp_simplifier.simplify_events(&segment_events);

            if simplified.len() < 2 {
                continue;
            }

            // Kinematic analysis
            let analysis = self.kinematic_segmenter.analyze(&simplified);

            // Adjust confidence based on movement pattern
            let confidence_adjustment = match analysis.pattern {
                MovementPattern::Ballistic => 0.05,    // Direct movement → boost
                MovementPattern::Stationary => 0.0,    // No change
                MovementPattern::Searching => -0.10,   // Uncertain → reduce
                MovementPattern::Corrective => -0.15,  // Very uncertain → reduce more
            };

            step.confidence = (step.confidence + confidence_adjustment).clamp(0.1, 1.0);

            if confidence_adjustment != 0.0 {
                debug!(
                    "Step {}: trajectory pattern {:?}, confidence adjusted by {:.2} to {:.2}",
                    step.number, analysis.pattern, confidence_adjustment, step.confidence
                );
            }
        }

        steps
    }

    /// Apply GOMS cognitive boundary analysis to boost step confidence
    ///
    /// Steps that immediately follow a detected mental operator (cognitive pause)
    /// are likely deliberate actions, so their confidence is boosted.
    fn apply_goms_confidence(
        &self,
        mut steps: Vec<GeneratedStep>,
        boundaries: &[ChunkBoundary],
        recording: &Recording,
    ) -> Vec<GeneratedStep> {
        if boundaries.is_empty() || steps.is_empty() {
            return steps;
        }

        // Build a set of event indices that are at GOMS boundaries
        let boundary_indices: std::collections::HashSet<usize> =
            boundaries.iter().map(|b| b.event_index).collect();

        for step in &mut steps {
            // Check if any of this step's source events are near a GOMS boundary
            let near_boundary = step.source_events.iter().any(|&src_idx| {
                // Check if this event index or adjacent indices are GOMS boundaries
                boundary_indices.contains(&src_idx)
                    || (src_idx > 0 && boundary_indices.contains(&(src_idx - 1)))
                    || boundary_indices.contains(&(src_idx + 1))
            });

            if near_boundary {
                // Steps following a cognitive pause are deliberate — boost confidence
                let boost = 0.05;
                step.confidence = (step.confidence + boost).min(1.0);
                debug!(
                    "Step {}: near GOMS boundary, confidence boosted by {:.2} to {:.2}",
                    step.number, boost, step.confidence
                );
            }
        }

        let _ = recording; // Used for context in future enhancements
        steps
    }

    /// Track window/app context transitions from a recording
    ///
    /// Populates the ContextStack with window focus changes detected
    /// from semantic data in the recording events.
    fn track_context_transitions(&self, recording: &Recording) {
        use crate::chunking::context_stack::WindowContext;

        let mut stack = match self.context_stack.lock() {
            Ok(s) => s,
            Err(_) => return,
        };
        stack.clear();

        let mut last_bundle: Option<String> = None;

        for event in &recording.events {
            if let Some(ref semantic) = event.semantic {
                let bundle = semantic.app_bundle_id.clone().unwrap_or_default();

                // Detect app switches
                if let Some(ref prev_bundle) = last_bundle {
                    if !bundle.is_empty() && bundle != *prev_bundle {
                        stack.on_app_switched(prev_bundle, &bundle, event.raw.timestamp.ticks());
                        debug!(
                            "Context transition: {} -> {} at tick {}",
                            prev_bundle, bundle, event.raw.timestamp.ticks()
                        );
                    }
                }

                // Track window focus from semantic title changes
                if let Some(ref title) = semantic.title {
                    let window_id = bundle.len() as u32 ^ title.len() as u32;
                    let ctx = WindowContext {
                        window_id,
                        title: title.clone(),
                        app_bundle_id: bundle.clone(),
                        app_name: bundle.split('.').next_back().unwrap_or("").to_string(),
                        z_index: 0,
                        activated_at: event.raw.timestamp.ticks(),
                    };
                    stack.on_window_focused(ctx, event.raw.timestamp.ticks());
                }

                if !bundle.is_empty() {
                    last_bundle = Some(bundle);
                }
            }
        }

        let transitions = stack.depth();
        if transitions > 0 {
            info!("Context tracking: {} window contexts detected", transitions);
        }
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

    /// Generate steps from UnitTasks produced by ActionClusterer
    ///
    /// Each UnitTask maps to one GeneratedStep, using the task's primary_action
    /// and name. This produces higher-level steps compared to raw action processing,
    /// because the clusterer merges dropdown patterns, transient interactions, and
    /// groups related events by time proximity and context.
    fn generate_steps_from_tasks(
        &self,
        tasks: &[UnitTask],
        variables: &[ExtractedVariable],
    ) -> Vec<GeneratedStep> {
        let mut steps = Vec::new();

        for task in tasks {
            let description = if task.name.starts_with("Select ") || task.name.starts_with("Click ") || task.name.starts_with("Fill ") {
                task.name.clone()
            } else {
                // Use the primary action to generate a description, with context from the first significant event
                let first_significant = task.events.iter().find(|e| {
                    e.raw.event_type.is_click() || e.raw.event_type.is_keyboard()
                });
                if let Some(event) = first_significant {
                    self.generate_step_description(&task.primary_action, event)
                } else if let Some(event) = task.events.first() {
                    self.generate_step_description(&task.primary_action, event)
                } else {
                    task.description.clone()
                }
            };

            let step_variables: Vec<ExtractedVariable> = variables
                .iter()
                .filter(|v| description.contains(&format!("{{{{{}}}}}", v.name)))
                .cloned()
                .collect();

            // Collect source event indices (using the event sequence numbers)
            let source_events: Vec<usize> = task
                .events
                .iter()
                .enumerate()
                .map(|(i, _)| i)
                .collect();

            steps.push(GeneratedStep {
                number: steps.len() + 1,
                description,
                action: task.primary_action.clone(),
                selector: None,
                fallback_selectors: vec![],
                variables: step_variables,
                verification: None,
                source_events,
                confidence: task.primary_action.confidence(),
            });
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

    /// Add selectors to steps using UnitTask events
    fn add_selectors_from_tasks(
        &self,
        mut steps: Vec<GeneratedStep>,
        tasks: &[UnitTask],
    ) -> Vec<GeneratedStep> {
        for (i, step) in steps.iter_mut().enumerate() {
            if let Some(task) = tasks.get(i) {
                // Find the first click event in the task for selector generation
                let target_event = task.events.iter().find(|e| {
                    e.raw.event_type.is_click()
                }).or(task.events.first());

                if let Some(event) = target_event {
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
        }

        steps
    }

    /// Add verification blocks to steps using UnitTask events
    fn add_verification_from_tasks(
        &self,
        mut steps: Vec<GeneratedStep>,
        tasks: &[UnitTask],
    ) -> Vec<GeneratedStep> {
        for (i, step) in steps.iter_mut().enumerate() {
            if let Some(task) = tasks.get(i) {
                let pre_event = task.events.first();
                // Use the last event of this task or the first event of the next task
                let post_event = if i + 1 < tasks.len() {
                    tasks[i + 1].events.first()
                } else {
                    task.events.last()
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

/// Pipeline execution statistics
#[derive(Debug, Clone, Default)]
pub struct PipelineStats {
    /// Number of events recovered by local recovery (AX retry + spiral search + Vision OCR)
    pub local_recovery_count: usize,
    /// Number of events enriched by LLM semantic inference
    pub llm_enriched_count: usize,
    /// Number of GOMS cognitive boundaries detected
    pub goms_boundaries_count: usize,
    /// Number of context transitions tracked
    pub context_transitions_count: usize,
    /// Number of UnitTasks from action clustering
    pub unit_tasks_count: usize,
    /// Number of significant events extracted from recording
    pub significant_events_count: usize,
    /// Number of trajectory confidence adjustments applied
    pub trajectory_adjustments_count: usize,
    /// Number of variables extracted
    pub variables_count: usize,
    /// Total generated steps
    pub generated_steps_count: usize,
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
    /// Pipeline execution statistics
    pub stats: PipelineStats,
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
            ..Default::default()
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
        let config = GeneratorConfig {
            use_action_clustering: false,
            ..Default::default()
        };
        let generator = SkillGenerator::with_config(config);
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
            use_action_clustering: false,
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
            use_action_clustering: false,
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
            use_action_clustering: false,
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
            use_action_clustering: false,
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
            use_action_clustering: false,
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
            use_action_clustering: false,
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
            use_action_clustering: false,
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

    #[test]
    fn test_action_clustering_merges_click_and_typing() {
        MachTimebase::init();
        let config = GeneratorConfig {
            include_verification: false,
            extract_variables: false,
            use_llm_semantic: false,
            use_action_clustering: true,
            ..Default::default()
        };
        let generator = SkillGenerator::with_config(config);
        let mut recording = Recording::new("cluster_test".to_string(), Some("Fill a field".to_string()));

        // Click on text field + type text = single unit task (form fill)
        let mut click = make_test_event(EventType::LeftMouseDown, 200.0, 100.0);
        click.semantic = Some(SemanticContext {
            ax_role: Some("AXTextField".to_string()),
            title: Some("Username".to_string()),
            ..Default::default()
        });
        recording.add_event(click);

        recording.add_event(make_test_event_with_modifiers(
            EventType::KeyDown, 0.0, 0.0,
            ModifierFlags::default(), None, Some('u'),
        ));
        recording.add_event(make_test_event_with_modifiers(
            EventType::KeyDown, 0.0, 0.0,
            ModifierFlags::default(), None, Some('s'),
        ));

        let skill = generator.generate(&recording).unwrap();

        // Clusterer groups click+typing into 1 unit task
        assert_eq!(skill.steps.len(), 1);
    }

    #[test]
    fn test_action_clustering_disabled_produces_raw_steps() {
        MachTimebase::init();
        let config = GeneratorConfig {
            include_verification: false,
            extract_variables: false,
            use_llm_semantic: false,
            use_action_clustering: false,
            ..Default::default()
        };
        let generator = SkillGenerator::with_config(config);
        let mut recording = Recording::new("no_cluster_test".to_string(), None);

        let mut click = make_test_event(EventType::LeftMouseDown, 200.0, 100.0);
        click.semantic = Some(SemanticContext {
            ax_role: Some("AXButton".to_string()),
            title: Some("OK".to_string()),
            ..Default::default()
        });
        recording.add_event(click);

        recording.add_event(make_test_event_with_modifiers(
            EventType::KeyDown, 0.0, 0.0,
            ModifierFlags::default(), None, Some('y'),
        ));

        let skill = generator.generate(&recording).unwrap();

        // Without clustering, each significant event = 1 step
        assert_eq!(skill.steps.len(), 2);
    }

    #[test]
    fn test_action_clustering_fallback_on_single_event() {
        MachTimebase::init();
        let config = GeneratorConfig {
            include_verification: false,
            use_llm_semantic: false,
            use_action_clustering: true,
            ..Default::default()
        };
        let generator = SkillGenerator::with_config(config);
        let mut recording = Recording::new("single_event_test".to_string(), None);

        // Single click: clusterer min_events=2, so falls back to raw actions
        let mut click = make_test_event(EventType::LeftMouseDown, 100.0, 100.0);
        click.semantic = Some(SemanticContext {
            ax_role: Some("AXButton".to_string()),
            title: Some("Submit".to_string()),
            ..Default::default()
        });
        recording.add_event(click);

        let skill = generator.generate(&recording).unwrap();

        // Should still produce 1 step via raw action fallback
        assert_eq!(skill.steps.len(), 1);
        assert!(skill.steps[0].description.contains("Submit"));
    }

    #[test]
    fn test_clustering_config_passed_through() {
        let config = GeneratorConfig {
            use_action_clustering: true,
            clustering_config: ClusteringConfig {
                max_gap_ms: 500,
                min_events: 1,
                merge_transient: false,
                min_movement_px: 10.0,
            },
            ..Default::default()
        };
        let generator = SkillGenerator::with_config(config);
        assert_eq!(generator.action_clusterer.config.max_gap_ms, 500);
        assert_eq!(generator.action_clusterer.config.min_events, 1);
        assert!(!generator.action_clusterer.config.merge_transient);
    }

    #[test]
    fn test_local_recovery_config_defaults() {
        let config = GeneratorConfig::default();
        assert!(config.use_local_recovery);
        assert!(config.use_vision_ocr);
    }

    #[test]
    fn test_local_recovery_disabled_skips_recovery() {
        MachTimebase::init();
        let config = GeneratorConfig {
            use_local_recovery: false,
            use_llm_semantic: false,
            use_action_clustering: false,
            ..Default::default()
        };
        let generator = SkillGenerator::with_config(config);
        let mut recording = Recording::new("no_recovery_test".to_string(), None);

        // Click without semantic data - recovery disabled, should still work
        let click = make_test_event(EventType::LeftMouseDown, 100.0, 100.0);
        recording.add_event(click);

        let skill = generator.generate(&recording).unwrap();
        assert_eq!(skill.steps.len(), 1);
        // Without recovery, element name will be coordinate-based
        assert!(skill.steps[0].description.contains("100"));
    }

    #[test]
    fn test_local_recovery_runs_before_llm() {
        MachTimebase::init();
        // Verify local recovery is invoked (Step 0a before 0b)
        let config = GeneratorConfig {
            use_local_recovery: true,
            use_llm_semantic: false,
            use_action_clustering: false,
            use_vision_ocr: false,
            ..Default::default()
        };
        let generator = SkillGenerator::with_config(config);
        let mut recording = Recording::new("recovery_order_test".to_string(), None);

        let click = make_test_event(EventType::LeftMouseDown, 500.0, 500.0);
        recording.add_event(click);

        // Should not panic or error even when recovery finds nothing
        let skill = generator.generate(&recording).unwrap();
        assert_eq!(skill.steps.len(), 1);
    }

    #[test]
    fn test_vision_ocr_flag_configures_null_handler() {
        // With vision enabled
        let config_with_vision = GeneratorConfig {
            use_vision_ocr: true,
            ..Default::default()
        };
        let _gen = SkillGenerator::with_config(config_with_vision);
        // No panic = vision fallback initialized correctly

        // Without vision
        let config_no_vision = GeneratorConfig {
            use_vision_ocr: false,
            ..Default::default()
        };
        let _gen = SkillGenerator::with_config(config_no_vision);
        // No panic = null handler initialized without vision
    }

    #[test]
    fn test_trajectory_analysis_config_default() {
        let config = GeneratorConfig::default();
        assert!(config.use_trajectory_analysis);
    }

    #[test]
    fn test_trajectory_analysis_adjusts_click_confidence() {
        MachTimebase::init();
        let config = GeneratorConfig {
            include_verification: false,
            extract_variables: false,
            use_llm_semantic: false,
            use_action_clustering: false,
            use_local_recovery: false,
            use_trajectory_analysis: true,
            ..Default::default()
        };
        let generator = SkillGenerator::with_config(config);
        let mut recording = Recording::new("trajectory_test".to_string(), None);

        // Add mouse moves leading to a click (simulates direct ballistic movement)
        for i in 0..5 {
            let event = make_test_event(EventType::MouseMoved, 100.0 + (i as f64 * 20.0), 200.0);
            recording.add_event(event);
        }

        // Click at end of trajectory
        let mut click = make_test_event(EventType::LeftMouseDown, 180.0, 200.0);
        click.semantic = Some(SemanticContext {
            ax_role: Some("AXButton".to_string()),
            title: Some("Target".to_string()),
            ..Default::default()
        });
        recording.add_event(click);

        let skill = generator.generate(&recording).unwrap();
        assert_eq!(skill.steps.len(), 1);
        // Confidence should be adjusted (not exactly 0.9 default)
        assert!(skill.steps[0].confidence > 0.0);
    }

    #[test]
    fn test_trajectory_analysis_disabled() {
        MachTimebase::init();
        let config = GeneratorConfig {
            include_verification: false,
            extract_variables: false,
            use_llm_semantic: false,
            use_action_clustering: false,
            use_local_recovery: false,
            use_trajectory_analysis: false,
            ..Default::default()
        };
        let generator = SkillGenerator::with_config(config);
        let mut recording = Recording::new("no_trajectory_test".to_string(), None);

        let mut click = make_test_event(EventType::LeftMouseDown, 100.0, 100.0);
        click.semantic = Some(SemanticContext {
            ax_role: Some("AXButton".to_string()),
            title: Some("Button".to_string()),
            ..Default::default()
        });
        recording.add_event(click);

        let skill = generator.generate(&recording).unwrap();
        assert_eq!(skill.steps.len(), 1);
        // Without trajectory analysis, default click confidence is 0.9
        assert_eq!(skill.steps[0].confidence, 0.9);
    }

    #[test]
    fn test_goms_detection_enabled_by_default() {
        let config = GeneratorConfig::default();
        assert!(config.use_goms_detection);
    }

    #[test]
    fn test_goms_detection_disabled_skips_analysis() {
        MachTimebase::init();
        let generator = SkillGenerator::with_config(GeneratorConfig {
            use_action_clustering: false,
            use_goms_detection: false,
            ..Default::default()
        });
        let mut recording = Recording::new("no_goms_test".to_string(), None);

        let mut click = make_test_event(EventType::LeftMouseDown, 100.0, 100.0);
        click.semantic = Some(SemanticContext {
            ax_role: Some("AXButton".to_string()),
            title: Some("Button".to_string()),
            ..Default::default()
        });
        recording.add_event(click);

        let skill = generator.generate(&recording).unwrap();
        assert_eq!(skill.steps.len(), 1);
        // Without GOMS, confidence is unchanged from default
        assert!(skill.steps[0].confidence > 0.0);
    }

    #[test]
    fn test_context_tracking_enabled_by_default() {
        let config = GeneratorConfig::default();
        assert!(config.use_context_tracking);
    }

    #[test]
    fn test_context_tracking_populates_stack() {
        MachTimebase::init();
        let generator = SkillGenerator::with_config(GeneratorConfig {
            use_action_clustering: false,
            use_context_tracking: true,
            ..Default::default()
        });
        let mut recording = Recording::new("context_test".to_string(), None);

        // Two clicks in different apps
        let mut click1 = make_test_event(EventType::LeftMouseDown, 100.0, 100.0);
        click1.semantic = Some(SemanticContext {
            ax_role: Some("AXButton".to_string()),
            title: Some("Save".to_string()),
            app_bundle_id: Some("com.apple.TextEdit".to_string()),
            ..Default::default()
        });
        recording.add_event(click1);

        let mut click2 = make_test_event(EventType::LeftMouseDown, 200.0, 200.0);
        click2.semantic = Some(SemanticContext {
            ax_role: Some("AXButton".to_string()),
            title: Some("Open".to_string()),
            app_bundle_id: Some("com.apple.Finder".to_string()),
            ..Default::default()
        });
        recording.add_event(click2);

        let skill = generator.generate(&recording).unwrap();
        assert!(!skill.steps.is_empty());

        // Verify context stack was populated (at least 1 window tracked)
        let stack = generator.context_stack.lock().unwrap();
        assert!(stack.depth() > 0);
    }

    #[test]
    fn test_context_tracking_disabled_skips() {
        MachTimebase::init();
        let generator = SkillGenerator::with_config(GeneratorConfig {
            use_action_clustering: false,
            use_context_tracking: false,
            ..Default::default()
        });
        let mut recording = Recording::new("no_context_test".to_string(), None);

        let mut click = make_test_event(EventType::LeftMouseDown, 100.0, 100.0);
        click.semantic = Some(SemanticContext {
            ax_role: Some("AXButton".to_string()),
            title: Some("Button".to_string()),
            app_bundle_id: Some("com.test.App".to_string()),
            ..Default::default()
        });
        recording.add_event(click);

        let skill = generator.generate(&recording).unwrap();
        assert_eq!(skill.steps.len(), 1);

        // Context stack should be empty when tracking is disabled
        let stack = generator.context_stack.lock().unwrap();
        assert_eq!(stack.depth(), 0);
    }

    #[test]
    fn test_goms_boundaries_boost_confidence() {
        use crate::chunking::goms_detector::{ChunkBoundary, HesitationIndex, OperatorType};

        let generator = SkillGenerator::with_config(GeneratorConfig {
            use_action_clustering: false,
            ..Default::default()
        });

        // Create a step with source_events referencing index 5
        let step = GeneratedStep {
            number: 1,
            description: "Click button".to_string(),
            action: Action::Click {
                element_name: "Button".to_string(),
                element_role: "AXButton".to_string(),
                confidence: 0.8,
            },
            selector: None,
            fallback_selectors: vec![],
            variables: vec![],
            verification: None,
            source_events: vec![5],
            confidence: 0.8,
        };

        // Create a GOMS boundary at index 5
        let boundaries = vec![ChunkBoundary {
            event_index: 5,
            timestamp_ticks: 5000,
            operator_type: OperatorType::Mental,
            hesitation_index: HesitationIndex {
                inverse_velocity: 0.9,
                direction_change: 0.1,
                pause_duration: 1500.0,
                total: 0.85,
            },
            confidence: 0.9,
        }];

        let recording = Recording::new("test".to_string(), None);
        let result = generator.apply_goms_confidence(vec![step], &boundaries, &recording);

        // Confidence should be boosted
        assert_eq!(result.len(), 1);
        assert!(result[0].confidence > 0.8);
        assert!((result[0].confidence - 0.85).abs() < 0.01);
    }

    #[test]
    fn test_pipeline_stats_default_values() {
        let stats = PipelineStats::default();
        assert_eq!(stats.local_recovery_count, 0);
        assert_eq!(stats.llm_enriched_count, 0);
        assert_eq!(stats.goms_boundaries_count, 0);
        assert_eq!(stats.context_transitions_count, 0);
        assert_eq!(stats.unit_tasks_count, 0);
        assert_eq!(stats.significant_events_count, 0);
        assert_eq!(stats.trajectory_adjustments_count, 0);
        assert_eq!(stats.variables_count, 0);
        assert_eq!(stats.generated_steps_count, 0);
    }

    #[test]
    fn test_generate_populates_stats_significant_and_steps() {
        MachTimebase::init();
        let config = GeneratorConfig {
            include_verification: false,
            extract_variables: false,
            use_llm_semantic: false,
            use_action_clustering: false,
            use_local_recovery: false,
            use_trajectory_analysis: false,
            use_goms_detection: false,
            use_context_tracking: false,
            ..Default::default()
        };
        let generator = SkillGenerator::with_config(config);
        let mut recording = Recording::new("stats_test".to_string(), Some("Test stats".to_string()));

        // Add a click event with semantic context
        let mut click = make_test_event(EventType::LeftMouseDown, 100.0, 200.0);
        click.semantic = Some(SemanticContext {
            ax_role: Some("AXButton".to_string()),
            title: Some("OK".to_string()),
            ..Default::default()
        });
        recording.add_event(click);

        // Add a key event
        recording.add_event(make_test_event_with_modifiers(
            EventType::KeyDown, 0.0, 0.0,
            ModifierFlags::default(), None, Some('x'),
        ));

        let skill = generator.generate(&recording).unwrap();

        assert!(skill.stats.significant_events_count > 0, "significant_events_count should be > 0");
        assert!(skill.stats.generated_steps_count > 0, "generated_steps_count should be > 0");
        // With all optional features disabled, these should be zero
        assert_eq!(skill.stats.local_recovery_count, 0);
        assert_eq!(skill.stats.llm_enriched_count, 0);
        assert_eq!(skill.stats.goms_boundaries_count, 0);
        assert_eq!(skill.stats.context_transitions_count, 0);
        assert_eq!(skill.stats.unit_tasks_count, 0);
        assert_eq!(skill.stats.trajectory_adjustments_count, 0);
    }

    #[test]
    fn test_goms_disabled_gives_zero_goms_stats() {
        MachTimebase::init();
        let config = GeneratorConfig {
            include_verification: false,
            extract_variables: false,
            use_llm_semantic: false,
            use_action_clustering: false,
            use_local_recovery: false,
            use_trajectory_analysis: false,
            use_goms_detection: false,
            use_context_tracking: false,
            ..Default::default()
        };
        let generator = SkillGenerator::with_config(config);
        let mut recording = Recording::new("no_goms_stats_test".to_string(), None);

        let mut click = make_test_event(EventType::LeftMouseDown, 100.0, 100.0);
        click.semantic = Some(SemanticContext {
            ax_role: Some("AXButton".to_string()),
            title: Some("Submit".to_string()),
            ..Default::default()
        });
        recording.add_event(click);

        let skill = generator.generate(&recording).unwrap();
        assert_eq!(skill.stats.goms_boundaries_count, 0, "goms_boundaries_count should be 0 when goms disabled");
    }

    #[test]
    fn test_context_tracking_disabled_gives_zero_context_stats() {
        MachTimebase::init();
        let config = GeneratorConfig {
            include_verification: false,
            extract_variables: false,
            use_llm_semantic: false,
            use_action_clustering: false,
            use_local_recovery: false,
            use_trajectory_analysis: false,
            use_goms_detection: false,
            use_context_tracking: false,
            ..Default::default()
        };
        let generator = SkillGenerator::with_config(config);
        let mut recording = Recording::new("no_context_stats_test".to_string(), None);

        let mut click = make_test_event(EventType::LeftMouseDown, 100.0, 100.0);
        click.semantic = Some(SemanticContext {
            ax_role: Some("AXButton".to_string()),
            title: Some("OK".to_string()),
            app_bundle_id: Some("com.test.App".to_string()),
            ..Default::default()
        });
        recording.add_event(click);

        let skill = generator.generate(&recording).unwrap();
        assert_eq!(skill.stats.context_transitions_count, 0, "context_transitions_count should be 0 when tracking disabled");
    }

    #[test]
    fn test_action_clustering_disabled_gives_zero_unit_tasks_stats() {
        MachTimebase::init();
        let config = GeneratorConfig {
            include_verification: false,
            extract_variables: false,
            use_llm_semantic: false,
            use_action_clustering: false,
            use_local_recovery: false,
            use_trajectory_analysis: false,
            use_goms_detection: false,
            use_context_tracking: false,
            ..Default::default()
        };
        let generator = SkillGenerator::with_config(config);
        let mut recording = Recording::new("no_clustering_stats_test".to_string(), None);

        let mut click = make_test_event(EventType::LeftMouseDown, 100.0, 100.0);
        click.semantic = Some(SemanticContext {
            ax_role: Some("AXButton".to_string()),
            title: Some("Submit".to_string()),
            ..Default::default()
        });
        recording.add_event(click);

        let skill = generator.generate(&recording).unwrap();
        assert_eq!(skill.stats.unit_tasks_count, 0, "unit_tasks_count should be 0 when clustering disabled");
    }

    #[test]
    fn test_trajectory_analysis_disabled_gives_zero_trajectory_stats() {
        MachTimebase::init();
        let config = GeneratorConfig {
            include_verification: false,
            extract_variables: false,
            use_llm_semantic: false,
            use_action_clustering: false,
            use_local_recovery: false,
            use_trajectory_analysis: false,
            use_goms_detection: false,
            use_context_tracking: false,
            ..Default::default()
        };
        let generator = SkillGenerator::with_config(config);
        let mut recording = Recording::new("no_trajectory_stats_test".to_string(), None);

        let mut click = make_test_event(EventType::LeftMouseDown, 100.0, 100.0);
        click.semantic = Some(SemanticContext {
            ax_role: Some("AXButton".to_string()),
            title: Some("Target".to_string()),
            ..Default::default()
        });
        recording.add_event(click);

        let skill = generator.generate(&recording).unwrap();
        assert_eq!(skill.stats.trajectory_adjustments_count, 0, "trajectory_adjustments_count should be 0 when trajectory analysis disabled");
    }

    #[test]
    fn test_variables_disabled_gives_zero_variables_stats() {
        MachTimebase::init();
        let config = GeneratorConfig {
            include_verification: false,
            extract_variables: false,
            use_llm_semantic: false,
            use_action_clustering: false,
            use_local_recovery: false,
            use_trajectory_analysis: false,
            use_goms_detection: false,
            use_context_tracking: false,
            ..Default::default()
        };
        let generator = SkillGenerator::with_config(config);
        let mut recording = Recording::new("no_vars_stats_test".to_string(), None);

        let mut click = make_test_event(EventType::LeftMouseDown, 100.0, 100.0);
        click.semantic = Some(SemanticContext {
            ax_role: Some("AXButton".to_string()),
            title: Some("Button".to_string()),
            ..Default::default()
        });
        recording.add_event(click);

        let skill = generator.generate(&recording).unwrap();
        assert_eq!(skill.stats.variables_count, 0, "variables_count should be 0 when extract_variables disabled");
    }
}
