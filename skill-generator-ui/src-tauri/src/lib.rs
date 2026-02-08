// Skill Generator UI - Tauri Backend
// Provides commands for recording user interactions and generating SKILL.md files

use serde::{Deserialize, Serialize};
use skill_generator::app::config::Config;
use skill_generator::capture::event_tap::{check_accessibility_permissions, check_screen_recording_permission};
use skill_generator::capture::ring_buffer::{EventRingBuffer, DEFAULT_CAPACITY};
use skill_generator::time::timebase::MachTimebase;
use skill_generator::workflow::generator::SkillGenerator;
use skill_generator::workflow::recording::Recording;
use skill_generator::capture::EventTap;
use std::sync::Mutex;
use tauri::State;
use tracing::{info, warn, error, debug};

/// Max events to drain per status poll (keeps UI responsive)
const STATUS_POLL_BATCH_SIZE: usize = 1000;
/// Max events to drain when stopping recording (capture everything remaining)
const STOP_DRAIN_BATCH_SIZE: usize = 10000;
/// Max tokens for LLM skill inference (short structured JSON response)
const INFERENCE_MAX_TOKENS: u32 = 512;
/// Max length for sanitized skill names
const MAX_SKILL_NAME_LEN: usize = 50;
/// HTTP timeout for Anthropic API calls
const API_TIMEOUT_SECS: u64 = 30;

/// Sanitize a user-provided name for safe use in file paths.
/// Removes path separators, traversal sequences, control chars, and null bytes.
fn sanitize_filename(name: &str) -> Result<String, String> {
    if name.contains("..") {
        return Err("Invalid name: path traversal not allowed".to_string());
    }
    let sanitized: String = name.chars()
        .filter(|c| !c.is_control() && *c != '/' && *c != '\\' && *c != '\0')
        .collect();
    if sanitized.is_empty() {
        return Err("Name cannot be empty".to_string());
    }
    if sanitized.len() > 255 {
        return Err("Name too long (max 255 characters)".to_string());
    }
    Ok(sanitized)
}

/// Application state shared across commands
pub struct AppState {
    /// Current recording (if any)
    recording: Mutex<Option<Recording>>,
    /// Event tap for capture
    event_tap: Mutex<Option<EventTap>>,
    /// Ring buffer consumer for draining events
    consumer: Mutex<Option<skill_generator::capture::ring_buffer::EventConsumer>>,
    /// Whether recording is in progress
    is_recording: Mutex<bool>,
    /// API key for LLM
    api_key: Mutex<Option<String>>,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            recording: Mutex::new(None),
            event_tap: Mutex::new(None),
            consumer: Mutex::new(None),
            is_recording: Mutex::new(false),
            api_key: Mutex::new(None),
        }
    }
}

/// Recording info for frontend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordingInfo {
    pub name: String,
    pub path: String,
    pub event_count: usize,
    pub duration_ms: u64,
    pub created_at: String,
    pub goal: Option<String>,
}

/// Recording status for frontend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordingStatus {
    pub is_recording: bool,
    pub event_count: usize,
    pub duration_seconds: f64,
    pub has_accessibility: bool,
    pub has_screen_recording: bool,
}

/// Pipeline statistics for the frontend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineStatsInfo {
    pub local_recovery_count: usize,
    pub llm_enriched_count: usize,
    pub goms_boundaries_count: usize,
    pub context_transitions_count: usize,
    pub unit_tasks_count: usize,
    pub significant_events_count: usize,
    pub trajectory_adjustments_count: usize,
    pub variables_count: usize,
    pub generated_steps_count: usize,
    pub warnings: Vec<String>,
}

/// Generated skill info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedSkillInfo {
    pub name: String,
    pub path: String,
    pub steps_count: usize,
    pub variables_count: usize,
    pub stats: PipelineStatsInfo,
}

/// Check if accessibility permissions are granted
#[tauri::command]
fn check_permissions() -> bool {
    check_accessibility_permissions()
}

/// Set the API key for LLM features
#[tauri::command]
fn set_api_key(state: State<AppState>, key: String) -> Result<(), String> {
    let mut api_key = state.api_key.lock().map_err(|e| e.to_string())?;
    if key.is_empty() {
        *api_key = None;
    } else {
        *api_key = Some(key);
    }
    Ok(())
}

/// Get current API key (masked)
#[tauri::command]
fn get_api_key(state: State<AppState>) -> Result<Option<String>, String> {
    let api_key = state.api_key.lock().map_err(|e| e.to_string())?;
    Ok(api_key.as_ref().map(|k| {
        if k.len() > 8 {
            format!("{}...{}", &k[..4], &k[k.len()-4..])
        } else {
            "****".to_string()
        }
    }))
}

/// Start recording user interactions
#[tauri::command]
fn start_recording(state: State<AppState>, name: String, goal: Option<String>) -> Result<(), String> {
    // Initialize timebase
    MachTimebase::init();

    // Check if already recording
    {
        let is_recording = state.is_recording.lock().map_err(|e| e.to_string())?;
        if *is_recording {
            return Err("Recording already in progress".to_string());
        }
    }

    // Check accessibility permissions
    if !check_accessibility_permissions() {
        return Err("Accessibility permissions not granted. Please enable in System Preferences > Security & Privacy > Privacy > Accessibility".to_string());
    }

    // Create ring buffer
    let buffer = EventRingBuffer::with_capacity(DEFAULT_CAPACITY);
    let (producer, consumer) = buffer.split();

    // Create event tap
    let mut event_tap = EventTap::new().map_err(|e| e.to_string())?;

    // Start the event tap
    event_tap.start(producer).map_err(|e| e.to_string())?;

    // Create recording
    let recording_name = if name.is_empty() {
        chrono::Local::now().format("recording_%Y%m%d_%H%M%S").to_string()
    } else {
        sanitize_filename(&name)?
    };
    let recording = Recording::new(recording_name, goal);

    // Store state
    {
        let mut rec = state.recording.lock().map_err(|e| e.to_string())?;
        *rec = Some(recording);
    }
    {
        let mut tap = state.event_tap.lock().map_err(|e| e.to_string())?;
        *tap = Some(event_tap);
    }
    {
        let mut cons = state.consumer.lock().map_err(|e| e.to_string())?;
        *cons = Some(consumer);
    }
    {
        let mut is_rec = state.is_recording.lock().map_err(|e| e.to_string())?;
        *is_rec = true;
    }

    Ok(())
}

/// Get current recording status
#[tauri::command]
fn get_recording_status(state: State<AppState>) -> Result<RecordingStatus, String> {
    let is_recording = *state.is_recording.lock().map_err(|e| e.to_string())?;

    if !is_recording {
        return Ok(RecordingStatus {
            is_recording: false,
            event_count: 0,
            duration_seconds: 0.0,
            has_accessibility: check_accessibility_permissions(),
            has_screen_recording: check_screen_recording_permission(),
        });
    }

    // Drain events from buffer
    let mut event_count = 0;
    {
        let mut consumer = state.consumer.lock().map_err(|e| e.to_string())?;
        let mut recording = state.recording.lock().map_err(|e| e.to_string())?;

        if let (Some(ref mut cons), Some(ref mut rec)) = (consumer.as_mut(), recording.as_mut()) {
            let batch = cons.pop_batch(STATUS_POLL_BATCH_SIZE);
            for slot in batch {
                rec.add_raw_event(slot.event);
            }
            event_count = rec.len();
        }
    }

    let duration = {
        let recording = state.recording.lock().map_err(|e| e.to_string())?;
        recording.as_ref().map(|r| {
            let start = r.metadata.started_at;
            let now = chrono::Utc::now();
            (now - start).num_milliseconds() as f64 / 1000.0
        }).unwrap_or(0.0)
    };

    Ok(RecordingStatus {
        is_recording: true,
        event_count,
        duration_seconds: duration,
        has_accessibility: check_accessibility_permissions(),
        has_screen_recording: check_screen_recording_permission(),
    })
}

/// Stop recording and save
#[tauri::command]
fn stop_recording(state: State<AppState>) -> Result<RecordingInfo, String> {
    // Check if recording
    {
        let is_recording = state.is_recording.lock().map_err(|e| e.to_string())?;
        if !*is_recording {
            return Err("No recording in progress".to_string());
        }
    }

    // Stop event tap
    {
        let mut event_tap = state.event_tap.lock().map_err(|e| e.to_string())?;
        if let Some(ref mut tap) = *event_tap {
            tap.stop();
        }
        *event_tap = None;
    }

    // Drain remaining events
    {
        let mut consumer = state.consumer.lock().map_err(|e| e.to_string())?;
        let mut recording = state.recording.lock().map_err(|e| e.to_string())?;

        if let (Some(ref mut cons), Some(ref mut rec)) = (consumer.as_mut(), recording.as_mut()) {
            let batch = cons.pop_batch(STOP_DRAIN_BATCH_SIZE);
            for slot in batch {
                rec.add_raw_event(slot.event);
            }
        }
        *consumer = None;
    }

    // Finalize and save recording
    let recording_info = {
        let mut recording = state.recording.lock().map_err(|e| e.to_string())?;

        if let Some(ref mut rec) = *recording {
            let duration = {
                let start = rec.metadata.started_at;
                let now = chrono::Utc::now();
                (now - start).num_milliseconds() as u64
            };
            rec.finalize(duration);

            // Save to file
            let recordings_dir = dirs::home_dir()
                .ok_or("Could not find home directory")?
                .join(".skill_generator")
                .join("recordings");
            std::fs::create_dir_all(&recordings_dir).map_err(|e| e.to_string())?;

            let file_path = recordings_dir.join(format!("{}.json", rec.metadata.name));
            rec.save(&file_path).map_err(|e| e.to_string())?;

            let info = RecordingInfo {
                name: rec.metadata.name.clone(),
                path: file_path.to_string_lossy().to_string(),
                event_count: rec.len(),
                duration_ms: rec.metadata.duration_ms,
                created_at: rec.metadata.started_at.to_rfc3339(),
                goal: rec.metadata.goal.clone(),
            };

            *recording = None;
            info
        } else {
            return Err("No recording to save".to_string());
        }
    };

    // Reset state
    {
        let mut is_rec = state.is_recording.lock().map_err(|e| e.to_string())?;
        *is_rec = false;
    }

    Ok(recording_info)
}

/// List all recordings
#[tauri::command]
fn list_recordings() -> Result<Vec<RecordingInfo>, String> {
    let recordings_dir = dirs::home_dir()
        .ok_or("Could not find home directory")?
        .join(".skill_generator")
        .join("recordings");

    if !recordings_dir.exists() {
        return Ok(vec![]);
    }

    let mut recordings = Vec::new();

    for entry in std::fs::read_dir(&recordings_dir).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();

        if path.extension().map(|e| e == "json").unwrap_or(false) {
            if let Ok(recording) = Recording::load(&path) {
                recordings.push(RecordingInfo {
                    name: recording.metadata.name.clone(),
                    path: path.to_string_lossy().to_string(),
                    event_count: recording.len(),
                    duration_ms: recording.metadata.duration_ms,
                    created_at: recording.metadata.started_at.to_rfc3339(),
                    goal: recording.metadata.goal.clone(),
                });
            }
        }
    }

    // Sort by creation time (newest first)
    recordings.sort_by(|a, b| b.created_at.cmp(&a.created_at));

    Ok(recordings)
}

/// Result from LLM skill inference
#[derive(Debug, Clone, Serialize, Deserialize)]
struct InferredSkillMeta {
    name: String,
    description: String,
    category: Option<String>,
}

/// Infer skill name and description by analyzing recording events
/// Extract the first JSON object from a text response.
/// Finds the outermost `{...}` and returns the substring.
fn extract_json_from_text(text: &str) -> Option<&str> {
    let start = text.find('{')?;
    let end = text.rfind('}')?;
    if end >= start {
        Some(&text[start..=end])
    } else {
        None
    }
}

/// Sanitize an AI-inferred skill name: keep only alphanumeric, hyphens, underscores.
/// Returns None if the result is empty or exceeds MAX_SKILL_NAME_LEN.
fn sanitize_skill_name(name: &str) -> Option<String> {
    let sanitized: String = name
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == '-' || *c == '_')
        .collect();
    if sanitized.is_empty() || sanitized.len() > MAX_SKILL_NAME_LEN {
        None
    } else {
        Some(sanitized.to_lowercase())
    }
}

/// Sanitize an AI-inferred category: keep only alphanumeric, hyphens, underscores, lowercase.
fn sanitize_category(category: Option<String>) -> Option<String> {
    category.map(|c| {
        c.chars()
            .filter(|ch| ch.is_alphanumeric() || *ch == '-' || *ch == '_')
            .collect::<String>()
            .to_lowercase()
    }).filter(|s| !s.is_empty())
}

fn infer_skill_from_recording(recording: &Recording, steps_summary: &str, model: &str, api_key: Option<&str>) -> Option<InferredSkillMeta> {
    let api_key = match api_key {
        Some(key) if !key.is_empty() => {
            let trimmed = key.trim().to_string();
            if trimmed.len() != key.len() {
                warn!(original_len = key.len(), trimmed_len = trimmed.len(), "API key had whitespace, trimmed");
            }
            trimmed
        },
        _ => {
            debug!("No API key provided, skipping AI skill inference");
            return None;
        }
    };

    info!("Starting AI skill inference");

    let user_goal = recording.metadata.goal.as_deref().unwrap_or("");

    let prompt = format!(
        r#"Analyze this UI automation recording and generate a skill name and description.

User's stated goal (may be empty): {}

Steps (first 5):
{}

Based on this analysis, infer what task the user was performing.

Respond in JSON format ONLY:
{{
  "name": "kebab-case-skill-name",
  "description": "One sentence describing what this skill does",
  "category": "kebab-case high-level category (e.g., email, github, files, browser)"
}}

Requirements for name:
- Use lowercase with hyphens (kebab-case)
- Keep it concise (2-4 words)
- Be specific about the action and target app if known
- Examples: "send-gmail-email", "create-github-issue", "search-chrome-tabs""#,
        user_goal,
        steps_summary
    );

    debug!(goal = user_goal, "Sending inference request");

    let rt = match tokio::runtime::Runtime::new() {
        Ok(rt) => rt,
        Err(e) => {
            error!(error = %e, "Failed to create tokio runtime");
            return None;
        }
    };

    rt.block_on(async {
        let client = match reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(API_TIMEOUT_SECS))
            .pool_max_idle_per_host(2)
            .build()
        {
            Ok(c) => c,
            Err(e) => {
                error!(error = %e, "Failed to build HTTP client");
                return None;
            }
        };

        let body = serde_json::json!({
            "model": model,
            "max_tokens": INFERENCE_MAX_TOKENS,
            "messages": [{"role": "user", "content": prompt}]
        });

        let response = match skill_generator::semantic::http_retry::send_with_retry(
            &client,
            |c| c.post("https://api.anthropic.com/v1/messages")
                .header("x-api-key", &api_key)
                .header("anthropic-version", "2023-06-01")
                .header("content-type", "application/json")
                .json(&body),
            3,
            "AI inference",
        ).await {
            Some(r) => r,
            None => return None,
        };

        #[derive(serde::Deserialize)]
        struct ApiResponse {
            content: Vec<ContentBlock>,
        }
        #[derive(serde::Deserialize)]
        struct ContentBlock {
            text: String,
        }
        #[derive(serde::Deserialize)]
        struct LlmResult {
            name: String,
            description: String,
            category: Option<String>,
        }

        let api_response: ApiResponse = match response.json().await {
            Ok(r) => r,
            Err(e) => {
                error!(error = %e, "Failed to parse AI inference response");
                return None;
            }
        };

        let text = match api_response.content.first() {
            Some(block) => block.text.clone(),
            None => {
                error!("Empty response from AI inference API");
                return None;
            }
        };

        debug!(response_len = text.len(), "Received AI inference response");

        let json_text = match extract_json_from_text(&text) {
            Some(j) => j,
            None => {
                error!("No JSON object found in AI response");
                return None;
            }
        };

        let result: LlmResult = match serde_json::from_str(json_text) {
            Ok(r) => r,
            Err(e) => {
                error!(error = %e, json = json_text, "Failed to parse AI inference JSON");
                return None;
            }
        };

        let sanitized_name = match sanitize_skill_name(&result.name) {
            Some(n) => n,
            None => {
                error!(name = result.name, "Invalid skill name after sanitization");
                return None;
            }
        };
        let sanitized_category = sanitize_category(result.category);

        info!(
            name = sanitized_name,
            category = ?sanitized_category,
            "AI inference complete"
        );

        Some(InferredSkillMeta {
            name: sanitized_name,
            description: result.description,
            category: sanitized_category,
        })
    })
}

/// Generate SKILL.md from a recording
#[tauri::command(rename_all = "camelCase")]
fn generate_skill(state: State<AppState>, recording_path: String) -> Result<GeneratedSkillInfo, String> {
    info!(path = recording_path, "Starting skill generation");

    // Get API key from state
    let api_key_value = state.api_key.lock().map_err(|e| e.to_string())?.clone();
    match &api_key_value {
        Some(key) if !key.is_empty() => {
            debug!(key_len = key.len(), "API key available");
        },
        _ => {
            warn!("No API key set, AI inference will be skipped");
        }
    }

    let path = std::path::Path::new(&recording_path);

    if !path.exists() {
        return Err(format!("Recording not found: {}", recording_path));
    }

    // Load recording
    let recording = Recording::load(path).map_err(|e| e.to_string())?;
    info!(event_count = recording.len(), "Recording loaded");

    // Load config for pipeline settings and model name
    let config = Config::load_default().unwrap_or_else(|e| {
        warn!("Failed to load config, using defaults: {}", e);
        Config::default()
    });

    // Create generator with API key and pipeline config from state
    let gen_config = skill_generator::workflow::generator::GeneratorConfig {
        api_key: api_key_value.clone(),
        use_action_clustering: config.pipeline.use_action_clustering,
        use_local_recovery: config.pipeline.use_local_recovery,
        use_vision_ocr: config.pipeline.use_vision_ocr,
        use_trajectory_analysis: config.pipeline.use_trajectory_analysis,
        use_goms_detection: config.pipeline.use_goms_detection,
        use_context_tracking: config.pipeline.use_context_tracking,
        ..Default::default()
    };
    let generator = SkillGenerator::with_config(gen_config);

    // Generate skill
    let mut skill = generator.generate(&recording).map_err(|e| e.to_string())?;
    info!(steps = skill.steps.len(), "Base skill generated");

    // Try to infer a better name, description, and category using AI
    let steps_summary: String = skill.steps.iter()
        .take(5)
        .map(|s| format!("- {}", s.description))
        .collect::<Vec<_>>()
        .join("\n");
    let inferred = infer_skill_from_recording(
        &recording,
        &steps_summary,
        &config.codegen.model,
        api_key_value.as_deref(),
    );

    if let Some(ref meta) = inferred {
        skill.name = meta.name.clone();
        if !meta.description.trim().is_empty() {
            skill.description = meta.description.clone();
        } else if let Some(goal) = recording.metadata.goal.clone() {
            skill.description = goal;
        }
    }

    // Save skill (category/name)
    let base_dir = dirs::home_dir()
        .ok_or("Could not find home directory")?
        .join(".claude")
        .join("skills");

    let category_dir = if let Some(ref meta) = inferred {
        meta.category.clone().unwrap_or_else(|| {
            skill.name.split('-').next().unwrap_or("misc").to_string()
        })
    } else {
        skill.name.split('-').next().unwrap_or("misc").to_string()
    };

    let skills_dir = base_dir.join(category_dir).join(&skill.name);
    std::fs::create_dir_all(&skills_dir).map_err(|e| e.to_string())?;

    let skill_path = skills_dir.join("SKILL.md");
    generator.save_skill(&skill, &skill_path).map_err(|e| e.to_string())?;

    Ok(GeneratedSkillInfo {
        name: skill.name.clone(),
        path: skill_path.to_string_lossy().to_string(),
        steps_count: skill.steps.len(),
        variables_count: skill.variables.len(),
        stats: PipelineStatsInfo {
            local_recovery_count: skill.stats.local_recovery_count,
            llm_enriched_count: skill.stats.llm_enriched_count,
            goms_boundaries_count: skill.stats.goms_boundaries_count,
            context_transitions_count: skill.stats.context_transitions_count,
            unit_tasks_count: skill.stats.unit_tasks_count,
            significant_events_count: skill.stats.significant_events_count,
            trajectory_adjustments_count: skill.stats.trajectory_adjustments_count,
            variables_count: skill.stats.variables_count,
            generated_steps_count: skill.stats.generated_steps_count,
            warnings: skill.stats.warnings.clone(),
        },
    })
}

/// Open a file or folder in the system file manager
#[tauri::command]
fn open_in_finder(path: String) -> Result<(), String> {
    std::process::Command::new("open")
        .arg("-R")
        .arg(&path)
        .spawn()
        .map_err(|e| e.to_string())?;
    Ok(())
}

/// Delete a recording
///
/// Accepts either:
/// - `name`: recording name (preferred)
/// - `recordingPath`: full path to the recording file (legacy frontend)
#[tauri::command(rename_all = "camelCase")]
fn delete_recording(name: Option<String>, recording_path: Option<String>) -> Result<(), String> {
    let recordings_dir = dirs::home_dir()
        .ok_or("Could not find home directory")?
        .join(".skill_generator")
        .join("recordings");

    // Support both the new API (name) and old API (recordingPath)
    let target = name
        .or(recording_path)
        .ok_or_else(|| "Missing parameter: name (or recordingPath)".to_string())?;

    // Resolve the path to delete
    let resolved_path = if target.contains('/') || target.contains('\\') || target.ends_with(".json") {
        // Full path provided - validate it's within the recordings directory
        let path = std::path::Path::new(&target);
        if !path.exists() {
            return Err(format!("Recording not found: {}", target));
        }
        let canonical = std::fs::canonicalize(path).map_err(|e| e.to_string())?;
        let canonical_dir = std::fs::canonicalize(&recordings_dir).map_err(|e| e.to_string())?;
        if !canonical.starts_with(&canonical_dir) {
            return Err("Access denied: path is outside the recordings directory".to_string());
        }
        canonical
    } else {
        // Treat as a name - look up in recordings dir
        let file_path = recordings_dir.join(format!("{}.json", target));
        if file_path.exists() {
            file_path
        } else {
            let dir_path = recordings_dir.join(&target);
            if dir_path.exists() && dir_path.is_dir() {
                dir_path
            } else {
                return Err(format!("Recording not found: {}", target));
            }
        }
    };

    if resolved_path.is_dir() {
        std::fs::remove_dir_all(&resolved_path).map_err(|e| e.to_string())?;
    } else {
        std::fs::remove_file(&resolved_path).map_err(|e| e.to_string())?;
    }

    Ok(())
}

/// Serializable config for the frontend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UiConfig {
    pub rdp_epsilon_px: f64,
    pub hesitation_threshold: f64,
    pub min_pause_ms: u64,
    pub model: String,
    pub temperature: f32,
    pub use_action_clustering: bool,
    pub use_local_recovery: bool,
    pub use_vision_ocr: bool,
    pub use_trajectory_analysis: bool,
    pub use_goms_detection: bool,
    pub use_context_tracking: bool,
}

/// Get current generator configuration
#[tauri::command]
fn get_config() -> Result<UiConfig, String> {
    let config = Config::load_default().map_err(|e| e.to_string())?;
    Ok(UiConfig {
        rdp_epsilon_px: config.analysis.rdp_epsilon_px,
        hesitation_threshold: config.analysis.hesitation_threshold,
        min_pause_ms: config.analysis.min_pause_ms,
        model: config.codegen.model,
        temperature: config.codegen.temperature,
        use_action_clustering: config.pipeline.use_action_clustering,
        use_local_recovery: config.pipeline.use_local_recovery,
        use_vision_ocr: config.pipeline.use_vision_ocr,
        use_trajectory_analysis: config.pipeline.use_trajectory_analysis,
        use_goms_detection: config.pipeline.use_goms_detection,
        use_context_tracking: config.pipeline.use_context_tracking,
    })
}

/// Save generator configuration
#[tauri::command]
fn save_config(config: UiConfig) -> Result<(), String> {
    let mut full_config = Config::load_default().map_err(|e| e.to_string())?;
    full_config.analysis.rdp_epsilon_px = config.rdp_epsilon_px;
    full_config.analysis.hesitation_threshold = config.hesitation_threshold;
    full_config.analysis.min_pause_ms = config.min_pause_ms;
    full_config.codegen.model = config.model;
    full_config.codegen.temperature = config.temperature;
    full_config.pipeline.use_action_clustering = config.use_action_clustering;
    full_config.pipeline.use_local_recovery = config.use_local_recovery;
    full_config.pipeline.use_vision_ocr = config.use_vision_ocr;
    full_config.pipeline.use_trajectory_analysis = config.use_trajectory_analysis;
    full_config.pipeline.use_goms_detection = config.use_goms_detection;
    full_config.pipeline.use_context_tracking = config.use_context_tracking;
    full_config.save_default().map_err(|e| e.to_string())?;
    info!("Configuration saved");
    Ok(())
}

/// Read a generated SKILL.md file for preview
#[tauri::command]
fn read_skill_file(path: String) -> Result<String, String> {
    std::fs::read_to_string(&path).map_err(|e| format!("Failed to read {}: {}", path, e))
}

/// Export the current config as a TOML string
#[tauri::command]
fn export_config() -> Result<String, String> {
    let config = Config::load_default().map_err(|e| e.to_string())?;
    config.to_toml().map_err(|e| e.to_string())
}

/// Import config from a TOML string, validate, and save
#[tauri::command(rename_all = "camelCase")]
fn import_config(toml_content: String) -> Result<(), String> {
    // Validate by parsing
    let config: Config = toml::from_str(&toml_content)
        .map_err(|e| format!("Invalid TOML config: {}", e))?;
    config.save_default().map_err(|e| e.to_string())?;
    info!("Config imported and saved");
    Ok(())
}

/// Read a recording JSON file for export
#[tauri::command(rename_all = "camelCase")]
fn read_recording_file(recording_path: String) -> Result<String, String> {
    let recordings_dir = dirs::home_dir()
        .ok_or("Could not find home directory")?
        .join(".skill_generator")
        .join("recordings");

    let path = std::path::Path::new(&recording_path);
    if !path.exists() {
        return Err(format!("Recording not found: {}", recording_path));
    }

    // Validate path is within recordings directory
    let canonical = std::fs::canonicalize(path).map_err(|e| e.to_string())?;
    let canonical_dir = std::fs::canonicalize(&recordings_dir).map_err(|e| e.to_string())?;
    if !canonical.starts_with(&canonical_dir) {
        return Err("Access denied: path is outside the recordings directory".to_string());
    }

    std::fs::read_to_string(path).map_err(|e| format!("Failed to read recording: {}", e))
}

/// Info about a generated skill for the frontend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillListEntry {
    pub name: String,
    pub category: String,
    pub path: String,
    pub size_bytes: u64,
    pub modified_at: String,
}

/// List all generated skills from ~/.claude/skills/
#[tauri::command]
fn list_generated_skills() -> Result<Vec<SkillListEntry>, String> {
    let skills_dir = dirs::home_dir()
        .ok_or("Could not find home directory")?
        .join(".claude")
        .join("skills");

    if !skills_dir.exists() {
        return Ok(vec![]);
    }

    let mut skills = Vec::new();

    // Walk category directories
    for category_entry in std::fs::read_dir(&skills_dir).map_err(|e| e.to_string())? {
        let category_entry = category_entry.map_err(|e| e.to_string())?;
        let category_path = category_entry.path();

        if !category_path.is_dir() {
            continue;
        }

        let category = category_path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();

        // Walk skill directories within category
        for skill_entry in std::fs::read_dir(&category_path).map_err(|e| e.to_string())? {
            let skill_entry = skill_entry.map_err(|e| e.to_string())?;
            let skill_path = skill_entry.path();

            if !skill_path.is_dir() {
                continue;
            }

            let skill_file = skill_path.join("SKILL.md");
            if !skill_file.exists() {
                continue;
            }

            let name = skill_path
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();

            let metadata = std::fs::metadata(&skill_file).map_err(|e| e.to_string())?;
            let modified = metadata
                .modified()
                .map(|t| {
                    let datetime: chrono::DateTime<chrono::Utc> = t.into();
                    datetime.to_rfc3339()
                })
                .unwrap_or_default();

            skills.push(SkillListEntry {
                name,
                category: category.clone(),
                path: skill_file.to_string_lossy().to_string(),
                size_bytes: metadata.len(),
                modified_at: modified,
            });
        }
    }

    // Sort by modification time (newest first)
    skills.sort_by(|a, b| b.modified_at.cmp(&a.modified_at));

    Ok(skills)
}

/// Delete a generated skill
#[tauri::command(rename_all = "camelCase")]
fn delete_skill(skill_path: String) -> Result<(), String> {
    let path = std::path::Path::new(&skill_path);

    // Validate path is within ~/.claude/skills/
    let skills_dir = dirs::home_dir()
        .ok_or("Could not find home directory")?
        .join(".claude")
        .join("skills");

    if !path.exists() {
        return Err(format!("Skill not found: {}", skill_path));
    }

    let canonical = std::fs::canonicalize(path).map_err(|e| e.to_string())?;
    let canonical_dir = std::fs::canonicalize(&skills_dir).map_err(|e| e.to_string())?;
    if !canonical.starts_with(&canonical_dir) {
        return Err("Access denied: path is outside the skills directory".to_string());
    }

    // Delete the skill directory (parent of SKILL.md)
    if let Some(parent) = path.parent() {
        if parent.starts_with(&skills_dir) && parent != skills_dir {
            std::fs::remove_dir_all(parent).map_err(|e| e.to_string())?;
        } else {
            std::fs::remove_file(path).map_err(|e| e.to_string())?;
        }
    }

    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // Initialize structured logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"))
        )
        .init();

    // Initialize timebase early
    MachTimebase::init();

    // Initialize state, reading API key from environment if available
    let state = AppState::default();
    if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
        if !key.is_empty() {
            match state.api_key.lock() {
                Ok(mut api_key) => *api_key = Some(key),
                Err(e) => warn!("Failed to set API key from environment: {}", e),
            }
        }
    }

    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_shell::init())
        .manage(state)
        .invoke_handler(tauri::generate_handler![
            check_permissions,
            set_api_key,
            get_api_key,
            start_recording,
            stop_recording,
            get_recording_status,
            list_recordings,
            generate_skill,
            open_in_finder,
            delete_recording,
            get_config,
            save_config,
            read_skill_file,
            list_generated_skills,
            delete_skill,
            export_config,
            import_config,
            read_recording_file,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- sanitize_filename tests ---

    #[test]
    fn test_sanitize_filename_valid_name() {
        assert_eq!(sanitize_filename("my-recording").unwrap(), "my-recording");
    }

    #[test]
    fn test_sanitize_filename_with_spaces() {
        assert_eq!(sanitize_filename("my recording").unwrap(), "my recording");
    }

    #[test]
    fn test_sanitize_filename_blocks_path_traversal() {
        assert!(sanitize_filename("../../../etc/passwd").is_err());
        assert!(sanitize_filename("foo/../bar").is_err());
        assert!(sanitize_filename("..").is_err());
    }

    #[test]
    fn test_sanitize_filename_strips_slashes() {
        assert_eq!(sanitize_filename("a/b").unwrap(), "ab");
        assert_eq!(sanitize_filename("a\\b").unwrap(), "ab");
    }

    #[test]
    fn test_sanitize_filename_strips_control_chars() {
        assert_eq!(sanitize_filename("abc\x00def").unwrap(), "abcdef");
        assert_eq!(sanitize_filename("abc\ndef").unwrap(), "abcdef");
    }

    #[test]
    fn test_sanitize_filename_rejects_empty() {
        assert!(sanitize_filename("").is_err());
    }

    #[test]
    fn test_sanitize_filename_rejects_only_slashes() {
        assert!(sanitize_filename("///").is_err());
    }

    #[test]
    fn test_sanitize_filename_rejects_too_long() {
        let long_name = "a".repeat(256);
        assert!(sanitize_filename(&long_name).is_err());
    }

    #[test]
    fn test_sanitize_filename_accepts_max_length() {
        let name = "a".repeat(255);
        assert_eq!(sanitize_filename(&name).unwrap().len(), 255);
    }

    #[test]
    fn test_sanitize_filename_unicode() {
        assert_eq!(sanitize_filename("日本語テスト").unwrap(), "日本語テスト");
    }

    // --- Struct serialization tests ---

    #[test]
    fn test_recording_status_serialization() {
        let status = RecordingStatus {
            is_recording: true,
            event_count: 42,
            duration_seconds: 3.14,
            has_accessibility: true,
            has_screen_recording: false,
        };
        let json = serde_json::to_string(&status).unwrap();
        let deserialized: RecordingStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.event_count, 42);
        assert!(deserialized.is_recording);
        assert!(!deserialized.has_screen_recording);
    }

    #[test]
    fn test_pipeline_stats_info_serialization() {
        let stats = PipelineStatsInfo {
            local_recovery_count: 1,
            llm_enriched_count: 2,
            goms_boundaries_count: 3,
            context_transitions_count: 4,
            unit_tasks_count: 5,
            significant_events_count: 6,
            trajectory_adjustments_count: 7,
            variables_count: 8,
            generated_steps_count: 9,
            warnings: vec!["test warning".to_string()],
        };
        let json = serde_json::to_string(&stats).unwrap();
        let deserialized: PipelineStatsInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.warnings.len(), 1);
        assert_eq!(deserialized.generated_steps_count, 9);
    }

    #[test]
    fn test_generated_skill_info_serialization() {
        let info = GeneratedSkillInfo {
            name: "test-skill".to_string(),
            path: "/tmp/test.md".to_string(),
            steps_count: 5,
            variables_count: 2,
            stats: PipelineStatsInfo {
                local_recovery_count: 0,
                llm_enriched_count: 0,
                goms_boundaries_count: 0,
                context_transitions_count: 0,
                unit_tasks_count: 0,
                significant_events_count: 10,
                trajectory_adjustments_count: 0,
                variables_count: 2,
                generated_steps_count: 5,
                warnings: vec![],
            },
        };
        let json = serde_json::to_string(&info).unwrap();
        let deserialized: GeneratedSkillInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.name, "test-skill");
        assert_eq!(deserialized.stats.significant_events_count, 10);
        assert!(deserialized.stats.warnings.is_empty());
    }

    #[test]
    fn test_recording_info_serialization() {
        let info = RecordingInfo {
            name: "test".to_string(),
            path: "/tmp/test.json".to_string(),
            event_count: 100,
            duration_ms: 5000,
            created_at: "2026-02-08T00:00:00Z".to_string(),
            goal: Some("Test goal".to_string()),
        };
        let json = serde_json::to_string(&info).unwrap();
        let deserialized: RecordingInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.event_count, 100);
        assert_eq!(deserialized.goal, Some("Test goal".to_string()));
    }

    #[test]
    fn test_app_state_default() {
        let state = AppState::default();
        assert!(state.recording.lock().unwrap().is_none());
        assert!(state.event_tap.lock().unwrap().is_none());
        assert!(state.consumer.lock().unwrap().is_none());
        assert!(!*state.is_recording.lock().unwrap());
        assert!(state.api_key.lock().unwrap().is_none());
    }

    #[test]
    fn test_ui_config_serialization_roundtrip() {
        let config = UiConfig {
            rdp_epsilon_px: 2.0,
            hesitation_threshold: 150.0,
            min_pause_ms: 300,
            model: "claude-sonnet-4-5-20250929".to_string(),
            temperature: 0.7,
            use_action_clustering: true,
            use_local_recovery: true,
            use_vision_ocr: false,
            use_trajectory_analysis: true,
            use_goms_detection: true,
            use_context_tracking: true,
        };
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: UiConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.model, "claude-sonnet-4-5-20250929");
        assert!(!deserialized.use_vision_ocr);
        assert!((deserialized.temperature - 0.7).abs() < f32::EPSILON);
    }

    // --- extract_json_from_text tests ---

    #[test]
    fn test_extract_json_simple() {
        let text = r#"Here is the result: {"name": "test"} done"#;
        assert_eq!(extract_json_from_text(text), Some(r#"{"name": "test"}"#));
    }

    #[test]
    fn test_extract_json_nested() {
        let text = r#"{"outer": {"inner": 1}}"#;
        assert_eq!(extract_json_from_text(text), Some(r#"{"outer": {"inner": 1}}"#));
    }

    #[test]
    fn test_extract_json_no_json() {
        assert_eq!(extract_json_from_text("no json here"), None);
    }

    #[test]
    fn test_extract_json_only_open_brace() {
        assert_eq!(extract_json_from_text("{ incomplete"), None);
    }

    // --- sanitize_skill_name tests ---

    #[test]
    fn test_sanitize_skill_name_valid() {
        assert_eq!(sanitize_skill_name("send-gmail-email"), Some("send-gmail-email".to_string()));
    }

    #[test]
    fn test_sanitize_skill_name_strips_special_chars() {
        assert_eq!(sanitize_skill_name("my skill! #1"), Some("myskill1".to_string()));
    }

    #[test]
    fn test_sanitize_skill_name_lowercases() {
        assert_eq!(sanitize_skill_name("Send-Gmail"), Some("send-gmail".to_string()));
    }

    #[test]
    fn test_sanitize_skill_name_empty_result() {
        assert_eq!(sanitize_skill_name("!@#$"), None);
    }

    #[test]
    fn test_sanitize_skill_name_too_long() {
        let long = "a".repeat(51);
        assert_eq!(sanitize_skill_name(&long), None);
    }

    // --- sanitize_category tests ---

    #[test]
    fn test_sanitize_category_valid() {
        assert_eq!(sanitize_category(Some("email".to_string())), Some("email".to_string()));
    }

    #[test]
    fn test_sanitize_category_strips_special() {
        assert_eq!(sanitize_category(Some("web & apps".to_string())), Some("webapps".to_string()));
    }

    #[test]
    fn test_sanitize_category_none() {
        assert_eq!(sanitize_category(None), None);
    }

    #[test]
    fn test_sanitize_category_empty_after_filter() {
        assert_eq!(sanitize_category(Some("!@#".to_string())), None);
    }
}
