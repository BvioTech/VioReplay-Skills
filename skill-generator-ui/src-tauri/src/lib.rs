// Skill Generator UI - Tauri Backend
// Provides commands for recording user interactions and generating SKILL.md files

use serde::{Deserialize, Serialize};
use skill_generator::app::config::Config;
use skill_generator::capture::event_tap::{check_accessibility_permissions, check_screen_recording_permission, was_hotkey_stopped};
use skill_generator::capture::ring_buffer::{EventRingBuffer, DEFAULT_CAPACITY};
use skill_generator::time::timebase::MachTimebase;
use skill_generator::workflow::generator::SkillGenerator;
use skill_generator::workflow::recording::Recording;
use skill_generator::capture::EventTap;
use skill_generator::capture::types::EventType;
use skill_generator::semantic::screenshot::{ScreenshotConfig, capture_full_screen_jpeg};
use skill_generator::semantic::screenshot_analysis::{
    self as sa, AnalysisConfig, AnalysisTier, RecordingAnalysis,
};
use std::sync::Mutex;
use std::sync::mpsc;
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

/// Minimum interval between screenshot captures (ms)
const SCREENSHOT_MIN_INTERVAL_MS: u128 = 100;

/// Screenshot capture request sent to background thread
enum ScreenshotRequest {
    /// Capture a screenshot with the given sequence number
    Capture { sequence: u32 },
    /// Stop the capture thread
    Stop,
}

/// Background screenshot capturer
struct ScreenshotCapturer {
    sender: mpsc::SyncSender<ScreenshotRequest>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl ScreenshotCapturer {
    /// Create and start a screenshot capture background thread.
    fn start(screenshots_dir: std::path::PathBuf, config: ScreenshotConfig) -> Self {
        let (sender, receiver) = mpsc::sync_channel::<ScreenshotRequest>(100);

        let handle = std::thread::spawn(move || {
            debug!("Screenshot capturer thread started");
            while let Ok(request) = receiver.recv() {
                match request {
                    ScreenshotRequest::Capture { sequence } => {
                        let filename = format!("{:04}.jpg", sequence);
                        let filepath = screenshots_dir.join(&filename);
                        if let Some(screenshot) = capture_full_screen_jpeg(&config) {
                            if let Err(e) = std::fs::write(&filepath, &screenshot.jpeg_data) {
                                warn!(error = %e, filename = %filename, "Failed to write screenshot");
                            }
                        }
                    }
                    ScreenshotRequest::Stop => {
                        debug!("Screenshot capturer thread stopping");
                        break;
                    }
                }
            }
        });

        Self {
            sender,
            handle: Some(handle),
        }
    }

    /// Send a capture request (non-blocking via try_send to avoid stalling the event loop).
    fn capture(&self, sequence: u32) {
        if let Err(e) = self.sender.try_send(ScreenshotRequest::Capture { sequence }) {
            warn!(sequence = sequence, error = %e, "Screenshot capture channel full or disconnected");
        }
    }

    /// Stop the capture thread and wait for it to finish.
    fn stop(self) {
        let _ = self.sender.send(ScreenshotRequest::Stop);
        if let Some(handle) = self.handle {
            if let Err(e) = handle.join() {
                warn!("Screenshot capturer thread panicked: {:?}", e);
            }
        }
    }
}

/// Check if an event type is significant enough to warrant a screenshot.
fn is_significant_event(event_type: EventType) -> bool {
    matches!(
        event_type,
        EventType::LeftMouseDown
            | EventType::RightMouseDown
            | EventType::OtherMouseDown
            | EventType::KeyDown
            | EventType::ScrollWheel
    )
}

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
    /// Event count at last checkpoint (for auto-save)
    last_checkpoint_count: Mutex<usize>,
    /// Screenshot capturer (if active)
    screenshot_capturer: Mutex<Option<ScreenshotCapturer>>,
    /// Screenshot sequence counter
    screenshot_sequence: Mutex<u32>,
    /// Last screenshot timestamp for rate limiting
    last_screenshot_time: Mutex<std::time::Instant>,
    /// Whether screenshot capture is enabled
    capture_screenshots: Mutex<bool>,
    /// Lock for serializing analysis file read-modify-write operations
    analysis_lock: Mutex<()>,
}

impl Default for AppState {
    fn default() -> Self {
        // Load capture_screenshots from persisted config (defaults to true)
        let capture_ss = Config::load_default()
            .map(|c| c.capture.capture_screenshots)
            .unwrap_or(true);
        Self {
            recording: Mutex::new(None),
            event_tap: Mutex::new(None),
            consumer: Mutex::new(None),
            is_recording: Mutex::new(false),
            api_key: Mutex::new(None),
            last_checkpoint_count: Mutex::new(0),
            screenshot_capturer: Mutex::new(None),
            screenshot_sequence: Mutex::new(0),
            last_screenshot_time: Mutex::new(std::time::Instant::now()),
            capture_screenshots: Mutex::new(capture_ss),
            analysis_lock: Mutex::new(()),
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
    pub has_screenshots: bool,
    pub has_analysis: bool,
}

/// Recording status for frontend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordingStatus {
    pub is_recording: bool,
    pub event_count: usize,
    pub duration_seconds: f64,
    pub has_accessibility: bool,
    pub has_screen_recording: bool,
    /// True when recording was stopped by the global hotkey (Cmd+Opt+Ctrl+S)
    pub hotkey_stopped: bool,
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
    pub noise_filtered_count: usize,
    pub variables_count: usize,
    pub generated_steps_count: usize,
    pub screenshot_enhanced_count: usize,
    pub state_diff_enriched_count: usize,
    pub error_correction_count: usize,
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

/// Analysis info for a single screenshot (frontend)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScreenshotAnalysisInfo {
    pub sequence: u64,
    pub filename: String,
    pub intent: String,
    pub confidence: f32,
    pub tier: String,
    pub user_edited: bool,
    pub image_path: String,
}

/// Full analysis result for the frontend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisInfo {
    pub recording_name: String,
    pub analyses: Vec<ScreenshotAnalysisInfo>,
    pub total_screenshots: usize,
    pub ocr_count: usize,
    pub vision_count: usize,
    pub edited_count: usize,
}

/// Screenshot file info for the frontend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScreenshotInfo {
    pub sequence: u32,
    pub filename: String,
    pub image_path: String,
}

/// Convert internal RecordingAnalysis to frontend AnalysisInfo
fn to_analysis_info(analysis: &RecordingAnalysis, screenshots_dir: &std::path::Path) -> AnalysisInfo {
    let analyses: Vec<ScreenshotAnalysisInfo> = analysis.analyses.iter().map(|a| {
        let tier_str = match a.tier {
            AnalysisTier::LocalOcr => "ocr",
            AnalysisTier::ClaudeVision => "vision",
            AnalysisTier::UserEdited => "edited",
        };
        ScreenshotAnalysisInfo {
            sequence: a.sequence,
            filename: a.filename.clone(),
            intent: a.intent.clone(),
            confidence: a.confidence,
            tier: tier_str.to_string(),
            user_edited: a.user_edited,
            image_path: screenshots_dir.join(&a.filename).to_string_lossy().to_string(),
        }
    }).collect();

    let ocr_count = analyses.iter().filter(|a| a.tier == "ocr").count();
    let vision_count = analyses.iter().filter(|a| a.tier == "vision").count();
    let edited_count = analyses.iter().filter(|a| a.user_edited).count();
    let total = analyses.len();

    AnalysisInfo {
        recording_name: analysis.recording_name.clone(),
        analyses,
        total_screenshots: total,
        ocr_count,
        vision_count,
        edited_count,
    }
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
    let recording = Recording::new(recording_name.clone(), goal);

    // Start screenshot capturer if enabled
    let should_capture = *state.capture_screenshots.lock().map_err(|e| e.to_string())?;
    if should_capture {
        let recordings_dir = dirs::home_dir()
            .ok_or("Could not find home directory")?
            .join(".skill_generator")
            .join("recordings");
        let screenshots_dir = recordings_dir.join(&recording_name).join("screenshots");
        std::fs::create_dir_all(&screenshots_dir).map_err(|e| e.to_string())?;

        let config = ScreenshotConfig {
            jpeg_quality: 0.8,
            max_width: 1024,
        };
        let capturer = ScreenshotCapturer::start(screenshots_dir, config);

        let mut ss_cap = state.screenshot_capturer.lock().map_err(|e| e.to_string())?;
        *ss_cap = Some(capturer);
    }

    // Reset screenshot sequence
    {
        let mut seq = state.screenshot_sequence.lock().map_err(|e| e.to_string())?;
        *seq = 0;
    }
    {
        let mut last_ss = state.last_screenshot_time.lock().map_err(|e| e.to_string())?;
        *last_ss = std::time::Instant::now();
    }

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
    {
        let mut last_cp = state.last_checkpoint_count.lock().map_err(|e| e.to_string())?;
        *last_cp = 0;
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
            hotkey_stopped: false,
        });
    }

    // Read checkpoint threshold before acquiring consumer/recording locks
    let last_cp_val = *state.last_checkpoint_count.lock().map_err(|e| e.to_string())?;

    // Drain events from buffer (consumer + recording locks only)
    let mut event_count = 0;
    let mut did_checkpoint = false;
    {
        let mut consumer = state.consumer.lock().map_err(|e| e.to_string())?;
        let mut recording = state.recording.lock().map_err(|e| e.to_string())?;

        if let (Some(ref mut cons), Some(ref mut rec)) = (consumer.as_mut(), recording.as_mut()) {
            let batch = cons.pop_batch(STATUS_POLL_BATCH_SIZE);

            // Get screenshot capturer reference
            let ss_capturer = state.screenshot_capturer.lock().ok();
            let mut ss_seq = state.screenshot_sequence.lock().unwrap_or_else(|e| {
                warn!("Screenshot sequence mutex was poisoned, recovering");
                e.into_inner()
            });
            let mut last_ss_time = state.last_screenshot_time.lock().unwrap_or_else(|e| {
                warn!("Screenshot timestamp mutex was poisoned, recovering");
                e.into_inner()
            });

            for slot in batch {
                let event_type = slot.event.event_type;
                rec.add_raw_event(slot.event);

                // Capture screenshot for significant events with rate limiting
                if is_significant_event(event_type) {
                    let now = std::time::Instant::now();
                    if now.duration_since(*last_ss_time).as_millis() >= SCREENSHOT_MIN_INTERVAL_MS {
                        if let Some(ref capturer_lock) = ss_capturer {
                            if let Some(ref capturer) = **capturer_lock {
                                let seq = *ss_seq;
                                capturer.capture(seq);
                                // Set screenshot filename on the last added event
                                if let Some(last_event) = rec.events.last_mut() {
                                    last_event.screenshot_filename =
                                        Some(format!("{:04}.jpg", seq));
                                }
                                *ss_seq += 1;
                                *last_ss_time = now;
                            }
                        }
                    }
                }
            }
            event_count = rec.len();

            // Auto-save checkpoint if needed
            if event_count >= last_cp_val + skill_generator::workflow::recording::CHECKPOINT_INTERVAL {
                let recordings_dir = dirs::home_dir()
                    .unwrap_or_default()
                    .join(".skill_generator")
                    .join("recordings");
                let rec_dir = recordings_dir.join(&rec.metadata.name);
                let _ = std::fs::create_dir_all(&rec_dir);
                let cp_path = rec_dir.join("recording.json");
                let _ = rec.save_checkpoint(&cp_path);
                did_checkpoint = true;
            }
        }
    }

    // Update checkpoint count after releasing consumer/recording locks
    if did_checkpoint {
        let mut last_cp = state.last_checkpoint_count.lock().map_err(|e| e.to_string())?;
        *last_cp = event_count;
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
        hotkey_stopped: was_hotkey_stopped(),
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

    // Stop screenshot capturer
    let had_screenshots = {
        let mut ss_cap = state.screenshot_capturer.lock().map_err(|e| e.to_string())?;
        let had = ss_cap.is_some();
        if let Some(capturer) = ss_cap.take() {
            capturer.stop();
        }
        had
    };

    // Drain remaining events
    {
        let mut consumer = state.consumer.lock().map_err(|e| e.to_string())?;
        let mut recording = state.recording.lock().map_err(|e| e.to_string())?;

        if let (Some(ref mut cons), Some(ref mut rec)) = (consumer.as_mut(), recording.as_mut()) {
            let batch = cons.pop_batch(STOP_DRAIN_BATCH_SIZE);
            for slot in batch {
                rec.add_raw_event(slot.event);
            }
            // Trim trailing FlagsChanged events caused by the stop hotkey modifier keys
            // (Cmd/Opt/Ctrl pressed before 'S'). These are recording pollution.
            while rec.events.last().is_some_and(|e| e.raw.event_type == EventType::FlagsChanged) {
                rec.events.pop();
            }
            // Mark recording as having screenshots
            if had_screenshots {
                rec.metadata.has_screenshots = true;
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

            // Save to directory format: {name}/recording.json
            let recordings_dir = dirs::home_dir()
                .ok_or("Could not find home directory")?
                .join(".skill_generator")
                .join("recordings");
            std::fs::create_dir_all(&recordings_dir).map_err(|e| e.to_string())?;

            let rec_dir = rec.save_to_dir(&recordings_dir).map_err(|e| e.to_string())?;
            let file_path = rec_dir.join("recording.json");
            // Remove any old-format checkpoint
            Recording::remove_checkpoint(&recordings_dir.join(format!("{}.json", rec.metadata.name)));

            let info = RecordingInfo {
                name: rec.metadata.name.clone(),
                path: file_path.to_string_lossy().to_string(),
                event_count: rec.len(),
                duration_ms: rec.metadata.duration_ms,
                created_at: rec.metadata.started_at.to_rfc3339(),
                goal: rec.metadata.goal.clone(),
                has_screenshots: rec.metadata.has_screenshots,
                has_analysis: false,
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

        // Directory format: {name}/recording.json
        if path.is_dir() {
            let recording_json = path.join("recording.json");
            if recording_json.exists() {
                if let Ok(recording) = Recording::load(&recording_json) {
                    let has_analysis = path.join("analysis.json").exists();
                    recordings.push(RecordingInfo {
                        name: recording.metadata.name.clone(),
                        path: recording_json.to_string_lossy().to_string(),
                        event_count: recording.len(),
                        duration_ms: recording.metadata.duration_ms,
                        created_at: recording.metadata.started_at.to_rfc3339(),
                        goal: recording.metadata.goal.clone(),
                        has_screenshots: recording.metadata.has_screenshots,
                        has_analysis,
                    });
                }
            }
            continue;
        }

        // Flat format: {name}.json (backward compat)
        if path.extension().map(|e| e == "json").unwrap_or(false) {
            if let Ok(recording) = Recording::load(&path) {
                recordings.push(RecordingInfo {
                    name: recording.metadata.name.clone(),
                    path: path.to_string_lossy().to_string(),
                    event_count: recording.len(),
                    duration_ms: recording.metadata.duration_ms,
                    created_at: recording.metadata.started_at.to_rfc3339(),
                    goal: recording.metadata.goal.clone(),
                    has_screenshots: false,
                    has_analysis: false,
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

/// Generate SKILL.md from a recording.
///
/// Runs on a background thread via `spawn_blocking` so the UI stays responsive.
#[tauri::command(rename_all = "camelCase")]
async fn generate_skill(state: State<'_, AppState>, recording_path: String) -> Result<GeneratedSkillInfo, String> {
    info!(path = recording_path, "Starting skill generation");

    // Extract state before the async boundary
    let api_key_value = state.api_key.lock().map_err(|e| e.to_string())?.clone();
    match &api_key_value {
        Some(key) if !key.is_empty() => {
            debug!(key_len = key.len(), "API key available");
        },
        _ => {
            warn!("No API key set, AI inference will be skipped");
        }
    }

    tokio::task::spawn_blocking(move || {
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

        // Check if screenshot analysis exists for this recording
        let screenshot_analysis = if let Some(parent) = path.parent() {
            match sa::load_analysis(parent) {
                Ok(analysis) => {
                    let intents: Vec<(u64, String)> = analysis.analyses.iter()
                        .filter(|a| !a.intent.trim().is_empty())
                        .map(|a| (a.sequence, a.intent.clone()))
                        .collect();
                    if intents.is_empty() {
                        debug!("Screenshot analysis loaded but no usable intents found");
                        None
                    } else {
                        info!(count = intents.len(), "Loaded screenshot analysis for generation");
                        Some(intents)
                    }
                }
                Err(e) => {
                    // Not an error if analysis.json simply doesn't exist
                    if e.kind() != std::io::ErrorKind::NotFound {
                        warn!(error = %e, "Failed to load screenshot analysis");
                    }
                    None
                }
            }
        } else {
            None
        };

        // Create generator with API key and pipeline config from state
        let gen_config = skill_generator::workflow::generator::GeneratorConfig {
            api_key: api_key_value.clone(),
            use_action_clustering: config.pipeline.use_action_clustering,
            use_local_recovery: config.pipeline.use_local_recovery,
            use_vision_ocr: config.pipeline.use_vision_ocr,
            use_trajectory_analysis: config.pipeline.use_trajectory_analysis,
            use_goms_detection: config.pipeline.use_goms_detection,
            use_context_tracking: config.pipeline.use_context_tracking,
            use_state_diff: config.pipeline.use_state_diff,
            use_error_correction_filter: config.pipeline.use_error_correction_filter,
            screenshot_analysis,
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
            .ok_or("Could not find home directory".to_string())?
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
                noise_filtered_count: skill.stats.noise_filtered_count,
                variables_count: skill.stats.variables_count,
                generated_steps_count: skill.stats.generated_steps_count,
                screenshot_enhanced_count: skill.stats.screenshot_enhanced_count,
                state_diff_enriched_count: skill.stats.state_diff_enriched_count,
                error_correction_count: skill.stats.error_correction_count,
                warnings: skill.stats.warnings.clone(),
            },
        })
    }).await.map_err(|e| format!("Generation task failed: {}", e))?
}

/// Run tiered AI analysis on a recording's screenshots.
///
/// Runs on a background thread via `spawn_blocking` so the UI stays responsive.
#[tauri::command(rename_all = "camelCase")]
async fn analyze_recording(state: State<'_, AppState>, recording_name: String) -> Result<AnalysisInfo, String> {
    let name = sanitize_filename(&recording_name)?;
    // Extract state before the async boundary
    let api_key_value = state.api_key.lock().map_err(|e| e.to_string())?.clone();

    info!(name = %name, "Starting screenshot analysis");

    tokio::task::spawn_blocking(move || {
        let recordings_dir = dirs::home_dir()
            .ok_or("Could not find home directory".to_string())?
            .join(".skill_generator")
            .join("recordings");

        let rec_dir = Recording::recording_dir(&recordings_dir, &name);
        let recording_json = rec_dir.join("recording.json");

        if !recording_json.exists() {
            return Err(format!("Recording not found: {}", name));
        }

        let screenshots_dir = Recording::screenshots_dir(&rec_dir);

        if !screenshots_dir.exists() {
            return Err("No screenshots found for this recording".to_string());
        }

        let recording = Recording::load(&recording_json).map_err(|e| e.to_string())?;

        let app_config = Config::load_default().unwrap_or_else(|e| {
            warn!("Failed to load config for analysis, using defaults: {}", e);
            Config::default()
        });
        let config = AnalysisConfig {
            claude_model: app_config.codegen.vision_model.clone(),
            ..AnalysisConfig::default()
        };

        let annotation_config = skill_generator::semantic::screenshot::AnnotationConfig {
            dot_radius: app_config.annotation.dot_radius as i32,
            dot_color: app_config.annotation.dot_color,
            box_color: [0, 255, 100],
            box_thickness: 2,
            crop_size: app_config.annotation.crop_size,
            jpeg_quality: 85,
            trajectory_ballistic_color: app_config.annotation.trajectory_ballistic_color,
            trajectory_searching_color: app_config.annotation.trajectory_searching_color,
            trajectory_thickness: app_config.annotation.trajectory_thickness,
        };

        let analysis = sa::analyze_recording(
            &rec_dir,
            &recording,
            api_key_value.as_deref(),
            &config,
            &annotation_config,
        );

        sa::save_analysis(&analysis, &rec_dir).map_err(|e| e.to_string())?;

        info!(
            name = %name,
            count = analysis.analyses.len(),
            "Analysis complete and saved"
        );

        Ok(to_analysis_info(&analysis, &screenshots_dir))
    }).await.map_err(|e| format!("Analysis task failed: {}", e))?
}

/// Get existing analysis for a recording (if available)
#[tauri::command(rename_all = "camelCase")]
fn get_analysis(recording_name: String) -> Result<Option<AnalysisInfo>, String> {
    let name = sanitize_filename(&recording_name)?;
    let recordings_dir = dirs::home_dir()
        .ok_or("Could not find home directory")?
        .join(".skill_generator")
        .join("recordings");

    let rec_dir = Recording::recording_dir(&recordings_dir, &name);
    let screenshots_dir = Recording::screenshots_dir(&rec_dir);

    match sa::load_analysis(&rec_dir) {
        Ok(analysis) => Ok(Some(to_analysis_info(&analysis, &screenshots_dir))),
        Err(_) => Ok(None),
    }
}

/// Update a single intent in the analysis
#[tauri::command(rename_all = "camelCase")]
fn update_intent(state: State<AppState>, recording_name: String, sequence: u64, new_intent: String) -> Result<(), String> {
    let name = sanitize_filename(&recording_name)?;
    let recordings_dir = dirs::home_dir()
        .ok_or("Could not find home directory")?
        .join(".skill_generator")
        .join("recordings");

    let rec_dir = Recording::recording_dir(&recordings_dir, &name);

    // Serialize analysis file access to prevent read-modify-write races
    let _lock = state.analysis_lock.lock().map_err(|e| e.to_string())?;

    let mut analysis = sa::load_analysis(&rec_dir).map_err(|e| e.to_string())?;

    let entry = analysis.analyses.iter_mut()
        .find(|a| a.sequence == sequence)
        .ok_or_else(|| format!("No analysis found for sequence {}", sequence))?;

    entry.intent = new_intent;
    entry.user_edited = true;
    entry.tier = AnalysisTier::UserEdited;

    sa::save_analysis(&analysis, &rec_dir).map_err(|e| e.to_string())?;

    Ok(())
}

/// List screenshot files for a recording
#[tauri::command(rename_all = "camelCase")]
fn list_screenshots(recording_name: String) -> Result<Vec<ScreenshotInfo>, String> {
    let name = sanitize_filename(&recording_name)?;
    let recordings_dir = dirs::home_dir()
        .ok_or("Could not find home directory")?
        .join(".skill_generator")
        .join("recordings");

    let rec_dir = Recording::recording_dir(&recordings_dir, &name);
    let screenshots_dir = Recording::screenshots_dir(&rec_dir);

    if !screenshots_dir.exists() {
        return Ok(vec![]);
    }

    let mut screenshots = Vec::new();

    for entry in std::fs::read_dir(&screenshots_dir).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();

        if path.extension().map(|e| e == "jpg").unwrap_or(false) {
            let filename = path.file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();

            // Parse sequence from filename (e.g., "0005.jpg" -> 5)
            let sequence = filename.trim_end_matches(".jpg")
                .parse::<u32>()
                .unwrap_or(0);

            screenshots.push(ScreenshotInfo {
                sequence,
                filename,
                image_path: path.to_string_lossy().to_string(),
            });
        }
    }

    screenshots.sort_by_key(|s| s.sequence);

    Ok(screenshots)
}

/// Open a file or folder in the system file manager.
///
/// Only allows paths within the recordings or skills directories.
#[tauri::command]
fn open_in_finder(path: String) -> Result<(), String> {
    let p = std::path::Path::new(&path);
    if !p.exists() {
        return Err(format!("Path not found: {}", path));
    }
    if p.is_symlink() {
        return Err("Cannot open symlinks".to_string());
    }

    // Validate path is within allowed directories
    let home = dirs::home_dir().ok_or("Could not find home directory")?;
    let recordings_dir = home.join(".skill_generator").join("recordings");
    let skills_dir = home.join(".claude").join("skills");

    let canonical = std::fs::canonicalize(p).map_err(|e| e.to_string())?;
    let in_recordings = recordings_dir.exists()
        && std::fs::canonicalize(&recordings_dir)
            .map(|d| canonical.starts_with(&d))
            .unwrap_or(false);
    let in_skills = skills_dir.exists()
        && std::fs::canonicalize(&skills_dir)
            .map(|d| canonical.starts_with(&d))
            .unwrap_or(false);

    if !in_recordings && !in_skills {
        return Err("Access denied: path is outside allowed directories".to_string());
    }

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
        // Reject symlinks to prevent path traversal attacks
        if path.is_symlink() {
            return Err("Cannot delete symlinks".to_string());
        }
        let canonical = std::fs::canonicalize(path).map_err(|e| e.to_string())?;
        let canonical_dir = std::fs::canonicalize(&recordings_dir).map_err(|e| e.to_string())?;
        if !canonical.starts_with(&canonical_dir) {
            return Err("Access denied: path is outside the recordings directory".to_string());
        }
        canonical
    } else {
        // Treat as a name - look up in recordings dir (flat file first, then directory)
        let file_path = recordings_dir.join(format!("{}.json", target));
        if file_path.exists() && !file_path.is_symlink() {
            file_path
        } else {
            let dir_path = recordings_dir.join(&target);
            if dir_path.is_symlink() {
                return Err("Cannot delete symlinks".to_string());
            }
            if dir_path.exists() && dir_path.is_dir() {
                // Validate it's actually a recording directory
                if !dir_path.join("recording.json").exists() {
                    return Err(format!("Not a valid recording directory: {}", target));
                }
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
    #[serde(default = "default_vision_model")]
    pub vision_model: String,
    pub temperature: f32,
    pub use_action_clustering: bool,
    pub use_local_recovery: bool,
    pub use_vision_ocr: bool,
    pub use_trajectory_analysis: bool,
    pub use_goms_detection: bool,
    pub use_context_tracking: bool,
    #[serde(default = "default_true")]
    pub use_state_diff: bool,
    #[serde(default = "default_true")]
    pub use_error_correction_filter: bool,
    #[serde(default = "default_capture_screenshots")]
    pub capture_screenshots: bool,
    #[serde(default = "default_dot_radius")]
    pub dot_radius: u32,
    #[serde(default = "default_dot_color")]
    pub dot_color: [u8; 3],
    #[serde(default = "default_trajectory_ballistic_color")]
    pub trajectory_ballistic_color: [u8; 3],
    #[serde(default = "default_trajectory_searching_color")]
    pub trajectory_searching_color: [u8; 3],
    #[serde(default = "default_trajectory_thickness")]
    pub trajectory_thickness: u32,
    #[serde(default = "default_crop_size")]
    pub crop_size: u32,
}

fn default_capture_screenshots() -> bool { true }
fn default_vision_model() -> String { "claude-haiku-4-5-20250929".to_string() }
fn default_true() -> bool { true }
fn default_dot_radius() -> u32 { 12 }
fn default_dot_color() -> [u8; 3] { [255, 40, 40] }
fn default_trajectory_ballistic_color() -> [u8; 3] { [40, 220, 40] }
fn default_trajectory_searching_color() -> [u8; 3] { [255, 160, 40] }
fn default_trajectory_thickness() -> u32 { 2 }
fn default_crop_size() -> u32 { 512 }

/// Get current generator configuration
#[tauri::command]
fn get_config(state: State<AppState>) -> Result<UiConfig, String> {
    let config = Config::load_default().map_err(|e| e.to_string())?;
    // Use in-memory value which stays in sync with the persisted config
    let capture_screenshots = *state.capture_screenshots.lock().map_err(|e| e.to_string())?;
    Ok(UiConfig {
        rdp_epsilon_px: config.analysis.rdp_epsilon_px,
        hesitation_threshold: config.analysis.hesitation_threshold,
        min_pause_ms: config.analysis.min_pause_ms,
        model: config.codegen.model,
        vision_model: config.codegen.vision_model,
        temperature: config.codegen.temperature,
        use_action_clustering: config.pipeline.use_action_clustering,
        use_local_recovery: config.pipeline.use_local_recovery,
        use_vision_ocr: config.pipeline.use_vision_ocr,
        use_trajectory_analysis: config.pipeline.use_trajectory_analysis,
        use_goms_detection: config.pipeline.use_goms_detection,
        use_context_tracking: config.pipeline.use_context_tracking,
        use_state_diff: config.pipeline.use_state_diff,
        use_error_correction_filter: config.pipeline.use_error_correction_filter,
        capture_screenshots,
        dot_radius: config.annotation.dot_radius,
        dot_color: config.annotation.dot_color,
        trajectory_ballistic_color: config.annotation.trajectory_ballistic_color,
        trajectory_searching_color: config.annotation.trajectory_searching_color,
        trajectory_thickness: config.annotation.trajectory_thickness,
        crop_size: config.annotation.crop_size,
    })
}

/// Save generator configuration
#[tauri::command]
fn save_config(state: State<AppState>, config: UiConfig) -> Result<(), String> {
    let mut full_config = Config::load_default().map_err(|e| e.to_string())?;
    full_config.analysis.rdp_epsilon_px = config.rdp_epsilon_px;
    full_config.analysis.hesitation_threshold = config.hesitation_threshold;
    full_config.analysis.min_pause_ms = config.min_pause_ms;
    full_config.codegen.model = config.model;
    full_config.codegen.vision_model = config.vision_model;
    full_config.codegen.temperature = config.temperature;
    full_config.pipeline.use_action_clustering = config.use_action_clustering;
    full_config.pipeline.use_local_recovery = config.use_local_recovery;
    full_config.pipeline.use_vision_ocr = config.use_vision_ocr;
    full_config.pipeline.use_trajectory_analysis = config.use_trajectory_analysis;
    full_config.pipeline.use_goms_detection = config.use_goms_detection;
    full_config.pipeline.use_context_tracking = config.use_context_tracking;
    full_config.pipeline.use_state_diff = config.use_state_diff;
    full_config.pipeline.use_error_correction_filter = config.use_error_correction_filter;
    full_config.capture.capture_screenshots = config.capture_screenshots;
    full_config.annotation.dot_radius = config.dot_radius;
    full_config.annotation.dot_color = config.dot_color;
    full_config.annotation.trajectory_ballistic_color = config.trajectory_ballistic_color;
    full_config.annotation.trajectory_searching_color = config.trajectory_searching_color;
    full_config.annotation.trajectory_thickness = config.trajectory_thickness;
    full_config.annotation.crop_size = config.crop_size;
    full_config.validate().map_err(|e| e.to_string())?;
    full_config.save_default().map_err(|e| e.to_string())?;

    // Update in-memory screenshot capture toggle
    let mut capture_ss = state.capture_screenshots.lock().map_err(|e| e.to_string())?;
    *capture_ss = config.capture_screenshots;

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
    // Parse and validate before saving
    let config: Config = toml::from_str(&toml_content)
        .map_err(|e| format!("Invalid TOML config: {}", e))?;
    config.validate().map_err(|e| e.to_string())?;
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
    if path.is_symlink() {
        return Err("Cannot delete symlinks".to_string());
    }

    let canonical = std::fs::canonicalize(path).map_err(|e| e.to_string())?;
    let canonical_dir = std::fs::canonicalize(&skills_dir).map_err(|e| e.to_string())?;
    if !canonical.starts_with(&canonical_dir) {
        return Err("Access denied: path is outside the skills directory".to_string());
    }

    // Delete the skill directory (parent of SKILL.md)
    if let Some(parent) = canonical.parent() {
        if parent.starts_with(&canonical_dir) && parent != canonical_dir {
            if parent.is_symlink() {
                return Err("Cannot delete symlinks".to_string());
            }
            std::fs::remove_dir_all(parent).map_err(|e| e.to_string())?;
        } else {
            std::fs::remove_file(&canonical).map_err(|e| e.to_string())?;
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
            analyze_recording,
            get_analysis,
            update_intent,
            list_screenshots,
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
            duration_seconds: 3.5,
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
            noise_filtered_count: 0,
            variables_count: 8,
            generated_steps_count: 9,
            screenshot_enhanced_count: 0,
            state_diff_enriched_count: 0,
            error_correction_count: 0,
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
                noise_filtered_count: 0,
                variables_count: 2,
                generated_steps_count: 5,
                screenshot_enhanced_count: 0,
                state_diff_enriched_count: 0,
                error_correction_count: 0,
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
            has_screenshots: true,
            has_analysis: false,
        };
        let json = serde_json::to_string(&info).unwrap();
        let deserialized: RecordingInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.event_count, 100);
        assert_eq!(deserialized.goal, Some("Test goal".to_string()));
        assert!(deserialized.has_screenshots);
        assert!(!deserialized.has_analysis);
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
            model: "claude-opus-4-6".to_string(),
            vision_model: "claude-haiku-4-5-20250929".to_string(),
            temperature: 0.7,
            use_action_clustering: true,
            use_local_recovery: true,
            use_vision_ocr: false,
            use_trajectory_analysis: true,
            use_goms_detection: true,
            use_context_tracking: true,
            use_state_diff: true,
            use_error_correction_filter: true,
            capture_screenshots: true,
            dot_radius: 12,
            dot_color: [255, 40, 40],
            trajectory_ballistic_color: [40, 220, 40],
            trajectory_searching_color: [255, 160, 40],
            trajectory_thickness: 2,
            crop_size: 512,
        };
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: UiConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.model, "claude-opus-4-6");
        assert_eq!(deserialized.vision_model, "claude-haiku-4-5-20250929");
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

    // --- Analysis response type tests ---

    #[test]
    fn test_screenshot_analysis_info_serialization() {
        let info = ScreenshotAnalysisInfo {
            sequence: 3,
            filename: "0003.jpg".to_string(),
            intent: "Click the Save button".to_string(),
            confidence: 0.85,
            tier: "ocr".to_string(),
            user_edited: false,
            image_path: "/tmp/screenshots/0003.jpg".to_string(),
        };
        let json = serde_json::to_string(&info).unwrap();
        let deserialized: ScreenshotAnalysisInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.sequence, 3);
        assert_eq!(deserialized.tier, "ocr");
        assert!(!deserialized.user_edited);
    }

    #[test]
    fn test_analysis_info_serialization() {
        let info = AnalysisInfo {
            recording_name: "test-recording".to_string(),
            analyses: vec![
                ScreenshotAnalysisInfo {
                    sequence: 0,
                    filename: "0000.jpg".to_string(),
                    intent: "Click button".to_string(),
                    confidence: 0.9,
                    tier: "vision".to_string(),
                    user_edited: false,
                    image_path: "/tmp/0000.jpg".to_string(),
                },
            ],
            total_screenshots: 1,
            ocr_count: 0,
            vision_count: 1,
            edited_count: 0,
        };
        let json = serde_json::to_string(&info).unwrap();
        let deserialized: AnalysisInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.recording_name, "test-recording");
        assert_eq!(deserialized.total_screenshots, 1);
        assert_eq!(deserialized.vision_count, 1);
    }

    #[test]
    fn test_screenshot_info_serialization() {
        let info = ScreenshotInfo {
            sequence: 5,
            filename: "0005.jpg".to_string(),
            image_path: "/tmp/screenshots/0005.jpg".to_string(),
        };
        let json = serde_json::to_string(&info).unwrap();
        let deserialized: ScreenshotInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.sequence, 5);
        assert_eq!(deserialized.filename, "0005.jpg");
    }

    // --- Screenshot analysis file I/O tests ---

    #[test]
    fn test_analysis_save_load_roundtrip() {
        use skill_generator::semantic::screenshot_analysis::{
            ScreenshotAnalysis, RecordingAnalysis, AnalysisTier,
        };

        let tmp = std::env::temp_dir().join("test_analysis_roundtrip");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        let analysis = RecordingAnalysis {
            recording_name: "roundtrip-test".to_string(),
            analyses: vec![
                ScreenshotAnalysis {
                    sequence: 0,
                    filename: "0000.jpg".to_string(),
                    intent: "Click the save button".to_string(),
                    ocr_text: Some("Save".to_string()),
                    confidence: 0.85,
                    tier: AnalysisTier::LocalOcr,
                    user_edited: false,
                },
            ],
            created_at: chrono::Utc::now(),
        };

        sa::save_analysis(&analysis, &tmp).unwrap();
        let loaded = sa::load_analysis(&tmp).unwrap();
        assert_eq!(loaded.recording_name, "roundtrip-test");
        assert_eq!(loaded.analyses.len(), 1);
        assert_eq!(loaded.analyses[0].intent, "Click the save button");
        assert_eq!(loaded.analyses[0].tier, AnalysisTier::LocalOcr);

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_analysis_update_intent_roundtrip() {
        use skill_generator::semantic::screenshot_analysis::{
            ScreenshotAnalysis, RecordingAnalysis, AnalysisTier,
        };

        let tmp = std::env::temp_dir().join("test_analysis_update");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        let analysis = RecordingAnalysis {
            recording_name: "update-test".to_string(),
            analyses: vec![
                ScreenshotAnalysis {
                    sequence: 0,
                    filename: "0000.jpg".to_string(),
                    intent: "Original intent".to_string(),
                    ocr_text: None,
                    confidence: 0.5,
                    tier: AnalysisTier::LocalOcr,
                    user_edited: false,
                },
                ScreenshotAnalysis {
                    sequence: 1,
                    filename: "0001.jpg".to_string(),
                    intent: "Second intent".to_string(),
                    ocr_text: None,
                    confidence: 0.6,
                    tier: AnalysisTier::LocalOcr,
                    user_edited: false,
                },
            ],
            created_at: chrono::Utc::now(),
        };

        sa::save_analysis(&analysis, &tmp).unwrap();

        // Simulate what update_intent does: load, modify, save
        let mut loaded = sa::load_analysis(&tmp).unwrap();
        let entry = loaded.analyses.iter_mut()
            .find(|a| a.sequence == 0)
            .unwrap();
        entry.intent = "Updated intent".to_string();
        entry.user_edited = true;
        entry.tier = AnalysisTier::UserEdited;
        sa::save_analysis(&loaded, &tmp).unwrap();

        // Verify the update persisted and didn't corrupt other entries
        let reloaded = sa::load_analysis(&tmp).unwrap();
        assert_eq!(reloaded.analyses[0].intent, "Updated intent");
        assert!(reloaded.analyses[0].user_edited);
        assert_eq!(reloaded.analyses[0].tier, AnalysisTier::UserEdited);
        assert_eq!(reloaded.analyses[1].intent, "Second intent");
        assert!(!reloaded.analyses[1].user_edited);

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_load_analysis_not_found() {
        let tmp = std::env::temp_dir().join("test_analysis_not_found");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        let result = sa::load_analysis(&tmp);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind(), std::io::ErrorKind::NotFound);

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_list_screenshots_from_disk() {
        let tmp = std::env::temp_dir().join("test_list_screenshots");
        let _ = std::fs::remove_dir_all(&tmp);
        let ss_dir = tmp.join("screenshots");
        std::fs::create_dir_all(&ss_dir).unwrap();

        // Create fake screenshot files
        std::fs::write(ss_dir.join("0000.jpg"), b"\xFF\xD8fake").unwrap();
        std::fs::write(ss_dir.join("0001.jpg"), b"\xFF\xD8fake").unwrap();
        std::fs::write(ss_dir.join("not-a-screenshot.txt"), b"nope").unwrap();

        // Read dir and filter .jpg files
        let mut screenshots: Vec<String> = std::fs::read_dir(&ss_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path().extension().map(|ext| ext == "jpg").unwrap_or(false)
            })
            .filter_map(|e| e.file_name().into_string().ok())
            .collect();
        screenshots.sort();

        assert_eq!(screenshots.len(), 2);
        assert_eq!(screenshots[0], "0000.jpg");
        assert_eq!(screenshots[1], "0001.jpg");

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_to_analysis_info_conversion() {
        use skill_generator::semantic::screenshot_analysis::{
            ScreenshotAnalysis, RecordingAnalysis, AnalysisTier,
        };

        let analysis = RecordingAnalysis {
            recording_name: "test".to_string(),
            analyses: vec![
                ScreenshotAnalysis {
                    sequence: 0,
                    filename: "0000.jpg".to_string(),
                    intent: "Click button".to_string(),
                    ocr_text: Some("OK".to_string()),
                    confidence: 0.8,
                    tier: AnalysisTier::LocalOcr,
                    user_edited: false,
                },
                ScreenshotAnalysis {
                    sequence: 1,
                    filename: "0001.jpg".to_string(),
                    intent: "Type in search field".to_string(),
                    ocr_text: None,
                    confidence: 0.9,
                    tier: AnalysisTier::ClaudeVision,
                    user_edited: false,
                },
                ScreenshotAnalysis {
                    sequence: 2,
                    filename: "0002.jpg".to_string(),
                    intent: "User corrected intent".to_string(),
                    ocr_text: None,
                    confidence: 1.0,
                    tier: AnalysisTier::UserEdited,
                    user_edited: true,
                },
            ],
            created_at: chrono::Utc::now(),
        };

        let screenshots_dir = std::path::Path::new("/tmp/screenshots");
        let info = to_analysis_info(&analysis, screenshots_dir);

        assert_eq!(info.recording_name, "test");
        assert_eq!(info.total_screenshots, 3);
        assert_eq!(info.ocr_count, 1);
        assert_eq!(info.vision_count, 1);
        assert_eq!(info.edited_count, 1);
        assert_eq!(info.analyses[0].tier, "ocr");
        assert_eq!(info.analyses[1].tier, "vision");
        assert_eq!(info.analyses[2].tier, "edited");
        assert!(info.analyses[2].user_edited);
        assert!(info.analyses[0].image_path.contains("0000.jpg"));
    }
}
