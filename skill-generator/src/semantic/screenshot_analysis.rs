//! Tiered AI Analysis for Screenshots
//!
//! Batch analysis of screenshots captured during recording. Uses a two-tier approach:
//! - Tier 1: Local Vision OCR (fast, free, works offline)
//! - Tier 2: Claude Vision API for low-confidence results (accurate, requires API key)

use crate::capture::types::{EnrichedEvent, EventType};
use crate::semantic::vision_fallback::VisionFallback;
use crate::workflow::recording::Recording;
use serde::{Deserialize, Serialize};
use std::path::Path;
use tracing::{debug, info, warn};

/// Map an RGB triplet to a human-readable color name for LLM prompts.
fn describe_rgb(r: u8, g: u8, b: u8) -> &'static str {
    // Dominant-channel heuristic — good enough for distinct annotation colors.
    let max = r.max(g).max(b);
    if max < 50 { return "dark"; }
    if r > 200 && g < 100 && b < 100 { return "red"; }
    if r < 100 && g > 150 && b < 100 { return "green"; }
    if r < 100 && g < 100 && b > 150 { return "blue"; }
    if r > 200 && g > 150 && b < 80 { return "orange"; }
    if r > 200 && g > 200 && b < 80 { return "yellow"; }
    if r > 150 && g < 100 && b > 150 { return "purple"; }
    if r > 200 && g > 200 && b > 200 { return "white"; }
    "colored"
}

/// Analysis tier indicating how the intent was determined
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AnalysisTier {
    /// Local Vision OCR analysis
    LocalOcr,
    /// Claude Vision API analysis
    ClaudeVision,
    /// User-edited intent
    UserEdited,
}

/// Analysis result for a single screenshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScreenshotAnalysis {
    /// Sequence number matching the event
    pub sequence: u64,
    /// Screenshot filename (e.g., "0000.jpg")
    pub filename: String,
    /// Human-readable intent description
    pub intent: String,
    /// OCR text extracted from the screenshot
    pub ocr_text: Option<String>,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Which tier produced this analysis
    pub tier: AnalysisTier,
    /// Whether the user has edited this intent
    pub user_edited: bool,
}

/// Complete analysis for a recording
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordingAnalysis {
    /// Recording name
    pub recording_name: String,
    /// Analysis results for each screenshot
    pub analyses: Vec<ScreenshotAnalysis>,
    /// When the analysis was created
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// Configuration for screenshot analysis
#[derive(Debug, Clone)]
pub struct AnalysisConfig {
    /// Confidence threshold below which to escalate to Claude Vision
    pub ocr_confidence_threshold: f32,
    /// Claude model to use for Vision analysis
    pub claude_model: String,
    /// Max tokens for Claude Vision response
    pub max_tokens: u32,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            ocr_confidence_threshold: 0.5,
            claude_model: "claude-haiku-4-5-20250929".to_string(),
            max_tokens: 256,
        }
    }
}

/// Describe what the user was doing based on event type and OCR text.
fn describe_intent_from_ocr(event: &EnrichedEvent, ocr_text: &str) -> String {
    let action = match event.raw.event_type {
        EventType::LeftMouseDown | EventType::LeftMouseUp => "Click",
        EventType::RightMouseDown | EventType::RightMouseUp => "Right-click",
        EventType::OtherMouseDown | EventType::OtherMouseUp => "Middle-click",
        EventType::KeyDown => "Type",
        EventType::ScrollWheel => "Scroll",
        _ => "Interact with",
    };

    let text_preview = if ocr_text.len() > 80 {
        // Find the last char boundary at or before byte 77 to avoid panicking on multi-byte UTF-8
        let truncate_at = ocr_text.char_indices()
            .take_while(|&(i, _)| i <= 77)
            .last()
            .map(|(i, c)| i + c.len_utf8())
            .unwrap_or(0);
        format!("{}...", &ocr_text[..truncate_at])
    } else {
        ocr_text.to_string()
    };

    if text_preview.is_empty() {
        format!("{} at ({:.0}, {:.0})", action, event.raw.coordinates.0, event.raw.coordinates.1)
    } else {
        format!("{} near \"{}\"", action, text_preview.trim())
    }
}

/// Run Tier 1: Local Vision OCR on screenshots.
///
/// For each JPEG in the screenshots directory, loads it as a CGImage and runs
/// VNRecognizeTextRequest to extract text, then generates an intent description.
pub fn run_local_ocr(
    screenshots_dir: &Path,
    events: &[EnrichedEvent],
) -> Vec<ScreenshotAnalysis> {
    let fallback = VisionFallback::new();
    let mut results = Vec::new();

    for event in events {
        let filename = match &event.screenshot_filename {
            Some(f) => f,
            None => continue,
        };

        let jpeg_path = screenshots_dir.join(filename);
        if !jpeg_path.exists() {
            debug!(filename = %filename, "Screenshot file not found, skipping");
            continue;
        }

        // Load JPEG as CGImage
        let jpeg_data = match std::fs::read(&jpeg_path) {
            Ok(data) => data,
            Err(e) => {
                warn!(filename = %filename, error = %e, "Failed to read screenshot");
                continue;
            }
        };

        let cgimage = match crate::semantic::screenshot::jpeg_to_cgimage(&jpeg_data) {
            Some(img) => img,
            None => {
                warn!(filename = %filename, "Failed to decode JPEG as CGImage");
                continue;
            }
        };

        // Run OCR
        let text_regions = fallback.perform_ocr_on_image(&cgimage);

        unsafe { crate::semantic::screenshot::release_cgimage(cgimage); }

        // Combine all text regions
        let ocr_text: String = text_regions
            .iter()
            .map(|r| r.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");

        let avg_confidence = if text_regions.is_empty() {
            0.0
        } else {
            text_regions.iter().map(|r| r.confidence).sum::<f32>() / text_regions.len() as f32
        };

        let intent = describe_intent_from_ocr(event, &ocr_text);

        results.push(ScreenshotAnalysis {
            sequence: event.sequence,
            filename: filename.clone(),
            intent,
            ocr_text: if ocr_text.is_empty() { None } else { Some(ocr_text) },
            confidence: avg_confidence,
            tier: AnalysisTier::LocalOcr,
            user_edited: false,
        });
    }

    info!(count = results.len(), "Local OCR analysis complete");
    results
}

/// Build the Claude Vision API request body for a low-confidence screenshot.
///
/// When `annotated` is `Some`, the request includes two images: a marked-up
/// full screenshot (with red dot + green AX box) and a foveated crop centred
/// on the interaction point. When `None`, falls back to the single raw image.
fn build_vision_request(
    analysis: &ScreenshotAnalysis,
    jpeg_data: &[u8],
    annotated: Option<&crate::semantic::screenshot::AnnotatedScreenshot>,
    event: &EnrichedEvent,
    goal: &str,
    config: &AnalysisConfig,
    annotation_config: &crate::semantic::screenshot::AnnotationConfig,
) -> serde_json::Value {
    use base64::Engine;
    let b64_encode = |data: &[u8]| base64::engine::general_purpose::STANDARD.encode(data);

    let action_desc = match event.raw.event_type {
        EventType::LeftMouseDown => format!("left click at ({:.0}, {:.0})", event.raw.coordinates.0, event.raw.coordinates.1),
        EventType::RightMouseDown => format!("right click at ({:.0}, {:.0})", event.raw.coordinates.0, event.raw.coordinates.1),
        EventType::KeyDown => {
            if let Some(ch) = event.raw.character {
                format!("keystroke '{}'", ch)
            } else {
                "keystroke".to_string()
            }
        }
        EventType::ScrollWheel => "scroll".to_string(),
        _ => format!("action at ({:.0}, {:.0})", event.raw.coordinates.0, event.raw.coordinates.1),
    };

    // Build content array: two annotated images when available, else one raw image
    let (content, prompt) = if let Some(ann) = annotated {
        let full_b64 = b64_encode(&ann.full_jpeg);
        let crop_b64 = b64_encode(&ann.crop_jpeg);

        // Describe trajectory colors dynamically so the prompt matches whatever the
        // user has configured in the annotation settings.
        let [br, bg, bb] = annotation_config.trajectory_ballistic_color;
        let [sr, sg, sb] = annotation_config.trajectory_searching_color;
        let ballistic_desc = describe_rgb(br, bg, bb);
        let searching_desc = describe_rgb(sr, sg, sb);

        let prompt = format!(
            "Image 1 is the full UI for macro context. The red dot and green box indicate \
             exactly what the user interacted with. If a colored line is visible, it shows the \
             mouse trajectory: a {} line means confident, direct movement; a {} line \
             means the user was searching or hesitating. Image 2 is a high-resolution crop of \
             that exact interaction point. Goal: '{}'. Action: {}. OCR text: '{}'. \
             Based on the text/UI elements in Image 2, what is the user clicking and why? \
             Describe the user's intent in one sentence.",
            ballistic_desc,
            searching_desc,
            goal,
            action_desc,
            analysis.ocr_text.as_deref().unwrap_or("(none)")
        );

        let content = serde_json::json!([
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": full_b64
                }
            },
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": crop_b64
                }
            },
            {
                "type": "text",
                "text": prompt
            }
        ]);
        (content, prompt)
    } else {
        let b64 = b64_encode(jpeg_data);
        let prompt = format!(
            "Screenshot during macOS workflow. Goal: '{}'. Action: {}. OCR text found: '{}'. \
             Describe the user's intent in one sentence.",
            goal,
            action_desc,
            analysis.ocr_text.as_deref().unwrap_or("(none)")
        );
        let content = serde_json::json!([
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": b64
                }
            },
            {
                "type": "text",
                "text": prompt
            }
        ]);
        (content, prompt)
    };

    let _ = prompt; // consumed by the json! macro above

    serde_json::json!({
        "model": config.claude_model,
        "max_tokens": config.max_tokens,
        "messages": [{
            "role": "user",
            "content": content
        }]
    })
}

/// Maximum concurrent Vision API requests
const VISION_CONCURRENCY: usize = 5;

/// Run Tier 2: Claude Vision API for low-confidence results.
///
/// Sends screenshots to Claude Vision API concurrently (up to VISION_CONCURRENCY
/// at a time) for much faster analysis of large batches.
pub async fn run_claude_vision_analysis(
    low_confidence: &[ScreenshotAnalysis],
    screenshots_dir: &Path,
    events: &[EnrichedEvent],
    goal: &str,
    api_key: &str,
    config: &AnalysisConfig,
    annotation_config: &crate::semantic::screenshot::AnnotationConfig,
    client: &reqwest::Client,
) -> Vec<ScreenshotAnalysis> {
    use std::sync::Arc;
    use tokio::sync::Semaphore;

    #[derive(serde::Deserialize)]
    struct ApiResponse {
        content: Vec<ContentBlock>,
    }
    #[derive(serde::Deserialize)]
    struct ContentBlock {
        text: String,
    }

    let semaphore = Arc::new(Semaphore::new(VISION_CONCURRENCY));
    let mut handles = Vec::new();

    // Obtain screen resolution once for coordinate mapping
    let screen_dims = crate::semantic::screenshot::get_screen_resolution();
    let ann_config = annotation_config.clone();

    // Pre-compute trajectory overlays for each click event
    let rdp = crate::analysis::rdp_simplification::RdpSimplifier::default();
    let kinematic = crate::analysis::kinematic_segmentation::KinematicSegmenter::default();
    let trajectory_map = build_trajectory_map(events, &rdp, &kinematic);

    for analysis in low_confidence {
        let analysis_owned = analysis.clone();

        // Find matching event
        let event = match events.iter().find(|e| e.sequence == analysis.sequence) {
            Some(e) => e.clone(),
            None => continue,
        };

        // Load screenshot
        let jpeg_path = screenshots_dir.join(&analysis.filename);
        let jpeg_data = match std::fs::read(&jpeg_path) {
            Ok(data) => data,
            Err(e) => {
                warn!(filename = %analysis.filename, error = %e, "Failed to read screenshot for Vision API");
                handles.push(tokio::spawn(async move { analysis_owned }));
                continue;
            }
        };

        // Look up pre-computed trajectory for this event
        let trajectory = trajectory_map.get(&analysis.sequence);

        // Annotate: red dot at click point, green AX bounding box, trajectory, foveated crop
        let ax_frame = event.semantic.as_ref().and_then(|s| s.frame);
        let annotated = crate::semantic::screenshot::annotate_screenshot(
            &jpeg_data,
            event.raw.coordinates.0,
            event.raw.coordinates.1,
            screen_dims,
            ax_frame,
            trajectory,
            &ann_config,
        );

        let body = build_vision_request(analysis, &jpeg_data, annotated.as_ref(), &event, goal, config, &ann_config);
        let analysis_clone = analysis.clone();
        let client = client.clone();
        let api_key = api_key.to_string();
        let sem = semaphore.clone();

        handles.push(tokio::spawn(async move {
            let _permit = match sem.acquire().await {
                Ok(p) => p,
                Err(_) => return analysis_clone,
            };

            let response = match crate::semantic::http_retry::send_with_retry(
                &client,
                |c| c.post("https://api.anthropic.com/v1/messages")
                    .header("x-api-key", &api_key)
                    .header("anthropic-version", "2023-06-01")
                    .header("content-type", "application/json")
                    .json(&body),
                1,
                "Claude Vision analysis",
            ).await {
                Some(r) => r,
                None => {
                    warn!(
                        sequence = analysis_clone.sequence,
                        "Claude Vision API request failed, keeping OCR result"
                    );
                    return analysis_clone;
                }
            };

            match response.json::<ApiResponse>().await {
                Ok(api_response) => {
                    let intent = api_response.content.first()
                        .map(|b| b.text.trim().to_string())
                        .filter(|s| !s.is_empty())
                        .unwrap_or_else(|| analysis_clone.intent.clone());

                    ScreenshotAnalysis {
                        sequence: analysis_clone.sequence,
                        filename: analysis_clone.filename.clone(),
                        intent,
                        ocr_text: analysis_clone.ocr_text.clone(),
                        confidence: 0.9,
                        tier: AnalysisTier::ClaudeVision,
                        user_edited: false,
                    }
                }
                Err(e) => {
                    warn!(error = %e, "Failed to parse Claude Vision response");
                    analysis_clone
                }
            }
        }));
    }

    let mut results = Vec::with_capacity(handles.len());
    for handle in handles {
        match handle.await {
            Ok(result) => results.push(result),
            Err(e) => warn!(error = %e, "Vision analysis task panicked"),
        }
    }

    info!(count = results.len(), "Claude Vision analysis complete");
    results
}

/// Run full tiered analysis on a recording's screenshots.
///
/// 1. Run local OCR on all screenshots
/// 2. Identify low-confidence results
/// 3. Send low-confidence results to Claude Vision API (if API key available)
/// 4. Merge results
pub fn analyze_recording(
    recording_dir: &Path,
    recording: &Recording,
    api_key: Option<&str>,
    config: &AnalysisConfig,
    annotation_config: &crate::semantic::screenshot::AnnotationConfig,
) -> RecordingAnalysis {
    let screenshots_dir = Recording::screenshots_dir(recording_dir);

    // Filter events that have screenshots
    let events_with_screenshots: Vec<&EnrichedEvent> = recording.events.iter()
        .filter(|e| e.screenshot_filename.is_some())
        .collect();

    info!(
        count = events_with_screenshots.len(),
        "Running screenshot analysis"
    );

    // Tier 1: Local OCR
    let mut all_analyses = run_local_ocr(&screenshots_dir, &recording.events);

    // Tier 2: Claude Vision for low-confidence results
    if let Some(key) = api_key {
        if !key.is_empty() {
            let low_confidence: Vec<ScreenshotAnalysis> = all_analyses.iter()
                .filter(|a| a.confidence < config.ocr_confidence_threshold)
                .cloned()
                .collect();

            if !low_confidence.is_empty() {
                info!(count = low_confidence.len(), "Escalating to Claude Vision API");

                let rt = match tokio::runtime::Runtime::new() {
                    Ok(rt) => rt,
                    Err(e) => {
                        warn!(error = %e, "Failed to create tokio runtime for Vision API");
                        return RecordingAnalysis {
                            recording_name: recording.metadata.name.clone(),
                            analyses: all_analyses,
                            created_at: chrono::Utc::now(),
                        };
                    }
                };

                let goal = recording.metadata.goal.as_deref().unwrap_or("");
                let client = match reqwest::Client::builder()
                    .timeout(std::time::Duration::from_secs(60))
                    .build()
                {
                    Ok(c) => c,
                    Err(e) => {
                        warn!(error = %e, "Failed to build HTTP client");
                        return RecordingAnalysis {
                            recording_name: recording.metadata.name.clone(),
                            analyses: all_analyses,
                            created_at: chrono::Utc::now(),
                        };
                    }
                };

                let vision_results = rt.block_on(run_claude_vision_analysis(
                    &low_confidence,
                    &screenshots_dir,
                    &recording.events,
                    goal,
                    key,
                    config,
                    annotation_config,
                    &client,
                ));

                // Merge: replace low-confidence results with Vision API results
                for vision_result in vision_results {
                    if let Some(existing) = all_analyses.iter_mut()
                        .find(|a| a.sequence == vision_result.sequence)
                    {
                        *existing = vision_result;
                    }
                }
            }
        }
    }

    RecordingAnalysis {
        recording_name: recording.metadata.name.clone(),
        analyses: all_analyses,
        created_at: chrono::Utc::now(),
    }
}

/// Save analysis results to JSON file.
pub fn save_analysis(analysis: &RecordingAnalysis, recording_dir: &Path) -> std::io::Result<()> {
    let path = recording_dir.join("analysis.json");
    let json = serde_json::to_string_pretty(analysis)
        .map_err(std::io::Error::other)?;
    std::fs::write(path, json)
}

/// Load analysis results from JSON file.
pub fn load_analysis(recording_dir: &Path) -> std::io::Result<RecordingAnalysis> {
    let path = recording_dir.join("analysis.json");
    let content = std::fs::read_to_string(path)?;
    serde_json::from_str(&content)
        .map_err(std::io::Error::other)
}

/// Build a map of sequence → TrajectoryOverlay for each click event.
///
/// For each click, extracts the preceding mouse-move events, simplifies with
/// RDP, classifies via kinematic segmentation, and produces a trajectory overlay
/// with screen-coordinate points and movement pattern.
fn build_trajectory_map(
    events: &[EnrichedEvent],
    rdp: &crate::analysis::rdp_simplification::RdpSimplifier,
    kinematic: &crate::analysis::kinematic_segmentation::KinematicSegmenter,
) -> std::collections::HashMap<u64, crate::semantic::screenshot::TrajectoryOverlay> {
    use crate::analysis::kinematic_segmentation::MovementPattern;

    let mut map = std::collections::HashMap::new();

    // Find click event indices
    let click_indices: Vec<usize> = events
        .iter()
        .enumerate()
        .filter(|(_, e)| e.raw.event_type.is_click())
        .map(|(i, _)| i)
        .collect();

    let mut prev_click_idx = 0;

    for &click_idx in &click_indices {
        let start = if click_idx > prev_click_idx { prev_click_idx } else { 0 };

        // Extract mouse-move + click events in the segment
        let segment_raw: Vec<crate::capture::types::RawEvent> = events[start..=click_idx]
            .iter()
            .filter(|e| e.raw.event_type.is_mouse_move() || e.raw.event_type.is_click())
            .map(|e| e.raw.clone())
            .collect();

        prev_click_idx = click_idx + 1;

        if segment_raw.len() < 3 {
            continue;
        }

        let simplified = rdp.simplify_events(&segment_raw);
        if simplified.len() < 2 {
            continue;
        }

        let analysis = kinematic.analyze(&simplified);

        // Skip stationary patterns — no visible trajectory to draw
        if analysis.pattern == MovementPattern::Stationary {
            continue;
        }

        // Convert simplified trajectory points to screen coordinates
        let points: Vec<(f64, f64)> = simplified
            .iter()
            .map(|pt| (pt.x, pt.y))
            .collect();

        let sequence = events[click_idx].sequence;
        map.insert(sequence, crate::semantic::screenshot::TrajectoryOverlay {
            points,
            pattern: analysis.pattern,
        });
    }

    map
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analysis_config_default() {
        let config = AnalysisConfig::default();
        assert!((config.ocr_confidence_threshold - 0.5).abs() < f32::EPSILON);
        assert_eq!(config.max_tokens, 256);
    }

    #[test]
    fn test_screenshot_analysis_json_roundtrip() {
        let analysis = ScreenshotAnalysis {
            sequence: 0,
            filename: "0000.jpg".to_string(),
            intent: "Click the Submit button".to_string(),
            ocr_text: Some("Submit".to_string()),
            confidence: 0.85,
            tier: AnalysisTier::LocalOcr,
            user_edited: false,
        };
        let json = serde_json::to_string(&analysis).unwrap();
        let loaded: ScreenshotAnalysis = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.sequence, 0);
        assert_eq!(loaded.intent, "Click the Submit button");
        assert_eq!(loaded.tier, AnalysisTier::LocalOcr);
        assert!(!loaded.user_edited);
    }

    #[test]
    fn test_recording_analysis_json_roundtrip() {
        let analysis = RecordingAnalysis {
            recording_name: "test".to_string(),
            analyses: vec![
                ScreenshotAnalysis {
                    sequence: 0,
                    filename: "0000.jpg".to_string(),
                    intent: "Click button".to_string(),
                    ocr_text: None,
                    confidence: 0.3,
                    tier: AnalysisTier::ClaudeVision,
                    user_edited: false,
                },
            ],
            created_at: chrono::Utc::now(),
        };
        let json = serde_json::to_string(&analysis).unwrap();
        let loaded: RecordingAnalysis = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.recording_name, "test");
        assert_eq!(loaded.analyses.len(), 1);
        assert_eq!(loaded.analyses[0].tier, AnalysisTier::ClaudeVision);
    }

    #[test]
    fn test_empty_screenshots_produces_empty_results() {
        let dir = tempfile::tempdir().unwrap();
        let screenshots_dir = dir.path().join("screenshots");
        std::fs::create_dir_all(&screenshots_dir).unwrap();

        let events: Vec<EnrichedEvent> = vec![];
        let results = run_local_ocr(&screenshots_dir, &events);
        assert!(results.is_empty());
    }

    #[test]
    fn test_save_and_load_analysis() {
        let dir = tempfile::tempdir().unwrap();
        let analysis = RecordingAnalysis {
            recording_name: "save_load_test".to_string(),
            analyses: vec![
                ScreenshotAnalysis {
                    sequence: 5,
                    filename: "0005.jpg".to_string(),
                    intent: "Type in search field".to_string(),
                    ocr_text: Some("Search...".to_string()),
                    confidence: 0.7,
                    tier: AnalysisTier::LocalOcr,
                    user_edited: false,
                },
            ],
            created_at: chrono::Utc::now(),
        };

        save_analysis(&analysis, dir.path()).unwrap();
        let loaded = load_analysis(dir.path()).unwrap();

        assert_eq!(loaded.recording_name, "save_load_test");
        assert_eq!(loaded.analyses.len(), 1);
        assert_eq!(loaded.analyses[0].sequence, 5);
        assert_eq!(loaded.analyses[0].intent, "Type in search field");
    }

    #[test]
    fn test_build_vision_request_produces_valid_json() {
        use crate::capture::types::{CursorState, ModifierFlags, RawEvent};
        use crate::time::timebase::{MachTimebase, Timestamp};

        MachTimebase::init();

        let event = EnrichedEvent {
            raw: RawEvent {
                timestamp: Timestamp::now(),
                event_type: EventType::LeftMouseDown,
                coordinates: (100.0, 200.0),
                cursor_state: CursorState::Arrow,
                key_code: None,
                character: None,
                modifiers: ModifierFlags::default(),
                scroll_delta: None,
                click_count: 1,
            },
            semantic: None,
            id: uuid::Uuid::new_v4(),
            sequence: 0,
            screenshot_filename: Some("0000.jpg".to_string()),
        };

        let analysis = ScreenshotAnalysis {
            sequence: 0,
            filename: "0000.jpg".to_string(),
            intent: "Click button".to_string(),
            ocr_text: Some("OK".to_string()),
            confidence: 0.3,
            tier: AnalysisTier::LocalOcr,
            user_edited: false,
        };

        let config = AnalysisConfig::default();
        let ann_config = crate::semantic::screenshot::AnnotationConfig::default();
        let jpeg_data = vec![0xFF, 0xD8, 0xFF, 0xE0]; // Minimal JPEG header

        // Test without annotation (fallback path)
        let body = build_vision_request(&analysis, &jpeg_data, None, &event, "Test goal", &config, &ann_config);

        // Verify structure
        assert_eq!(body["model"], "claude-haiku-4-5-20250929");
        assert!(body["messages"][0]["content"][0]["source"]["data"].is_string());
        assert_eq!(body["messages"][0]["content"][0]["type"], "image");
        assert!(body["messages"][0]["content"][1]["text"].as_str().unwrap().contains("Test goal"));
    }

    #[test]
    fn test_describe_intent_from_ocr_with_text() {
        use crate::capture::types::{CursorState, ModifierFlags, RawEvent};
        use crate::time::timebase::{MachTimebase, Timestamp};

        MachTimebase::init();

        let event = EnrichedEvent {
            raw: RawEvent {
                timestamp: Timestamp::now(),
                event_type: EventType::LeftMouseDown,
                coordinates: (100.0, 200.0),
                cursor_state: CursorState::Arrow,
                key_code: None,
                character: None,
                modifiers: ModifierFlags::default(),
                scroll_delta: None,
                click_count: 1,
            },
            semantic: None,
            id: uuid::Uuid::new_v4(),
            sequence: 0,
            screenshot_filename: None,
        };

        let intent = describe_intent_from_ocr(&event, "Submit");
        assert!(intent.contains("Click"));
        assert!(intent.contains("Submit"));
    }

    #[test]
    fn test_describe_intent_from_ocr_empty_text() {
        use crate::capture::types::{CursorState, ModifierFlags, RawEvent};
        use crate::time::timebase::{MachTimebase, Timestamp};

        MachTimebase::init();

        let event = EnrichedEvent {
            raw: RawEvent {
                timestamp: Timestamp::now(),
                event_type: EventType::ScrollWheel,
                coordinates: (50.0, 100.0),
                cursor_state: CursorState::Arrow,
                key_code: None,
                character: None,
                modifiers: ModifierFlags::default(),
                scroll_delta: Some((0.0, -3.0)),
                click_count: 0,
            },
            semantic: None,
            id: uuid::Uuid::new_v4(),
            sequence: 0,
            screenshot_filename: None,
        };

        let intent = describe_intent_from_ocr(&event, "");
        assert!(intent.contains("Scroll"));
        assert!(intent.contains("50"));
    }

    #[test]
    fn test_describe_intent_from_ocr_multibyte_truncation() {
        use crate::capture::types::{CursorState, ModifierFlags, RawEvent};
        use crate::time::timebase::{MachTimebase, Timestamp};

        MachTimebase::init();

        let event = EnrichedEvent {
            raw: RawEvent {
                timestamp: Timestamp::now(),
                event_type: EventType::LeftMouseDown,
                coordinates: (100.0, 200.0),
                cursor_state: CursorState::Arrow,
                key_code: None,
                character: None,
                modifiers: ModifierFlags::default(),
                scroll_delta: None,
                click_count: 1,
            },
            semantic: None,
            id: uuid::Uuid::new_v4(),
            sequence: 0,
            screenshot_filename: None,
        };

        // 30 CJK characters = 90 bytes in UTF-8 (each is 3 bytes), exceeds 80-byte threshold
        let cjk_text = "日本語テスト文字列を使用して多バイト文字の切り捨てをテストする長い文字列です";
        // This should NOT panic — the old code would panic on byte index 77
        let intent = describe_intent_from_ocr(&event, cjk_text);
        assert!(intent.contains("Click"));
        assert!(intent.contains("..."));
    }

    #[test]
    fn test_analysis_tier_serialization() {
        let tier = AnalysisTier::UserEdited;
        let json = serde_json::to_string(&tier).unwrap();
        let loaded: AnalysisTier = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded, AnalysisTier::UserEdited);
    }
}
