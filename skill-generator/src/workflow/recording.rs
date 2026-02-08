//! Recording Data Structures
//!
//! Defines the serialization format for captured event recordings.

use crate::capture::types::{EnrichedEvent, RawEvent, SemanticContext};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::Path;
use uuid::Uuid;

/// Current recording format version
pub const CURRENT_FORMAT_VERSION: &str = "1.0";

/// Checkpoint interval: save every N events
pub const CHECKPOINT_INTERVAL: usize = 100;

/// Get the checkpoint (temporary) path for a recording file
fn checkpoint_path(final_path: &Path) -> std::path::PathBuf {
    final_path.with_extension("json.tmp")
}

/// Recording metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RecordingMetadata {
    /// Unique recording ID
    pub id: Uuid,
    /// Recording name
    pub name: String,
    /// User-provided goal description
    pub goal: Option<String>,
    /// Application context (bundle ID)
    pub app_context: Option<String>,
    /// Recording start time
    pub started_at: DateTime<Utc>,
    /// Recording end time
    pub ended_at: Option<DateTime<Utc>>,
    /// Total event count
    pub event_count: usize,
    /// Recording duration in milliseconds
    pub duration_ms: u64,
    /// Version of the recording format
    pub format_version: String,
}

impl RecordingMetadata {
    /// Create new metadata for a recording
    pub fn new(name: String, goal: Option<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            name,
            goal,
            app_context: None,
            started_at: Utc::now(),
            ended_at: None,
            event_count: 0,
            duration_ms: 0,
            format_version: CURRENT_FORMAT_VERSION.to_string(),
        }
    }

    /// Finalize the recording with end time and event count
    pub fn finalize(&mut self, event_count: usize, duration_ms: u64) {
        self.ended_at = Some(Utc::now());
        self.event_count = event_count;
        self.duration_ms = duration_ms;
    }
}

impl Default for RecordingMetadata {
    fn default() -> Self {
        Self {
            id: Uuid::new_v4(),
            name: String::new(),
            goal: None,
            app_context: None,
            started_at: Utc::now(),
            ended_at: None,
            event_count: 0,
            duration_ms: 0,
            format_version: CURRENT_FORMAT_VERSION.to_string(),
        }
    }
}

/// A complete recording of user interactions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recording {
    /// Recording metadata
    pub metadata: RecordingMetadata,
    /// Enriched events with semantic data
    pub events: Vec<EnrichedEvent>,
}

impl Recording {
    /// Create a new empty recording
    pub fn new(name: String, goal: Option<String>) -> Self {
        Self {
            metadata: RecordingMetadata::new(name, goal),
            events: Vec::new(),
        }
    }

    /// Add an event to the recording
    pub fn add_event(&mut self, event: EnrichedEvent) {
        self.events.push(event);
    }

    /// Add a raw event (will be converted to enriched)
    pub fn add_raw_event(&mut self, raw: RawEvent) {
        let sequence = self.events.len() as u64;
        let enriched = EnrichedEvent::new(raw, sequence);
        self.events.push(enriched);
    }

    /// Add a raw event with semantic context
    pub fn add_enriched_raw(&mut self, raw: RawEvent, semantic: Option<SemanticContext>) {
        let sequence = self.events.len() as u64;
        let mut enriched = EnrichedEvent::new(raw, sequence);
        enriched.semantic = semantic;
        self.events.push(enriched);
    }

    /// Finalize the recording
    pub fn finalize(&mut self, duration_ms: u64) {
        self.metadata.finalize(self.events.len(), duration_ms);
    }

    /// Save recording to a file
    pub fn save(&self, path: &Path) -> crate::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Save a checkpoint to a temporary file for crash recovery.
    ///
    /// Writes to `<path>.tmp` so that if the process crashes, the recording
    /// can be recovered on next launch.
    pub fn save_checkpoint(&self, final_path: &Path) -> crate::Result<()> {
        let tmp_path = checkpoint_path(final_path);
        let json = serde_json::to_string(self)?; // compact JSON for speed
        std::fs::write(&tmp_path, json)?;
        Ok(())
    }

    /// Finalize a checkpoint by renaming `.tmp` to the final path.
    ///
    /// This is an atomic operation on most filesystems, ensuring no data loss.
    pub fn finalize_checkpoint(final_path: &Path) -> crate::Result<()> {
        let tmp_path = checkpoint_path(final_path);
        if tmp_path.exists() {
            std::fs::rename(&tmp_path, final_path)?;
        }
        Ok(())
    }

    /// Remove a checkpoint file if it exists (e.g., after successful save).
    pub fn remove_checkpoint(final_path: &Path) {
        let tmp_path = checkpoint_path(final_path);
        let _ = std::fs::remove_file(tmp_path);
    }

    /// Find and recover any orphaned checkpoint files in a directory.
    ///
    /// Returns a list of (checkpoint_path, recovered_recording) pairs.
    pub fn recover_checkpoints(dir: &Path) -> Vec<(std::path::PathBuf, Recording)> {
        let mut recovered = Vec::new();
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().map(|e| e == "tmp").unwrap_or(false) {
                    if let Ok(content) = std::fs::read_to_string(&path) {
                        if let Ok(recording) = serde_json::from_str::<Recording>(&content) {
                            recovered.push((path, recording));
                        }
                    }
                }
            }
        }
        recovered
    }

    /// Load recording from a file.
    ///
    /// Logs a warning if the recording was saved with an unknown format version,
    /// but still attempts to deserialize it (forward-compatible via `#[serde(default)]`).
    pub fn load(path: &Path) -> crate::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let recording: Recording = serde_json::from_str(&content)?;
        if recording.metadata.format_version != CURRENT_FORMAT_VERSION {
            tracing::warn!(
                name = %recording.metadata.name,
                found = %recording.metadata.format_version,
                expected = CURRENT_FORMAT_VERSION,
                "Recording has different format version; some fields may use default values"
            );
        }
        Ok(recording)
    }

    /// Get the number of events
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Check if recording is empty
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Get events by type filter
    pub fn events_of_type(
        &self,
        filter: impl Fn(&EnrichedEvent) -> bool,
    ) -> Vec<&EnrichedEvent> {
        self.events.iter().filter(|e| filter(e)).collect()
    }

    /// Get click events only
    pub fn click_events(&self) -> Vec<&EnrichedEvent> {
        self.events_of_type(|e| e.raw.event_type.is_click())
    }

    /// Get keyboard events only
    pub fn keyboard_events(&self) -> Vec<&EnrichedEvent> {
        self.events_of_type(|e| e.raw.event_type.is_keyboard())
    }
}

impl Default for Recording {
    fn default() -> Self {
        Self::new("untitled".to_string(), None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capture::types::{CursorState, EventType, ModifierFlags, SemanticSource};
    use crate::time::timebase::{MachTimebase, Timestamp};
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn make_test_raw_event(event_type: EventType, x: f64, y: f64) -> RawEvent {
        MachTimebase::init();
        RawEvent {
            timestamp: Timestamp::now(),
            event_type,
            coordinates: (x, y),
            cursor_state: CursorState::Arrow,
            key_code: None,
            character: None,
            modifiers: ModifierFlags::default(),
            scroll_delta: None,
            click_count: 0,
        }
    }

    fn make_test_raw_event_with_keyboard(key_code: u16, character: char) -> RawEvent {
        MachTimebase::init();
        RawEvent {
            timestamp: Timestamp::now(),
            event_type: EventType::KeyDown,
            coordinates: (0.0, 0.0),
            cursor_state: CursorState::Arrow,
            key_code: Some(key_code),
            character: Some(character),
            modifiers: ModifierFlags::default(),
            scroll_delta: None,
            click_count: 0,
        }
    }

    #[test]
    fn test_recording_creation() {
        let recording = Recording::new("test".to_string(), Some("Test goal".to_string()));
        assert_eq!(recording.metadata.name, "test");
        assert_eq!(recording.metadata.goal, Some("Test goal".to_string()));
        assert!(recording.is_empty());
    }

    #[test]
    fn test_add_events() {
        MachTimebase::init();
        let mut recording = Recording::new("test".to_string(), None);

        for i in 0..5 {
            let raw = make_test_raw_event(EventType::MouseMoved, i as f64, 0.0);
            recording.add_raw_event(raw);
        }

        assert_eq!(recording.len(), 5);

        // Add click event
        let click = make_test_raw_event(EventType::LeftMouseDown, 100.0, 100.0);
        recording.add_raw_event(click);

        assert_eq!(recording.click_events().len(), 1);
    }

    #[test]
    fn test_finalize() {
        let mut recording = Recording::new("test".to_string(), None);
        recording.finalize(1000);

        assert!(recording.metadata.ended_at.is_some());
        assert_eq!(recording.metadata.duration_ms, 1000);
    }

    #[test]
    fn test_serialization() {
        MachTimebase::init();
        let mut recording = Recording::new("test".to_string(), Some("Goal".to_string()));

        let raw = make_test_raw_event(EventType::LeftMouseDown, 100.0, 200.0);
        recording.add_raw_event(raw);
        recording.finalize(500);

        let json = serde_json::to_string(&recording).unwrap();
        let loaded: Recording = serde_json::from_str(&json).unwrap();

        assert_eq!(loaded.metadata.name, "test");
        assert_eq!(loaded.len(), 1);
    }

    #[test]
    fn test_recording_metadata_creation() {
        let metadata = RecordingMetadata::new("test_recording".to_string(), Some("Test goal".to_string()));
        assert_eq!(metadata.name, "test_recording");
        assert_eq!(metadata.goal, Some("Test goal".to_string()));
        assert!(metadata.ended_at.is_none());
        assert_eq!(metadata.event_count, 0);
        assert_eq!(metadata.duration_ms, 0);
        assert_eq!(metadata.format_version, "1.0");
    }

    #[test]
    fn test_recording_metadata_finalize() {
        let mut metadata = RecordingMetadata::new("test".to_string(), None);
        assert!(metadata.ended_at.is_none());

        metadata.finalize(42, 5000);

        assert!(metadata.ended_at.is_some());
        assert_eq!(metadata.event_count, 42);
        assert_eq!(metadata.duration_ms, 5000);
    }

    #[test]
    fn test_add_enriched_raw_event() {
        MachTimebase::init();
        let mut recording = Recording::new("test".to_string(), None);

        let raw = make_test_raw_event(EventType::LeftMouseDown, 100.0, 200.0);
        let semantic = SemanticContext {
            ax_role: Some("AXButton".to_string()),
            title: Some("Click Me".to_string()),
            ..Default::default()
        };

        recording.add_enriched_raw(raw, Some(semantic.clone()));

        assert_eq!(recording.len(), 1);
        assert!(recording.events[0].semantic.is_some());
        assert_eq!(
            recording.events[0].semantic.as_ref().unwrap().title,
            semantic.title
        );
    }

    #[test]
    fn test_events_of_type_filter() {
        MachTimebase::init();
        let mut recording = Recording::new("test".to_string(), None);

        recording.add_raw_event(make_test_raw_event(EventType::MouseMoved, 10.0, 10.0));
        recording.add_raw_event(make_test_raw_event(EventType::LeftMouseDown, 20.0, 20.0));
        recording.add_raw_event(make_test_raw_event(EventType::RightMouseDown, 30.0, 30.0));
        recording.add_raw_event(make_test_raw_event(EventType::KeyDown, 0.0, 0.0));

        let clicks = recording.events_of_type(|e| e.raw.event_type.is_click());
        assert_eq!(clicks.len(), 2);

        let mouse_moves = recording.events_of_type(|e| e.raw.event_type.is_mouse_move());
        assert_eq!(mouse_moves.len(), 1);
    }

    #[test]
    fn test_click_events_filter() {
        MachTimebase::init();
        let mut recording = Recording::new("test".to_string(), None);

        recording.add_raw_event(make_test_raw_event(EventType::LeftMouseDown, 10.0, 10.0));
        recording.add_raw_event(make_test_raw_event(EventType::LeftMouseUp, 10.0, 10.0));
        recording.add_raw_event(make_test_raw_event(EventType::MouseMoved, 20.0, 20.0));
        recording.add_raw_event(make_test_raw_event(EventType::RightMouseDown, 30.0, 30.0));
        recording.add_raw_event(make_test_raw_event(EventType::KeyDown, 0.0, 0.0));

        let clicks = recording.click_events();
        assert_eq!(clicks.len(), 3); // left down, left up, right down
    }

    #[test]
    fn test_keyboard_events_filter() {
        MachTimebase::init();
        let mut recording = Recording::new("test".to_string(), None);

        recording.add_raw_event(make_test_raw_event_with_keyboard(0, 'a'));
        recording.add_raw_event(make_test_raw_event_with_keyboard(1, 'b'));
        recording.add_raw_event(make_test_raw_event(EventType::LeftMouseDown, 10.0, 10.0));
        recording.add_raw_event(make_test_raw_event_with_keyboard(2, 'c'));

        let keyboard_events = recording.keyboard_events();
        assert_eq!(keyboard_events.len(), 3);
    }

    #[test]
    fn test_save_and_load_recording() {
        MachTimebase::init();
        let mut recording = Recording::new("save_test".to_string(), Some("Test save/load".to_string()));

        recording.add_raw_event(make_test_raw_event(EventType::LeftMouseDown, 100.0, 200.0));
        recording.add_raw_event(make_test_raw_event_with_keyboard(0, 'x'));
        recording.finalize(1500);

        // Create a temporary file
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        // Save recording
        recording.save(path).unwrap();

        // Load recording
        let loaded = Recording::load(path).unwrap();

        assert_eq!(loaded.metadata.name, "save_test");
        assert_eq!(loaded.metadata.goal, Some("Test save/load".to_string()));
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded.metadata.duration_ms, 1500);
        assert!(loaded.metadata.ended_at.is_some());
    }

    #[test]
    fn test_recording_default() {
        let recording = Recording::default();
        assert_eq!(recording.metadata.name, "untitled");
        assert!(recording.metadata.goal.is_none());
        assert!(recording.is_empty());
    }

    #[test]
    fn test_load_invalid_file() {
        let result = Recording::load(Path::new("/nonexistent/file.json"));
        assert!(result.is_err());
    }

    #[test]
    fn test_load_malformed_json() {
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(b"{ invalid json }").unwrap();
        temp_file.flush().unwrap();

        let result = Recording::load(temp_file.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_sequential_event_ordering() {
        MachTimebase::init();
        let mut recording = Recording::new("sequence_test".to_string(), None);

        // Add events in order
        recording.add_raw_event(make_test_raw_event(EventType::LeftMouseDown, 10.0, 10.0));
        recording.add_raw_event(make_test_raw_event(EventType::LeftMouseUp, 10.0, 10.0));
        recording.add_raw_event(make_test_raw_event(EventType::KeyDown, 0.0, 0.0));

        // Verify sequence numbers
        assert_eq!(recording.events[0].sequence, 0);
        assert_eq!(recording.events[1].sequence, 1);
        assert_eq!(recording.events[2].sequence, 2);
    }

    #[test]
    fn test_finalize_updates_metadata() {
        MachTimebase::init();
        let mut recording = Recording::new("finalize_test".to_string(), None);

        recording.add_raw_event(make_test_raw_event(EventType::LeftMouseDown, 10.0, 10.0));
        recording.add_raw_event(make_test_raw_event(EventType::LeftMouseUp, 10.0, 10.0));

        assert_eq!(recording.metadata.event_count, 0);
        assert_eq!(recording.metadata.duration_ms, 0);

        recording.finalize(2500);

        assert_eq!(recording.metadata.event_count, 2);
        assert_eq!(recording.metadata.duration_ms, 2500);
    }

    #[test]
    fn test_empty_recording_operations() {
        let recording = Recording::new("empty".to_string(), None);

        assert_eq!(recording.len(), 0);
        assert!(recording.is_empty());
        assert_eq!(recording.click_events().len(), 0);
        assert_eq!(recording.keyboard_events().len(), 0);
        assert_eq!(recording.events_of_type(|_| true).len(), 0);
    }

    #[test]
    fn test_recording_with_large_event_set() {
        MachTimebase::init();
        let mut recording = Recording::new("large_test".to_string(), None);

        // Add 1000 events
        for i in 0..1000 {
            let event_type = if i % 3 == 0 {
                EventType::LeftMouseDown
            } else if i % 3 == 1 {
                EventType::KeyDown
            } else {
                EventType::MouseMoved
            };
            recording.add_raw_event(make_test_raw_event(event_type, i as f64, i as f64));
        }

        assert_eq!(recording.len(), 1000);
        assert!(!recording.is_empty());

        // Verify filtering works correctly
        let clicks = recording.click_events();
        let keyboard = recording.keyboard_events();

        // Approximately 333 of each type
        assert!(clicks.len() >= 330 && clicks.len() <= 340);
        assert!(keyboard.len() >= 330 && keyboard.len() <= 340);
    }

    #[test]
    fn test_checkpoint_save_and_recover() {
        MachTimebase::init();
        let dir = tempfile::tempdir().unwrap();
        let final_path = dir.path().join("test_recording.json");

        let mut recording = Recording::new("checkpoint_test".to_string(), Some("Testing checkpoints".to_string()));
        recording.add_raw_event(make_test_raw_event(EventType::LeftMouseDown, 10.0, 20.0));
        recording.add_raw_event(make_test_raw_event(EventType::KeyDown, 0.0, 0.0));

        // Save checkpoint
        recording.save_checkpoint(&final_path).unwrap();

        // Checkpoint file should exist, final file should not
        let tmp_path = final_path.with_extension("json.tmp");
        assert!(tmp_path.exists());
        assert!(!final_path.exists());

        // Recover checkpoints from directory
        let recovered = Recording::recover_checkpoints(dir.path());
        assert_eq!(recovered.len(), 1);
        assert_eq!(recovered[0].1.metadata.name, "checkpoint_test");
        assert_eq!(recovered[0].1.len(), 2);
    }

    #[test]
    fn test_finalize_checkpoint_renames() {
        MachTimebase::init();
        let dir = tempfile::tempdir().unwrap();
        let final_path = dir.path().join("rename_test.json");

        let recording = Recording::new("rename_test".to_string(), None);
        recording.save_checkpoint(&final_path).unwrap();

        // Finalize: rename .tmp to .json
        Recording::finalize_checkpoint(&final_path).unwrap();

        assert!(final_path.exists());
        assert!(!final_path.with_extension("json.tmp").exists());

        // Load the finalized file
        let loaded = Recording::load(&final_path).unwrap();
        assert_eq!(loaded.metadata.name, "rename_test");
    }

    #[test]
    fn test_remove_checkpoint() {
        let dir = tempfile::tempdir().unwrap();
        let final_path = dir.path().join("remove_test.json");
        let tmp_path = final_path.with_extension("json.tmp");

        // Create a fake checkpoint file
        std::fs::write(&tmp_path, "{}").unwrap();
        assert!(tmp_path.exists());

        // Remove it
        Recording::remove_checkpoint(&final_path);
        assert!(!tmp_path.exists());
    }

    #[test]
    fn test_recover_empty_directory() {
        let dir = tempfile::tempdir().unwrap();
        let recovered = Recording::recover_checkpoints(dir.path());
        assert!(recovered.is_empty());
    }

    #[test]
    fn test_recover_ignores_invalid_tmp_files() {
        let dir = tempfile::tempdir().unwrap();
        // Write an invalid .tmp file
        std::fs::write(dir.path().join("bad.json.tmp"), "not valid json").unwrap();

        let recovered = Recording::recover_checkpoints(dir.path());
        assert!(recovered.is_empty());
    }

    #[test]
    fn test_checkpoint_path_helper() {
        let path = Path::new("/tmp/my_recording.json");
        assert_eq!(checkpoint_path(path), Path::new("/tmp/my_recording.json.tmp"));
    }

    #[test]
    fn test_recording_metadata_default() {
        let meta = RecordingMetadata::default();
        assert!(meta.name.is_empty());
        assert!(meta.goal.is_none());
        assert!(meta.app_context.is_none());
        assert!(meta.ended_at.is_none());
        assert_eq!(meta.event_count, 0);
        assert_eq!(meta.duration_ms, 0);
        assert_eq!(meta.format_version, CURRENT_FORMAT_VERSION);
    }

    #[test]
    fn test_backward_compat_metadata_missing_fields() {
        // Simulate a v0.x recording that lacked format_version and app_context
        let json = r#"{
            "id": "00000000-0000-0000-0000-000000000001",
            "name": "old_recording",
            "goal": null,
            "started_at": "2025-01-01T00:00:00Z",
            "ended_at": null,
            "event_count": 0,
            "duration_ms": 0
        }"#;
        let meta: RecordingMetadata = serde_json::from_str(json).unwrap();
        assert_eq!(meta.name, "old_recording");
        // Missing fields should get defaults
        assert!(meta.app_context.is_none());
        assert_eq!(meta.format_version, CURRENT_FORMAT_VERSION);
    }

    #[test]
    fn test_backward_compat_enriched_event_missing_fields() {
        // Simulate a recording event that's missing the `semantic` and `sequence` fields
        let json = r#"{
            "raw": {
                "timestamp": 1000,
                "event_type": "LeftMouseDown",
                "coordinates": [100.0, 200.0],
                "cursor_state": "Arrow",
                "key_code": null,
                "character": null,
                "modifiers": {"shift": false, "control": false, "option": false, "command": false, "caps_lock": false, "function": false},
                "scroll_delta": null,
                "click_count": 1
            },
            "id": "00000000-0000-0000-0000-000000000042"
        }"#;
        let event: EnrichedEvent = serde_json::from_str(json).unwrap();
        assert_eq!(event.raw.event_type, EventType::LeftMouseDown);
        assert!(event.semantic.is_none());
        assert_eq!(event.sequence, 0); // default
    }

    #[test]
    fn test_backward_compat_raw_event_missing_fields() {
        // Simulate a raw event missing optional fields that were added later
        let json = r#"{
            "timestamp": 500,
            "event_type": "KeyDown",
            "coordinates": [0.0, 0.0],
            "modifiers": {"shift": false, "control": false, "option": false, "command": false, "caps_lock": false, "function": false}
        }"#;
        let event: RawEvent = serde_json::from_str(json).unwrap();
        assert_eq!(event.event_type, EventType::KeyDown);
        assert_eq!(event.cursor_state, CursorState::Arrow); // default
        assert!(event.key_code.is_none());
        assert!(event.scroll_delta.is_none());
        assert_eq!(event.click_count, 0);
    }

    #[test]
    fn test_backward_compat_semantic_context_missing_fields() {
        // Simulate a semantic context from before ancestors/ocr_text were added
        let json = r#"{
            "ax_role": "AXButton",
            "title": "OK"
        }"#;
        let ctx: SemanticContext = serde_json::from_str(json).unwrap();
        assert_eq!(ctx.ax_role, Some("AXButton".to_string()));
        assert_eq!(ctx.title, Some("OK".to_string()));
        assert!(ctx.ocr_text.is_none());
        assert!(ctx.ancestors.is_empty());
        assert_eq!(ctx.confidence, 1.0);
        assert_eq!(ctx.source, SemanticSource::Accessibility);
    }

    #[test]
    fn test_version_mismatch_still_loads() {
        MachTimebase::init();
        let mut recording = Recording::new("versioned".to_string(), None);
        recording.add_raw_event(make_test_raw_event(EventType::LeftMouseDown, 10.0, 20.0));
        recording.metadata.format_version = "2.0".to_string();

        let temp_file = NamedTempFile::new().unwrap();
        recording.save(temp_file.path()).unwrap();

        // Loading a future version should still succeed (forward-compat via serde defaults)
        let loaded = Recording::load(temp_file.path()).unwrap();
        assert_eq!(loaded.metadata.format_version, "2.0");
        assert_eq!(loaded.len(), 1);
    }

    #[test]
    fn test_current_format_version_constant() {
        assert_eq!(CURRENT_FORMAT_VERSION, "1.0");
        let meta = RecordingMetadata::new("test".to_string(), None);
        assert_eq!(meta.format_version, CURRENT_FORMAT_VERSION);
    }
}
