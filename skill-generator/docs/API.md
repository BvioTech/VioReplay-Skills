# API Reference

This document describes the programmatic interface for integrating the Skill Generator into other tools.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Core Types](#core-types)
3. [Event Capture](#event-capture)
4. [Recording](#recording)
5. [Skill Generation](#skill-generation)
6. [Validation](#validation)
7. [Configuration](#configuration)
8. [Error Handling](#error-handling)

---

## Quick Start

```rust
use skill_generator::{Recording, SkillGenerator, MachTimebase};
use skill_generator::capture::types::{EventType, EnrichedEvent};

// Initialize the timebase (required once per process)
MachTimebase::init();

// Create a recording
let mut recording = Recording::new(
    "my_skill".to_string(),
    Some("Demonstrate a workflow".to_string()),
);

// ... add events to the recording ...

// Generate a skill
let generator = SkillGenerator::new();
let skill = generator.generate(&recording).expect("Failed to generate");

// Render to markdown
let markdown = generator.render_to_markdown(&skill);
println!("{}", markdown);
```

---

## Core Types

### `RawEvent`

Represents a raw input event as captured from the event tap.

```rust
pub struct RawEvent {
    /// Monotonic timestamp (mach_absolute_time ticks)
    pub timestamp: Timestamp,
    /// Event type
    pub event_type: EventType,
    /// Mouse coordinates (screen position)
    pub coordinates: (f64, f64),
    /// Cursor state at time of event
    pub cursor_state: CursorState,
    /// Keyboard key code (for key events)
    pub key_code: Option<u16>,
    /// Unicode character (for key events)
    pub character: Option<char>,
    /// Modifier flags
    pub modifiers: ModifierFlags,
    /// Scroll delta (for scroll events)
    pub scroll_delta: Option<(f64, f64)>,
    /// Click count (for mouse events)
    pub click_count: u8,
}
```

#### Factory Methods

```rust
// Create a mouse event
let event = RawEvent::mouse(
    timestamp,
    EventType::LeftMouseDown,
    x, y,
    CursorState::Arrow,
    ModifierFlags::default(),
    click_count,
);

// Create a keyboard event
let event = RawEvent::keyboard(
    timestamp,
    EventType::KeyDown,
    key_code,
    Some('a'),
    ModifierFlags::default(),
    cursor_position,
);

// Create a scroll event
let event = RawEvent::scroll(
    timestamp,
    x, y,
    delta_x, delta_y,
    ModifierFlags::default(),
);
```

### `EventType`

Enumeration of captured event types.

```rust
pub enum EventType {
    MouseMoved,
    LeftMouseDown,
    LeftMouseUp,
    RightMouseDown,
    RightMouseUp,
    ScrollWheel,
    KeyDown,
    KeyUp,
    FlagsChanged,
    LeftMouseDragged,
    RightMouseDragged,
    OtherMouseDown,
    OtherMouseUp,
    OtherMouseDragged,
}
```

#### Helper Methods

```rust
impl EventType {
    /// Check if this is a click event (down or up)
    pub fn is_click(&self) -> bool;
    
    /// Check if this is a mouse movement event
    pub fn is_mouse_move(&self) -> bool;
    
    /// Check if this is a keyboard event
    pub fn is_keyboard(&self) -> bool;
}
```

### `CursorState`

Current cursor appearance.

```rust
pub enum CursorState {
    Arrow,          // Standard arrow
    IBeam,          // Text input
    PointingHand,   // Clickable link/button
    OpenHand,       // Draggable
    ClosedHand,     // Dragging
    Crosshair,      // Precise selection
    ResizeLeftRight,
    ResizeUpDown,
    Wait,           // Busy
    Progress,       // Spinning
    NotAllowed,
    Unknown,
}
```

#### Helper Methods

```rust
impl CursorState {
    /// Check if cursor indicates system is busy
    pub fn is_busy(&self) -> bool;
    
    /// Check if cursor indicates text input context
    pub fn is_text_input(&self) -> bool;
    
    /// Check if cursor indicates clickable element
    pub fn is_clickable(&self) -> bool;
}
```

### `ModifierFlags`

Keyboard modifier state.

```rust
pub struct ModifierFlags {
    pub shift: bool,
    pub control: bool,
    pub option: bool,
    pub command: bool,
    pub caps_lock: bool,
    pub function: bool,
}
```

#### Factory Methods

```rust
// Create from CGEventFlags bitmask
let flags = ModifierFlags::from_cg_flags(cg_flags);

// Check if any modifier is active
if flags.any_active() {
    // Handle keyboard shortcut
}
```

### `EnrichedEvent`

Event with semantic context attached.

```rust
pub struct EnrichedEvent {
    /// The raw event data
    pub raw: RawEvent,
    /// Semantic context (if available)
    pub semantic: Option<SemanticContext>,
    /// Unique event ID
    pub id: uuid::Uuid,
    /// Sequence number in recording
    pub sequence: u64,
}
```

### `SemanticContext`

Accessibility information about the UI element at event location.

```rust
pub struct SemanticContext {
    /// Accessibility role (AXButton, AXTextField, etc.)
    pub ax_role: Option<String>,
    /// Element title/label
    pub title: Option<String>,
    /// Element identifier
    pub identifier: Option<String>,
    /// Element value (for text fields, etc.)
    pub value: Option<String>,
    /// Parent element role
    pub parent_role: Option<String>,
    /// Parent element title
    pub parent_title: Option<String>,
    /// Window title
    pub window_title: Option<String>,
    /// Application bundle ID
    pub app_bundle_id: Option<String>,
    /// Application name
    pub app_name: Option<String>,
    /// Process ID
    pub pid: Option<i32>,
    /// Window ID
    pub window_id: Option<u32>,
    /// Element frame (x, y, width, height)
    pub frame: Option<(f64, f64, f64, f64)>,
    /// Source of this semantic data
    pub source: SemanticSource,
    /// Confidence score (0.0-1.0)
    pub confidence: f32,
    /// OCR text (if from vision fallback)
    pub ocr_text: Option<String>,
    /// Ancestor chain (role, title) pairs
    pub ancestors: Vec<(String, Option<String>)>,
}
```

### `SemanticSource`

Origin of semantic data.

```rust
pub enum SemanticSource {
    Accessibility,          // From macOS Accessibility API
    Vision,                 // From OCR/Vision fallback
    Inferred,               // From context reconstruction
    InjectedAccessibility,  // From AXManualAccessibility
}
```

---

## Event Capture

### `EventRingBuffer`

Lock-free SPSC ring buffer for event capture.

```rust
use skill_generator::EventRingBuffer;

// Create with default capacity (8192)
let buffer = EventRingBuffer::new();

// Create with custom capacity (must be power of 2)
let buffer = EventRingBuffer::with_capacity(4096);

// Split into producer and consumer
let (producer, consumer) = buffer.split();

// Get statistics
let stats = buffer.stats();
```

### `EventProducer`

Producer half for pushing events.

```rust
impl EventProducer {
    /// Push an event (never blocks, may drop if full)
    pub fn push(&mut self, event: RawEvent) -> bool;
    
    /// Try to push, returning sequence number or original event
    pub fn try_push(&mut self, event: RawEvent) -> Result<u64, RawEvent>;
    
    /// Check available slots
    pub fn available_slots(&self) -> usize;
    
    /// Check if buffer is full
    pub fn is_full(&self) -> bool;
    
    /// Get current sequence number
    pub fn sequence(&self) -> u64;
}
```

### `EventConsumer`

Consumer half for reading events.

```rust
impl EventConsumer {
    /// Pop an event
    pub fn pop(&mut self) -> Option<EventSlot>;
    
    /// Peek at next event without removing
    pub fn peek(&self) -> Option<&EventSlot>;
    
    /// Check if empty
    pub fn is_empty(&self) -> bool;
    
    /// Get number of available events
    pub fn available(&self) -> usize;
    
    /// Pop multiple events at once
    pub fn pop_batch(&mut self, max_count: usize) -> Vec<EventSlot>;
}
```

### `RingBufferStats`

Statistics for monitoring.

```rust
pub struct RingBufferStats {
    pub events_pushed: AtomicU64,
    pub events_dropped: AtomicU64,
    pub events_consumed: AtomicU64,
    pub peak_occupancy: AtomicU64,
}
```

### `MachTimebase`

High-precision timing utilities.

```rust
use skill_generator::MachTimebase;

// Initialize (required once per process)
MachTimebase::init();

// Get current time
let timestamp = Timestamp::now();

// Convert ticks to nanoseconds
let nanos = MachTimebase::ticks_to_nanos(ticks);

// Calculate elapsed time
let elapsed_ns = MachTimebase::elapsed_nanos(start_ticks, end_ticks);
let elapsed_ms = MachTimebase::elapsed_millis(start_ticks, end_ticks);
```

### `Timestamp`

Wrapper for mach_absolute_time values.

```rust
impl Timestamp {
    /// Get current timestamp
    pub fn now() -> Self;
    
    /// Create from raw ticks
    pub fn from_ticks(ticks: u64) -> Self;
    
    /// Get raw ticks
    pub fn ticks(&self) -> u64;
    
    /// Convert to nanoseconds
    pub fn to_nanos(&self) -> u64;
}
```

---

## Recording

### `Recording`

Stores captured events with metadata.

```rust
use skill_generator::Recording;

// Create new recording
let mut recording = Recording::new(
    "my_workflow".to_string(),
    Some("Click buttons and fill forms".to_string()),
);

// Add events
recording.add_event(enriched_event);
recording.add_raw_event(raw_event);
recording.add_enriched_raw(raw_event, Some(semantic_context));

// Finalize
recording.finalize(duration_ms);

// Save to file
recording.save(&PathBuf::from("recording.json"))?;

// Load from file
let loaded = Recording::load(&PathBuf::from("recording.json"))?;

// Query events
let click_events = recording.click_events();
let keyboard_events = recording.keyboard_events();
let filtered = recording.events_of_type(|e| e.raw.event_type == EventType::ScrollWheel);
```

### `RecordingMetadata`

Metadata about the recording session.

```rust
pub struct RecordingMetadata {
    pub id: Uuid,
    pub name: String,
    pub goal: Option<String>,
    pub app_context: Option<String>,
    pub started_at: DateTime<Utc>,
    pub ended_at: Option<DateTime<Utc>>,
    pub event_count: usize,
    pub duration_ms: u64,
    pub format_version: String,
}
```

---

## Skill Generation

### `SkillGenerator`

Main interface for generating skills from recordings.

```rust
use skill_generator::SkillGenerator;
use skill_generator::workflow::generator::GeneratorConfig;

// Create with default config
let generator = SkillGenerator::new();

// Create with custom config
let config = GeneratorConfig {
    min_step_confidence: 0.7,
    include_verification: true,
    extract_variables: true,
    selector_chain_depth: 3,
    use_llm_semantic: true,
    api_key: None,
};
let generator = SkillGenerator::with_config(config);

// Generate skill from recording
let skill = generator.generate(&recording)?;

// Render to markdown
let markdown = generator.render_to_markdown(&skill);

// Validate
let validation = generator.validate(&skill);

// Save to file
generator.save_skill(&skill, &PathBuf::from("SKILL.md"))?;
```

### `GeneratorConfig`

Configuration for skill generation.

```rust
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
```

### `GeneratedSkill`

A complete generated skill.

```rust
pub struct GeneratedSkill {
    pub name: String,
    pub description: String,
    pub context: String,                    // "fork" or "inline"
    pub allowed_tools: Vec<String>,
    pub variables: Vec<ExtractedVariable>,
    pub steps: Vec<GeneratedStep>,
    pub source_recording_id: Uuid,
}
```

### `GeneratedStep`

A single step in the skill.

```rust
pub struct GeneratedStep {
    pub number: usize,
    pub description: String,
    pub action: Action,
    pub selector: Option<RankedSelector>,
    pub fallback_selectors: Vec<RankedSelector>,
    pub variables: Vec<ExtractedVariable>,
    pub verification: Option<VerificationBlock>,
    pub source_events: Vec<usize>,
    pub confidence: f32,
}
```

### `Action`

Inferred user action.

```rust
pub enum Action {
    Click { element_name: String, element_role: String, confidence: f32 },
    DoubleClick { element_name: String, element_role: String, confidence: f32 },
    RightClick { element_name: String, element_role: String, confidence: f32 },
    Fill { field_name: String, value: String, confidence: f32 },
    Select { menu_name: String, option: String, confidence: f32 },
    Type { text: String, confidence: f32 },
    Shortcut { keys: Vec<String>, confidence: f32 },
    Scroll { direction: ScrollDirection, magnitude: f64, confidence: f32 },
    DragDrop { source: String, target: String, confidence: f32 },
    Search { candidate_elements: Vec<String>, confidence: f32 },
    Unknown { description: String, confidence: f32 },
}

impl Action {
    /// Get confidence score
    pub fn confidence(&self) -> f32;
    
    /// Get human-readable description
    pub fn description(&self) -> String;
}
```

### `ExtractedVariable`

A detected variable for parameterization.

```rust
pub struct ExtractedVariable {
    pub name: String,
    pub detected_value: String,
    pub source: VariableSource,
    pub confidence: f32,
    pub type_hint: Option<String>,
    pub description: Option<String>,
}

pub enum VariableSource {
    GoalMatch,      // Matched from goal text
    TypedString,    // High-entropy typed string
    Implicit,       // Matches current date, etc.
    LlmInferred,    // LLM inference
}
```

### `RankedSelector`

UI element selector with stability ranking.

```rust
pub struct RankedSelector {
    pub selector_type: SelectorType,
    pub value: String,
    pub stability: f32,
    pub specificity: f32,
    pub rank_score: f32,
}

pub enum SelectorType {
    AxIdentifier,
    TextContent,
    CssSelector,
    XPath,
    RelativePosition,
}
```

---

## Validation

### `SkillValidator`

Validates generated skills.

```rust
use skill_generator::codegen::validation::SkillValidator;

let validator = SkillValidator::new();

// Validate a compiled skill
let result = validator.validate(&skill);

// Validate markdown directly
let result = validator.validate_markdown(&markdown_content);

if result.passed {
    println!("Validation passed!");
} else {
    for error in &result.errors {
        println!("Error: {:?} - {}", error.error_type, error.message);
    }
}
```

### `ValidationResult`

Result of validation.

```rust
pub struct ValidationResult {
    pub passed: bool,
    pub errors: Vec<ValidationError>,
    pub warnings: Vec<String>,
}
```

### `ValidationError`

A validation error.

```rust
pub struct ValidationError {
    pub error_type: ValidationErrorType,
    pub message: String,
    pub location: Option<String>,
}

pub enum ValidationErrorType {
    YamlSyntax,
    HardcodedLiteral,
    UndefinedVariable,
    MissingVerification,
    ArgumentMismatch,
    InvalidFrontmatter,
}
```

---

## Configuration

### `Config`

Application configuration.

```rust
use skill_generator::app::config::Config;

// Load from file
let config = Config::load(&PathBuf::from("config.toml"))?;

// Load from default location (~/.skill_generator/config.toml)
let config = Config::load_default()?;

// Save to file
config.save(&PathBuf::from("config.toml"))?;

// Get default path
let path = Config::default_path();

// Serialize to TOML
let toml = config.to_toml();
```

### Configuration Sections

```rust
pub struct Config {
    pub capture: CaptureConfig,
    pub analysis: AnalysisConfig,
    pub codegen: CodegenConfig,
}

pub struct CaptureConfig {
    pub ring_buffer_size: usize,    // Must be power of 2
    pub sampling_rate_hz: u32,      // Screenshot sampling (0 = disabled)
    pub vision_fallback: bool,      // Enable vision fallback
}

pub struct AnalysisConfig {
    pub rdp_epsilon_px: f64,        // RDP simplification threshold
    pub hesitation_threshold: f64,  // GOMS mental operator threshold
    pub min_pause_ms: u64,          // Minimum pause for chunk boundary
}

pub struct CodegenConfig {
    pub model: String,              // Claude model for synthesis
    pub temperature: f32,           // LLM temperature
    pub include_screenshots: bool,  // Include screenshots in skill
}
```

---

## Error Handling

### `Error`

Main error type.

```rust
pub enum Error {
    Capture(String),
    RingBuffer(String),
    Accessibility(String),
    Vision(String),
    Analysis(String),
    Chunking(String),
    Synthesis(String),
    Codegen(String),
    Validation(String),
    Config(String),
    Io(std::io::Error),
    Serialization(serde_json::Error),
}
```

### `Result`

Type alias for results.

```rust
pub type Result<T> = std::result::Result<T, Error>;
```

### `GeneratorError`

Skill generation specific errors.

```rust
pub enum GeneratorError {
    NoSignificantEvents,
    ValidationFailed(Vec<String>),
    IoError(String),
}
```

---

## Example: Complete Workflow

```rust
use skill_generator::{
    Recording, SkillGenerator, MachTimebase,
    capture::types::{RawEvent, EventType, CursorState, ModifierFlags, SemanticContext, SemanticSource},
    time::timebase::Timestamp,
};

fn main() -> anyhow::Result<()> {
    // Initialize
    MachTimebase::init();
    
    // Create recording
    let mut recording = Recording::new(
        "login_workflow".to_string(),
        Some("Log in to the application".to_string()),
    );
    
    // Simulate captured events
    let mut timestamp = Timestamp::now();
    
    // Click username field
    let mut click_event = skill_generator::capture::types::EnrichedEvent::new(
        RawEvent::mouse(
            timestamp,
            EventType::LeftMouseDown,
            400.0, 200.0,
            CursorState::IBeam,
            ModifierFlags::default(),
            1,
        ),
        0,
    );
    click_event.semantic = Some(SemanticContext {
        ax_role: Some("AXTextField".to_string()),
        title: Some("Username".to_string()),
        source: SemanticSource::Accessibility,
        confidence: 0.95,
        ..Default::default()
    });
    recording.add_event(click_event);
    
    // Type username
    for c in "admin".chars() {
        let key_event = skill_generator::capture::types::EnrichedEvent::new(
            RawEvent::keyboard(
                Timestamp::now(),
                EventType::KeyDown,
                0,
                Some(c),
                ModifierFlags::default(),
                (400.0, 200.0),
            ),
            recording.len() as u64,
        );
        recording.add_event(key_event);
    }
    
    // Click login button
    let mut login_click = skill_generator::capture::types::EnrichedEvent::new(
        RawEvent::mouse(
            Timestamp::now(),
            EventType::LeftMouseDown,
            400.0, 350.0,
            CursorState::PointingHand,
            ModifierFlags::default(),
            1,
        ),
        recording.len() as u64,
    );
    login_click.semantic = Some(SemanticContext {
        ax_role: Some("AXButton".to_string()),
        title: Some("Login".to_string()),
        identifier: Some("login-btn".to_string()),
        source: SemanticSource::Accessibility,
        confidence: 0.95,
        ..Default::default()
    });
    recording.add_event(login_click);
    
    // Finalize recording
    recording.finalize(5000);
    
    // Generate skill
    let generator = SkillGenerator::new();
    let skill = generator.generate(&recording)?;
    
    // Validate
    let validation = generator.validate(&skill);
    if !validation.passed {
        for error in &validation.errors {
            eprintln!("Validation error: {}", error.message);
        }
    }
    
    // Render to markdown
    let markdown = generator.render_to_markdown(&skill);
    println!("{}", markdown);
    
    // Save to file
    std::fs::write("login_workflow/SKILL.md", &markdown)?;
    
    Ok(())
}
```
