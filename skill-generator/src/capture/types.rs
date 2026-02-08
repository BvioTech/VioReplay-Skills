//! Core types for event capture
//!
//! Defines the fundamental data structures used throughout the capture pipeline.

use crate::time::timebase::Timestamp;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU8, Ordering};

/// Event types captured by the system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum EventType {
    /// Mouse moved (includes drag)
    MouseMoved = 0,
    /// Left mouse button pressed
    LeftMouseDown = 1,
    /// Left mouse button released
    LeftMouseUp = 2,
    /// Right mouse button pressed
    RightMouseDown = 3,
    /// Right mouse button released
    RightMouseUp = 4,
    /// Mouse scroll (wheel or trackpad)
    ScrollWheel = 5,
    /// Key pressed
    KeyDown = 6,
    /// Key released
    KeyUp = 7,
    /// Modifier flags changed (shift, ctrl, etc.)
    FlagsChanged = 8,
    /// Left mouse dragged
    LeftMouseDragged = 9,
    /// Right mouse dragged
    RightMouseDragged = 10,
    /// Other mouse button pressed (middle click, etc.)
    OtherMouseDown = 11,
    /// Other mouse button released
    OtherMouseUp = 12,
    /// Other mouse button dragged
    OtherMouseDragged = 13,
}

impl EventType {
    /// Check if this is a click event (down or up)
    pub fn is_click(&self) -> bool {
        matches!(
            self,
            EventType::LeftMouseDown
                | EventType::LeftMouseUp
                | EventType::RightMouseDown
                | EventType::RightMouseUp
                | EventType::OtherMouseDown
                | EventType::OtherMouseUp
        )
    }

    /// Check if this is a mouse movement event
    pub fn is_mouse_move(&self) -> bool {
        matches!(
            self,
            EventType::MouseMoved
                | EventType::LeftMouseDragged
                | EventType::RightMouseDragged
                | EventType::OtherMouseDragged
        )
    }

    /// Check if this is a keyboard event
    pub fn is_keyboard(&self) -> bool {
        matches!(
            self,
            EventType::KeyDown | EventType::KeyUp | EventType::FlagsChanged
        )
    }
}

impl TryFrom<u32> for EventType {
    type Error = ();

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        // CGEventType values from Core Graphics
        match value {
            5 => Ok(EventType::MouseMoved), // kCGEventMouseMoved
            1 => Ok(EventType::LeftMouseDown), // kCGEventLeftMouseDown
            2 => Ok(EventType::LeftMouseUp), // kCGEventLeftMouseUp
            3 => Ok(EventType::RightMouseDown), // kCGEventRightMouseDown
            4 => Ok(EventType::RightMouseUp), // kCGEventRightMouseUp
            22 => Ok(EventType::ScrollWheel), // kCGEventScrollWheel
            10 => Ok(EventType::KeyDown),   // kCGEventKeyDown
            11 => Ok(EventType::KeyUp),     // kCGEventKeyUp
            12 => Ok(EventType::FlagsChanged), // kCGEventFlagsChanged
            6 => Ok(EventType::LeftMouseDragged), // kCGEventLeftMouseDragged
            7 => Ok(EventType::RightMouseDragged), // kCGEventRightMouseDragged
            25 => Ok(EventType::OtherMouseDown), // kCGEventOtherMouseDown
            26 => Ok(EventType::OtherMouseUp),   // kCGEventOtherMouseUp
            27 => Ok(EventType::OtherMouseDragged), // kCGEventOtherMouseDragged
            _ => Err(()),
        }
    }
}

/// Cursor state/glyph for context inference
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[repr(u8)]
pub enum CursorState {
    /// Standard arrow cursor
    #[default]
    Arrow = 0,
    /// I-beam cursor (text input)
    IBeam = 1,
    /// Pointing hand (clickable link/button)
    PointingHand = 2,
    /// Open hand (draggable)
    OpenHand = 3,
    /// Closed hand (dragging)
    ClosedHand = 4,
    /// Crosshair (precise selection)
    Crosshair = 5,
    /// Resize horizontal
    ResizeLeftRight = 6,
    /// Resize vertical
    ResizeUpDown = 7,
    /// Wait/busy cursor
    Wait = 8,
    /// Progress/spinning cursor
    Progress = 9,
    /// Not allowed
    NotAllowed = 10,
    /// Unknown cursor type
    Unknown = 255,
}

impl CursorState {
    /// Check if cursor indicates system is busy
    pub fn is_busy(&self) -> bool {
        matches!(self, CursorState::Wait | CursorState::Progress)
    }

    /// Check if cursor indicates text input context
    pub fn is_text_input(&self) -> bool {
        matches!(self, CursorState::IBeam)
    }

    /// Check if cursor indicates clickable element
    pub fn is_clickable(&self) -> bool {
        matches!(self, CursorState::PointingHand)
    }
}

/// Semantic state of an event slot
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum SemanticState {
    /// Slot is empty/unused
    Empty = 0,
    /// Event captured, semantic data pending
    Pending = 1,
    /// Semantic data successfully filled
    Filled = 2,
    /// Semantic lookup failed
    Failed = 3,
}

impl SemanticState {
    pub fn from_u8(value: u8) -> Self {
        match value {
            0 => SemanticState::Empty,
            1 => SemanticState::Pending,
            2 => SemanticState::Filled,
            3 => SemanticState::Failed,
            _ => SemanticState::Empty,
        }
    }
}

/// Source of semantic data
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SemanticSource {
    /// Data from Accessibility API
    Accessibility,
    /// Data from Vision/OCR fallback
    Vision,
    /// Data inferred from context
    Inferred,
    /// Data from AXManualAccessibility injection
    InjectedAccessibility,
}

/// Semantic context extracted from accessibility/vision
#[derive(Debug, Clone, Serialize, Deserialize)]
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

impl Default for SemanticContext {
    fn default() -> Self {
        Self {
            ax_role: None,
            title: None,
            identifier: None,
            value: None,
            parent_role: None,
            parent_title: None,
            window_title: None,
            app_bundle_id: None,
            app_name: None,
            pid: None,
            window_id: None,
            frame: None,
            source: SemanticSource::Accessibility,
            confidence: 1.0,
            ocr_text: None,
            ancestors: Vec::new(),
        }
    }
}

/// Keyboard modifier flags
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct ModifierFlags {
    pub shift: bool,
    pub control: bool,
    pub option: bool,
    pub command: bool,
    pub caps_lock: bool,
    pub function: bool,
}

impl ModifierFlags {
    /// Create from CGEventFlags bitmask
    pub fn from_cg_flags(flags: u64) -> Self {
        Self {
            shift: (flags & 0x00020000) != 0,      // kCGEventFlagMaskShift
            control: (flags & 0x00040000) != 0,   // kCGEventFlagMaskControl
            option: (flags & 0x00080000) != 0,    // kCGEventFlagMaskAlternate
            command: (flags & 0x00100000) != 0,   // kCGEventFlagMaskCommand
            caps_lock: (flags & 0x00010000) != 0, // kCGEventFlagMaskAlphaShift
            function: (flags & 0x00800000) != 0,  // kCGEventFlagMaskSecondaryFn
        }
    }

    /// Check if any modifier is active
    pub fn any_active(&self) -> bool {
        self.shift || self.control || self.option || self.command
    }
}

/// Raw event as captured from the event tap
#[derive(Debug, Clone, Serialize, Deserialize)]
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

impl RawEvent {
    /// Create a new mouse event
    pub fn mouse(
        timestamp: Timestamp,
        event_type: EventType,
        x: f64,
        y: f64,
        cursor_state: CursorState,
        modifiers: ModifierFlags,
        click_count: u8,
    ) -> Self {
        Self {
            timestamp,
            event_type,
            coordinates: (x, y),
            cursor_state,
            key_code: None,
            character: None,
            modifiers,
            scroll_delta: None,
            click_count,
        }
    }

    /// Create a new keyboard event
    pub fn keyboard(
        timestamp: Timestamp,
        event_type: EventType,
        key_code: u16,
        character: Option<char>,
        modifiers: ModifierFlags,
        cursor_pos: (f64, f64),
    ) -> Self {
        Self {
            timestamp,
            event_type,
            coordinates: cursor_pos,
            cursor_state: CursorState::Arrow,
            key_code: Some(key_code),
            character,
            modifiers,
            scroll_delta: None,
            click_count: 0,
        }
    }

    /// Create a scroll event
    pub fn scroll(
        timestamp: Timestamp,
        x: f64,
        y: f64,
        delta_x: f64,
        delta_y: f64,
        modifiers: ModifierFlags,
    ) -> Self {
        Self {
            timestamp,
            event_type: EventType::ScrollWheel,
            coordinates: (x, y),
            cursor_state: CursorState::Arrow,
            key_code: None,
            character: None,
            modifiers,
            scroll_delta: Some((delta_x, delta_y)),
            click_count: 0,
        }
    }
}

/// Event with associated semantic context
#[derive(Debug, Clone, Serialize, Deserialize)]
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

impl EnrichedEvent {
    /// Create a new enriched event
    pub fn new(raw: RawEvent, sequence: u64) -> Self {
        Self {
            raw,
            semantic: None,
            id: uuid::Uuid::new_v4(),
            sequence,
        }
    }

    /// Add semantic context
    pub fn with_semantic(mut self, semantic: SemanticContext) -> Self {
        self.semantic = Some(semantic);
        self
    }
}

/// Atomic wrapper for SemanticState
#[derive(Debug)]
pub struct AtomicSemanticState(AtomicU8);

impl AtomicSemanticState {
    pub const fn new(state: SemanticState) -> Self {
        Self(AtomicU8::new(state as u8))
    }

    pub fn load(&self, ordering: Ordering) -> SemanticState {
        SemanticState::from_u8(self.0.load(ordering))
    }

    pub fn store(&self, state: SemanticState, ordering: Ordering) {
        self.0.store(state as u8, ordering);
    }

    pub fn compare_exchange(
        &self,
        current: SemanticState,
        new: SemanticState,
        success: Ordering,
        failure: Ordering,
    ) -> Result<SemanticState, SemanticState> {
        self.0
            .compare_exchange(current as u8, new as u8, success, failure)
            .map(SemanticState::from_u8)
            .map_err(SemanticState::from_u8)
    }
}

impl Default for AtomicSemanticState {
    fn default() -> Self {
        Self::new(SemanticState::Empty)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time::timebase::Timestamp;

    #[test]
    fn test_event_type_conversion() {
        assert_eq!(EventType::try_from(5u32), Ok(EventType::MouseMoved));
        assert_eq!(EventType::try_from(1u32), Ok(EventType::LeftMouseDown));
        assert_eq!(EventType::try_from(10u32), Ok(EventType::KeyDown));
        assert!(EventType::try_from(999u32).is_err());
    }

    #[test]
    fn test_event_type_categories() {
        assert!(EventType::LeftMouseDown.is_click());
        assert!(!EventType::MouseMoved.is_click());
        assert!(EventType::MouseMoved.is_mouse_move());
        assert!(EventType::KeyDown.is_keyboard());
    }

    #[test]
    fn test_cursor_state() {
        assert!(CursorState::Wait.is_busy());
        assert!(CursorState::Progress.is_busy());
        assert!(!CursorState::Arrow.is_busy());
        assert!(CursorState::IBeam.is_text_input());
        assert!(CursorState::PointingHand.is_clickable());
    }

    #[test]
    fn test_modifier_flags() {
        let flags = ModifierFlags::from_cg_flags(0x00120000); // Shift + Command
        assert!(flags.shift);
        assert!(flags.command);
        assert!(!flags.control);
        assert!(flags.any_active());
    }

    #[test]
    fn test_atomic_semantic_state() {
        let state = AtomicSemanticState::new(SemanticState::Empty);
        assert_eq!(state.load(Ordering::SeqCst), SemanticState::Empty);

        state.store(SemanticState::Pending, Ordering::SeqCst);
        assert_eq!(state.load(Ordering::SeqCst), SemanticState::Pending);

        let result = state.compare_exchange(
            SemanticState::Pending,
            SemanticState::Filled,
            Ordering::SeqCst,
            Ordering::SeqCst,
        );
        assert_eq!(result, Ok(SemanticState::Pending));
        assert_eq!(state.load(Ordering::SeqCst), SemanticState::Filled);
    }

    #[test]
    fn test_event_type_all_conversions() {
        assert_eq!(EventType::try_from(5u32), Ok(EventType::MouseMoved));
        assert_eq!(EventType::try_from(1u32), Ok(EventType::LeftMouseDown));
        assert_eq!(EventType::try_from(2u32), Ok(EventType::LeftMouseUp));
        assert_eq!(EventType::try_from(3u32), Ok(EventType::RightMouseDown));
        assert_eq!(EventType::try_from(4u32), Ok(EventType::RightMouseUp));
        assert_eq!(EventType::try_from(22u32), Ok(EventType::ScrollWheel));
        assert_eq!(EventType::try_from(10u32), Ok(EventType::KeyDown));
        assert_eq!(EventType::try_from(11u32), Ok(EventType::KeyUp));
        assert_eq!(EventType::try_from(12u32), Ok(EventType::FlagsChanged));
        assert_eq!(EventType::try_from(6u32), Ok(EventType::LeftMouseDragged));
        assert_eq!(EventType::try_from(7u32), Ok(EventType::RightMouseDragged));
        assert_eq!(EventType::try_from(25u32), Ok(EventType::OtherMouseDown));
        assert_eq!(EventType::try_from(26u32), Ok(EventType::OtherMouseUp));
        assert_eq!(EventType::try_from(27u32), Ok(EventType::OtherMouseDragged));
    }

    #[test]
    fn test_event_type_click_detection() {
        assert!(EventType::LeftMouseDown.is_click());
        assert!(EventType::LeftMouseUp.is_click());
        assert!(EventType::RightMouseDown.is_click());
        assert!(EventType::RightMouseUp.is_click());
        assert!(EventType::OtherMouseDown.is_click());
        assert!(EventType::OtherMouseUp.is_click());

        assert!(!EventType::MouseMoved.is_click());
        assert!(!EventType::LeftMouseDragged.is_click());
        assert!(!EventType::KeyDown.is_click());
    }

    #[test]
    fn test_event_type_mouse_move_detection() {
        assert!(EventType::MouseMoved.is_mouse_move());
        assert!(EventType::LeftMouseDragged.is_mouse_move());
        assert!(EventType::RightMouseDragged.is_mouse_move());
        assert!(EventType::OtherMouseDragged.is_mouse_move());

        assert!(!EventType::LeftMouseDown.is_mouse_move());
        assert!(!EventType::KeyDown.is_mouse_move());
    }

    #[test]
    fn test_event_type_keyboard_detection() {
        assert!(EventType::KeyDown.is_keyboard());
        assert!(EventType::KeyUp.is_keyboard());
        assert!(EventType::FlagsChanged.is_keyboard());

        assert!(!EventType::MouseMoved.is_keyboard());
        assert!(!EventType::LeftMouseDown.is_keyboard());
    }

    #[test]
    fn test_cursor_state_default() {
        let cursor = CursorState::default();
        assert_eq!(cursor, CursorState::Arrow);
    }

    #[test]
    fn test_cursor_state_all_types() {
        assert!(!CursorState::Arrow.is_busy());
        assert!(!CursorState::Arrow.is_text_input());
        assert!(!CursorState::Arrow.is_clickable());

        assert!(CursorState::IBeam.is_text_input());
        assert!(!CursorState::IBeam.is_busy());

        assert!(CursorState::PointingHand.is_clickable());
        assert!(!CursorState::PointingHand.is_busy());

        assert!(CursorState::Wait.is_busy());
        assert!(CursorState::Progress.is_busy());
        assert!(!CursorState::Unknown.is_busy());
    }

    #[test]
    fn test_semantic_state_from_u8() {
        assert_eq!(SemanticState::from_u8(0), SemanticState::Empty);
        assert_eq!(SemanticState::from_u8(1), SemanticState::Pending);
        assert_eq!(SemanticState::from_u8(2), SemanticState::Filled);
        assert_eq!(SemanticState::from_u8(3), SemanticState::Failed);
        assert_eq!(SemanticState::from_u8(99), SemanticState::Empty);
    }

    #[test]
    fn test_modifier_flags_default() {
        let flags = ModifierFlags::default();
        assert!(!flags.shift);
        assert!(!flags.control);
        assert!(!flags.option);
        assert!(!flags.command);
        assert!(!flags.caps_lock);
        assert!(!flags.function);
        assert!(!flags.any_active());
    }

    #[test]
    fn test_modifier_flags_from_cg_flags() {
        // Test shift
        let flags = ModifierFlags::from_cg_flags(0x00020000);
        assert!(flags.shift);
        assert!(!flags.control);
        assert!(flags.any_active());

        // Test control
        let flags = ModifierFlags::from_cg_flags(0x00040000);
        assert!(flags.control);
        assert!(!flags.shift);

        // Test option
        let flags = ModifierFlags::from_cg_flags(0x00080000);
        assert!(flags.option);

        // Test command
        let flags = ModifierFlags::from_cg_flags(0x00100000);
        assert!(flags.command);

        // Test caps lock
        let flags = ModifierFlags::from_cg_flags(0x00010000);
        assert!(flags.caps_lock);

        // Test function
        let flags = ModifierFlags::from_cg_flags(0x00800000);
        assert!(flags.function);

        // Test multiple modifiers
        let flags = ModifierFlags::from_cg_flags(0x00120000);
        assert!(flags.shift);
        assert!(flags.command);
        assert!(flags.any_active());
    }

    #[test]
    fn test_semantic_context_default() {
        let ctx = SemanticContext::default();
        assert!(ctx.ax_role.is_none());
        assert!(ctx.title.is_none());
        assert_eq!(ctx.source, SemanticSource::Accessibility);
        assert_eq!(ctx.confidence, 1.0);
        assert!(ctx.ancestors.is_empty());
    }

    #[test]
    fn test_raw_event_mouse() {
        let event = RawEvent::mouse(
            Timestamp::from_ticks(1000),
            EventType::LeftMouseDown,
            100.0,
            200.0,
            CursorState::Arrow,
            ModifierFlags::default(),
            1,
        );

        assert_eq!(event.event_type, EventType::LeftMouseDown);
        assert_eq!(event.coordinates, (100.0, 200.0));
        assert_eq!(event.cursor_state, CursorState::Arrow);
        assert_eq!(event.click_count, 1);
        assert!(event.key_code.is_none());
        assert!(event.scroll_delta.is_none());
    }

    #[test]
    fn test_raw_event_keyboard() {
        let event = RawEvent::keyboard(
            Timestamp::from_ticks(1000),
            EventType::KeyDown,
            42,
            Some('a'),
            ModifierFlags::default(),
            (100.0, 200.0),
        );

        assert_eq!(event.event_type, EventType::KeyDown);
        assert_eq!(event.key_code, Some(42));
        assert_eq!(event.character, Some('a'));
        assert_eq!(event.cursor_state, CursorState::Arrow);
        assert!(event.scroll_delta.is_none());
    }

    #[test]
    fn test_raw_event_scroll() {
        let event = RawEvent::scroll(
            Timestamp::from_ticks(1000),
            100.0,
            200.0,
            5.0,
            -10.0,
            ModifierFlags::default(),
        );

        assert_eq!(event.event_type, EventType::ScrollWheel);
        assert_eq!(event.coordinates, (100.0, 200.0));
        assert_eq!(event.scroll_delta, Some((5.0, -10.0)));
        assert!(event.key_code.is_none());
    }

    #[test]
    fn test_enriched_event_new() {
        let raw = RawEvent::mouse(
            Timestamp::from_ticks(1000),
            EventType::LeftMouseDown,
            100.0,
            200.0,
            CursorState::Arrow,
            ModifierFlags::default(),
            1,
        );

        let enriched = EnrichedEvent::new(raw.clone(), 42);
        assert_eq!(enriched.sequence, 42);
        assert!(enriched.semantic.is_none());
        assert_eq!(enriched.raw.event_type, EventType::LeftMouseDown);
    }

    #[test]
    fn test_enriched_event_with_semantic() {
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
            title: Some("Click Me".to_string()),
            ..Default::default()
        };

        let enriched = EnrichedEvent::new(raw, 42).with_semantic(semantic.clone());
        assert!(enriched.semantic.is_some());
        assert_eq!(enriched.semantic.as_ref().unwrap().ax_role, Some("AXButton".to_string()));
        assert_eq!(enriched.semantic.as_ref().unwrap().title, Some("Click Me".to_string()));
    }

    #[test]
    fn test_atomic_semantic_state_default() {
        let state = AtomicSemanticState::default();
        assert_eq!(state.load(Ordering::SeqCst), SemanticState::Empty);
    }

    #[test]
    fn test_atomic_semantic_state_compare_exchange_failure() {
        let state = AtomicSemanticState::new(SemanticState::Pending);

        let result = state.compare_exchange(
            SemanticState::Empty,
            SemanticState::Filled,
            Ordering::SeqCst,
            Ordering::SeqCst,
        );

        assert_eq!(result, Err(SemanticState::Pending));
        assert_eq!(state.load(Ordering::SeqCst), SemanticState::Pending);
    }

    #[test]
    fn test_event_type_serialization() {
        let event_type = EventType::LeftMouseDown;
        let serialized = serde_json::to_string(&event_type).unwrap();
        let deserialized: EventType = serde_json::from_str(&serialized).unwrap();
        assert_eq!(event_type, deserialized);
    }

    #[test]
    fn test_cursor_state_serialization() {
        let cursor = CursorState::IBeam;
        let serialized = serde_json::to_string(&cursor).unwrap();
        let deserialized: CursorState = serde_json::from_str(&serialized).unwrap();
        assert_eq!(cursor, deserialized);
    }

    #[test]
    fn test_modifier_flags_serialization() {
        let flags = ModifierFlags {
            shift: true,
            command: true,
            ..Default::default()
        };

        let serialized = serde_json::to_string(&flags).unwrap();
        let deserialized: ModifierFlags = serde_json::from_str(&serialized).unwrap();
        assert_eq!(flags.shift, deserialized.shift);
        assert_eq!(flags.command, deserialized.command);
    }

    #[test]
    fn test_raw_event_serialization() {
        let event = RawEvent::mouse(
            Timestamp::from_raw(1000),
            EventType::LeftMouseDown,
            100.0,
            200.0,
            CursorState::Arrow,
            ModifierFlags::default(),
            1,
        );

        let serialized = serde_json::to_string(&event).unwrap();
        let deserialized: RawEvent = serde_json::from_str(&serialized).unwrap();
        assert_eq!(event.event_type, deserialized.event_type);
        assert_eq!(event.coordinates, deserialized.coordinates);
    }
}
