//! Cursor Shape Detection
//!
//! This module queries NSCursor to determine the current cursor shape.
//! The cursor shape provides valuable context for understanding user intent.

use super::types::CursorState;
use objc::runtime::{Class, Object};
use objc::{msg_send, sel, sel_impl};

/// Get the current cursor state by querying NSCursor
///
/// This function queries the AppKit NSCursor class to determine
/// the current system cursor shape.
pub fn get_current_cursor() -> CursorState {
    unsafe {
        // Get NSCursor class
        let ns_cursor_class = match Class::get("NSCursor") {
            Some(cls) => cls,
            None => return CursorState::Unknown,
        };

        // Get current cursor: [NSCursor currentCursor]
        let current_cursor: *mut Object = msg_send![ns_cursor_class, currentCursor];
        if current_cursor.is_null() {
            return CursorState::Unknown;
        }

        // Compare against known cursor types
        // We need to check against each standard cursor

        // Arrow cursor
        let arrow_cursor: *mut Object = msg_send![ns_cursor_class, arrowCursor];
        if cursors_equal(current_cursor, arrow_cursor) {
            return CursorState::Arrow;
        }

        // IBeam cursor (text)
        let ibeam_cursor: *mut Object = msg_send![ns_cursor_class, IBeamCursor];
        if cursors_equal(current_cursor, ibeam_cursor) {
            return CursorState::IBeam;
        }

        // Pointing hand cursor
        let pointing_cursor: *mut Object = msg_send![ns_cursor_class, pointingHandCursor];
        if cursors_equal(current_cursor, pointing_cursor) {
            return CursorState::PointingHand;
        }

        // Open hand cursor
        let open_hand_cursor: *mut Object = msg_send![ns_cursor_class, openHandCursor];
        if cursors_equal(current_cursor, open_hand_cursor) {
            return CursorState::OpenHand;
        }

        // Closed hand cursor
        let closed_hand_cursor: *mut Object = msg_send![ns_cursor_class, closedHandCursor];
        if cursors_equal(current_cursor, closed_hand_cursor) {
            return CursorState::ClosedHand;
        }

        // Crosshair cursor
        let crosshair_cursor: *mut Object = msg_send![ns_cursor_class, crosshairCursor];
        if cursors_equal(current_cursor, crosshair_cursor) {
            return CursorState::Crosshair;
        }

        // Resize left-right cursor
        let resize_lr_cursor: *mut Object = msg_send![ns_cursor_class, resizeLeftRightCursor];
        if cursors_equal(current_cursor, resize_lr_cursor) {
            return CursorState::ResizeLeftRight;
        }

        // Resize up-down cursor
        let resize_ud_cursor: *mut Object = msg_send![ns_cursor_class, resizeUpDownCursor];
        if cursors_equal(current_cursor, resize_ud_cursor) {
            return CursorState::ResizeUpDown;
        }

        // Operation not allowed cursor
        let not_allowed_cursor: *mut Object = msg_send![ns_cursor_class, operationNotAllowedCursor];
        if cursors_equal(current_cursor, not_allowed_cursor) {
            return CursorState::NotAllowed;
        }

        // If none matched, return Unknown
        CursorState::Unknown
    }
}

/// Compare two NSCursor objects for equality
///
/// NSCursor objects are singletons, so we can compare by pointer
unsafe fn cursors_equal(a: *mut Object, b: *mut Object) -> bool {
    if a.is_null() || b.is_null() {
        return false;
    }
    // Use isEqual: for proper comparison
    let result: bool = msg_send![a, isEqual: b];
    result
}

/// Cursor context for semantic enrichment
#[derive(Debug, Clone)]
pub struct CursorContext {
    /// Current cursor state
    pub state: CursorState,
    /// Whether the cursor indicates text input mode
    pub is_text_mode: bool,
    /// Whether the cursor indicates a clickable element
    pub is_clickable: bool,
    /// Whether the cursor indicates the system is busy
    pub is_busy: bool,
    /// Whether the cursor indicates a drag operation
    pub is_dragging: bool,
}

impl CursorContext {
    /// Create cursor context from current state
    pub fn from_current() -> Self {
        let state = get_current_cursor();
        Self::from_state(state)
    }

    /// Create cursor context from a given state
    pub fn from_state(state: CursorState) -> Self {
        Self {
            state,
            is_text_mode: state.is_text_input(),
            is_clickable: state.is_clickable(),
            is_busy: state.is_busy(),
            is_dragging: matches!(state, CursorState::ClosedHand),
        }
    }
}

impl Default for CursorContext {
    fn default() -> Self {
        Self::from_state(CursorState::Arrow)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cursor_context_from_state() {
        let ctx = CursorContext::from_state(CursorState::IBeam);
        assert!(ctx.is_text_mode);
        assert!(!ctx.is_clickable);

        let ctx = CursorContext::from_state(CursorState::PointingHand);
        assert!(ctx.is_clickable);
        assert!(!ctx.is_text_mode);

        let ctx = CursorContext::from_state(CursorState::Wait);
        assert!(ctx.is_busy);
    }

    #[test]
    fn test_cursor_context_default() {
        let ctx = CursorContext::default();
        assert_eq!(ctx.state, CursorState::Arrow);
        assert!(!ctx.is_text_mode);
        assert!(!ctx.is_clickable);
        assert!(!ctx.is_busy);
    }

    #[test]
    fn test_get_current_cursor() {
        // This test will return Unknown in headless/CI environments
        // but should not panic
        let _cursor = get_current_cursor();
    }

    #[test]
    fn test_cursor_context_ibeam_state() {
        let ctx = CursorContext::from_state(CursorState::IBeam);
        assert_eq!(ctx.state, CursorState::IBeam);
        assert!(ctx.is_text_mode);
        assert!(!ctx.is_clickable);
        assert!(!ctx.is_busy);
        assert!(!ctx.is_dragging);
    }

    #[test]
    fn test_cursor_context_pointing_hand_state() {
        let ctx = CursorContext::from_state(CursorState::PointingHand);
        assert_eq!(ctx.state, CursorState::PointingHand);
        assert!(!ctx.is_text_mode);
        assert!(ctx.is_clickable);
        assert!(!ctx.is_busy);
        assert!(!ctx.is_dragging);
    }

    #[test]
    fn test_cursor_context_open_hand_state() {
        let ctx = CursorContext::from_state(CursorState::OpenHand);
        assert_eq!(ctx.state, CursorState::OpenHand);
        assert!(!ctx.is_text_mode);
        assert!(!ctx.is_clickable);
        assert!(!ctx.is_busy);
        assert!(!ctx.is_dragging);
    }

    #[test]
    fn test_cursor_context_closed_hand_state() {
        let ctx = CursorContext::from_state(CursorState::ClosedHand);
        assert_eq!(ctx.state, CursorState::ClosedHand);
        assert!(!ctx.is_text_mode);
        assert!(!ctx.is_clickable);
        assert!(!ctx.is_busy);
        assert!(ctx.is_dragging);
    }

    #[test]
    fn test_cursor_context_wait_state() {
        let ctx = CursorContext::from_state(CursorState::Wait);
        assert_eq!(ctx.state, CursorState::Wait);
        assert!(!ctx.is_text_mode);
        assert!(!ctx.is_clickable);
        assert!(ctx.is_busy);
        assert!(!ctx.is_dragging);
    }

    #[test]
    fn test_cursor_context_progress_state() {
        let ctx = CursorContext::from_state(CursorState::Progress);
        assert_eq!(ctx.state, CursorState::Progress);
        assert!(!ctx.is_text_mode);
        assert!(!ctx.is_clickable);
        assert!(ctx.is_busy);
        assert!(!ctx.is_dragging);
    }

    #[test]
    fn test_cursor_context_crosshair_state() {
        let ctx = CursorContext::from_state(CursorState::Crosshair);
        assert_eq!(ctx.state, CursorState::Crosshair);
        assert!(!ctx.is_text_mode);
        assert!(!ctx.is_clickable);
        assert!(!ctx.is_busy);
        assert!(!ctx.is_dragging);
    }

    #[test]
    fn test_cursor_context_resize_states() {
        let ctx = CursorContext::from_state(CursorState::ResizeLeftRight);
        assert_eq!(ctx.state, CursorState::ResizeLeftRight);
        assert!(!ctx.is_text_mode);
        assert!(!ctx.is_clickable);
        assert!(!ctx.is_busy);
        assert!(!ctx.is_dragging);

        let ctx = CursorContext::from_state(CursorState::ResizeUpDown);
        assert_eq!(ctx.state, CursorState::ResizeUpDown);
        assert!(!ctx.is_text_mode);
        assert!(!ctx.is_clickable);
        assert!(!ctx.is_busy);
        assert!(!ctx.is_dragging);
    }

    #[test]
    fn test_cursor_context_not_allowed_state() {
        let ctx = CursorContext::from_state(CursorState::NotAllowed);
        assert_eq!(ctx.state, CursorState::NotAllowed);
        assert!(!ctx.is_text_mode);
        assert!(!ctx.is_clickable);
        assert!(!ctx.is_busy);
        assert!(!ctx.is_dragging);
    }

    #[test]
    fn test_cursor_context_unknown_state() {
        let ctx = CursorContext::from_state(CursorState::Unknown);
        assert_eq!(ctx.state, CursorState::Unknown);
        assert!(!ctx.is_text_mode);
        assert!(!ctx.is_clickable);
        assert!(!ctx.is_busy);
        assert!(!ctx.is_dragging);
    }

    #[test]
    fn test_cursor_context_clone() {
        let ctx = CursorContext::from_state(CursorState::PointingHand);
        let cloned = ctx.clone();

        assert_eq!(ctx.state, cloned.state);
        assert_eq!(ctx.is_text_mode, cloned.is_text_mode);
        assert_eq!(ctx.is_clickable, cloned.is_clickable);
        assert_eq!(ctx.is_busy, cloned.is_busy);
        assert_eq!(ctx.is_dragging, cloned.is_dragging);
    }

    #[test]
    fn test_cursor_context_from_current_does_not_panic() {
        // This test verifies that from_current doesn't panic
        // It may return Unknown in headless environments
        let ctx = CursorContext::from_current();
        // Just verify we got some state back
        let _ = ctx.state;
    }
}
