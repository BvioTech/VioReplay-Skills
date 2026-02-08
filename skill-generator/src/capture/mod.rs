//! Event capture module
//!
//! This module provides lock-free event capture using macOS Quartz Event Tap.
//! The architecture ensures zero blocking in the event callback path.

pub mod types;
pub mod ring_buffer;
pub mod event_tap;
pub mod accessibility;
pub mod cursor;

pub use types::*;
pub use ring_buffer::EventRingBuffer;
pub use event_tap::{EventTap, check_accessibility_permissions, request_accessibility_permissions, check_screen_recording_permission, request_screen_recording_permission};
pub use cursor::{get_current_cursor, CursorContext};
