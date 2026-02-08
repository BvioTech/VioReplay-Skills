//! Quartz Event Tap Handler
//!
//! This module implements the macOS event capture using CGEventTap.
//! It captures mouse movements, clicks, keyboard events, and scroll events
//! with microsecond precision using mach_absolute_time.
//!
//! # Permissions
//!
//! Requires Accessibility permissions in System Preferences > Security & Privacy.

use super::ring_buffer::EventProducer;
use super::types::{CursorState, EventType, ModifierFlags, RawEvent};
use crate::time::timebase::{MachTimebase, Timestamp};
use core_foundation::base::{CFRelease, CFTypeRef, TCFType};
use core_foundation::runloop::kCFRunLoopCommonModes;
use std::cell::UnsafeCell;
use std::ffi::c_void;
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicPtr, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use tracing::{error, info, trace};

// Core Graphics event types
type CGEventRef = CFTypeRef;
type CGEventTapProxy = *const c_void;
type CGEventMask = u64;

// CGEventTap location
#[repr(u32)]
#[derive(Copy, Clone)]
#[allow(dead_code, clippy::enum_variant_names)]
enum CGEventTapLocation {
    HIDEventTap = 0,
    SessionEventTap = 1,
    AnnotatedSessionEventTap = 2,
}

// CGEventTap placement
#[repr(u32)]
#[derive(Copy, Clone)]
#[allow(dead_code, clippy::enum_variant_names)]
enum CGEventTapPlacement {
    HeadInsertEventTap = 0,
    TailAppendEventTap = 1,
}

// CGEventTap options
#[repr(u32)]
#[derive(Copy, Clone)]
#[allow(dead_code)]
enum CGEventTapOptions {
    DefaultTap = 0,
    ListenOnly = 1,
}

// macOS CoreGraphics event constants — full API surface defined for completeness
#[allow(dead_code)]
mod cg_constants {
    // CGEventType values
    pub const CG_EVENT_NULL: u32 = 0;
    pub const CG_EVENT_LEFT_MOUSE_DOWN: u32 = 1;
    pub const CG_EVENT_LEFT_MOUSE_UP: u32 = 2;
    pub const CG_EVENT_RIGHT_MOUSE_DOWN: u32 = 3;
    pub const CG_EVENT_RIGHT_MOUSE_UP: u32 = 4;
    pub const CG_EVENT_MOUSE_MOVED: u32 = 5;
    pub const CG_EVENT_LEFT_MOUSE_DRAGGED: u32 = 6;
    pub const CG_EVENT_RIGHT_MOUSE_DRAGGED: u32 = 7;
    pub const CG_EVENT_KEY_DOWN: u32 = 10;
    pub const CG_EVENT_KEY_UP: u32 = 11;
    pub const CG_EVENT_FLAGS_CHANGED: u32 = 12;
    pub const CG_EVENT_SCROLL_WHEEL: u32 = 22;
    pub const CG_EVENT_OTHER_MOUSE_DOWN: u32 = 25;
    pub const CG_EVENT_OTHER_MOUSE_UP: u32 = 26;
    pub const CG_EVENT_OTHER_MOUSE_DRAGGED: u32 = 27;

    // CGEventField values for querying event data
    pub const CG_MOUSE_EVENT_NUMBER: u32 = 0;
    pub const CG_MOUSE_EVENT_CLICK_STATE: u32 = 1;
    pub const CG_MOUSE_EVENT_PRESSURE: u32 = 2;
    pub const CG_MOUSE_EVENT_BUTTON_NUMBER: u32 = 3;
    pub const CG_MOUSE_EVENT_DELTA_X: u32 = 4;
    pub const CG_MOUSE_EVENT_DELTA_Y: u32 = 5;
    pub const CG_SCROLL_WHEEL_EVENT_DELTA_AXIS_1: u32 = 11;
    pub const CG_SCROLL_WHEEL_EVENT_DELTA_AXIS_2: u32 = 12;
    pub const CG_KEYBOARD_EVENT_KEYCODE: u32 = 9;
    pub const CG_KEYBOARD_EVENT_AUTOREPEAT: u32 = 8;
}
use cg_constants::*;

// CGEventFlags
const CG_EVENT_FLAG_MASK_SHIFT: u64 = 0x00020000;
const CG_EVENT_FLAG_MASK_CONTROL: u64 = 0x00040000;
const CG_EVENT_FLAG_MASK_ALTERNATE: u64 = 0x00080000;
const CG_EVENT_FLAG_MASK_COMMAND: u64 = 0x00100000;
const CG_EVENT_FLAG_MASK_CAPS_LOCK: u64 = 0x00010000;
const CG_EVENT_FLAG_MASK_FN: u64 = 0x00800000;

// Event mask for all events we want to capture
fn create_event_mask() -> CGEventMask {
    (1 << CG_EVENT_LEFT_MOUSE_DOWN)
        | (1 << CG_EVENT_LEFT_MOUSE_UP)
        | (1 << CG_EVENT_RIGHT_MOUSE_DOWN)
        | (1 << CG_EVENT_RIGHT_MOUSE_UP)
        | (1 << CG_EVENT_MOUSE_MOVED)
        | (1 << CG_EVENT_LEFT_MOUSE_DRAGGED)
        | (1 << CG_EVENT_RIGHT_MOUSE_DRAGGED)
        | (1 << CG_EVENT_KEY_DOWN)
        | (1 << CG_EVENT_KEY_UP)
        | (1 << CG_EVENT_FLAGS_CHANGED)
        | (1 << CG_EVENT_SCROLL_WHEEL)
        | (1 << CG_EVENT_OTHER_MOUSE_DOWN)
        | (1 << CG_EVENT_OTHER_MOUSE_UP)
        | (1 << CG_EVENT_OTHER_MOUSE_DRAGGED)
}

// FFI declarations for Core Graphics
#[link(name = "CoreGraphics", kind = "framework")]
extern "C" {
    fn CGEventTapCreate(
        tap: CGEventTapLocation,
        place: CGEventTapPlacement,
        options: CGEventTapOptions,
        events_of_interest: CGEventMask,
        callback: extern "C" fn(CGEventTapProxy, u32, CGEventRef, *mut c_void) -> CGEventRef,
        user_info: *mut c_void,
    ) -> CFTypeRef;

    fn CGEventTapEnable(tap: CFTypeRef, enable: bool);

    fn CGEventGetLocation(event: CGEventRef) -> CGPoint;
    fn CGEventGetIntegerValueField(event: CGEventRef, field: u32) -> i64;
    fn CGEventGetFlags(event: CGEventRef) -> u64;
    fn CGEventGetTimestamp(event: CGEventRef) -> u64;
}

// FFI declarations for Core Foundation
#[link(name = "CoreFoundation", kind = "framework")]
extern "C" {
    fn CFMachPortCreateRunLoopSource(
        allocator: CFTypeRef,
        port: CFTypeRef,
        order: i64,
    ) -> CFTypeRef;

    fn CFRunLoopGetCurrent() -> CFTypeRef;
    fn CFRunLoopAddSource(rl: CFTypeRef, source: CFTypeRef, mode: CFTypeRef);
    fn CFRunLoopRun();
    fn CFRunLoopStop(rl: CFTypeRef);
}

// FFI declarations for Accessibility
extern "C" {
    fn AXIsProcessTrusted() -> bool;
    fn AXIsProcessTrustedWithOptions(options: CFTypeRef) -> bool;
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct CGPoint {
    x: f64,
    y: f64,
}

/// Context for the event tap callback.
///
/// Safety: The `producer` field uses UnsafeCell because the EventProducer (rtrb::Producer)
/// is `!Sync`, but we guarantee single-thread access: the callback only runs on the
/// dedicated CFRunLoop thread. This avoids Mutex overhead in the real-time event path.
struct EventTapContext {
    producer: UnsafeCell<EventProducer>,
    running: Arc<AtomicBool>,
    event_count: std::sync::atomic::AtomicU64,
}

// Safety: EventTapContext is only accessed from the CFRunLoop thread (via the callback)
// and from the main thread (to create/destroy it, never concurrently with the callback).
// The UnsafeCell<EventProducer> is only accessed in the callback on the CFRunLoop thread.
unsafe impl Sync for EventTapContext {}

/// Global context pointer for the callback
/// This is necessary because CGEventTapCreate's callback can't capture Rust closures
static CONTEXT_PTR: AtomicPtr<EventTapContext> = AtomicPtr::new(ptr::null_mut());
static RUN_LOOP_PTR: AtomicPtr<c_void> = AtomicPtr::new(ptr::null_mut());

/// Quartz Event Tap for capturing user interactions
pub struct EventTap {
    /// Thread handle for the run loop
    thread_handle: Option<JoinHandle<()>>,
    /// Flag to signal stop
    running: Arc<AtomicBool>,
    /// Context for the callback — kept alive to prevent dangling pointer in CONTEXT_PTR.
    /// Field is read implicitly via Drop to deallocate the Box.
    _context: Option<Box<EventTapContext>>,
}

impl EventTap {
    /// Create a new event tap
    ///
    /// # Errors
    /// Returns error if event tap creation fails
    pub fn new() -> Result<Self, crate::Error> {
        // Ensure timebase is initialized
        MachTimebase::init();

        let running = Arc::new(AtomicBool::new(false));

        Ok(Self {
            thread_handle: None,
            running,
            _context: None,
        })
    }

    /// Start capturing events
    ///
    /// This spawns a dedicated thread for the CFRunLoop that processes events.
    pub fn start(&mut self, producer: EventProducer) -> Result<(), crate::Error> {
        if self.running.swap(true, Ordering::SeqCst) {
            return Err(crate::Error::Capture("Event tap already running".into()));
        }

        // Check accessibility permissions first
        if !check_accessibility_permissions() {
            self.running.store(false, Ordering::SeqCst);
            return Err(crate::Error::Capture(
                "Accessibility permissions not granted. Please enable in System Preferences > Security & Privacy > Privacy > Accessibility".into()
            ));
        }

        // Create context
        let context = Box::new(EventTapContext {
            producer: UnsafeCell::new(producer),
            running: Arc::clone(&self.running),
            event_count: std::sync::atomic::AtomicU64::new(0),
        });

        // Store context pointer for callback
        let context_ptr = Box::into_raw(context);
        CONTEXT_PTR.store(context_ptr, Ordering::SeqCst);

        let running = Arc::clone(&self.running);

        // Spawn the event tap thread
        let handle = thread::Builder::new()
            .name("event-tap".into())
            .spawn(move || {
                if let Err(e) = run_event_tap_loop(running) {
                    error!("Event tap error: {}", e);
                }
            })
            .map_err(|e| {
                // Clean up on failure
                unsafe {
                    let _ = Box::from_raw(context_ptr);
                }
                CONTEXT_PTR.store(ptr::null_mut(), Ordering::SeqCst);
                crate::Error::Capture(format!("Failed to spawn event tap thread: {}", e))
            })?;

        self.thread_handle = Some(handle);
        info!("Event tap started");

        Ok(())
    }

    /// Stop capturing events
    pub fn stop(&mut self) {
        if !self.running.swap(false, Ordering::SeqCst) {
            return; // Already stopped
        }

        // Stop the run loop
        let run_loop = RUN_LOOP_PTR.swap(ptr::null_mut(), Ordering::SeqCst);
        if !run_loop.is_null() {
            unsafe {
                CFRunLoopStop(run_loop as _);
            }
        }

        // Wait for thread to finish
        if let Some(handle) = self.thread_handle.take() {
            let _ = handle.join();
        }

        // Clean up context
        let context_ptr = CONTEXT_PTR.swap(ptr::null_mut(), Ordering::SeqCst);
        if !context_ptr.is_null() {
            unsafe {
                let ctx = Box::from_raw(context_ptr);
                let count = ctx.event_count.load(Ordering::Relaxed);
                info!("Event tap stopped after capturing {} events", count);
            }
        } else {
            info!("Event tap stopped");
        }
    }

    /// Check if event tap is running
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// Get event count (if available)
    pub fn event_count(&self) -> u64 {
        let ctx = CONTEXT_PTR.load(Ordering::SeqCst);
        if ctx.is_null() {
            0
        } else {
            unsafe { (*ctx).event_count.load(Ordering::Relaxed) }
        }
    }
}

impl Drop for EventTap {
    fn drop(&mut self) {
        self.stop();
    }
}

/// The event tap callback - called for each captured event
extern "C" fn event_tap_callback(
    _proxy: CGEventTapProxy,
    event_type: u32,
    event: CGEventRef,
    _user_info: *mut c_void,
) -> CGEventRef {
    // Get context
    let ctx = CONTEXT_PTR.load(Ordering::SeqCst);
    if ctx.is_null() {
        return event;
    }

    let context = unsafe { &*ctx };

    // Check if we should stop
    if !context.running.load(Ordering::Relaxed) {
        return event;
    }

    // Convert event type
    let our_event_type = match convert_cg_event_type(event_type) {
        Some(t) => t,
        None => return event, // Skip unknown events
    };

    // Get event location
    let location = unsafe { CGEventGetLocation(event) };

    // Get timestamp (in mach absolute time)
    let timestamp_mach = unsafe { CGEventGetTimestamp(event) };
    let timestamp = Timestamp::from_ticks(timestamp_mach);

    // Get flags
    let flags = unsafe { CGEventGetFlags(event) };
    let modifiers = convert_flags_to_modifiers(flags);

    // Get event-specific data
    let (key_code, character, scroll_delta, click_count) = match our_event_type {
        EventType::KeyDown | EventType::KeyUp => {
            let key_code = unsafe { CGEventGetIntegerValueField(event, CG_KEYBOARD_EVENT_KEYCODE) };
            (Some(key_code as u16), None, None, 0u8)
        }
        EventType::ScrollWheel => {
            let delta_y =
                unsafe { CGEventGetIntegerValueField(event, CG_SCROLL_WHEEL_EVENT_DELTA_AXIS_1) };
            let delta_x =
                unsafe { CGEventGetIntegerValueField(event, CG_SCROLL_WHEEL_EVENT_DELTA_AXIS_2) };
            (None, None, Some((delta_x as f64, delta_y as f64)), 0u8)
        }
        EventType::LeftMouseDown
        | EventType::LeftMouseUp
        | EventType::RightMouseDown
        | EventType::RightMouseUp => {
            let click_state =
                unsafe { CGEventGetIntegerValueField(event, CG_MOUSE_EVENT_CLICK_STATE) };
            (None, None, None, click_state as u8)
        }
        _ => (None, None, None, 0u8),
    };

    // Get cursor state (simplified - full implementation would query NSCursor)
    let cursor_state = get_current_cursor_state();

    // Create raw event
    let raw_event = RawEvent {
        timestamp,
        event_type: our_event_type,
        coordinates: (location.x, location.y),
        cursor_state,
        key_code,
        character,
        modifiers,
        scroll_delta,
        click_count,
    };

    // Push to ring buffer (lock-free, no Mutex needed - callback runs on single CFRunLoop thread)
    // Safety: UnsafeCell access is safe because the callback is only invoked on the
    // dedicated CFRunLoop thread, guaranteeing single-threaded access to the producer.
    let producer = unsafe { &mut *context.producer.get() };
    if producer.push(raw_event) {
        context.event_count.fetch_add(1, Ordering::Relaxed);
        trace!(
            "Captured {:?} at ({:.1}, {:.1})",
            our_event_type,
            location.x,
            location.y
        );
    } else {
        // Buffer full - this is logged but not an error
        trace!("Ring buffer full, dropping event");
    }

    // Return the event unchanged (we're listen-only)
    event
}

/// RAII guard for a CGEventTap handle. Disables and releases the tap on drop.
struct EventTapGuard(CFTypeRef);

impl Drop for EventTapGuard {
    fn drop(&mut self) {
        unsafe {
            CGEventTapEnable(self.0, false);
            CFRelease(self.0);
        }
    }
}

/// RAII guard for a CFRunLoopSource. Releases the source on drop.
struct RunLoopSourceGuard(CFTypeRef);

impl Drop for RunLoopSourceGuard {
    fn drop(&mut self) {
        unsafe {
            CFRelease(self.0);
        }
    }
}

/// RAII guard that clears RUN_LOOP_PTR on drop.
struct RunLoopPtrGuard;

impl Drop for RunLoopPtrGuard {
    fn drop(&mut self) {
        RUN_LOOP_PTR.store(ptr::null_mut(), Ordering::SeqCst);
    }
}

/// Run the event tap loop
fn run_event_tap_loop(_running: Arc<AtomicBool>) -> Result<(), crate::Error> {
    info!("Event tap loop starting...");

    // Create event mask for all events we want
    let event_mask = create_event_mask();

    // Create the event tap
    let tap = unsafe {
        CGEventTapCreate(
            CGEventTapLocation::SessionEventTap,
            CGEventTapPlacement::HeadInsertEventTap,
            CGEventTapOptions::ListenOnly,
            event_mask,
            event_tap_callback,
            ptr::null_mut(),
        )
    };

    if tap.is_null() {
        return Err(crate::Error::Capture(
            "Failed to create event tap. Ensure accessibility permissions are granted.".into(),
        ));
    }

    // RAII guard: disables and releases tap on any exit (including panic)
    let _tap_guard = EventTapGuard(tap);

    // Create run loop source
    let run_loop_source = unsafe { CFMachPortCreateRunLoopSource(ptr::null(), tap, 0) };

    if run_loop_source.is_null() {
        return Err(crate::Error::Capture(
            "Failed to create run loop source".into(),
        ));
    }

    // RAII guard: releases source on any exit
    let _source_guard = RunLoopSourceGuard(run_loop_source);

    // Get current run loop and store for stopping
    let run_loop = unsafe { CFRunLoopGetCurrent() };
    RUN_LOOP_PTR.store(run_loop as *mut c_void, Ordering::SeqCst);
    let _ptr_guard = RunLoopPtrGuard;

    // Add source to run loop
    unsafe {
        CFRunLoopAddSource(
            run_loop,
            run_loop_source,
            kCFRunLoopCommonModes as CFTypeRef,
        );
    }

    // Enable the event tap
    unsafe {
        CGEventTapEnable(tap, true);
    }

    info!("Event tap loop running");

    // Run the loop until stopped
    // CFRunLoopRun will return when CFRunLoopStop is called
    unsafe {
        CFRunLoopRun();
    }

    // Guards handle cleanup automatically via Drop

    info!("Event tap loop stopped");
    Ok(())
}

/// Convert CGEventType to our EventType
fn convert_cg_event_type(cg_type: u32) -> Option<EventType> {
    match cg_type {
        CG_EVENT_LEFT_MOUSE_DOWN => Some(EventType::LeftMouseDown),
        CG_EVENT_LEFT_MOUSE_UP => Some(EventType::LeftMouseUp),
        CG_EVENT_RIGHT_MOUSE_DOWN => Some(EventType::RightMouseDown),
        CG_EVENT_RIGHT_MOUSE_UP => Some(EventType::RightMouseUp),
        CG_EVENT_MOUSE_MOVED => Some(EventType::MouseMoved),
        CG_EVENT_LEFT_MOUSE_DRAGGED => Some(EventType::LeftMouseDragged),
        CG_EVENT_RIGHT_MOUSE_DRAGGED => Some(EventType::RightMouseDragged),
        CG_EVENT_KEY_DOWN => Some(EventType::KeyDown),
        CG_EVENT_KEY_UP => Some(EventType::KeyUp),
        CG_EVENT_FLAGS_CHANGED => Some(EventType::FlagsChanged),
        CG_EVENT_SCROLL_WHEEL => Some(EventType::ScrollWheel),
        CG_EVENT_OTHER_MOUSE_DOWN => Some(EventType::OtherMouseDown),
        CG_EVENT_OTHER_MOUSE_UP => Some(EventType::OtherMouseUp),
        CG_EVENT_OTHER_MOUSE_DRAGGED => Some(EventType::OtherMouseDragged),
        _ => None,
    }
}

/// Convert CG event flags to our ModifierFlags
fn convert_flags_to_modifiers(flags: u64) -> ModifierFlags {
    let mut modifiers = ModifierFlags::default();

    if flags & CG_EVENT_FLAG_MASK_SHIFT != 0 {
        modifiers.shift = true;
    }
    if flags & CG_EVENT_FLAG_MASK_CONTROL != 0 {
        modifiers.control = true;
    }
    if flags & CG_EVENT_FLAG_MASK_ALTERNATE != 0 {
        modifiers.option = true;
    }
    if flags & CG_EVENT_FLAG_MASK_COMMAND != 0 {
        modifiers.command = true;
    }
    if flags & CG_EVENT_FLAG_MASK_CAPS_LOCK != 0 {
        modifiers.caps_lock = true;
    }
    if flags & CG_EVENT_FLAG_MASK_FN != 0 {
        modifiers.function = true;
    }

    modifiers
}

/// Check if accessibility permissions are granted
pub fn check_accessibility_permissions() -> bool {
    unsafe { AXIsProcessTrusted() }
}

/// Request accessibility permissions (shows system dialog)
pub fn request_accessibility_permissions() -> bool {
    use core_foundation::dictionary::CFDictionary;
    use core_foundation::string::CFString;

    // Create options dictionary with kAXTrustedCheckOptionPrompt = true
    let key = CFString::new("AXTrustedCheckOptionPrompt");
    let value = core_foundation::boolean::CFBoolean::true_value();

    let options = CFDictionary::from_CFType_pairs(&[(key.as_CFType(), value.as_CFType())]);

    unsafe { AXIsProcessTrustedWithOptions(options.as_concrete_TypeRef() as CFTypeRef) }
}

/// Check if screen recording permission is granted (macOS 10.15+).
///
/// Required for CGWindowListCreateImage used by the Vision OCR fallback.
/// Uses CGPreflightScreenCaptureAccess on macOS 14+ and falls back to
/// a small test capture on older systems.
pub fn check_screen_recording_permission() -> bool {
    unsafe {
        // macOS 14+ provides CGPreflightScreenCaptureAccess
        extern "C" {
            fn CGPreflightScreenCaptureAccess() -> bool;
        }

        // Try the modern API first. If the symbol isn't available (pre-14),
        // this will still be resolved since we link CoreGraphics, but we
        // guard with a fallback approach.
        let result = CGPreflightScreenCaptureAccess();
        if result {
            return true;
        }

        // Fallback: attempt a 1x1 pixel capture as a probe
        extern "C" {
            fn CGWindowListCreateImage(
                screenBounds: CGRect,
                listOption: u32,
                windowID: u32,
                imageOption: u32,
            ) -> *mut std::ffi::c_void;
            fn CGImageRelease(image: *mut std::ffi::c_void);
        }

        use core_graphics::display::{CGPoint, CGRect, CGSize};

        let rect = CGRect::new(&CGPoint::new(0.0, 0.0), &CGSize::new(1.0, 1.0));
        let image = CGWindowListCreateImage(rect, 1, 0, 0);
        if !image.is_null() {
            CGImageRelease(image);
            true
        } else {
            false
        }
    }
}

/// Request screen recording permission (macOS 14+).
///
/// Shows the system permission dialog. Returns true if already granted.
pub fn request_screen_recording_permission() -> bool {
    unsafe {
        extern "C" {
            fn CGRequestScreenCaptureAccess() -> bool;
        }
        CGRequestScreenCaptureAccess()
    }
}

/// Convert CGEventType to our EventType (public API)
pub fn convert_event_type(cg_type: u32) -> Option<EventType> {
    convert_cg_event_type(cg_type)
}

/// Get current cursor state
///
/// Queries NSCursor through the Objective-C runtime to determine
/// the current system cursor shape.
pub fn get_current_cursor_state() -> CursorState {
    super::cursor::get_current_cursor()
}

/// Create a RawEvent from captured event data (for testing)
#[allow(clippy::too_many_arguments)]
pub fn create_raw_event(
    event_type: EventType,
    x: f64,
    y: f64,
    modifiers: ModifierFlags,
    key_code: Option<u16>,
    character: Option<char>,
    scroll_delta: Option<(f64, f64)>,
    click_count: u8,
) -> RawEvent {
    let timestamp = Timestamp::now();
    let cursor_state = get_current_cursor_state();

    RawEvent {
        timestamp,
        event_type,
        coordinates: (x, y),
        cursor_state,
        key_code,
        character,
        modifiers,
        scroll_delta,
        click_count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accessibility_check() {
        // This will return false in CI, but shouldn't panic
        let _has_access = check_accessibility_permissions();
    }

    #[test]
    fn test_convert_event_type() {
        assert_eq!(convert_event_type(5), Some(EventType::MouseMoved));
        assert_eq!(convert_event_type(1), Some(EventType::LeftMouseDown));
        assert_eq!(convert_event_type(2), Some(EventType::LeftMouseUp));
        assert_eq!(convert_event_type(3), Some(EventType::RightMouseDown));
        assert_eq!(convert_event_type(10), Some(EventType::KeyDown));
        assert_eq!(convert_event_type(22), Some(EventType::ScrollWheel));
        assert_eq!(convert_event_type(999), None);
    }

    #[test]
    fn test_create_raw_event() {
        MachTimebase::init();

        let event = create_raw_event(
            EventType::MouseMoved,
            100.0,
            200.0,
            ModifierFlags::default(),
            None,
            None,
            None,
            0,
        );

        assert_eq!(event.event_type, EventType::MouseMoved);
        assert_eq!(event.coordinates, (100.0, 200.0));
    }

    #[test]
    fn test_convert_flags_to_modifiers() {
        let flags = CG_EVENT_FLAG_MASK_SHIFT | CG_EVENT_FLAG_MASK_COMMAND;
        let modifiers = convert_flags_to_modifiers(flags);

        assert!(modifiers.shift);
        assert!(modifiers.command);
        assert!(!modifiers.control);
        assert!(!modifiers.option);
    }

    #[test]
    fn test_event_mask_creation() {
        let mask = create_event_mask();
        // Should include left mouse down
        assert!(mask & (1 << CG_EVENT_LEFT_MOUSE_DOWN) != 0);
        // Should include key down
        assert!(mask & (1 << CG_EVENT_KEY_DOWN) != 0);
        // Should include scroll wheel
        assert!(mask & (1 << CG_EVENT_SCROLL_WHEEL) != 0);
    }

    #[test]
    fn test_event_tap_creation() {
        MachTimebase::init();

        // Should succeed even without permissions (creation just stores state)
        let result = EventTap::new();
        assert!(result.is_ok());

        let tap = result.unwrap();
        assert!(!tap.is_running());
    }

    #[test]
    fn test_convert_all_event_types() {
        assert_eq!(convert_event_type(CG_EVENT_LEFT_MOUSE_DOWN), Some(EventType::LeftMouseDown));
        assert_eq!(convert_event_type(CG_EVENT_LEFT_MOUSE_UP), Some(EventType::LeftMouseUp));
        assert_eq!(convert_event_type(CG_EVENT_RIGHT_MOUSE_DOWN), Some(EventType::RightMouseDown));
        assert_eq!(convert_event_type(CG_EVENT_RIGHT_MOUSE_UP), Some(EventType::RightMouseUp));
        assert_eq!(convert_event_type(CG_EVENT_MOUSE_MOVED), Some(EventType::MouseMoved));
        assert_eq!(convert_event_type(CG_EVENT_LEFT_MOUSE_DRAGGED), Some(EventType::LeftMouseDragged));
        assert_eq!(convert_event_type(CG_EVENT_RIGHT_MOUSE_DRAGGED), Some(EventType::RightMouseDragged));
        assert_eq!(convert_event_type(CG_EVENT_KEY_DOWN), Some(EventType::KeyDown));
        assert_eq!(convert_event_type(CG_EVENT_KEY_UP), Some(EventType::KeyUp));
        assert_eq!(convert_event_type(CG_EVENT_FLAGS_CHANGED), Some(EventType::FlagsChanged));
        assert_eq!(convert_event_type(CG_EVENT_SCROLL_WHEEL), Some(EventType::ScrollWheel));
        assert_eq!(convert_event_type(CG_EVENT_OTHER_MOUSE_DOWN), Some(EventType::OtherMouseDown));
        assert_eq!(convert_event_type(CG_EVENT_OTHER_MOUSE_UP), Some(EventType::OtherMouseUp));
        assert_eq!(convert_event_type(CG_EVENT_OTHER_MOUSE_DRAGGED), Some(EventType::OtherMouseDragged));
    }

    #[test]
    fn test_convert_flags_shift_only() {
        let modifiers = convert_flags_to_modifiers(CG_EVENT_FLAG_MASK_SHIFT);
        assert!(modifiers.shift);
        assert!(!modifiers.control);
        assert!(!modifiers.option);
        assert!(!modifiers.command);
        assert!(!modifiers.caps_lock);
        assert!(!modifiers.function);
    }

    #[test]
    fn test_convert_flags_control_only() {
        let modifiers = convert_flags_to_modifiers(CG_EVENT_FLAG_MASK_CONTROL);
        assert!(!modifiers.shift);
        assert!(modifiers.control);
        assert!(!modifiers.option);
        assert!(!modifiers.command);
    }

    #[test]
    fn test_convert_flags_option_only() {
        let modifiers = convert_flags_to_modifiers(CG_EVENT_FLAG_MASK_ALTERNATE);
        assert!(!modifiers.shift);
        assert!(!modifiers.control);
        assert!(modifiers.option);
        assert!(!modifiers.command);
    }

    #[test]
    fn test_convert_flags_command_only() {
        let modifiers = convert_flags_to_modifiers(CG_EVENT_FLAG_MASK_COMMAND);
        assert!(!modifiers.shift);
        assert!(!modifiers.control);
        assert!(!modifiers.option);
        assert!(modifiers.command);
    }

    #[test]
    fn test_convert_flags_caps_lock() {
        let modifiers = convert_flags_to_modifiers(CG_EVENT_FLAG_MASK_CAPS_LOCK);
        assert!(!modifiers.shift);
        assert!(!modifiers.control);
        assert!(!modifiers.option);
        assert!(!modifiers.command);
        assert!(modifiers.caps_lock);
        assert!(!modifiers.function);
    }

    #[test]
    fn test_convert_flags_function_key() {
        let modifiers = convert_flags_to_modifiers(CG_EVENT_FLAG_MASK_FN);
        assert!(!modifiers.shift);
        assert!(!modifiers.control);
        assert!(!modifiers.option);
        assert!(!modifiers.command);
        assert!(!modifiers.caps_lock);
        assert!(modifiers.function);
    }

    #[test]
    fn test_convert_flags_multiple_modifiers() {
        let flags = CG_EVENT_FLAG_MASK_SHIFT | CG_EVENT_FLAG_MASK_CONTROL | CG_EVENT_FLAG_MASK_ALTERNATE;
        let modifiers = convert_flags_to_modifiers(flags);
        assert!(modifiers.shift);
        assert!(modifiers.control);
        assert!(modifiers.option);
        assert!(!modifiers.command);
    }

    #[test]
    fn test_convert_flags_all_modifiers() {
        let flags = CG_EVENT_FLAG_MASK_SHIFT | CG_EVENT_FLAG_MASK_CONTROL
                  | CG_EVENT_FLAG_MASK_ALTERNATE | CG_EVENT_FLAG_MASK_COMMAND
                  | CG_EVENT_FLAG_MASK_CAPS_LOCK | CG_EVENT_FLAG_MASK_FN;
        let modifiers = convert_flags_to_modifiers(flags);
        assert!(modifiers.shift);
        assert!(modifiers.control);
        assert!(modifiers.option);
        assert!(modifiers.command);
        assert!(modifiers.caps_lock);
        assert!(modifiers.function);
    }

    #[test]
    fn test_convert_flags_none() {
        let modifiers = convert_flags_to_modifiers(0);
        assert!(!modifiers.shift);
        assert!(!modifiers.control);
        assert!(!modifiers.option);
        assert!(!modifiers.command);
        assert!(!modifiers.caps_lock);
        assert!(!modifiers.function);
    }

    #[test]
    fn test_event_mask_includes_all_event_types() {
        let mask = create_event_mask();

        // Mouse events
        assert!(mask & (1 << CG_EVENT_LEFT_MOUSE_DOWN) != 0);
        assert!(mask & (1 << CG_EVENT_LEFT_MOUSE_UP) != 0);
        assert!(mask & (1 << CG_EVENT_RIGHT_MOUSE_DOWN) != 0);
        assert!(mask & (1 << CG_EVENT_RIGHT_MOUSE_UP) != 0);
        assert!(mask & (1 << CG_EVENT_MOUSE_MOVED) != 0);
        assert!(mask & (1 << CG_EVENT_LEFT_MOUSE_DRAGGED) != 0);
        assert!(mask & (1 << CG_EVENT_RIGHT_MOUSE_DRAGGED) != 0);
        assert!(mask & (1 << CG_EVENT_OTHER_MOUSE_DOWN) != 0);
        assert!(mask & (1 << CG_EVENT_OTHER_MOUSE_UP) != 0);
        assert!(mask & (1 << CG_EVENT_OTHER_MOUSE_DRAGGED) != 0);

        // Keyboard events
        assert!(mask & (1 << CG_EVENT_KEY_DOWN) != 0);
        assert!(mask & (1 << CG_EVENT_KEY_UP) != 0);
        assert!(mask & (1 << CG_EVENT_FLAGS_CHANGED) != 0);

        // Scroll events
        assert!(mask & (1 << CG_EVENT_SCROLL_WHEEL) != 0);
    }

    #[test]
    fn test_create_raw_event_with_keyboard_data() {
        MachTimebase::init();

        let modifiers = ModifierFlags {
            shift: true,
            ..Default::default()
        };

        let event = create_raw_event(
            EventType::KeyDown,
            100.0,
            200.0,
            modifiers,
            Some(42),
            Some('a'),
            None,
            0,
        );

        assert_eq!(event.event_type, EventType::KeyDown);
        assert_eq!(event.coordinates, (100.0, 200.0));
        assert_eq!(event.key_code, Some(42));
        assert_eq!(event.character, Some('a'));
        assert!(event.modifiers.shift);
        assert!(event.scroll_delta.is_none());
        assert_eq!(event.click_count, 0);
    }

    #[test]
    fn test_create_raw_event_with_scroll_data() {
        MachTimebase::init();

        let event = create_raw_event(
            EventType::ScrollWheel,
            100.0,
            200.0,
            ModifierFlags::default(),
            None,
            None,
            Some((5.0, -10.0)),
            0,
        );

        assert_eq!(event.event_type, EventType::ScrollWheel);
        assert_eq!(event.coordinates, (100.0, 200.0));
        assert_eq!(event.scroll_delta, Some((5.0, -10.0)));
        assert!(event.key_code.is_none());
        assert!(event.character.is_none());
    }

    #[test]
    fn test_create_raw_event_with_click_count() {
        MachTimebase::init();

        let event = create_raw_event(
            EventType::LeftMouseDown,
            100.0,
            200.0,
            ModifierFlags::default(),
            None,
            None,
            None,
            2,
        );

        assert_eq!(event.event_type, EventType::LeftMouseDown);
        assert_eq!(event.click_count, 2);
    }

    #[test]
    fn test_event_tap_initial_state() {
        MachTimebase::init();

        let tap = EventTap::new().unwrap();

        assert!(!tap.is_running());
        assert_eq!(tap.event_count(), 0);
    }

    #[test]
    fn test_event_tap_stop_when_not_running() {
        MachTimebase::init();

        let mut tap = EventTap::new().unwrap();

        // Should not panic when stopping a non-running tap
        tap.stop();
        assert!(!tap.is_running());
    }

    #[test]
    fn test_cg_event_type_constants() {
        // Verify constants match expected values
        assert_eq!(CG_EVENT_LEFT_MOUSE_DOWN, 1);
        assert_eq!(CG_EVENT_LEFT_MOUSE_UP, 2);
        assert_eq!(CG_EVENT_RIGHT_MOUSE_DOWN, 3);
        assert_eq!(CG_EVENT_RIGHT_MOUSE_UP, 4);
        assert_eq!(CG_EVENT_MOUSE_MOVED, 5);
        assert_eq!(CG_EVENT_LEFT_MOUSE_DRAGGED, 6);
        assert_eq!(CG_EVENT_RIGHT_MOUSE_DRAGGED, 7);
        assert_eq!(CG_EVENT_KEY_DOWN, 10);
        assert_eq!(CG_EVENT_KEY_UP, 11);
        assert_eq!(CG_EVENT_FLAGS_CHANGED, 12);
        assert_eq!(CG_EVENT_SCROLL_WHEEL, 22);
        assert_eq!(CG_EVENT_OTHER_MOUSE_DOWN, 25);
        assert_eq!(CG_EVENT_OTHER_MOUSE_UP, 26);
        assert_eq!(CG_EVENT_OTHER_MOUSE_DRAGGED, 27);
    }

    #[test]
    fn test_create_raw_event_with_negative_coordinates() {
        MachTimebase::init();

        let event = create_raw_event(
            EventType::MouseMoved,
            -100.0,
            -200.0,
            ModifierFlags::default(),
            None,
            None,
            None,
            0,
        );

        assert_eq!(event.coordinates, (-100.0, -200.0));
    }

    #[test]
    fn test_create_raw_event_with_large_coordinates() {
        MachTimebase::init();

        let event = create_raw_event(
            EventType::MouseMoved,
            10000.0,
            20000.0,
            ModifierFlags::default(),
            None,
            None,
            None,
            0,
        );

        assert_eq!(event.coordinates, (10000.0, 20000.0));
    }

    #[test]
    fn test_get_current_cursor_state() {
        // Should not panic when called
        let _cursor = get_current_cursor_state();
    }
}
