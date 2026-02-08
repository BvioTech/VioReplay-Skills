//! Async Semantic Backfill using Accessibility API
//!
//! This module provides asynchronous semantic enrichment of captured events
//! by querying the macOS Accessibility API. It runs in a background thread
//! and never blocks the event capture pipeline.
//!
//! # Architecture
//!
//! The consumer thread continuously drains the ring buffer and queries
//! AXUIElementCopyElementAtPosition for each event's coordinates. The
//! semantic data is then written back to the event slot.
//!
//! # Failure Handling
//!
//! When AX queries fail (common with Electron apps), we attempt:
//! 1. AXManualAccessibility injection for Electron
//! 2. Spiral search around the original coordinates
//! 3. Mark as Failed if all attempts fail

use super::ring_buffer::{EventConsumer, ProcessedEventStore};
use super::types::{EventType, SemanticContext, SemanticSource, SemanticState};
use core_foundation::base::{CFRelease, CFTypeRef, TCFType};
use core_foundation::string::CFString;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;
use tracing::{debug, error, info, trace};

// Accessibility API types
type AXUIElementRef = CFTypeRef;
type AXError = i32;

// macOS Accessibility error codes — full API surface for error handling
#[allow(dead_code)]
mod ax_errors {
    use super::AXError;
    pub const K_AX_ERROR_SUCCESS: AXError = 0;
    pub const K_AX_ERROR_FAILURE: AXError = -25200;
    pub const K_AX_ERROR_ILLEGAL_ARGUMENT: AXError = -25201;
    pub const K_AX_ERROR_INVALID_UI_ELEMENT: AXError = -25202;
    pub const K_AX_ERROR_INVALID_UI_ELEMENT_OBSERVER: AXError = -25203;
    pub const K_AX_ERROR_CANNOT_COMPLETE: AXError = -25204;
    pub const K_AX_ERROR_NOT_IMPLEMENTED: AXError = -25205;
    pub const K_AX_ERROR_NOT_UI_ELEMENT: AXError = -25206;
    pub const K_AX_ERROR_NO_VALUE: AXError = -25212;
}
use ax_errors::*;

// FFI declarations for Accessibility API
extern "C" {
    fn AXUIElementCreateSystemWide() -> AXUIElementRef;
    fn AXUIElementCopyElementAtPosition(
        element: AXUIElementRef,
        x: f32,
        y: f32,
        element_at_position: *mut AXUIElementRef,
    ) -> AXError;
    fn AXUIElementCopyAttributeValue(
        element: AXUIElementRef,
        attribute: CFTypeRef,
        value: *mut CFTypeRef,
    ) -> AXError;
    fn AXUIElementGetPid(element: AXUIElementRef, pid: *mut i32) -> AXError;
    fn AXUIElementSetAttributeValue(
        element: AXUIElementRef,
        attribute: CFTypeRef,
        value: CFTypeRef,
    ) -> AXError;
}

/// Accessibility attribute names — full API surface for macOS AX queries
#[allow(dead_code)]
mod attributes {
    use core_foundation::string::CFString;

    pub fn role() -> CFString {
        CFString::new("AXRole")
    }

    pub fn title() -> CFString {
        CFString::new("AXTitle")
    }

    pub fn description() -> CFString {
        CFString::new("AXDescription")
    }

    pub fn value() -> CFString {
        CFString::new("AXValue")
    }

    pub fn identifier() -> CFString {
        CFString::new("AXIdentifier")
    }

    pub fn parent() -> CFString {
        CFString::new("AXParent")
    }

    pub fn window() -> CFString {
        CFString::new("AXWindow")
    }

    pub fn focused_window() -> CFString {
        CFString::new("AXFocusedWindow")
    }

    pub fn position() -> CFString {
        CFString::new("AXPosition")
    }

    pub fn size() -> CFString {
        CFString::new("AXSize")
    }

    pub fn frame() -> CFString {
        CFString::new("AXFrame")
    }

    pub fn manual_accessibility() -> CFString {
        CFString::new("AXManualAccessibility")
    }
}

/// Semantic backfill worker
pub struct SemanticBackfill {
    /// Worker thread handle
    thread_handle: Option<JoinHandle<()>>,
    /// Stop signal
    running: Arc<AtomicBool>,
    /// Whether to use vision fallback for OCR recovery
    vision_fallback_enabled: bool,
}

impl SemanticBackfill {
    /// Create a new semantic backfill worker
    pub fn new() -> Self {
        Self {
            thread_handle: None,
            running: Arc::new(AtomicBool::new(false)),
            vision_fallback_enabled: false,
        }
    }

    /// Create with vision fallback enabled
    pub fn with_vision_fallback(mut self) -> Self {
        self.vision_fallback_enabled = true;
        self
    }

    /// Start the background worker
    pub fn start(
        &mut self,
        consumer: EventConsumer,
        store: Arc<ProcessedEventStore>,
    ) -> Result<(), crate::Error> {
        if self.running.swap(true, Ordering::SeqCst) {
            return Err(crate::Error::Accessibility(
                "Semantic backfill already running".into(),
            ));
        }

        let running = Arc::clone(&self.running);
        let vision_enabled = self.vision_fallback_enabled;

        let handle = thread::Builder::new()
            .name("semantic-backfill".into())
            .spawn(move || {
                run_backfill_loop(consumer, store, running, vision_enabled);
            })
            .map_err(|e| {
                crate::Error::Accessibility(format!("Failed to spawn backfill thread: {}", e))
            })?;

        self.thread_handle = Some(handle);
        info!("Semantic backfill started (vision_fallback={})", self.vision_fallback_enabled);

        Ok(())
    }

    /// Stop the background worker
    pub fn stop(&mut self) {
        if !self.running.swap(false, Ordering::SeqCst) {
            return;
        }

        if let Some(handle) = self.thread_handle.take() {
            let _ = handle.join();
        }

        info!("Semantic backfill stopped");
    }

    /// Check if worker is running
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }
}

impl Default for SemanticBackfill {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for SemanticBackfill {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Main backfill loop
fn run_backfill_loop(
    mut consumer: EventConsumer,
    store: Arc<ProcessedEventStore>,
    running: Arc<AtomicBool>,
    vision_enabled: bool,
) {
    // Create system-wide element for position queries
    let system_wide = unsafe { AXUIElementCreateSystemWide() };
    if system_wide.is_null() {
        error!("Failed to create system-wide accessibility element");
        return;
    }

    // Create vision fallback if enabled
    let vision_fallback = if vision_enabled {
        Some(crate::semantic::vision_fallback::VisionFallback::new())
    } else {
        None
    };

    info!("Semantic backfill loop started");

    while running.load(Ordering::SeqCst) {
        // Process available events
        let batch = consumer.pop_batch(100);

        if batch.is_empty() {
            // No events, sleep briefly to avoid spinning
            thread::sleep(Duration::from_millis(5));
            continue;
        }

        for slot in batch {
            // Only process events that need semantic data
            if slot.semantic_state.load(Ordering::Acquire) != SemanticState::Pending {
                store.store(slot);
                continue;
            }

            // Only query for click events (optimization)
            if !should_query_semantics(&slot.event.event_type) {
                slot.semantic_state
                    .store(SemanticState::Filled, Ordering::Release);
                store.store(slot);
                continue;
            }

            // Query accessibility
            let (x, y) = slot.event.coordinates;
            match query_element_at_position(system_wide, x as f32, y as f32) {
                Ok(context) => {
                    // Safety: We're the only writer to semantic_data
                    unsafe {
                        slot.set_semantic(context);
                    }
                    trace!("Semantic data filled for event at ({}, {})", x, y);
                }
                Err(e) => {
                    // Try recovery strategies (spiral search)
                    if let Some(context) = try_recovery_strategies(system_wide, x as f32, y as f32)
                    {
                        unsafe {
                            slot.set_semantic(context);
                        }
                        debug!("Semantic data recovered via spiral search for ({}, {})", x, y);
                    } else if let Some(context) = try_vision_fallback(&vision_fallback, x, y) {
                        // Vision OCR fallback
                        unsafe {
                            slot.set_semantic(context);
                        }
                        debug!("Semantic data recovered via Vision OCR for ({}, {})", x, y);
                    } else {
                        slot.mark_failed();
                        debug!(
                            "Semantic lookup failed for ({}, {}): {:?}",
                            x, y, e
                        );
                    }
                }
            }

            store.store(slot);
        }
    }

    // Cleanup
    unsafe {
        CFRelease(system_wide);
    }

    info!("Semantic backfill loop stopped");
}

/// Try vision OCR fallback for failed accessibility queries
fn try_vision_fallback(
    vision: &Option<crate::semantic::vision_fallback::VisionFallback>,
    x: f64,
    y: f64,
) -> Option<SemanticContext> {
    let vision = vision.as_ref()?;
    let result = vision.run(x, y);
    result.semantic
}

/// Check if we should query semantics for this event type
fn should_query_semantics(event_type: &EventType) -> bool {
    matches!(
        event_type,
        EventType::LeftMouseDown
            | EventType::LeftMouseUp
            | EventType::RightMouseDown
            | EventType::RightMouseUp
    )
}

/// Query the UI element at a given position
fn query_element_at_position(
    system_wide: AXUIElementRef,
    x: f32,
    y: f32,
) -> Result<SemanticContext, AXError> {
    let mut element: AXUIElementRef = std::ptr::null_mut();

    let error = unsafe { AXUIElementCopyElementAtPosition(system_wide, x, y, &mut element) };

    if error != K_AX_ERROR_SUCCESS || element.is_null() {
        return Err(error);
    }

    // Extract semantic data from the element
    let context = extract_semantic_context(element);

    // Release the element
    unsafe {
        CFRelease(element);
    }

    Ok(context)
}

/// Extract semantic context from an accessibility element
#[allow(clippy::field_reassign_with_default)]
fn extract_semantic_context(element: AXUIElementRef) -> SemanticContext {
    let mut context = SemanticContext::default();

    // Get role
    context.ax_role = get_string_attribute(element, &attributes::role());

    // Get title
    context.title = get_string_attribute(element, &attributes::title())
        .or_else(|| get_string_attribute(element, &attributes::description()));

    // Get identifier
    context.identifier = get_string_attribute(element, &attributes::identifier());

    // Get value
    context.value = get_string_attribute(element, &attributes::value());

    // Get PID
    let mut pid: i32 = 0;
    if unsafe { AXUIElementGetPid(element, &mut pid) } == K_AX_ERROR_SUCCESS {
        context.pid = Some(pid);
    }

    // Get parent info
    if let Some(parent) = get_element_attribute(element, &attributes::parent()) {
        context.parent_role = get_string_attribute(parent, &attributes::role());
        context.parent_title = get_string_attribute(parent, &attributes::title());
        unsafe {
            CFRelease(parent);
        }
    }

    // Get window info
    if let Some(window) = get_element_attribute(element, &attributes::window()) {
        context.window_title = get_string_attribute(window, &attributes::title());
        unsafe {
            CFRelease(window);
        }
    }

    // Build ancestor chain (up to 5 levels)
    let mut ancestors = Vec::new();
    let mut current = element;
    let mut depth = 0;

    while depth < 5 {
        if let Some(parent) = get_element_attribute(current, &attributes::parent()) {
            let role = get_string_attribute(parent, &attributes::role());
            let title = get_string_attribute(parent, &attributes::title());

            if let Some(r) = role {
                ancestors.push((r, title));
            }

            if current != element {
                unsafe {
                    CFRelease(current);
                }
            }
            current = parent;
            depth += 1;
        } else {
            break;
        }
    }

    if current != element && !current.is_null() {
        unsafe {
            CFRelease(current);
        }
    }

    context.ancestors = ancestors;
    context.source = SemanticSource::Accessibility;
    context.confidence = 1.0;

    context
}

/// Get a string attribute from an element
fn get_string_attribute(element: AXUIElementRef, attribute: &CFString) -> Option<String> {
    let mut value: CFTypeRef = std::ptr::null_mut();

    let error = unsafe {
        AXUIElementCopyAttributeValue(element, attribute.as_concrete_TypeRef() as *const c_void, &mut value)
    };

    if error != K_AX_ERROR_SUCCESS || value.is_null() {
        return None;
    }

    // Convert CFString to Rust String
    let result = unsafe {
        if core_foundation::base::CFGetTypeID(value)
            == core_foundation::string::CFString::type_id()
        {
            let cf_string = core_foundation::string::CFString::wrap_under_get_rule(value as _);
            Some(cf_string.to_string())
        } else {
            None
        }
    };

    unsafe {
        CFRelease(value);
    }

    result
}

/// Get an element attribute from an element
fn get_element_attribute(element: AXUIElementRef, attribute: &CFString) -> Option<AXUIElementRef> {
    let mut value: CFTypeRef = std::ptr::null_mut();

    let error = unsafe {
        AXUIElementCopyAttributeValue(element, attribute.as_concrete_TypeRef() as *const c_void, &mut value)
    };

    if error != K_AX_ERROR_SUCCESS || value.is_null() {
        return None;
    }

    Some(value)
}

/// Try recovery strategies for failed accessibility queries
fn try_recovery_strategies(
    system_wide: AXUIElementRef,
    x: f32,
    y: f32,
) -> Option<SemanticContext> {
    // Strategy 1: Spiral search around the original coordinates
    let offsets = [
        (0.0, 5.0),
        (5.0, 0.0),
        (0.0, -5.0),
        (-5.0, 0.0),
        (5.0, 5.0),
        (-5.0, 5.0),
        (5.0, -5.0),
        (-5.0, -5.0),
        (0.0, 10.0),
        (10.0, 0.0),
    ];

    for (dx, dy) in offsets.iter() {
        if let Ok(mut context) = query_element_at_position(system_wide, x + dx, y + dy) {
            // Mark as recovered via spatial search
            context.confidence = 0.9;
            return Some(context);
        }
    }

    // Strategy 2: Try AXManualAccessibility injection (for Electron apps)
    // This requires identifying the application first
    // For now, we skip this as it requires more infrastructure

    None
}

/// Attempt to enable AXManualAccessibility for Electron apps
///
/// # Safety
/// The `app_element` must be a valid AXUIElementRef obtained from the Accessibility API.
pub unsafe fn inject_manual_accessibility(app_element: AXUIElementRef) -> bool {
    let attribute = attributes::manual_accessibility();
    let value = core_foundation::boolean::CFBoolean::true_value();

    let error = AXUIElementSetAttributeValue(
        app_element,
        attribute.as_concrete_TypeRef() as *const c_void,
        value.as_concrete_TypeRef() as *const c_void,
    );

    error == K_AX_ERROR_SUCCESS
}

/// Get the frontmost application's accessibility element
pub fn get_frontmost_application() -> Option<AXUIElementRef> {
    let system_wide = unsafe { AXUIElementCreateSystemWide() };
    if system_wide.is_null() {
        return None;
    }

    let focused_app_attr = CFString::new("AXFocusedApplication");
    let app = get_element_attribute(system_wide, &focused_app_attr);

    unsafe {
        CFRelease(system_wide);
    }

    app
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn test_ax_error_codes() {
        assert_eq!(K_AX_ERROR_SUCCESS, 0);
        assert!(K_AX_ERROR_CANNOT_COMPLETE < 0);
    }

    #[test]
    fn test_should_query_semantics() {
        assert!(should_query_semantics(&EventType::LeftMouseDown));
        assert!(should_query_semantics(&EventType::RightMouseUp));
        assert!(!should_query_semantics(&EventType::MouseMoved));
        assert!(!should_query_semantics(&EventType::KeyDown));
    }

    #[test]
    fn test_semantic_context_default() {
        let context = SemanticContext::default();
        assert!(context.ax_role.is_none());
        assert_eq!(context.confidence, 1.0);
        assert_eq!(context.source, SemanticSource::Accessibility);
    }

    #[test]
    fn test_semantic_backfill_lifecycle() {
        let backfill = SemanticBackfill::new();
        assert!(!backfill.is_running());

        // Test default construction
        let default_backfill = SemanticBackfill::default();
        assert!(!default_backfill.is_running());
    }

    #[test]
    fn test_semantic_backfill_double_start() {
        use crate::time::timebase::MachTimebase;
        use super::super::ring_buffer::{EventRingBuffer, ProcessedEventStore};

        MachTimebase::init();
        let mut backfill = SemanticBackfill::new();
        let buffer = EventRingBuffer::with_capacity(64);
        let (_, consumer) = buffer.split();
        let store = Arc::new(ProcessedEventStore::new(100));

        // Start once - should succeed
        let result1 = backfill.start(consumer, store.clone());
        assert!(result1.is_ok());
        assert!(backfill.is_running());

        // Try to start again - should fail
        let buffer2 = EventRingBuffer::with_capacity(64);
        let (_, consumer2) = buffer2.split();
        let result2 = backfill.start(consumer2, store);
        assert!(result2.is_err());

        backfill.stop();
        assert!(!backfill.is_running());
    }

    #[test]
    fn test_semantic_backfill_stop_when_not_running() {
        let mut backfill = SemanticBackfill::new();
        assert!(!backfill.is_running());

        // Stopping when not running should be safe
        backfill.stop();
        assert!(!backfill.is_running());
    }

    #[test]
    fn test_should_query_semantics_comprehensive() {
        // Should query for all mouse down/up events
        assert!(should_query_semantics(&EventType::LeftMouseDown));
        assert!(should_query_semantics(&EventType::LeftMouseUp));
        assert!(should_query_semantics(&EventType::RightMouseDown));
        assert!(should_query_semantics(&EventType::RightMouseUp));

        // Should not query for other events
        assert!(!should_query_semantics(&EventType::MouseMoved));
        assert!(!should_query_semantics(&EventType::LeftMouseDragged));
        assert!(!should_query_semantics(&EventType::RightMouseDragged));
        assert!(!should_query_semantics(&EventType::KeyDown));
        assert!(!should_query_semantics(&EventType::KeyUp));
        assert!(!should_query_semantics(&EventType::ScrollWheel));
        assert!(!should_query_semantics(&EventType::FlagsChanged));
        assert!(!should_query_semantics(&EventType::OtherMouseDown));
        assert!(!should_query_semantics(&EventType::OtherMouseUp));
    }

    #[test]
    fn test_semantic_context_confidence_variation() {
        let mut context = SemanticContext::default();
        assert_eq!(context.confidence, 1.0);

        // Test confidence adjustment for recovery
        context.confidence = 0.9;
        assert_eq!(context.confidence, 0.9);

        context.confidence = 0.5;
        assert!(context.confidence < 1.0);
    }

    #[test]
    fn test_semantic_context_sources() {
        let mut context = SemanticContext::default();
        assert_eq!(context.source, SemanticSource::Accessibility);

        context.source = SemanticSource::Vision;
        assert_eq!(context.source, SemanticSource::Vision);

        context.source = SemanticSource::Inferred;
        assert_eq!(context.source, SemanticSource::Inferred);

        context.source = SemanticSource::InjectedAccessibility;
        assert_eq!(context.source, SemanticSource::InjectedAccessibility);
    }

    #[test]
    fn test_semantic_context_ancestor_chain() {
        let mut context = SemanticContext::default();
        assert!(context.ancestors.is_empty());

        // Build an ancestor chain
        context.ancestors.push(("AXGroup".to_string(), Some("Container".to_string())));
        context.ancestors.push(("AXWindow".to_string(), Some("Main Window".to_string())));
        context.ancestors.push(("AXApplication".to_string(), None));

        assert_eq!(context.ancestors.len(), 3);
        assert_eq!(context.ancestors[0].0, "AXGroup");
        assert_eq!(context.ancestors[1].1, Some("Main Window".to_string()));
        assert_eq!(context.ancestors[2].1, None);
    }

    #[test]
    fn test_semantic_backfill_drop_cleanup() {
        use crate::time::timebase::MachTimebase;
        use super::super::ring_buffer::{EventRingBuffer, ProcessedEventStore};

        MachTimebase::init();
        let mut backfill = SemanticBackfill::new();
        let buffer = EventRingBuffer::with_capacity(64);
        let (_, consumer) = buffer.split();
        let store = Arc::new(ProcessedEventStore::new(100));

        let _ = backfill.start(consumer, store);
        assert!(backfill.is_running());

        // Drop should stop the worker
        drop(backfill);
        // Test passes if no panic occurs
    }

    #[test]
    fn test_ax_error_code_constants() {
        // Verify all error codes are properly defined
        assert_eq!(K_AX_ERROR_SUCCESS, 0);
        assert_eq!(K_AX_ERROR_FAILURE, -25200);
        assert_eq!(K_AX_ERROR_ILLEGAL_ARGUMENT, -25201);
        assert_eq!(K_AX_ERROR_INVALID_UI_ELEMENT, -25202);
        assert_eq!(K_AX_ERROR_INVALID_UI_ELEMENT_OBSERVER, -25203);
        assert_eq!(K_AX_ERROR_CANNOT_COMPLETE, -25204);
        assert_eq!(K_AX_ERROR_NOT_IMPLEMENTED, -25205);
        assert_eq!(K_AX_ERROR_NOT_UI_ELEMENT, -25206);
        assert_eq!(K_AX_ERROR_NO_VALUE, -25212);
    }

    #[test]
    fn test_attribute_names() {
        // Test that attribute name constructors work
        let role = attributes::role();
        assert_eq!(role.to_string(), "AXRole");

        let title = attributes::title();
        assert_eq!(title.to_string(), "AXTitle");

        let description = attributes::description();
        assert_eq!(description.to_string(), "AXDescription");

        let value = attributes::value();
        assert_eq!(value.to_string(), "AXValue");

        let identifier = attributes::identifier();
        assert_eq!(identifier.to_string(), "AXIdentifier");

        let parent = attributes::parent();
        assert_eq!(parent.to_string(), "AXParent");

        let window = attributes::window();
        assert_eq!(window.to_string(), "AXWindow");

        let manual = attributes::manual_accessibility();
        assert_eq!(manual.to_string(), "AXManualAccessibility");
    }

    #[test]
    fn test_attribute_names_all_variants() {
        // Test all attribute name constructors
        let focused_window = attributes::focused_window();
        assert_eq!(focused_window.to_string(), "AXFocusedWindow");

        let position = attributes::position();
        assert_eq!(position.to_string(), "AXPosition");

        let size = attributes::size();
        assert_eq!(size.to_string(), "AXSize");

        let frame = attributes::frame();
        assert_eq!(frame.to_string(), "AXFrame");
    }

    #[test]
    fn test_should_query_semantics_all_event_types() {
        // Test that only click events are queried
        assert!(should_query_semantics(&EventType::LeftMouseDown));
        assert!(should_query_semantics(&EventType::LeftMouseUp));
        assert!(should_query_semantics(&EventType::RightMouseDown));
        assert!(should_query_semantics(&EventType::RightMouseUp));

        // All other event types should not be queried
        assert!(!should_query_semantics(&EventType::MouseMoved));
        assert!(!should_query_semantics(&EventType::LeftMouseDragged));
        assert!(!should_query_semantics(&EventType::RightMouseDragged));
        assert!(!should_query_semantics(&EventType::OtherMouseDragged));
        assert!(!should_query_semantics(&EventType::KeyDown));
        assert!(!should_query_semantics(&EventType::KeyUp));
        assert!(!should_query_semantics(&EventType::FlagsChanged));
        assert!(!should_query_semantics(&EventType::ScrollWheel));
        assert!(!should_query_semantics(&EventType::OtherMouseDown));
        assert!(!should_query_semantics(&EventType::OtherMouseUp));
    }

    #[test]
    fn test_semantic_context_building() {
        let mut context = SemanticContext::default();

        // Build a complete semantic context
        context.ax_role = Some("AXButton".to_string());
        context.title = Some("Submit".to_string());
        context.identifier = Some("submit-btn".to_string());
        context.value = Some("enabled".to_string());
        context.parent_role = Some("AXGroup".to_string());
        context.parent_title = Some("Form".to_string());
        context.window_title = Some("Registration".to_string());
        context.app_bundle_id = Some("com.example.app".to_string());
        context.app_name = Some("Example App".to_string());
        context.pid = Some(1234);
        context.window_id = Some(5678);
        context.frame = Some((100.0, 200.0, 50.0, 30.0));
        context.source = SemanticSource::Accessibility;
        context.confidence = 1.0;

        // Verify all fields are set correctly
        assert_eq!(context.ax_role, Some("AXButton".to_string()));
        assert_eq!(context.title, Some("Submit".to_string()));
        assert_eq!(context.identifier, Some("submit-btn".to_string()));
        assert_eq!(context.value, Some("enabled".to_string()));
        assert_eq!(context.pid, Some(1234));
        assert_eq!(context.window_id, Some(5678));
        assert_eq!(context.frame, Some((100.0, 200.0, 50.0, 30.0)));
        assert_eq!(context.source, SemanticSource::Accessibility);
        assert_eq!(context.confidence, 1.0);
    }

    #[test]
    fn test_semantic_context_ancestor_chain_building() {
        let mut context = SemanticContext::default();

        // Build a realistic ancestor chain
        context.ancestors.push(("AXButton".to_string(), Some("Click me".to_string())));
        context.ancestors.push(("AXGroup".to_string(), Some("Button group".to_string())));
        context.ancestors.push(("AXScrollArea".to_string(), None));
        context.ancestors.push(("AXWindow".to_string(), Some("Main Window".to_string())));
        context.ancestors.push(("AXApplication".to_string(), Some("MyApp".to_string())));

        assert_eq!(context.ancestors.len(), 5);

        // Verify first element
        assert_eq!(context.ancestors[0].0, "AXButton");
        assert_eq!(context.ancestors[0].1, Some("Click me".to_string()));

        // Verify element without title
        assert_eq!(context.ancestors[2].0, "AXScrollArea");
        assert_eq!(context.ancestors[2].1, None);

        // Verify last element
        assert_eq!(context.ancestors[4].0, "AXApplication");
    }

    #[test]
    fn test_semantic_source_variants() {
        // Test all semantic source variants
        assert_eq!(SemanticSource::Accessibility, SemanticSource::Accessibility);
        assert_eq!(SemanticSource::Vision, SemanticSource::Vision);
        assert_eq!(SemanticSource::Inferred, SemanticSource::Inferred);
        assert_eq!(SemanticSource::InjectedAccessibility, SemanticSource::InjectedAccessibility);

        // Test inequality
        assert_ne!(SemanticSource::Accessibility, SemanticSource::Vision);
        assert_ne!(SemanticSource::Inferred, SemanticSource::InjectedAccessibility);
    }

    #[test]
    fn test_semantic_context_ocr_text() {
        let mut context = SemanticContext::default();
        context.source = SemanticSource::Vision;
        context.ocr_text = Some("Hello World".to_string());
        context.confidence = 0.8;

        assert_eq!(context.source, SemanticSource::Vision);
        assert_eq!(context.ocr_text, Some("Hello World".to_string()));
        assert_eq!(context.confidence, 0.8);
    }

    #[test]
    fn test_semantic_backfill_multiple_stops() {
        let mut backfill = SemanticBackfill::new();
        assert!(!backfill.is_running());

        // Stop when not running should be safe
        backfill.stop();
        assert!(!backfill.is_running());

        // Multiple stops should be safe
        backfill.stop();
        backfill.stop();
        assert!(!backfill.is_running());
    }

    #[test]
    fn test_semantic_context_with_frame_bounds() {
        let mut context = SemanticContext::default();

        // Test various frame configurations
        context.frame = Some((0.0, 0.0, 100.0, 100.0)); // Top-left origin
        assert_eq!(context.frame, Some((0.0, 0.0, 100.0, 100.0)));

        context.frame = Some((1920.0, 1080.0, 200.0, 150.0)); // Bottom-right
        assert_eq!(context.frame, Some((1920.0, 1080.0, 200.0, 150.0)));

        context.frame = Some((0.5, 0.5, 1.5, 1.5)); // Fractional coordinates
        assert_eq!(context.frame, Some((0.5, 0.5, 1.5, 1.5)));
    }

    #[test]
    fn test_semantic_context_confidence_range() {
        let mut context = SemanticContext::default();

        // Test various confidence levels
        context.confidence = 1.0;
        assert_eq!(context.confidence, 1.0);

        context.confidence = 0.9;
        assert_eq!(context.confidence, 0.9);

        context.confidence = 0.5;
        assert_eq!(context.confidence, 0.5);

        context.confidence = 0.0;
        assert_eq!(context.confidence, 0.0);
    }

    #[test]
    fn test_semantic_context_serialization() {
        let mut context = SemanticContext::default();
        context.ax_role = Some("AXButton".to_string());
        context.title = Some("Click Me".to_string());
        context.confidence = 0.95;
        context.ancestors.push(("AXWindow".to_string(), Some("Main".to_string())));

        // Test serialization
        let serialized = serde_json::to_string(&context).unwrap();
        assert!(serialized.contains("AXButton"));
        assert!(serialized.contains("Click Me"));

        // Test deserialization
        let deserialized: SemanticContext = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized.ax_role, Some("AXButton".to_string()));
        assert_eq!(deserialized.title, Some("Click Me".to_string()));
        assert_eq!(deserialized.confidence, 0.95);
        assert_eq!(deserialized.ancestors.len(), 1);
    }

    #[test]
    fn test_ax_error_code_all_values() {
        // Verify all error code constants are negative (except SUCCESS)
        assert_eq!(K_AX_ERROR_SUCCESS, 0);
        assert!(K_AX_ERROR_FAILURE < 0);
        assert!(K_AX_ERROR_ILLEGAL_ARGUMENT < 0);
        assert!(K_AX_ERROR_INVALID_UI_ELEMENT < 0);
        assert!(K_AX_ERROR_INVALID_UI_ELEMENT_OBSERVER < 0);
        assert!(K_AX_ERROR_CANNOT_COMPLETE < 0);
        assert!(K_AX_ERROR_NOT_IMPLEMENTED < 0);
        assert!(K_AX_ERROR_NOT_UI_ELEMENT < 0);
        assert!(K_AX_ERROR_NO_VALUE < 0);

        // Verify they are all distinct
        let error_codes = vec![
            K_AX_ERROR_SUCCESS,
            K_AX_ERROR_FAILURE,
            K_AX_ERROR_ILLEGAL_ARGUMENT,
            K_AX_ERROR_INVALID_UI_ELEMENT,
            K_AX_ERROR_INVALID_UI_ELEMENT_OBSERVER,
            K_AX_ERROR_CANNOT_COMPLETE,
            K_AX_ERROR_NOT_IMPLEMENTED,
            K_AX_ERROR_NOT_UI_ELEMENT,
            K_AX_ERROR_NO_VALUE,
        ];
        let unique_codes: std::collections::HashSet<_> = error_codes.iter().collect();
        assert_eq!(unique_codes.len(), error_codes.len());
    }

    #[test]
    fn test_semantic_source_debug_format() {
        let source = SemanticSource::Accessibility;
        let debug_str = format!("{:?}", source);
        assert!(debug_str.contains("Accessibility"));

        let vision_source = SemanticSource::Vision;
        let vision_debug = format!("{:?}", vision_source);
        assert!(vision_debug.contains("Vision"));
    }

    #[test]
    fn test_semantic_context_clone() {
        let mut context = SemanticContext::default();
        context.ax_role = Some("AXButton".to_string());
        context.title = Some("Test".to_string());
        context.ancestors.push(("AXWindow".to_string(), None));

        let cloned = context.clone();
        assert_eq!(cloned.ax_role, context.ax_role);
        assert_eq!(cloned.title, context.title);
        assert_eq!(cloned.ancestors.len(), context.ancestors.len());
    }
}
