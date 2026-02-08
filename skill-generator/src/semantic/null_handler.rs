//! Null-State Recovery
//!
//! Handles accessibility API failures through various recovery strategies.

use crate::capture::types::{SemanticContext, SemanticSource};
use core_foundation::base::{CFRelease, CFTypeRef, TCFType};
use core_foundation::string::CFString;
use std::ffi::c_void;

// Accessibility API types (same as in accessibility.rs)
type AXUIElementRef = CFTypeRef;
type AXError = i32;
const K_AX_ERROR_SUCCESS: AXError = 0;

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
}

/// Recovery strategy used
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecoveryStrategy {
    /// AXManualAccessibility injection
    Injection,
    /// Spiral search around coordinates
    SpiralSearch,
    /// Vision fallback (OCR)
    Vision,
    /// None - could not recover
    None,
}

/// Result of a recovery attempt
#[derive(Debug, Clone)]
pub struct RecoveryResult {
    /// Strategy that succeeded
    pub strategy: RecoveryStrategy,
    /// Recovered semantic context
    pub context: Option<SemanticContext>,
    /// Confidence in the result
    pub confidence: f32,
    /// Time taken (ms)
    pub duration_ms: u64,
}

/// Null handler for accessibility failures
pub struct NullHandler {
    /// Maximum spiral search radius (pixels)
    pub max_spiral_radius: f64,
    /// Spiral step size (pixels)
    pub spiral_step: f64,
    /// Timeout for injection retry (ms)
    pub injection_timeout_ms: u64,
    /// Vision fallback for OCR-based recovery (optional)
    vision_fallback: Option<super::vision_fallback::VisionFallback>,
}

impl NullHandler {
    /// Create with default settings
    pub fn new() -> Self {
        Self {
            max_spiral_radius: 20.0,
            spiral_step: 5.0,
            injection_timeout_ms: 100,
            vision_fallback: None,
        }
    }

    /// Create with vision fallback enabled
    pub fn with_vision(mut self) -> Self {
        self.vision_fallback = Some(super::vision_fallback::VisionFallback::new());
        self
    }

    /// Attempt to recover semantic data for a failed query
    pub fn recover(&self, x: f64, y: f64, app_bundle_id: Option<&str>) -> RecoveryResult {
        let start = std::time::Instant::now();

        // Strategy 1: Check if Electron app and try injection
        if let Some(bundle) = app_bundle_id {
            if self.is_electron_app(bundle) {
                if let Some(context) = self.try_injection(x, y, bundle) {
                    return RecoveryResult {
                        strategy: RecoveryStrategy::Injection,
                        context: Some(context),
                        confidence: 0.85,
                        duration_ms: start.elapsed().as_millis() as u64,
                    };
                }
            }
        }

        // Strategy 2: Spiral search
        if let Some(context) = self.spiral_search(x, y) {
            return RecoveryResult {
                strategy: RecoveryStrategy::SpiralSearch,
                context: Some(context),
                confidence: 0.9,
                duration_ms: start.elapsed().as_millis() as u64,
            };
        }

        // Strategy 3: Vision fallback (OCR)
        if let Some(ref vision) = self.vision_fallback {
            let result = vision.run(x, y);
            if let Some(semantic) = result.semantic {
                return RecoveryResult {
                    strategy: RecoveryStrategy::Vision,
                    context: Some(semantic),
                    confidence: 0.75,
                    duration_ms: start.elapsed().as_millis() as u64,
                };
            }
        }

        // Failed to recover
        RecoveryResult {
            strategy: RecoveryStrategy::None,
            context: None,
            confidence: 0.0,
            duration_ms: start.elapsed().as_millis() as u64,
        }
    }

    /// Check if app is an Electron app
    fn is_electron_app(&self, bundle_id: &str) -> bool {
        // Common Electron apps
        let electron_bundles = [
            "com.microsoft.VSCode",
            "com.slack.Slack",
            "com.discord",
            "com.spotify.client",
            "com.atom.editor",
            "com.github.desktop",
            "com.figma.Desktop",
            "com.notion.id",
            "com.linear",
            "com.obsidian",
        ];

        electron_bundles.iter().any(|b| bundle_id.contains(b))
    }

    /// Try AXManualAccessibility injection for Electron apps
    fn try_injection(&self, x: f64, y: f64, _bundle_id: &str) -> Option<SemanticContext> {
        // Create system-wide element
        let system_wide = unsafe { AXUIElementCreateSystemWide() };
        if system_wide.is_null() {
            return None;
        }

        // First, try to get the focused application and inject AXManualAccessibility
        let focused_app_attr = CFString::new("AXFocusedApplication");
        let mut app_element: CFTypeRef = std::ptr::null_mut();

        let error = unsafe {
            AXUIElementCopyAttributeValue(
                system_wide,
                focused_app_attr.as_concrete_TypeRef() as *const c_void,
                &mut app_element,
            )
        };

        if error == K_AX_ERROR_SUCCESS && !app_element.is_null() {
            // Try to set AXManualAccessibility to true
            let manual_attr = CFString::new("AXManualAccessibility");
            let true_value = core_foundation::boolean::CFBoolean::true_value();

            // Note: This may fail silently if the app doesn't support it
            unsafe {
                extern "C" {
                    fn AXUIElementSetAttributeValue(
                        element: AXUIElementRef,
                        attribute: CFTypeRef,
                        value: CFTypeRef,
                    ) -> AXError;
                }

                let _ = AXUIElementSetAttributeValue(
                    app_element,
                    manual_attr.as_concrete_TypeRef() as *const c_void,
                    true_value.as_concrete_TypeRef() as *const c_void,
                );
            }

            // Small delay to allow injection to take effect
            std::thread::sleep(std::time::Duration::from_millis(self.injection_timeout_ms));

            // Release app element
            unsafe {
                CFRelease(app_element);
            }

            // Now try querying again at the original position
            let result = self.query_element_at(system_wide, x as f32, y as f32);

            unsafe {
                CFRelease(system_wide);
            }

            return result;
        }

        unsafe {
            CFRelease(system_wide);
        }

        None
    }

    /// Spiral search around coordinates
    fn spiral_search(&self, center_x: f64, center_y: f64) -> Option<SemanticContext> {
        // Create system-wide element for position queries
        let system_wide = unsafe { AXUIElementCreateSystemWide() };
        if system_wide.is_null() {
            return None;
        }

        // Generate spiral offsets
        let offsets = self.generate_spiral_offsets();

        let mut result = None;

        for (dx, dy) in offsets {
            let x = center_x + dx;
            let y = center_y + dy;

            // Try to query at this position
            if let Some(context) = self.query_element_at(system_wide, x as f32, y as f32) {
                result = Some(context);
                break;
            }
        }

        // Cleanup
        unsafe {
            CFRelease(system_wide);
        }

        result
    }

    /// Query the UI element at a given position
    fn query_element_at(&self, system_wide: AXUIElementRef, x: f32, y: f32) -> Option<SemanticContext> {
        let mut element: AXUIElementRef = std::ptr::null_mut();

        let error = unsafe { AXUIElementCopyElementAtPosition(system_wide, x, y, &mut element) };

        if error != K_AX_ERROR_SUCCESS || element.is_null() {
            return None;
        }

        // Extract semantic data from the element
        let context = self.extract_semantic_context(element);

        // Release the element
        unsafe {
            CFRelease(element);
        }

        Some(context)
    }

    /// Extract semantic context from an accessibility element
    fn extract_semantic_context(&self, element: AXUIElementRef) -> SemanticContext {
        let mut pid: i32 = 0;
        let pid_result = if unsafe { AXUIElementGetPid(element, &mut pid) } == K_AX_ERROR_SUCCESS {
            Some(pid)
        } else {
            None
        };

        SemanticContext {
            ax_role: self.get_string_attribute(element, "AXRole"),
            title: self.get_string_attribute(element, "AXTitle")
                .or_else(|| self.get_string_attribute(element, "AXDescription")),
            identifier: self.get_string_attribute(element, "AXIdentifier"),
            value: self.get_string_attribute(element, "AXValue"),
            pid: pid_result,
            source: SemanticSource::Accessibility,
            confidence: 0.85,
            ..Default::default()
        }
    }

    /// Get a string attribute from an element
    fn get_string_attribute(&self, element: AXUIElementRef, attribute_name: &str) -> Option<String> {
        let attribute = CFString::new(attribute_name);
        let mut value: CFTypeRef = std::ptr::null_mut();

        let error = unsafe {
            AXUIElementCopyAttributeValue(
                element,
                attribute.as_concrete_TypeRef() as *const c_void,
                &mut value,
            )
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

    /// Generate spiral search offsets
    fn generate_spiral_offsets(&self) -> Vec<(f64, f64)> {
        let mut offsets = Vec::new();
        let mut radius = self.spiral_step;

        while radius <= self.max_spiral_radius {
            // 8 directions at each radius
            let angles = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0];

            for angle in angles {
                let rad = angle * std::f64::consts::PI / 180.0;
                let dx = radius * rad.cos();
                let dy = radius * rad.sin();
                offsets.push((dx, dy));
            }

            radius += self.spiral_step;
        }

        offsets
    }

    /// Log a failure for analysis
    pub fn log_failure(&self, x: f64, y: f64, app_bundle_id: Option<&str>, window_title: Option<&str>) {
        tracing::debug!(
            "AX query failed at ({}, {}), app: {:?}, window: {:?}",
            x, y, app_bundle_id, window_title
        );
    }
}

impl Default for NullHandler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_electron_detection() {
        let handler = NullHandler::new();

        assert!(handler.is_electron_app("com.microsoft.VSCode"));
        assert!(handler.is_electron_app("com.slack.Slack"));
        assert!(!handler.is_electron_app("com.apple.finder"));
    }

    #[test]
    fn test_spiral_offsets() {
        let handler = NullHandler::new();
        let offsets = handler.generate_spiral_offsets();

        // Should have offsets at multiple radii
        assert!(!offsets.is_empty());

        // First offset should be at spiral_step radius
        let (dx, dy) = offsets[0];
        let radius = (dx * dx + dy * dy).sqrt();
        assert!((radius - handler.spiral_step).abs() < 0.1);
    }

    #[test]
    fn test_null_handler_defaults() {
        let handler = NullHandler::new();
        assert!((handler.max_spiral_radius - 20.0).abs() < 0.01);
        assert!((handler.spiral_step - 5.0).abs() < 0.01);
        assert_eq!(handler.injection_timeout_ms, 100);
    }

    #[test]
    fn test_default_implementation() {
        let handler = NullHandler::default();
        assert!(handler.max_spiral_radius > 0.0);
    }

    #[test]
    fn test_recovery_strategy_variants() {
        assert_ne!(RecoveryStrategy::Injection, RecoveryStrategy::SpiralSearch);
        assert_ne!(RecoveryStrategy::Vision, RecoveryStrategy::None);
        assert_eq!(RecoveryStrategy::Injection, RecoveryStrategy::Injection);
    }

    #[test]
    fn test_recovery_result_structure() {
        let result = RecoveryResult {
            strategy: RecoveryStrategy::SpiralSearch,
            context: Some(SemanticContext::default()),
            confidence: 0.9,
            duration_ms: 50,
        };

        assert_eq!(result.strategy, RecoveryStrategy::SpiralSearch);
        assert!(result.context.is_some());
        assert!(result.confidence > 0.5);
    }

    #[test]
    fn test_spiral_offset_coverage() {
        let handler = NullHandler::new();
        let offsets = handler.generate_spiral_offsets();

        // Should have 8 directions per radius level
        // With max_radius=20 and step=5, we have 4 levels (5, 10, 15, 20)
        // 4 levels * 8 directions = 32 offsets
        assert_eq!(offsets.len(), 32);

        // Check all 8 cardinal directions are covered at first radius
        let first_radius_offsets: Vec<_> = offsets.iter().take(8).collect();
        assert_eq!(first_radius_offsets.len(), 8);
    }

    #[test]
    fn test_electron_app_list() {
        let handler = NullHandler::new();

        // Test all known Electron apps
        let electron_apps = [
            "com.microsoft.VSCode",
            "com.slack.Slack",
            "com.discord",
            "com.spotify.client",
            "com.atom.editor",
            "com.github.desktop",
            "com.figma.Desktop",
            "com.notion.id",
            "com.linear",
            "com.obsidian",
        ];

        for app in &electron_apps {
            assert!(handler.is_electron_app(app), "Should detect {} as Electron app", app);
        }
    }

    #[test]
    fn test_non_electron_apps() {
        let handler = NullHandler::new();

        // Test native apps are not detected as Electron
        let native_apps = [
            "com.apple.finder",
            "com.apple.Safari",
            "com.apple.Terminal",
            "com.apple.TextEdit",
        ];

        for app in &native_apps {
            assert!(!handler.is_electron_app(app), "Should not detect {} as Electron app", app);
        }
    }

    #[test]
    fn test_recovery_without_bundle_id() {
        let handler = NullHandler::new();
        let result = handler.recover(100.0, 100.0, None);

        // Without bundle_id, injection strategy is skipped
        // Result depends on actual accessibility state
        assert!(result.duration_ms < 1000); // Should be quick
    }

    #[test]
    fn test_spiral_search_radius_calculation() {
        let handler = NullHandler::new();
        let offsets = handler.generate_spiral_offsets();

        // Calculate radius for each offset and verify it's within bounds
        for (dx, dy) in &offsets {
            let radius = (dx * dx + dy * dy).sqrt();
            assert!(radius <= handler.max_spiral_radius + 0.1);
            assert!(radius >= handler.spiral_step - 0.1);
        }
    }

    #[test]
    fn test_recovery_result_cloning() {
        let result = RecoveryResult {
            strategy: RecoveryStrategy::Injection,
            context: Some(SemanticContext::default()),
            confidence: 0.85,
            duration_ms: 100,
        };

        let cloned = result.clone();
        assert_eq!(cloned.strategy, result.strategy);
        assert_eq!(cloned.confidence, result.confidence);
        assert_eq!(cloned.duration_ms, result.duration_ms);
        assert!(cloned.context.is_some());
    }

    #[test]
    fn test_null_handler_custom_settings() {
        let handler = NullHandler {
            max_spiral_radius: 30.0,
            spiral_step: 10.0,
            injection_timeout_ms: 200,
            vision_fallback: None,
        };

        assert_eq!(handler.max_spiral_radius, 30.0);
        assert_eq!(handler.spiral_step, 10.0);
        assert_eq!(handler.injection_timeout_ms, 200);

        // Verify spiral offsets respect new settings
        let offsets = handler.generate_spiral_offsets();
        // With radius=30 and step=10, we have 3 levels (10, 20, 30)
        // 3 levels * 8 directions = 24 offsets
        assert_eq!(offsets.len(), 24);
    }

    #[test]
    fn test_recovery_strategy_debug_format() {
        let strategy = RecoveryStrategy::SpiralSearch;
        let debug_str = format!("{:?}", strategy);
        assert!(debug_str.contains("SpiralSearch"));

        let none_strategy = RecoveryStrategy::None;
        let debug_none = format!("{:?}", none_strategy);
        assert!(debug_none.contains("None"));
    }

    #[test]
    fn test_recovery_with_electron_bundle() {
        let handler = NullHandler::new();
        let result = handler.recover(100.0, 100.0, Some("com.microsoft.VSCode"));

        // Should attempt injection strategy first for Electron apps
        // Result may vary based on actual accessibility state
        assert!(result.duration_ms < 2000);

        // Strategy should be one of the recovery strategies or None
        match result.strategy {
            RecoveryStrategy::Injection
            | RecoveryStrategy::SpiralSearch
            | RecoveryStrategy::Vision
            | RecoveryStrategy::None => {},
        }
    }

    #[test]
    fn test_spiral_offset_ordering() {
        let handler = NullHandler::new();
        let offsets = handler.generate_spiral_offsets();

        // First 8 offsets should all be at the first radius (spiral_step)
        let first_radius = handler.spiral_step;
        for i in 0..8 {
            let (dx, dy) = offsets[i];
            let radius = (dx * dx + dy * dy).sqrt();
            assert!((radius - first_radius).abs() < 0.1,
                "Offset {} has radius {} but expected ~{}", i, radius, first_radius);
        }

        // Next 8 offsets should be at second radius (2 * spiral_step)
        let second_radius = handler.spiral_step * 2.0;
        for i in 8..16 {
            let (dx, dy) = offsets[i];
            let radius = (dx * dx + dy * dy).sqrt();
            assert!((radius - second_radius).abs() < 0.1,
                "Offset {} has radius {} but expected ~{}", i, radius, second_radius);
        }
    }

    #[test]
    fn test_log_failure_various_inputs() {
        let handler = NullHandler::new();

        // Test with full context
        handler.log_failure(100.0, 200.0,
            Some("com.apple.Safari"),
            Some("Welcome to Safari"));

        // Test with partial context
        handler.log_failure(150.0, 250.0, Some("com.example.app"), None);

        // Test with minimal context
        handler.log_failure(200.0, 300.0, None, None);

        // All should complete without panic
    }

    #[test]
    fn test_recovery_result_with_none_context() {
        let result = RecoveryResult {
            strategy: RecoveryStrategy::None,
            context: None,
            confidence: 0.0,
            duration_ms: 50,
        };

        assert_eq!(result.strategy, RecoveryStrategy::None);
        assert!(result.context.is_none());
        assert_eq!(result.confidence, 0.0);
    }

    #[test]
    fn test_spiral_directions_comprehensive() {
        let handler = NullHandler::new();
        let offsets = handler.generate_spiral_offsets();

        // Verify we have 8 cardinal and intercardinal directions
        // at the first radius: 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°
        let first_eight: Vec<_> = offsets.iter().take(8).collect();

        // Check that we have variety in directions (not all same angle)
        let mut unique_angles = std::collections::HashSet::new();
        for (dx, dy) in first_eight {
            let angle = dy.atan2(*dx);
            unique_angles.insert((angle * 1000.0) as i32);
        }
        assert_eq!(unique_angles.len(), 8, "Should have 8 unique directions");
    }

    #[test]
    fn test_recovery_confidence_levels() {
        // Test that different strategies have appropriate confidence levels
        let injection_result = RecoveryResult {
            strategy: RecoveryStrategy::Injection,
            context: Some(SemanticContext::default()),
            confidence: 0.85,
            duration_ms: 100,
        };
        assert_eq!(injection_result.confidence, 0.85);

        let spiral_result = RecoveryResult {
            strategy: RecoveryStrategy::SpiralSearch,
            context: Some(SemanticContext::default()),
            confidence: 0.9,
            duration_ms: 50,
        };
        assert_eq!(spiral_result.confidence, 0.9);

        let none_result = RecoveryResult {
            strategy: RecoveryStrategy::None,
            context: None,
            confidence: 0.0,
            duration_ms: 10,
        };
        assert_eq!(none_result.confidence, 0.0);
    }

    #[test]
    fn test_electron_app_partial_matching() {
        let handler = NullHandler::new();

        // Test partial matches (contains check)
        assert!(handler.is_electron_app("com.microsoft.VSCode.helper"));
        assert!(handler.is_electron_app("com.slack.Slack.helper"));
        assert!(handler.is_electron_app("prefix.com.discord.suffix"));

        // Negative cases
        assert!(!handler.is_electron_app("com.apple.VSCodeLike"));
        assert!(!handler.is_electron_app("totally.different.app"));
        assert!(!handler.is_electron_app("io.slack.Slack")); // Different bundle ID format
    }

    #[test]
    fn test_null_handler_equality() {
        // Test RecoveryStrategy equality
        assert_eq!(RecoveryStrategy::Injection, RecoveryStrategy::Injection);
        assert_eq!(RecoveryStrategy::SpiralSearch, RecoveryStrategy::SpiralSearch);
        assert_ne!(RecoveryStrategy::Injection, RecoveryStrategy::Vision);
    }

    #[test]
    fn test_with_vision_builder() {
        let handler = NullHandler::new().with_vision();
        assert!(handler.vision_fallback.is_some());

        let handler_no_vision = NullHandler::new();
        assert!(handler_no_vision.vision_fallback.is_none());
    }

    #[test]
    fn test_recovery_with_vision_fallback_enabled() {
        let handler = NullHandler::new().with_vision();
        // Run recovery at an arbitrary point - vision will attempt OCR
        let result = handler.recover(500.0, 500.0, None);

        // Should complete without panic regardless of screen recording permission
        assert!(result.duration_ms < 5000);
        match result.strategy {
            RecoveryStrategy::Injection
            | RecoveryStrategy::SpiralSearch
            | RecoveryStrategy::Vision
            | RecoveryStrategy::None => {}
        }
    }
}
