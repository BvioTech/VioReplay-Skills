//! OCR & Icon Recognition Vision Fallback
//!
//! When accessibility queries fail repeatedly, use vision (OCR) to extract
//! semantic information from the screen.
//!
//! # Architecture
//!
//! This module provides a vision-based fallback for accessibility queries.
//! When enabled, it uses Apple's Vision framework for OCR text recognition.
//!
//! # Requirements
//!
//! - macOS 10.15+ for Vision framework
//! - Screen Recording permission for screenshot capture

use crate::capture::types::{SemanticContext, SemanticSource};
use core_graphics::display::{CGPoint, CGRect, CGSize};
use objc::runtime::{Class, Object, BOOL, YES};
use objc::{msg_send, sel, sel_impl};
use std::ffi::c_void;

/// Vision fallback configuration
#[derive(Debug, Clone)]
pub struct VisionConfig {
    /// Capture region size (pixels)
    pub capture_size: u32,
    /// OCR confidence threshold
    pub ocr_threshold: f32,
    /// Maximum latency allowed (ms)
    pub max_latency_ms: u64,
    /// Number of consecutive failures to trigger vision
    pub failure_threshold: usize,
}

impl Default for VisionConfig {
    fn default() -> Self {
        Self {
            capture_size: 512,
            ocr_threshold: 0.7,
            max_latency_ms: 150,
            failure_threshold: 3,
        }
    }
}

/// Detected text region
#[derive(Debug, Clone)]
pub struct TextRegion {
    /// Recognized text
    pub text: String,
    /// Bounding box (x, y, width, height)
    pub bounds: (f64, f64, f64, f64),
    /// Confidence score
    pub confidence: f32,
}

/// Detected icon
#[derive(Debug, Clone)]
pub struct DetectedIcon {
    /// Icon type/name
    pub icon_type: String,
    /// Bounding box
    pub bounds: (f64, f64, f64, f64),
    /// Confidence score
    pub confidence: f32,
}

/// Vision fallback result
#[derive(Debug, Clone)]
pub struct VisionResult {
    /// Detected text regions
    pub text_regions: Vec<TextRegion>,
    /// Detected icons
    pub icons: Vec<DetectedIcon>,
    /// Best match semantic context
    pub semantic: Option<SemanticContext>,
    /// Total processing time (ms)
    pub processing_time_ms: u64,
}

/// Vision fallback handler
pub struct VisionFallback {
    /// Configuration
    pub config: VisionConfig,
    /// Consecutive failure counter per window
    failure_counts: std::collections::HashMap<u32, usize>,
}

impl VisionFallback {
    /// Create with default config
    pub fn new() -> Self {
        Self {
            config: VisionConfig::default(),
            failure_counts: std::collections::HashMap::new(),
        }
    }

    /// Create with custom config
    pub fn with_config(config: VisionConfig) -> Self {
        Self {
            config,
            failure_counts: std::collections::HashMap::new(),
        }
    }

    /// Record a failure for a window
    pub fn record_failure(&mut self, window_id: u32) {
        *self.failure_counts.entry(window_id).or_insert(0) += 1;
    }

    /// Reset failure count for a window
    pub fn reset_failures(&mut self, window_id: u32) {
        self.failure_counts.remove(&window_id);
    }

    /// Check if vision should be triggered for a window
    pub fn should_trigger(&self, window_id: u32) -> bool {
        self.failure_counts
            .get(&window_id)
            .map(|&count| count >= self.config.failure_threshold)
            .unwrap_or(false)
    }

    /// Check if screen recording permission is available for vision capture
    pub fn has_screen_recording_permission() -> bool {
        crate::capture::check_screen_recording_permission()
    }

    /// Run vision fallback at a position
    pub fn run(&self, x: f64, y: f64) -> VisionResult {
        let start = std::time::Instant::now();

        // Verify screen recording permission before capture
        if !Self::has_screen_recording_permission() {
            tracing::debug!("Screen Recording permission not granted, skipping Vision OCR");
            return VisionResult {
                text_regions: Vec::new(),
                icons: Vec::new(),
                semantic: None,
                processing_time_ms: start.elapsed().as_millis() as u64,
            };
        }

        // Step 1: Capture screenshot of region
        let capture_region = self.calculate_capture_region(x, y);

        // Step 2: Run OCR
        let text_regions = self.run_ocr(&capture_region);

        // Step 3: Run icon detection (optional)
        let icons = self.run_icon_detection(&capture_region);

        // Step 4: Fuse results into semantic context
        let semantic = self.fuse_results(x, y, &text_regions, &icons);

        VisionResult {
            text_regions,
            icons,
            semantic,
            processing_time_ms: start.elapsed().as_millis() as u64,
        }
    }

    /// Calculate capture region around cursor
    fn calculate_capture_region(&self, x: f64, y: f64) -> (f64, f64, f64, f64) {
        let half_size = self.config.capture_size as f64 / 2.0;
        (
            x - half_size,
            y - half_size,
            self.config.capture_size as f64,
            self.config.capture_size as f64,
        )
    }

    /// Run OCR on a region
    ///
    /// This captures a screenshot of the region and performs OCR using
    /// Apple's Vision framework when available.
    fn run_ocr(&self, region: &(f64, f64, f64, f64)) -> Vec<TextRegion> {
        // Step 1: Capture screenshot of the region
        let screenshot = match self.capture_region_screenshot(region) {
            Some(img) => img,
            None => {
                tracing::debug!("Failed to capture screenshot for OCR");
                return Vec::new();
            }
        };

        // Step 2: Run Vision framework OCR
        // Note: Full Vision framework integration requires Objective-C bridging
        // via objc crate. The screenshot capture is implemented, but Vision
        // API calls would need additional FFI bindings.
        //
        // For production use, consider using the `vision` crate or implementing
        // VNRecognizeTextRequest bindings.

        let result = self.perform_ocr_on_image(&screenshot);

        // Release the screenshot
        unsafe {
            extern "C" {
                fn CGImageRelease(image: *mut c_void);
            }
            CGImageRelease(screenshot);
        }

        result
    }

    /// Capture a screenshot of a specific region
    ///
    /// Returns the raw image reference that can be used with Vision framework.
    fn capture_region_screenshot(&self, region: &(f64, f64, f64, f64)) -> Option<*mut c_void> {
        let (x, y, width, height) = *region;

        // Ensure coordinates are non-negative
        let x = x.max(0.0);
        let y = y.max(0.0);

        let rect = CGRect::new(
            &CGPoint::new(x, y),
            &CGSize::new(width, height),
        );

        // Capture from main display
        // CGWindowListCreateImage captures the specified rectangle
        unsafe {
            extern "C" {
                fn CGWindowListCreateImage(
                    screenBounds: CGRect,
                    listOption: u32,
                    windowID: u32,
                    imageOption: u32,
                ) -> *mut c_void;
            }

            // kCGWindowListOptionOnScreenOnly = 1
            // kCGNullWindowID = 0
            // kCGWindowImageDefault = 0
            let image_ref = CGWindowListCreateImage(rect, 1, 0, 0);
            if image_ref.is_null() {
                return None;
            }

            Some(image_ref)
        }
    }

    /// Perform OCR on a captured image using Apple's Vision framework.
    ///
    /// Uses VNRecognizeTextRequest with accurate recognition level to extract
    /// text regions with bounding boxes and confidence scores.
    fn perform_ocr_on_image(&self, image_ref: &*mut c_void) -> Vec<TextRegion> {
        let image = *image_ref;
        if image.is_null() {
            return Vec::new();
        }

        unsafe {
            // Step 1: Create VNImageRequestHandler from CGImage
            let handler_class = match Class::get("VNImageRequestHandler") {
                Some(cls) => cls,
                None => {
                    tracing::warn!("VNImageRequestHandler class not found - Vision framework unavailable");
                    return Vec::new();
                }
            };

            let handler: *mut Object = msg_send![handler_class, alloc];
            let handler: *mut Object = msg_send![handler, initWithCGImage:image options:std::ptr::null::<Object>()];
            if handler.is_null() {
                tracing::debug!("Failed to create VNImageRequestHandler");
                return Vec::new();
            }

            // Step 2: Create VNRecognizeTextRequest
            let request_class = match Class::get("VNRecognizeTextRequest") {
                Some(cls) => cls,
                None => {
                    let _: () = msg_send![handler, release];
                    tracing::warn!("VNRecognizeTextRequest class not found");
                    return Vec::new();
                }
            };

            let request: *mut Object = msg_send![request_class, alloc];
            let request: *mut Object = msg_send![request, init];
            if request.is_null() {
                let _: () = msg_send![handler, release];
                return Vec::new();
            }

            // Set recognition level to accurate (VNRequestTextRecognitionLevelAccurate = 1)
            let _: () = msg_send![request, setRecognitionLevel:1_isize];
            // Enable automatic language correction
            let _: () = msg_send![request, setUsesLanguageCorrection:YES];

            // Step 3: Create NSArray with the request and perform
            let nsarray_class = match Class::get("NSArray") {
                Some(cls) => cls,
                None => {
                    let _: () = msg_send![request, release];
                    let _: () = msg_send![handler, release];
                    tracing::warn!("NSArray class not found");
                    return Vec::new();
                }
            };
            let requests_array: *mut Object = msg_send![nsarray_class, arrayWithObject:request];

            let mut error: *mut Object = std::ptr::null_mut();
            let success: BOOL = msg_send![handler, performRequests:requests_array error:&mut error];

            if success != YES {
                if !error.is_null() {
                    let description: *mut Object = msg_send![error, localizedDescription];
                    let utf8: *const std::os::raw::c_char = msg_send![description, UTF8String];
                    if !utf8.is_null() {
                        let err_str = std::ffi::CStr::from_ptr(utf8).to_string_lossy();
                        tracing::debug!("Vision OCR failed: {}", err_str);
                    }
                }
                let _: () = msg_send![request, release];
                let _: () = msg_send![handler, release];
                return Vec::new();
            }

            // Step 4: Extract results from VNRecognizedTextObservation array
            let results: *mut Object = msg_send![request, results];
            if results.is_null() {
                let _: () = msg_send![request, release];
                let _: () = msg_send![handler, release];
                return Vec::new();
            }

            let count: usize = msg_send![results, count];
            let mut text_regions = Vec::with_capacity(count);
            let threshold = self.config.ocr_threshold;

            for i in 0..count {
                let observation: *mut Object = msg_send![results, objectAtIndex:i];

                // Get confidence
                let confidence: f32 = msg_send![observation, confidence];
                if confidence < threshold {
                    continue;
                }

                // Get top candidate text
                let candidates: *mut Object = msg_send![observation, topCandidates:1_usize];
                let candidate_count: usize = msg_send![candidates, count];
                if candidate_count == 0 {
                    continue;
                }

                let candidate: *mut Object = msg_send![candidates, objectAtIndex:0_usize];
                let ns_string: *mut Object = msg_send![candidate, string];
                if ns_string.is_null() {
                    continue;
                }

                let utf8: *const std::os::raw::c_char = msg_send![ns_string, UTF8String];
                if utf8.is_null() {
                    continue;
                }
                let text = std::ffi::CStr::from_ptr(utf8)
                    .to_string_lossy()
                    .into_owned();

                // Get bounding box (normalized coordinates 0..1, origin at bottom-left)
                // VNRectangleObservation boundingBox returns CGRect
                #[repr(C)]
                struct VNCGRect {
                    origin_x: f64,
                    origin_y: f64,
                    size_w: f64,
                    size_h: f64,
                }
                let bbox: VNCGRect = msg_send![observation, boundingBox];

                // Convert from normalized bottom-left to pixel top-left coordinates
                // relative to the capture region
                let capture_w = self.config.capture_size as f64;
                let capture_h = self.config.capture_size as f64;
                let x = bbox.origin_x * capture_w;
                let y = (1.0 - bbox.origin_y - bbox.size_h) * capture_h;
                let w = bbox.size_w * capture_w;
                let h = bbox.size_h * capture_h;

                text_regions.push(TextRegion {
                    text,
                    bounds: (x, y, w, h),
                    confidence,
                });
            }

            tracing::debug!("Vision OCR found {} text regions", text_regions.len());

            // Cleanup
            let _: () = msg_send![request, release];
            let _: () = msg_send![handler, release];

            text_regions
        }
    }

    /// Run icon detection using Vision framework rectangle detection and geometric heuristics.
    ///
    /// Uses VNDetectRectanglesRequest to find rectangular UI elements in the capture region,
    /// then classifies them by aspect ratio and size into common UI element types
    /// (buttons, checkboxes, text fields, close buttons, icons).
    fn run_icon_detection(&self, region: &(f64, f64, f64, f64)) -> Vec<DetectedIcon> {
        let screenshot = match self.capture_region_screenshot(region) {
            Some(img) => img,
            None => {
                tracing::debug!("Failed to capture screenshot for icon detection");
                return Vec::new();
            }
        };

        let icons = self.detect_rectangles_as_icons(&screenshot, region);

        unsafe {
            extern "C" {
                fn CGImageRelease(image: *mut c_void);
            }
            CGImageRelease(screenshot);
        }

        tracing::debug!("Icon detection found {} UI elements", icons.len());
        icons
    }

    /// Detect rectangular UI elements using VNDetectRectanglesRequest and classify them.
    fn detect_rectangles_as_icons(
        &self,
        image_ref: &*mut c_void,
        region: &(f64, f64, f64, f64),
    ) -> Vec<DetectedIcon> {
        let image = *image_ref;
        if image.is_null() {
            return Vec::new();
        }

        unsafe {
            // Create VNImageRequestHandler
            let handler_class = match Class::get("VNImageRequestHandler") {
                Some(cls) => cls,
                None => return Vec::new(),
            };

            let handler: *mut Object = msg_send![handler_class, alloc];
            let handler: *mut Object = msg_send![handler, initWithCGImage:image options:std::ptr::null::<Object>()];
            if handler.is_null() {
                return Vec::new();
            }

            // Create VNDetectRectanglesRequest
            let request_class = match Class::get("VNDetectRectanglesRequest") {
                Some(cls) => cls,
                None => {
                    let _: () = msg_send![handler, release];
                    tracing::debug!("VNDetectRectanglesRequest not available");
                    return Vec::new();
                }
            };

            let request: *mut Object = msg_send![request_class, alloc];
            let request: *mut Object = msg_send![request, init];
            if request.is_null() {
                let _: () = msg_send![handler, release];
                return Vec::new();
            }

            // Configure: detect up to 10 rectangles, min size 2% of image
            let _: () = msg_send![request, setMaximumObservations:10_usize];
            let _: () = msg_send![request, setMinimumSize:0.02_f32];
            let _: () = msg_send![request, setMinimumConfidence:0.5_f32];

            // Perform request
            let nsarray_class = match Class::get("NSArray") {
                Some(cls) => cls,
                None => {
                    let _: () = msg_send![request, release];
                    let _: () = msg_send![handler, release];
                    return Vec::new();
                }
            };
            let requests_array: *mut Object = msg_send![nsarray_class, arrayWithObject:request];

            let mut error: *mut Object = std::ptr::null_mut();
            let success: BOOL = msg_send![handler, performRequests:requests_array error:&mut error];

            if success != YES {
                let _: () = msg_send![request, release];
                let _: () = msg_send![handler, release];
                return Vec::new();
            }

            // Extract results
            let results: *mut Object = msg_send![request, results];
            if results.is_null() {
                let _: () = msg_send![request, release];
                let _: () = msg_send![handler, release];
                return Vec::new();
            }

            let count: usize = msg_send![results, count];
            let capture_w = region.2;
            let capture_h = region.3;
            let mut icons = Vec::with_capacity(count);

            for i in 0..count {
                let observation: *mut Object = msg_send![results, objectAtIndex:i];
                let confidence: f32 = msg_send![observation, confidence];

                // Get bounding box (normalized, bottom-left origin)
                #[repr(C)]
                struct VNCGRect {
                    origin_x: f64,
                    origin_y: f64,
                    size_w: f64,
                    size_h: f64,
                }
                let bbox: VNCGRect = msg_send![observation, boundingBox];

                // Convert to pixel coordinates (top-left origin)
                let x = bbox.origin_x * capture_w;
                let y = (1.0 - bbox.origin_y - bbox.size_h) * capture_h;
                let w = bbox.size_w * capture_w;
                let h = bbox.size_h * capture_h;

                // Classify by geometry
                let icon_type = Self::classify_rectangle(w, h, capture_w, capture_h);

                icons.push(DetectedIcon {
                    icon_type,
                    bounds: (x, y, w, h),
                    confidence,
                });
            }

            let _: () = msg_send![request, release];
            let _: () = msg_send![handler, release];

            icons
        }
    }

    /// Classify a detected rectangle into a UI element type based on geometry.
    fn classify_rectangle(width: f64, height: f64, region_w: f64, region_h: f64) -> String {
        let aspect_ratio = width / height.max(1.0);
        let rel_w = width / region_w.max(1.0);
        let rel_h = height / region_h.max(1.0);
        let area_ratio = rel_w * rel_h;

        if (aspect_ratio - 1.0).abs() < 0.15 && area_ratio < 0.01 {
            // Small square: checkbox or radio button
            "Checkbox".to_string()
        } else if (aspect_ratio - 1.0).abs() < 0.2 && area_ratio < 0.005 {
            // Tiny square: close/minimize button
            "CloseButton".to_string()
        } else if area_ratio > 0.2 {
            // Large area: panel or dialog
            "Panel".to_string()
        } else if rel_w > 0.5 && rel_h < 0.15 {
            // Very wide, spans >50% of region: toolbar or menu bar
            "Toolbar".to_string()
        } else if aspect_ratio > 2.5 && rel_w > 0.3 {
            // Wide and spans significant region width: text field or search bar
            "TextField".to_string()
        } else if aspect_ratio > 1.3 && area_ratio < 0.05 {
            // Moderately wide, small: button
            "Button".to_string()
        } else {
            // Generic UI element
            "UIElement".to_string()
        }
    }

    /// Fuse OCR and icon results into semantic context
    fn fuse_results(
        &self,
        cursor_x: f64,
        cursor_y: f64,
        text_regions: &[TextRegion],
        icons: &[DetectedIcon],
    ) -> Option<SemanticContext> {
        // Find text closest to cursor
        let closest_text = text_regions
            .iter()
            .min_by(|a, b| {
                let dist_a = self.distance_to_region(cursor_x, cursor_y, &a.bounds);
                let dist_b = self.distance_to_region(cursor_x, cursor_y, &b.bounds);
                dist_a.partial_cmp(&dist_b).unwrap_or(std::cmp::Ordering::Equal)
            });

        // Find icon closest to cursor
        let closest_icon = icons
            .iter()
            .min_by(|a, b| {
                let dist_a = self.distance_to_region(cursor_x, cursor_y, &a.bounds);
                let dist_b = self.distance_to_region(cursor_x, cursor_y, &b.bounds);
                dist_a.partial_cmp(&dist_b).unwrap_or(std::cmp::Ordering::Equal)
            });

        if closest_text.is_none() && closest_icon.is_none() {
            return None;
        }

        // Determine confidence: max of text and icon confidence
        let text_conf = closest_text.map(|t| t.confidence).unwrap_or(0.0);
        let icon_conf = closest_icon.map(|i| i.confidence).unwrap_or(0.0);

        Some(SemanticContext {
            source: SemanticSource::Vision,
            ocr_text: closest_text.map(|t| t.text.clone()),
            title: closest_text.map(|t| t.text.clone()),
            ax_role: closest_icon.map(|i| format!("AX{}", i.icon_type)),
            confidence: text_conf.max(icon_conf),
            ..Default::default()
        })
    }

    /// Calculate distance from point to region center
    fn distance_to_region(&self, x: f64, y: f64, bounds: &(f64, f64, f64, f64)) -> f64 {
        let center_x = bounds.0 + bounds.2 / 2.0;
        let center_y = bounds.1 + bounds.3 / 2.0;
        let dx = x - center_x;
        let dy = y - center_y;
        (dx * dx + dy * dy).sqrt()
    }
}

impl Default for VisionFallback {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vision_config_default() {
        let config = VisionConfig::default();
        assert_eq!(config.capture_size, 512);
        assert!(config.ocr_threshold > 0.0);
    }

    #[test]
    fn test_failure_tracking() {
        let mut fallback = VisionFallback::new();

        assert!(!fallback.should_trigger(1));

        for _ in 0..3 {
            fallback.record_failure(1);
        }

        assert!(fallback.should_trigger(1));
    }

    #[test]
    fn test_capture_region() {
        let fallback = VisionFallback::new();
        let region = fallback.calculate_capture_region(500.0, 400.0);

        assert_eq!(region.2, 512.0); // width
        assert_eq!(region.3, 512.0); // height
    }

    #[test]
    fn test_vision_config_custom() {
        let config = VisionConfig {
            capture_size: 256,
            ocr_threshold: 0.8,
            max_latency_ms: 100,
            failure_threshold: 5,
        };

        assert_eq!(config.capture_size, 256);
        assert!((config.ocr_threshold - 0.8).abs() < 0.01);
        assert_eq!(config.max_latency_ms, 100);
        assert_eq!(config.failure_threshold, 5);
    }

    #[test]
    fn test_with_custom_config() {
        let config = VisionConfig {
            capture_size: 1024,
            ocr_threshold: 0.9,
            max_latency_ms: 200,
            failure_threshold: 2,
        };

        let fallback = VisionFallback::with_config(config);
        assert_eq!(fallback.config.capture_size, 1024);
        assert_eq!(fallback.config.failure_threshold, 2);
    }

    #[test]
    fn test_reset_failures() {
        let mut fallback = VisionFallback::new();

        // Record failures
        for _ in 0..5 {
            fallback.record_failure(1);
        }
        assert!(fallback.should_trigger(1));

        // Reset
        fallback.reset_failures(1);
        assert!(!fallback.should_trigger(1));
    }

    #[test]
    fn test_multiple_windows() {
        let mut fallback = VisionFallback::new();

        // Window 1: 3 failures (should trigger)
        for _ in 0..3 {
            fallback.record_failure(1);
        }

        // Window 2: 2 failures (should not trigger)
        for _ in 0..2 {
            fallback.record_failure(2);
        }

        assert!(fallback.should_trigger(1));
        assert!(!fallback.should_trigger(2));
        assert!(!fallback.should_trigger(3)); // Never seen
    }

    #[test]
    fn test_distance_to_region() {
        let fallback = VisionFallback::new();

        // Region at (0, 0) with size 100x100, center at (50, 50)
        let bounds = (0.0, 0.0, 100.0, 100.0);

        // Point at center
        let dist = fallback.distance_to_region(50.0, 50.0, &bounds);
        assert!(dist < 0.01);

        // Point at (0, 50) - distance should be 50
        let dist = fallback.distance_to_region(0.0, 50.0, &bounds);
        assert!((dist - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_text_region_structure() {
        let region = TextRegion {
            text: "Submit".to_string(),
            bounds: (100.0, 200.0, 80.0, 30.0),
            confidence: 0.95,
        };

        assert_eq!(region.text, "Submit");
        assert!(region.confidence > 0.9);
    }

    #[test]
    fn test_detected_icon_structure() {
        let icon = DetectedIcon {
            icon_type: "Button".to_string(),
            bounds: (50.0, 50.0, 24.0, 24.0),
            confidence: 0.88,
        };

        assert_eq!(icon.icon_type, "Button");
        assert!(icon.confidence > 0.8);
    }

    #[test]
    fn test_vision_result_structure() {
        let result = VisionResult {
            text_regions: vec![TextRegion {
                text: "OK".to_string(),
                bounds: (10.0, 10.0, 40.0, 20.0),
                confidence: 0.9,
            }],
            icons: vec![],
            semantic: None,
            processing_time_ms: 50,
        };

        assert_eq!(result.text_regions.len(), 1);
        assert!(result.icons.is_empty());
        assert!(result.processing_time_ms < 100);
    }

    #[test]
    fn test_fuse_results_with_text() {
        let fallback = VisionFallback::new();

        let text_regions = vec![TextRegion {
            text: "Login".to_string(),
            bounds: (100.0, 100.0, 60.0, 20.0),
            confidence: 0.92,
        }];

        let context = fallback.fuse_results(130.0, 110.0, &text_regions, &[]);

        assert!(context.is_some());
        let ctx = context.unwrap();
        assert_eq!(ctx.title, Some("Login".to_string()));
        assert_eq!(ctx.source, SemanticSource::Vision);
    }

    #[test]
    fn test_fuse_results_with_icon() {
        let fallback = VisionFallback::new();

        let icons = vec![DetectedIcon {
            icon_type: "Checkbox".to_string(),
            bounds: (50.0, 50.0, 20.0, 20.0),
            confidence: 0.85,
        }];

        let context = fallback.fuse_results(60.0, 60.0, &[], &icons);

        assert!(context.is_some());
        let ctx = context.unwrap();
        assert_eq!(ctx.ax_role, Some("AXCheckbox".to_string()));
    }

    #[test]
    fn test_fuse_results_empty() {
        let fallback = VisionFallback::new();

        let context = fallback.fuse_results(100.0, 100.0, &[], &[]);
        assert!(context.is_none());
    }

    #[test]
    fn test_default_implementation() {
        let fallback = VisionFallback::default();
        assert_eq!(fallback.config.capture_size, 512);
    }

    #[test]
    fn test_run_returns_result() {
        let fallback = VisionFallback::new();
        let result = fallback.run(500.0, 400.0);

        // Should return a valid result structure even if OCR is empty
        assert!(result.processing_time_ms < 10000); // Sanity check
    }

    #[test]
    fn test_capture_region_at_origin() {
        let fallback = VisionFallback::new();
        let region = fallback.calculate_capture_region(0.0, 0.0);

        // Half the region should be negative
        assert!(region.0 < 0.0);
        assert!(region.1 < 0.0);
        assert_eq!(region.2, 512.0);
        assert_eq!(region.3, 512.0);
    }

    #[test]
    fn test_closest_text_selection() {
        let fallback = VisionFallback::new();

        let text_regions = vec![
            TextRegion {
                text: "Far".to_string(),
                bounds: (0.0, 0.0, 50.0, 20.0), // Center at (25, 10)
                confidence: 0.9,
            },
            TextRegion {
                text: "Near".to_string(),
                bounds: (100.0, 100.0, 50.0, 20.0), // Center at (125, 110)
                confidence: 0.85,
            },
        ];

        // Cursor closer to "Near"
        let context = fallback.fuse_results(130.0, 115.0, &text_regions, &[]);

        assert!(context.is_some());
        let ctx = context.unwrap();
        assert_eq!(ctx.title, Some("Near".to_string()));
    }

    #[test]
    fn test_perform_ocr_with_null_image() {
        let fallback = VisionFallback::new();
        let null_ptr: *mut c_void = std::ptr::null_mut();
        let result = fallback.perform_ocr_on_image(&null_ptr);
        assert!(result.is_empty(), "Null image should produce empty results");
    }

    #[test]
    fn test_vision_framework_class_availability() {
        // Verify Vision framework classes are loadable on macOS
        use objc::runtime::Class;

        let handler_class = Class::get("VNImageRequestHandler");
        let request_class = Class::get("VNRecognizeTextRequest");

        // On macOS 10.15+, these should be available
        assert!(handler_class.is_some(), "VNImageRequestHandler should be available");
        assert!(request_class.is_some(), "VNRecognizeTextRequest should be available");
    }

    #[test]
    fn test_ocr_threshold_filtering() {
        // Verify that the threshold is used correctly in the config
        let config = VisionConfig {
            capture_size: 512,
            ocr_threshold: 0.9, // High threshold
            max_latency_ms: 150,
            failure_threshold: 3,
        };
        let fallback = VisionFallback::with_config(config);

        // Results below threshold should be filtered
        // (tested through the full pipeline with real images,
        // but we verify the config is wired up)
        assert!((fallback.config.ocr_threshold - 0.9).abs() < 0.01);
    }

    #[test]
    fn test_classify_rectangle_checkbox() {
        // Small square: ~1:1 aspect ratio, tiny area
        let icon_type = VisionFallback::classify_rectangle(10.0, 10.0, 512.0, 512.0);
        assert_eq!(icon_type, "Checkbox");
    }

    #[test]
    fn test_classify_rectangle_button() {
        // Moderately wide, small area relative to region
        let icon_type = VisionFallback::classify_rectangle(80.0, 30.0, 512.0, 512.0);
        assert_eq!(icon_type, "Button");
    }

    #[test]
    fn test_classify_rectangle_text_field() {
        // Wide, spans >30% of region, high aspect ratio
        let icon_type = VisionFallback::classify_rectangle(200.0, 30.0, 512.0, 512.0);
        assert_eq!(icon_type, "TextField");
    }

    #[test]
    fn test_classify_rectangle_toolbar() {
        // Very wide, spans >50% of region width
        let icon_type = VisionFallback::classify_rectangle(400.0, 40.0, 512.0, 512.0);
        assert_eq!(icon_type, "Toolbar");
    }

    #[test]
    fn test_classify_rectangle_panel() {
        // Large area
        let icon_type = VisionFallback::classify_rectangle(400.0, 350.0, 512.0, 512.0);
        assert_eq!(icon_type, "Panel");
    }

    #[test]
    fn test_classify_rectangle_generic() {
        // Medium rectangle that doesn't match specific categories
        let icon_type = VisionFallback::classify_rectangle(100.0, 80.0, 512.0, 512.0);
        assert_eq!(icon_type, "UIElement");
    }

    #[test]
    fn test_detect_rectangles_with_null_image() {
        let fallback = VisionFallback::new();
        let null_ptr: *mut c_void = std::ptr::null_mut();
        let region = (0.0, 0.0, 512.0, 512.0);
        let result = fallback.detect_rectangles_as_icons(&null_ptr, &region);
        assert!(result.is_empty(), "Null image should produce empty icon results");
    }

    #[test]
    fn test_rectangle_detection_class_availability() {
        use objc::runtime::Class;
        let rect_class = Class::get("VNDetectRectanglesRequest");
        assert!(rect_class.is_some(), "VNDetectRectanglesRequest should be available");
    }

    #[test]
    fn test_fuse_results_with_text_and_icons() {
        let fallback = VisionFallback::new();

        let text_regions = vec![TextRegion {
            text: "Submit".to_string(),
            bounds: (100.0, 100.0, 60.0, 20.0),
            confidence: 0.9,
        }];

        let icons = vec![DetectedIcon {
            icon_type: "Button".to_string(),
            bounds: (95.0, 95.0, 70.0, 30.0),
            confidence: 0.85,
        }];

        let context = fallback.fuse_results(130.0, 110.0, &text_regions, &icons);
        assert!(context.is_some());
        let ctx = context.unwrap();
        // Should have both text (from OCR) and role (from icon detection)
        assert_eq!(ctx.title, Some("Submit".to_string()));
        assert_eq!(ctx.ax_role, Some("AXButton".to_string()));
        assert_eq!(ctx.source, SemanticSource::Vision);
    }
}
