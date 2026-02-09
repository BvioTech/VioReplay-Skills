//! CGImage-to-JPEG Conversion Module
//!
//! Captures screenshots and converts CGImageRef to JPEG bytes using macOS ImageIO
//! framework FFI. Supports configurable quality and automatic downscaling.

use core_graphics::display::{CGRect, CGPoint, CGSize};
use std::ffi::c_void;

/// Screenshot capture configuration
#[derive(Debug, Clone)]
pub struct ScreenshotConfig {
    /// JPEG quality (0.0 = worst, 1.0 = best)
    pub jpeg_quality: f32,
    /// Maximum image width in pixels (aspect ratio preserved)
    pub max_width: u32,
}

impl Default for ScreenshotConfig {
    fn default() -> Self {
        Self {
            jpeg_quality: 0.8,
            max_width: 1024,
        }
    }
}

/// A captured screenshot as JPEG data
#[derive(Debug, Clone)]
pub struct CapturedScreenshot {
    /// JPEG image data
    pub jpeg_data: Vec<u8>,
    /// Image width in pixels (after scaling)
    pub width: u32,
    /// Image height in pixels (after scaling)
    pub height: u32,
}

// CoreFoundation FFI types
type CFTypeRef = *const c_void;
type CFMutableDataRef = *mut c_void;
type CFDictionaryRef = *const c_void;
type CFStringRef = *const c_void;
type CFNumberRef = *const c_void;
type CGImageRef = *mut c_void;
type CGImageDestinationRef = *mut c_void;
type CGContextRef = *mut c_void;
type CGColorSpaceRef = *mut c_void;
type CGImageSourceRef = *mut c_void;
type CFDataRef = *const c_void;

// CFNumber type constants
const K_CF_NUMBER_FLOAT32_TYPE: i64 = 3;

extern "C" {
    // CoreFoundation
    fn CFDataCreateMutable(allocator: CFTypeRef, capacity: i64) -> CFMutableDataRef;
    fn CFDataGetBytePtr(data: CFDataRef) -> *const u8;
    fn CFDataGetLength(data: CFDataRef) -> i64;
    fn CFRelease(cf: CFTypeRef);
    fn CFDictionaryCreate(
        allocator: CFTypeRef,
        keys: *const CFTypeRef,
        values: *const CFTypeRef,
        num_values: i64,
        key_callbacks: CFTypeRef,
        value_callbacks: CFTypeRef,
    ) -> CFDictionaryRef;
    fn CFNumberCreate(
        allocator: CFTypeRef,
        the_type: i64,
        value_ptr: *const c_void,
    ) -> CFNumberRef;

    // kCFTypeDictionaryKeyCallBacks / kCFTypeDictionaryValueCallBacks
    static kCFTypeDictionaryKeyCallBacks: c_void;
    static kCFTypeDictionaryValueCallBacks: c_void;

    // ImageIO - CGImageDestination
    fn CGImageDestinationCreateWithData(
        data: CFMutableDataRef,
        type_: CFStringRef,
        count: usize,
        options: CFDictionaryRef,
    ) -> CGImageDestinationRef;
    fn CGImageDestinationAddImage(
        dest: CGImageDestinationRef,
        image: CGImageRef,
        properties: CFDictionaryRef,
    );
    fn CGImageDestinationFinalize(dest: CGImageDestinationRef) -> bool;

    // ImageIO - CGImageSource (for loading JPEG back to CGImage)
    fn CGImageSourceCreateWithData(
        data: CFDataRef,
        options: CFDictionaryRef,
    ) -> CGImageSourceRef;
    fn CGImageSourceCreateImageAtIndex(
        source: CGImageSourceRef,
        index: usize,
        options: CFDictionaryRef,
    ) -> CGImageRef;

    // CoreFoundation string constant for JPEG UTI
    static kCGImageDestinationLossyCompressionQuality: CFStringRef;

    // CoreGraphics
    fn CGImageGetWidth(image: CGImageRef) -> usize;
    fn CGImageGetHeight(image: CGImageRef) -> usize;
    fn CGImageRelease(image: CGImageRef);
    fn CGColorSpaceCreateDeviceRGB() -> CGColorSpaceRef;
    fn CGColorSpaceRelease(space: CGColorSpaceRef);
    fn CGBitmapContextCreate(
        data: *mut c_void,
        width: usize,
        height: usize,
        bits_per_component: usize,
        bytes_per_row: usize,
        color_space: CGColorSpaceRef,
        bitmap_info: u32,
    ) -> CGContextRef;
    fn CGBitmapContextCreateImage(context: CGContextRef) -> CGImageRef;
    fn CGContextDrawImage(context: CGContextRef, rect: CGRect, image: CGImageRef);
    fn CGContextRelease(context: CGContextRef);

    // Screen capture
    fn CGWindowListCreateImage(
        screen_bounds: CGRect,
        list_option: u32,
        window_id: u32,
        image_option: u32,
    ) -> CGImageRef;

    // CFData from raw bytes
    fn CFDataCreate(
        allocator: CFTypeRef,
        bytes: *const u8,
        length: i64,
    ) -> CFDataRef;
}

// JPEG UTI string: "public.jpeg"
fn jpeg_uti() -> CFStringRef {
    extern "C" {
        // This is available as a constant in ImageIO
        static kUTTypeJPEG: CFStringRef;
    }
    unsafe { kUTTypeJPEG }
}

// kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big
const BITMAP_INFO: u32 = 1 | (2 << 12);

/// Capture the entire screen as a JPEG image.
///
/// Returns `None` if screen capture fails (e.g., no Screen Recording permission).
pub fn capture_full_screen_jpeg(config: &ScreenshotConfig) -> Option<CapturedScreenshot> {
    unsafe {
        // Capture full screen: CGRectInfinite is (0,0,0,0) for "entire screen"
        let null_rect = CGRect::new(
            &CGPoint::new(f64::INFINITY, f64::INFINITY),
            &CGSize::new(0.0, 0.0),
        );
        // kCGWindowListOptionOnScreenOnly = 1, kCGNullWindowID = 0, kCGWindowImageDefault = 0
        let image = CGWindowListCreateImage(null_rect, 1, 0, 0);
        if image.is_null() {
            return None;
        }

        let result = cgimage_to_jpeg(image, config);
        CGImageRelease(image);
        result.map(|(jpeg_data, width, height)| CapturedScreenshot {
            jpeg_data,
            width,
            height,
        })
    }
}

/// Convert a CGImageRef to JPEG bytes, scaling if necessary.
///
/// # Safety
/// The caller must ensure `image` is a valid CGImageRef or null.
/// The caller retains ownership of `image` — this function does NOT release it.
pub unsafe fn cgimage_to_jpeg(image: *mut c_void, config: &ScreenshotConfig) -> Option<(Vec<u8>, u32, u32)> {
    if image.is_null() {
        return None;
    }

    let orig_w = CGImageGetWidth(image);
    let orig_h = CGImageGetHeight(image);

    if orig_w == 0 || orig_h == 0 {
        return None;
    }

    // Scale down if wider than max_width
    let (final_image, needs_release) = if orig_w > config.max_width as usize {
        match scale_cgimage(image, config.max_width) {
            Some(scaled) => (scaled, true),
            None => (image, false),
        }
    } else {
        (image, false)
    };

    let final_w = CGImageGetWidth(final_image) as u32;
    let final_h = CGImageGetHeight(final_image) as u32;

    let jpeg_data = write_cgimage_as_jpeg(final_image, config.jpeg_quality);

    if needs_release {
        CGImageRelease(final_image);
    }

    jpeg_data.map(|data| (data, final_w, final_h))
}

/// Release a CGImageRef.
///
/// # Safety
/// The caller must ensure `image` is a valid, non-null CGImageRef that has not
/// already been released. After calling this function, the pointer is invalid.
pub unsafe fn release_cgimage(image: *mut c_void) {
    CGImageRelease(image);
}

/// Load JPEG bytes back into a CGImageRef.
///
/// The caller is responsible for releasing the returned CGImageRef via CGImageRelease.
/// Returns `None` if the data cannot be parsed as an image.
pub fn jpeg_to_cgimage(jpeg_data: &[u8]) -> Option<*mut c_void> {
    if jpeg_data.is_empty() {
        return None;
    }
    unsafe {
        let cf_data = CFDataCreate(
            std::ptr::null(),
            jpeg_data.as_ptr(),
            jpeg_data.len() as i64,
        );
        if cf_data.is_null() {
            return None;
        }

        let source = CGImageSourceCreateWithData(cf_data, std::ptr::null());
        CFRelease(cf_data);
        if source.is_null() {
            return None;
        }

        let image = CGImageSourceCreateImageAtIndex(source, 0, std::ptr::null());
        CFRelease(source as CFTypeRef);

        if image.is_null() {
            None
        } else {
            Some(image)
        }
    }
}

/// Write a CGImageRef as JPEG bytes using ImageIO CGImageDestination.
unsafe fn write_cgimage_as_jpeg(image: CGImageRef, quality: f32) -> Option<Vec<u8>> {
    // Create mutable CFData to hold output
    let data = CFDataCreateMutable(std::ptr::null(), 0);
    if data.is_null() {
        return None;
    }

    // Create image destination for JPEG
    let dest = CGImageDestinationCreateWithData(data, jpeg_uti(), 1, std::ptr::null());
    if dest.is_null() {
        CFRelease(data);
        return None;
    }

    // Create quality properties dictionary
    let quality_key = kCGImageDestinationLossyCompressionQuality;
    let quality_value = CFNumberCreate(
        std::ptr::null(),
        K_CF_NUMBER_FLOAT32_TYPE,
        &quality as *const f32 as *const c_void,
    );

    let keys = [quality_key as CFTypeRef];
    let values = [quality_value as CFTypeRef];

    let properties = CFDictionaryCreate(
        std::ptr::null(),
        keys.as_ptr(),
        values.as_ptr(),
        1,
        &kCFTypeDictionaryKeyCallBacks as *const c_void as CFTypeRef,
        &kCFTypeDictionaryValueCallBacks as *const c_void as CFTypeRef,
    );

    // Add image with properties
    CGImageDestinationAddImage(dest, image, properties);

    // Finalize
    let success = CGImageDestinationFinalize(dest);

    // Cleanup
    CFRelease(properties as CFTypeRef);
    CFRelease(quality_value as CFTypeRef);
    CFRelease(dest as CFTypeRef);

    if !success {
        CFRelease(data);
        return None;
    }

    // Extract bytes from CFData
    let ptr = CFDataGetBytePtr(data);
    let len = CFDataGetLength(data) as usize;

    if ptr.is_null() || len == 0 {
        CFRelease(data);
        return None;
    }

    let bytes = std::slice::from_raw_parts(ptr, len).to_vec();
    CFRelease(data);

    Some(bytes)
}

/// Scale a CGImageRef to the specified max width, preserving aspect ratio.
unsafe fn scale_cgimage(image: CGImageRef, max_width: u32) -> Option<CGImageRef> {
    let orig_w = CGImageGetWidth(image);
    let orig_h = CGImageGetHeight(image);

    if orig_w == 0 || orig_h == 0 {
        return None;
    }

    let scale = max_width as f64 / orig_w as f64;
    let new_w = max_width as usize;
    let new_h = (orig_h as f64 * scale).round() as usize;

    if new_w == 0 || new_h == 0 {
        return None;
    }

    let color_space = CGColorSpaceCreateDeviceRGB();
    if color_space.is_null() {
        return None;
    }

    let bytes_per_row = new_w * 4;
    let context = CGBitmapContextCreate(
        std::ptr::null_mut(),
        new_w,
        new_h,
        8,
        bytes_per_row,
        color_space,
        BITMAP_INFO,
    );
    CGColorSpaceRelease(color_space);

    if context.is_null() {
        return None;
    }

    let draw_rect = CGRect::new(
        &CGPoint::new(0.0, 0.0),
        &CGSize::new(new_w as f64, new_h as f64),
    );
    CGContextDrawImage(context, draw_rect, image);

    let scaled_image = CGBitmapContextCreateImage(context);
    CGContextRelease(context);

    if scaled_image.is_null() {
        None
    } else {
        Some(scaled_image)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_null_pointer_returns_none() {
        let config = ScreenshotConfig::default();
        let result = unsafe { cgimage_to_jpeg(std::ptr::null_mut(), &config) };
        assert!(result.is_none(), "Null image should return None");
    }

    #[test]
    fn test_default_config_values() {
        let config = ScreenshotConfig::default();
        assert!((config.jpeg_quality - 0.8).abs() < f32::EPSILON);
        assert_eq!(config.max_width, 1024);
    }

    #[test]
    fn test_custom_config() {
        let config = ScreenshotConfig {
            jpeg_quality: 0.5,
            max_width: 512,
        };
        assert!((config.jpeg_quality - 0.5).abs() < f32::EPSILON);
        assert_eq!(config.max_width, 512);
    }

    #[test]
    fn test_capture_full_screen_produces_jpeg() {
        // This test requires Screen Recording permission; skip if capture fails
        let config = ScreenshotConfig::default();
        if let Some(screenshot) = capture_full_screen_jpeg(&config) {
            // Check JPEG magic bytes
            assert!(screenshot.jpeg_data.len() >= 2);
            assert_eq!(screenshot.jpeg_data[0], 0xFF);
            assert_eq!(screenshot.jpeg_data[1], 0xD8);
            // Check dimensions are reasonable
            assert!(screenshot.width > 0);
            assert!(screenshot.height > 0);
            assert!(screenshot.width <= config.max_width);
        }
    }

    #[test]
    fn test_scaling_produces_width_within_max() {
        let config = ScreenshotConfig {
            jpeg_quality: 0.8,
            max_width: 256,
        };
        if let Some(screenshot) = capture_full_screen_jpeg(&config) {
            assert!(screenshot.width <= 256);
            assert!(screenshot.height > 0);
        }
    }

    #[test]
    fn test_jpeg_roundtrip_via_cgimage_source() {
        // Capture a screenshot, convert to JPEG, then load back as CGImage
        let config = ScreenshotConfig::default();
        if let Some(screenshot) = capture_full_screen_jpeg(&config) {
            let cgimage = jpeg_to_cgimage(&screenshot.jpeg_data);
            assert!(cgimage.is_some(), "Should be able to load JPEG back as CGImage");
            if let Some(img) = cgimage {
                unsafe { CGImageRelease(img); }
            }
        }
    }

    #[test]
    fn test_jpeg_to_cgimage_empty_data() {
        let result = jpeg_to_cgimage(&[]);
        assert!(result.is_none(), "Empty data should return None");
    }

    #[test]
    fn test_jpeg_to_cgimage_invalid_data() {
        let result = jpeg_to_cgimage(&[0x00, 0x01, 0x02, 0x03]);
        assert!(result.is_none(), "Invalid data should return None");
    }
}
