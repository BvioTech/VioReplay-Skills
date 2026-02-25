//! CGImage-to-JPEG Conversion & Visual Annotation Module
//!
//! Captures screenshots and converts CGImageRef to JPEG bytes using macOS ImageIO
//! framework FFI. Supports configurable quality and automatic downscaling.
//!
//! Also provides visual anchoring: draws interaction markers (red dot, AX bounding
//! box) and generates foveated crops for higher-accuracy LLM analysis.

use core_graphics::display::{CGRect, CGPoint, CGSize};
use image::RgbImage;
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

// ---------------------------------------------------------------------------
// Visual Annotation — draw interaction markers and generate foveated crops
// ---------------------------------------------------------------------------

/// Configuration for visual annotations drawn onto screenshots.
#[derive(Debug, Clone)]
pub struct AnnotationConfig {
    /// Radius of the interaction-point dot in image pixels.
    pub dot_radius: i32,
    /// Color of the interaction dot (R, G, B).
    pub dot_color: [u8; 3],
    /// Color of the AX bounding box (R, G, B).
    pub box_color: [u8; 3],
    /// Thickness of the bounding box in image pixels.
    pub box_thickness: u32,
    /// Side length of the foveated crop in image pixels.
    pub crop_size: u32,
    /// JPEG quality for annotated output (1–100).
    pub jpeg_quality: u8,
    /// Color for ballistic (confident) trajectory lines (R, G, B).
    pub trajectory_ballistic_color: [u8; 3],
    /// Color for searching/corrective trajectory lines (R, G, B).
    pub trajectory_searching_color: [u8; 3],
    /// Thickness of trajectory lines in image pixels.
    pub trajectory_thickness: u32,
}

impl Default for AnnotationConfig {
    fn default() -> Self {
        Self {
            dot_radius: 12,
            dot_color: [255, 40, 40],       // bright red
            box_color: [0, 255, 100],        // neon green
            box_thickness: 2,
            crop_size: 512,
            jpeg_quality: 85,
            trajectory_ballistic_color: [40, 220, 40],   // green — confident
            trajectory_searching_color: [255, 160, 40],  // orange — hesitant
            trajectory_thickness: 2,
        }
    }
}

/// Trajectory data to overlay on a screenshot.
#[derive(Debug, Clone)]
pub struct TrajectoryOverlay {
    /// Points in screen-pixel coordinates.
    pub points: Vec<(f64, f64)>,
    /// Movement classification.
    pub pattern: crate::analysis::kinematic_segmentation::MovementPattern,
}

/// Annotated screenshot: the marked-up full image plus a foveated crop.
pub struct AnnotatedScreenshot {
    /// JPEG bytes of the full screenshot with visual markers.
    pub full_jpeg: Vec<u8>,
    /// JPEG bytes of the unscaled crop centred on the interaction point.
    pub crop_jpeg: Vec<u8>,
    /// Width / height of the crop (may be smaller than `crop_size` at edges).
    pub crop_width: u32,
    pub crop_height: u32,
}

/// Decode JPEG bytes into an `RgbImage` via the `image` crate.
pub fn decode_jpeg_to_rgb(jpeg_data: &[u8]) -> Option<RgbImage> {
    use image::ImageReader;
    use std::io::Cursor;

    ImageReader::new(Cursor::new(jpeg_data))
        .with_guessed_format()
        .ok()?
        .decode()
        .ok()
        .map(|img| img.into_rgb8())
}

/// Encode an `RgbImage` as JPEG bytes.
pub fn encode_rgb_to_jpeg(image: &RgbImage, quality: u8) -> Option<Vec<u8>> {
    use image::codecs::jpeg::JpegEncoder;
    use image::ImageEncoder;

    let mut buf: Vec<u8> = Vec::new();
    let encoder = JpegEncoder::new_with_quality(&mut buf, quality);
    encoder
        .write_image(
            image.as_raw(),
            image.width(),
            image.height(),
            image::ExtendedColorType::Rgb8,
        )
        .ok()?;
    Some(buf)
}

/// Draw a filled circle at `(cx, cy)` in image-pixel coordinates.
///
/// Uses manual pixel blending (75 % opacity) so the underlying UI remains
/// visible through the marker.
pub fn draw_interaction_point(image: &mut RgbImage, cx: i32, cy: i32, config: &AnnotationConfig) {
    let r = config.dot_radius;
    let [dr, dg, db] = config.dot_color;
    let (w, h) = (image.width() as i32, image.height() as i32);

    for dy in -r..=r {
        for dx in -r..=r {
            if dx * dx + dy * dy <= r * r {
                let px = cx + dx;
                let py = cy + dy;
                if px >= 0 && px < w && py >= 0 && py < h {
                    let pixel = image.get_pixel_mut(px as u32, py as u32);
                    // 75 % dot, 25 % background — simple alpha blend
                    pixel.0[0] = ((dr as u16 * 3 + pixel.0[0] as u16) / 4) as u8;
                    pixel.0[1] = ((dg as u16 * 3 + pixel.0[1] as u16) / 4) as u8;
                    pixel.0[2] = ((db as u16 * 3 + pixel.0[2] as u16) / 4) as u8;
                }
            }
        }
    }
}

/// Draw a rectangle outline for the AX element bounding box.
///
/// `frame` is `(x, y, width, height)` in **image-pixel** coordinates.
pub fn draw_ax_bounds(
    image: &mut RgbImage,
    frame: (i32, i32, u32, u32),
    config: &AnnotationConfig,
) {
    let (fx, fy, fw, fh) = frame;
    let color = image::Rgb(config.box_color);
    let t = config.box_thickness as i32;
    let (iw, ih) = (image.width() as i32, image.height() as i32);

    // Helper: draw a filled rectangle clipped to image bounds.
    let mut fill_rect = |x0: i32, y0: i32, x1: i32, y1: i32| {
        let x0 = x0.max(0).min(iw);
        let y0 = y0.max(0).min(ih);
        let x1 = x1.max(0).min(iw);
        let y1 = y1.max(0).min(ih);
        for py in y0..y1 {
            for px in x0..x1 {
                image.put_pixel(px as u32, py as u32, color);
            }
        }
    };

    let right = fx + fw as i32;
    let bottom = fy + fh as i32;

    // Top edge
    fill_rect(fx, fy, right, fy + t);
    // Bottom edge
    fill_rect(fx, bottom - t, right, bottom);
    // Left edge
    fill_rect(fx, fy, fx + t, bottom);
    // Right edge
    fill_rect(right - t, fy, right, bottom);
}

/// Draw a mouse trajectory path onto the screenshot.
///
/// `points` are in **image-pixel** coordinates. The line color is determined
/// by the movement pattern: green for ballistic (confident), orange for
/// searching or corrective (hesitant).
pub fn draw_mouse_trajectory(
    image: &mut RgbImage,
    points: &[(i32, i32)],
    pattern: crate::analysis::kinematic_segmentation::MovementPattern,
    config: &AnnotationConfig,
) {
    use crate::analysis::kinematic_segmentation::MovementPattern;

    if points.len() < 2 {
        return;
    }

    let color_rgb = match pattern {
        MovementPattern::Ballistic => config.trajectory_ballistic_color,
        MovementPattern::Searching | MovementPattern::Corrective => {
            config.trajectory_searching_color
        }
        MovementPattern::Stationary => return, // nothing to draw
    };
    let color = image::Rgb(color_rgb);
    let (iw, ih) = (image.width() as i32, image.height() as i32);
    let t = config.trajectory_thickness as i32;

    // Draw thick lines by offsetting in both axes
    for window in points.windows(2) {
        let (x0, y0) = window[0];
        let (x1, y1) = window[1];

        for offset in -(t / 2)..=(t / 2) {
            // Offset perpendicular to the dominant axis for thickness
            let dx = (x1 - x0).abs();
            let dy = (y1 - y0).abs();

            let (ax, ay, bx, by) = if dx >= dy {
                // More horizontal — offset vertically
                (x0, y0 + offset, x1, y1 + offset)
            } else {
                // More vertical — offset horizontally
                (x0 + offset, y0, x1 + offset, y1)
            };

            // Clamp to image bounds for the line drawing
            let ax = ax.max(0).min(iw - 1) as f32;
            let ay = ay.max(0).min(ih - 1) as f32;
            let bx = bx.max(0).min(iw - 1) as f32;
            let by = by.max(0).min(ih - 1) as f32;

            imageproc::drawing::draw_line_segment_mut(image, (ax, ay), (bx, by), color);
        }
    }
}

/// Extract a square crop centred on `(cx, cy)` in image-pixel coordinates.
///
/// The crop may be smaller than `crop_size` if the interaction point is near
/// an image edge.
pub fn generate_local_crop(
    image: &RgbImage,
    cx: u32,
    cy: u32,
    config: &AnnotationConfig,
) -> RgbImage {
    let half = config.crop_size / 2;
    let (iw, ih) = (image.width(), image.height());

    // Compute crop bounds, clamped to image edges
    let x0 = (cx as i64 - half as i64).max(0) as u32;
    let y0 = (cy as i64 - half as i64).max(0) as u32;
    let x1 = (cx + half).min(iw);
    let y1 = (cy + half).min(ih);

    let crop_w = x1.saturating_sub(x0).max(1);
    let crop_h = y1.saturating_sub(y0).max(1);

    image::imageops::crop_imm(image, x0, y0, crop_w, crop_h).to_image()
}

/// Get the main display resolution in pixels (width, height).
///
/// Falls back to (1920, 1080) if the query fails.
pub fn get_screen_resolution() -> (f64, f64) {
    extern "C" {
        fn CGMainDisplayID() -> u32;
        fn CGDisplayPixelsWide(display: u32) -> usize;
        fn CGDisplayPixelsHigh(display: u32) -> usize;
    }
    unsafe {
        let display = CGMainDisplayID();
        let w = CGDisplayPixelsWide(display);
        let h = CGDisplayPixelsHigh(display);
        if w == 0 || h == 0 {
            (1920.0, 1080.0)
        } else {
            (w as f64, h as f64)
        }
    }
}

/// High-level: annotate a screenshot JPEG with interaction markers and produce
/// both a marked-up full image and a foveated crop.
///
/// Coordinates `screen_x`, `screen_y` and `ax_frame` are in screen-pixel space.
/// `screen_dims` is `(screen_width, screen_height)` used to compute the scale
/// transform into image-pixel space. `trajectory` optionally draws the mouse
/// path leading up to the click.
pub fn annotate_screenshot(
    jpeg_data: &[u8],
    screen_x: f64,
    screen_y: f64,
    screen_dims: (f64, f64),
    ax_frame: Option<(f64, f64, f64, f64)>,
    trajectory: Option<&TrajectoryOverlay>,
    config: &AnnotationConfig,
) -> Option<AnnotatedScreenshot> {
    let mut img = decode_jpeg_to_rgb(jpeg_data)?;
    let (iw, ih) = (img.width() as f64, img.height() as f64);
    let (sw, sh) = screen_dims;

    // Scale factor: image pixels per screen pixel
    let sx = iw / sw;
    let sy = ih / sh;

    // Map screen coordinates to image coordinates
    let ix = (screen_x * sx).round() as i32;
    let iy = (screen_y * sy).round() as i32;

    // Draw the mouse trajectory first (underneath the dot)
    if let Some(traj) = trajectory {
        if traj.points.len() >= 2 {
            let img_points: Vec<(i32, i32)> = traj
                .points
                .iter()
                .map(|(px, py)| ((px * sx).round() as i32, (py * sy).round() as i32))
                .collect();
            draw_mouse_trajectory(&mut img, &img_points, traj.pattern, config);
        }
    }

    // Draw the interaction dot
    draw_interaction_point(&mut img, ix, iy, config);

    // Draw AX bounding box if available
    if let Some((fx, fy, fw, fh)) = ax_frame {
        let img_fx = (fx * sx).round() as i32;
        let img_fy = (fy * sy).round() as i32;
        let img_fw = (fw * sx).round() as u32;
        let img_fh = (fh * sy).round() as u32;
        if img_fw > 0 && img_fh > 0 {
            draw_ax_bounds(&mut img, (img_fx, img_fy, img_fw, img_fh), config);
        }
    }

    // Generate foveated crop (from the annotated image so the dot is visible)
    let crop = generate_local_crop(
        &img,
        ix.max(0) as u32,
        iy.max(0) as u32,
        config,
    );
    let crop_w = crop.width();
    let crop_h = crop.height();

    let full_jpeg = encode_rgb_to_jpeg(&img, config.jpeg_quality)?;
    let crop_jpeg = encode_rgb_to_jpeg(&crop, config.jpeg_quality)?;

    Some(AnnotatedScreenshot {
        full_jpeg,
        crop_jpeg,
        crop_width: crop_w,
        crop_height: crop_h,
    })
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

    // -- Visual annotation tests --

    /// Create a synthetic 200x100 RGB test image and return its JPEG bytes.
    fn make_test_jpeg(width: u32, height: u32) -> Vec<u8> {
        let img = RgbImage::from_fn(width, height, |x, y| {
            // Chequerboard pattern so annotations are visible
            if (x / 20 + y / 20) % 2 == 0 {
                image::Rgb([200, 200, 200])
            } else {
                image::Rgb([100, 100, 100])
            }
        });
        encode_rgb_to_jpeg(&img, 90).expect("encode test image")
    }

    #[test]
    fn test_annotation_config_default() {
        let cfg = AnnotationConfig::default();
        assert_eq!(cfg.dot_radius, 12);
        assert_eq!(cfg.crop_size, 512);
        assert_eq!(cfg.dot_color, [255, 40, 40]);
    }

    #[test]
    fn test_decode_encode_roundtrip() {
        let jpeg = make_test_jpeg(200, 100);
        let img = decode_jpeg_to_rgb(&jpeg).expect("decode");
        assert_eq!(img.width(), 200);
        assert_eq!(img.height(), 100);
        let re_encoded = encode_rgb_to_jpeg(&img, 80).expect("re-encode");
        assert!(re_encoded.len() > 2);
        assert_eq!(re_encoded[0], 0xFF);
        assert_eq!(re_encoded[1], 0xD8);
    }

    #[test]
    fn test_draw_interaction_point_modifies_center() {
        let mut img = RgbImage::from_pixel(100, 100, image::Rgb([255, 255, 255]));
        let cfg = AnnotationConfig::default();
        draw_interaction_point(&mut img, 50, 50, &cfg);
        let center = img.get_pixel(50, 50);
        // After blending 75% red + 25% white: R ≈ 216, G ≈ 93, B ≈ 93
        assert!(center.0[0] > 150, "red channel should be bright");
        assert!(center.0[1] < 120, "green channel should be dim");
    }

    #[test]
    fn test_draw_interaction_point_clamps_to_bounds() {
        let mut img = RgbImage::from_pixel(50, 50, image::Rgb([0, 0, 0]));
        let cfg = AnnotationConfig { dot_radius: 20, ..AnnotationConfig::default() };
        // Point at corner — should not panic
        draw_interaction_point(&mut img, 0, 0, &cfg);
        draw_interaction_point(&mut img, 49, 49, &cfg);
    }

    #[test]
    fn test_draw_ax_bounds_draws_rectangle() {
        let mut img = RgbImage::from_pixel(200, 200, image::Rgb([0, 0, 0]));
        let cfg = AnnotationConfig::default();
        draw_ax_bounds(&mut img, (10, 10, 50, 30), &cfg);
        // Top-left corner of the box should be box_color
        let px = img.get_pixel(10, 10);
        assert_eq!(px.0, cfg.box_color);
    }

    #[test]
    fn test_generate_local_crop_centered() {
        let img = RgbImage::from_pixel(1024, 768, image::Rgb([128, 128, 128]));
        let cfg = AnnotationConfig { crop_size: 100, ..AnnotationConfig::default() };
        let crop = generate_local_crop(&img, 512, 384, &cfg);
        assert_eq!(crop.width(), 100);
        assert_eq!(crop.height(), 100);
    }

    #[test]
    fn test_generate_local_crop_at_edge() {
        let img = RgbImage::from_pixel(200, 200, image::Rgb([128, 128, 128]));
        let cfg = AnnotationConfig { crop_size: 200, ..AnnotationConfig::default() };
        let crop = generate_local_crop(&img, 0, 0, &cfg);
        // Crop should be clamped — starts at 0, extends up to 100
        assert!(crop.width() <= 200);
        assert!(crop.height() <= 200);
        assert!(crop.width() > 0);
    }

    #[test]
    fn test_annotate_screenshot_full_pipeline() {
        let jpeg = make_test_jpeg(400, 300);
        let cfg = AnnotationConfig {
            dot_radius: 6,
            crop_size: 64,
            ..AnnotationConfig::default()
        };
        // Screen coords: pretend screen is 800x600, image is 400x300 → 0.5 scale
        let result = annotate_screenshot(
            &jpeg,
            400.0, 300.0,       // centre of screen
            (800.0, 600.0),
            Some((380.0, 280.0, 40.0, 40.0)),
            None,
            &cfg,
        );
        let ann = result.expect("annotation should succeed");
        assert!(ann.full_jpeg.len() > 2);
        assert!(ann.crop_jpeg.len() > 2);
        assert!(ann.crop_width <= 64);
        assert!(ann.crop_height <= 64);
    }

    #[test]
    fn test_annotate_screenshot_no_ax_frame() {
        let jpeg = make_test_jpeg(400, 300);
        let cfg = AnnotationConfig { crop_size: 50, ..AnnotationConfig::default() };
        let result = annotate_screenshot(
            &jpeg, 100.0, 50.0, (400.0, 300.0), None, None, &cfg,
        );
        assert!(result.is_some());
    }

    #[test]
    fn test_get_screen_resolution_returns_nonzero() {
        let (w, h) = get_screen_resolution();
        assert!(w > 0.0);
        assert!(h > 0.0);
    }

    // -- Trajectory drawing tests --

    #[test]
    fn test_draw_mouse_trajectory_ballistic() {
        use crate::analysis::kinematic_segmentation::MovementPattern;
        let mut img = RgbImage::from_pixel(200, 200, image::Rgb([0, 0, 0]));
        let cfg = AnnotationConfig::default();
        let points = vec![(10, 10), (50, 50), (100, 100)];
        draw_mouse_trajectory(&mut img, &points, MovementPattern::Ballistic, &cfg);
        // Check that a pixel along the line was painted ballistic green
        let px = img.get_pixel(30, 30);
        assert!(px.0[1] > 100, "green channel should be bright for ballistic");
    }

    #[test]
    fn test_draw_mouse_trajectory_searching() {
        use crate::analysis::kinematic_segmentation::MovementPattern;
        let mut img = RgbImage::from_pixel(200, 200, image::Rgb([0, 0, 0]));
        let cfg = AnnotationConfig::default();
        let points = vec![(10, 100), (50, 100), (100, 100)];
        draw_mouse_trajectory(&mut img, &points, MovementPattern::Searching, &cfg);
        // Check that a pixel along the line was painted orange
        let px = img.get_pixel(30, 100);
        assert!(px.0[0] > 150, "red channel should be bright for searching");
        assert!(px.0[1] > 80, "green channel should be moderate for orange");
    }

    #[test]
    fn test_draw_mouse_trajectory_stationary_noop() {
        use crate::analysis::kinematic_segmentation::MovementPattern;
        let mut img = RgbImage::from_pixel(100, 100, image::Rgb([0, 0, 0]));
        let cfg = AnnotationConfig::default();
        let points = vec![(50, 50), (51, 51)];
        draw_mouse_trajectory(&mut img, &points, MovementPattern::Stationary, &cfg);
        // Stationary should not draw anything
        let px = img.get_pixel(50, 50);
        assert_eq!(px.0, [0, 0, 0]);
    }

    #[test]
    fn test_draw_mouse_trajectory_single_point_noop() {
        use crate::analysis::kinematic_segmentation::MovementPattern;
        let mut img = RgbImage::from_pixel(100, 100, image::Rgb([0, 0, 0]));
        let cfg = AnnotationConfig::default();
        draw_mouse_trajectory(&mut img, &[(50, 50)], MovementPattern::Ballistic, &cfg);
        // Single point — nothing drawn
        let px = img.get_pixel(50, 50);
        assert_eq!(px.0, [0, 0, 0]);
    }

    #[test]
    fn test_draw_mouse_trajectory_clamps_to_bounds() {
        use crate::analysis::kinematic_segmentation::MovementPattern;
        let mut img = RgbImage::from_pixel(50, 50, image::Rgb([0, 0, 0]));
        let cfg = AnnotationConfig::default();
        // Points outside image bounds should not panic
        let points = vec![(-10, -10), (25, 25), (60, 60)];
        draw_mouse_trajectory(&mut img, &points, MovementPattern::Ballistic, &cfg);
    }

    #[test]
    fn test_annotate_screenshot_with_trajectory() {
        use crate::analysis::kinematic_segmentation::MovementPattern;
        let jpeg = make_test_jpeg(400, 300);
        let cfg = AnnotationConfig {
            dot_radius: 6,
            crop_size: 64,
            ..AnnotationConfig::default()
        };
        let trajectory = TrajectoryOverlay {
            points: vec![(100.0, 100.0), (300.0, 200.0), (400.0, 300.0)],
            pattern: MovementPattern::Ballistic,
        };
        let result = annotate_screenshot(
            &jpeg,
            400.0, 300.0,
            (800.0, 600.0),
            None,
            Some(&trajectory),
            &cfg,
        );
        let ann = result.expect("annotation with trajectory should succeed");
        assert!(ann.full_jpeg.len() > 2);
        assert!(ann.crop_jpeg.len() > 2);
    }

    #[test]
    fn test_trajectory_overlay_config_defaults() {
        let cfg = AnnotationConfig::default();
        assert_eq!(cfg.trajectory_ballistic_color, [40, 220, 40]);
        assert_eq!(cfg.trajectory_searching_color, [255, 160, 40]);
        assert_eq!(cfg.trajectory_thickness, 2);
    }
}
