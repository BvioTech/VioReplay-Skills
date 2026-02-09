//! Semantic Void Handling & Vision Fallback
//!
//! Provides resilience against opaque UIs (Electron, Chromium, custom canvas)
//! through null recovery, vision fallback, and context reconstruction.

pub mod null_handler;
pub mod vision_fallback;
pub mod context_reconstruction;
pub mod http_retry;
pub mod screenshot;
pub mod screenshot_analysis;

pub use null_handler::NullHandler;
pub use vision_fallback::VisionFallback;
pub use context_reconstruction::ContextReconstructor;
pub use screenshot::{ScreenshotConfig, CapturedScreenshot, capture_full_screen_jpeg};
pub use screenshot_analysis::{
    AnalysisConfig as ScreenshotAnalysisConfig, AnalysisTier, ScreenshotAnalysis,
    RecordingAnalysis, analyze_recording as analyze_screenshots,
    save_analysis, load_analysis,
};
