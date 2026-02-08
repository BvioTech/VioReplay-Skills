//! Semantic Void Handling & Vision Fallback
//!
//! Provides resilience against opaque UIs (Electron, Chromium, custom canvas)
//! through null recovery, vision fallback, and context reconstruction.

pub mod null_handler;
pub mod vision_fallback;
pub mod context_reconstruction;

pub use null_handler::NullHandler;
pub use vision_fallback::VisionFallback;
pub use context_reconstruction::ContextReconstructor;
