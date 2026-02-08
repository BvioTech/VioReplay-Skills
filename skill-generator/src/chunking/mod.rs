//! GOMS-Based Action Segmentation
//!
//! Divides continuous action streams into high-level Unit Tasks using
//! cognitive signal detection and context tracking.

pub mod goms_detector;
pub mod context_stack;
pub mod action_clustering;

pub use goms_detector::GomsDetector;
pub use context_stack::ContextStack;
pub use action_clustering::ActionClusterer;
