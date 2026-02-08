//! Trajectory analysis and intent inference
//!
//! This module transforms noisy mouse point clouds into clean,
//! intent-bearing trajectories using:
//! - Ramer-Douglas-Peucker polyline simplification
//! - Fitts' Law kinematic segmentation
//! - Intent binding from trajectory + semantics

pub mod rdp_simplification;
pub mod kinematic_segmentation;
pub mod intent_binding;

pub use rdp_simplification::RdpSimplifier;
pub use kinematic_segmentation::KinematicSegmenter;
pub use intent_binding::{Action, IntentBinder};
