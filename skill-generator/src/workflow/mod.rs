//! Workflow Module
//!
//! Orchestrates the complete skill generation workflow from recording to SKILL.md output.

pub mod recording;
pub mod generator;

pub use recording::{Recording, RecordingMetadata};
pub use generator::SkillGenerator;
