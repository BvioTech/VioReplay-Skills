//! # Skill Generator
//!
//! A production-grade macOS Skill Generation Engine that transforms user interaction
//! telemetry into executable Claude Code SKILL.md artifacts.
//!
//! ## Overview
//!
//! This library captures mouse movements, clicks, keyboard events, and scroll events
//! using the macOS Quartz Event Tap API. Events are enriched with semantic context
//! from the Accessibility API, then transformed into reusable skills.
//!
//! ## Quick Start
//!
//! ```no_run
//! use skill_generator::{Recording, SkillGenerator, MachTimebase};
//! use skill_generator::capture::types::{EventType, EnrichedEvent};
//!
//! // Initialize the timebase (required once)
//! MachTimebase::init();
//!
//! // Create a recording
//! let mut recording = Recording::new(
//!     "my_skill".to_string(),
//!     Some("Demonstrate a workflow".to_string()),
//! );
//!
//! // ... add events to the recording ...
//!
//! // Generate a skill
//! let generator = SkillGenerator::new();
//! let skill = generator.generate(&recording).expect("Failed to generate");
//!
//! // Render to markdown
//! let markdown = generator.render_to_markdown(&skill);
//! println!("{}", markdown);
//! ```
//!
//! ## Architecture
//!
//! The system is organized into the following modules:
//!
//! - [`capture`]: Lock-free event capture using Quartz Event Tap
//! - [`time`]: High-precision timing using mach_absolute_time
//! - [`analysis`]: Trajectory analysis and intent inference
//! - [`chunking`]: GOMS-based action segmentation
//! - [`semantic`]: Accessibility API integration and vision fallback
//! - [`synthesis`]: Variable extraction and generalization
//! - [`codegen`]: SKILL.md generation and validation
//! - [`verification`]: Postcondition extraction and Hoare triple generation
//! - [`workflow`]: High-level recording and skill generation
//! - [`app`]: CLI and configuration management
//!
//! ## Event Pipeline
//!
//! ```text
//! ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
//! │  CGEventTap │───▶│ Ring Buffer │───▶│  Semantic   │───▶│  Recording  │
//! │  (capture)  │    │ (lock-free) │    │  Enrichment │    │             │
//! └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
//!                                                                 │
//!                                                                 ▼
//! ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
//! │  SKILL.md   │◀───│  Codegen &  │◀───│  Variable   │◀───│   Intent    │
//! │   Output    │    │ Validation  │    │ Extraction  │    │  Inference  │
//! └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
//! ```
//!
//! ## Permissions
//!
//! This library requires Accessibility permissions on macOS:
//! System Preferences → Security & Privacy → Accessibility

pub mod time;
pub mod capture;
pub mod analysis;
pub mod chunking;
pub mod semantic;
pub mod synthesis;
pub mod codegen;
pub mod verification;
pub mod app;
pub mod workflow;

// Re-export commonly used types
pub use capture::types::{RawEvent, EventType, CursorState, SemanticContext, SemanticState};
pub use capture::ring_buffer::EventRingBuffer;
pub use time::timebase::MachTimebase;
pub use workflow::{Recording, SkillGenerator};

/// Result type alias for the skill generator
pub type Result<T> = std::result::Result<T, Error>;

/// Error types for the skill generator
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Event capture error: {0}")]
    Capture(String),

    #[error("Ring buffer error: {0}")]
    RingBuffer(String),

    #[error("Accessibility API error: {0}")]
    Accessibility(String),

    #[error("Vision fallback error: {0}")]
    Vision(String),

    #[error("Analysis error: {0}")]
    Analysis(String),

    #[error("Chunking error: {0}")]
    Chunking(String),

    #[error("Synthesis error: {0}")]
    Synthesis(String),

    #[error("Codegen error: {0}")]
    Codegen(String),

    #[error("Validation error: {0}")]
    Validation(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}
