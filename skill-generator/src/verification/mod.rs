//! Verification-Centric Design
//!
//! Ensures every step in skill.md is deterministically verifiable.

pub mod postcondition_extractor;
pub mod hoare_triple_generator;

pub use postcondition_extractor::PostconditionExtractor;
pub use hoare_triple_generator::{HoareTripleGenerator, VerificationBlock};
