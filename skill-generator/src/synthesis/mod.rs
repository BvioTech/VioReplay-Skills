//! Variable Extraction & Generalization
//!
//! Transforms static recordings into reusable, parameterized skills.

pub mod variable_extraction;
pub mod selector_ranking;
pub mod llm_semantic_synthesis;

pub use variable_extraction::{VariableExtractor, ExtractedVariable};
pub use selector_ranking::{SelectorRanker, RankedSelector};
pub use llm_semantic_synthesis::LlmSynthesizer;
