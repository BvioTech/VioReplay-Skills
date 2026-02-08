//! SKILL.md Generation & Validation
//!
//! Generates production-grade, agent-executable SKILL.md files.

pub mod skill_compiler;
pub mod markdown_builder;
pub mod validation;

pub use skill_compiler::SkillCompiler;
pub use markdown_builder::MarkdownBuilder;
pub use validation::SkillValidator;
