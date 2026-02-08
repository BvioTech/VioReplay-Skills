//! Robust UI Targeting with Selector Ranking

use crate::capture::types::SemanticContext;
use serde::{Deserialize, Serialize};

/// Selector type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SelectorType {
    /// AX identifier (highest stability)
    AxIdentifier,
    /// Text content match
    TextContent,
    /// Relative position to anchor
    RelativePosition,
    /// CSS selector
    CssSelector,
    /// XPath
    XPath,
}

/// Stability score for a selector type
impl SelectorType {
    pub fn stability_score(&self) -> f32 {
        match self {
            SelectorType::AxIdentifier => 0.95,
            SelectorType::TextContent => 0.75,
            SelectorType::RelativePosition => 0.85,
            SelectorType::CssSelector => 0.5,
            SelectorType::XPath => 0.6,
        }
    }
}

/// A ranked selector for targeting UI elements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankedSelector {
    /// Selector type
    pub selector_type: SelectorType,
    /// The selector value
    pub value: String,
    /// Stability score (0-1)
    pub stability: f32,
    /// Specificity score (0-1)
    pub specificity: f32,
    /// Combined rank score
    pub rank_score: f32,
}

/// Selector chain for fallback
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectorChain {
    /// Primary selector (best ranked)
    pub primary: RankedSelector,
    /// Fallback selectors in order
    pub fallbacks: Vec<RankedSelector>,
}

impl SelectorChain {
    /// Get all selectors in rank order
    pub fn all_selectors(&self) -> impl Iterator<Item = &RankedSelector> {
        std::iter::once(&self.primary).chain(self.fallbacks.iter())
    }
}

/// Selector ranker
pub struct SelectorRanker {
    /// Stability weight in ranking
    pub stability_weight: f32,
    /// Specificity weight in ranking
    pub specificity_weight: f32,
}

impl SelectorRanker {
    /// Create with default weights
    pub fn new() -> Self {
        Self {
            stability_weight: 0.7,
            specificity_weight: 0.3,
        }
    }

    /// Generate and rank selectors for a semantic context
    pub fn generate_selectors(&self, semantic: &SemanticContext) -> SelectorChain {
        let mut selectors = Vec::new();

        // AX Identifier (if available)
        if let Some(id) = &semantic.identifier {
            selectors.push(RankedSelector {
                selector_type: SelectorType::AxIdentifier,
                value: id.clone(),
                stability: SelectorType::AxIdentifier.stability_score(),
                specificity: 1.0,
                rank_score: 0.0, // Will be calculated
            });
        }

        // Text content
        if let Some(title) = &semantic.title {
            selectors.push(RankedSelector {
                selector_type: SelectorType::TextContent,
                value: title.clone(),
                stability: SelectorType::TextContent.stability_score(),
                specificity: self.calculate_text_specificity(title),
                rank_score: 0.0,
            });
        }

        // Relative position (if parent info available)
        if let (Some(parent_title), Some(role)) = (&semantic.parent_title, &semantic.ax_role) {
            let relative = format!("{} {} of '{}'", role, "child", parent_title);
            selectors.push(RankedSelector {
                selector_type: SelectorType::RelativePosition,
                value: relative,
                stability: SelectorType::RelativePosition.stability_score(),
                specificity: 0.8,
                rank_score: 0.0,
            });
        }

        // CSS-like selector (simulated)
        if let Some(role) = &semantic.ax_role {
            let css = self.generate_css_selector(semantic, role);
            selectors.push(RankedSelector {
                selector_type: SelectorType::CssSelector,
                value: css,
                stability: SelectorType::CssSelector.stability_score(),
                specificity: 0.9,
                rank_score: 0.0,
            });
        }

        // Calculate rank scores
        for selector in &mut selectors {
            selector.rank_score = self.calculate_rank(selector);
        }

        // Sort by rank (descending)
        selectors.sort_by(|a, b| b.rank_score.partial_cmp(&a.rank_score).unwrap_or(std::cmp::Ordering::Equal));

        // Build chain
        if selectors.is_empty() {
            // Create a placeholder selector
            let placeholder = RankedSelector {
                selector_type: SelectorType::TextContent,
                value: "unknown".to_string(),
                stability: 0.1,
                specificity: 0.1,
                rank_score: 0.1,
            };
            SelectorChain {
                primary: placeholder,
                fallbacks: vec![],
            }
        } else {
            SelectorChain {
                primary: selectors.remove(0),
                fallbacks: selectors.into_iter().take(2).collect(), // Keep top 2 as fallbacks
            }
        }
    }

    /// Calculate rank score for a selector
    fn calculate_rank(&self, selector: &RankedSelector) -> f32 {
        self.stability_weight * selector.stability + self.specificity_weight * selector.specificity
    }

    /// Calculate specificity for text content
    fn calculate_text_specificity(&self, text: &str) -> f32 {
        // Longer, more unique text = higher specificity
        let length_score = (text.len() as f32 / 50.0).min(1.0);

        // Presence of numbers increases specificity
        let has_numbers = text.chars().any(|c| c.is_ascii_digit());
        let number_bonus = if has_numbers { 0.1 } else { 0.0 };

        (0.5 + length_score * 0.4 + number_bonus).min(1.0)
    }

    /// Generate a CSS-like selector
    fn generate_css_selector(&self, semantic: &SemanticContext, role: &str) -> String {
        let mut parts = Vec::new();

        // Add role
        parts.push(role.to_string());

        // Add identifier if present
        if let Some(id) = &semantic.identifier {
            parts.push(format!("#'{}'", id));
        } else if let Some(title) = &semantic.title {
            // Use title as attribute selector
            parts.push(format!("[title='{}']", title));
        }

        parts.join("")
    }
}

impl Default for SelectorRanker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_selector_type_stability() {
        assert!(SelectorType::AxIdentifier.stability_score() > SelectorType::CssSelector.stability_score());
    }

    #[test]
    fn test_selector_type_stability_ordering() {
        // Test all selector types are properly ordered
        assert!(SelectorType::AxIdentifier.stability_score() > SelectorType::RelativePosition.stability_score());
        assert!(SelectorType::RelativePosition.stability_score() > SelectorType::TextContent.stability_score());
        assert!(SelectorType::TextContent.stability_score() > SelectorType::XPath.stability_score());
        assert!(SelectorType::XPath.stability_score() > SelectorType::CssSelector.stability_score());
    }

    #[test]
    fn test_generate_selectors() {
        let ranker = SelectorRanker::new();

        let semantic = SemanticContext {
            ax_role: Some("AXButton".to_string()),
            title: Some("Submit".to_string()),
            identifier: Some("btn-submit".to_string()),
            ..Default::default()
        };

        let chain = ranker.generate_selectors(&semantic);

        // Primary should be AX identifier (highest stability)
        assert_eq!(chain.primary.selector_type, SelectorType::AxIdentifier);
        assert!(!chain.fallbacks.is_empty());
    }

    #[test]
    fn test_generate_selectors_no_identifier() {
        let ranker = SelectorRanker::new();

        let semantic = SemanticContext {
            ax_role: Some("AXButton".to_string()),
            title: Some("Submit".to_string()),
            identifier: None,
            ..Default::default()
        };

        let chain = ranker.generate_selectors(&semantic);

        // Without identifier, should use text content or CSS
        assert_ne!(chain.primary.selector_type, SelectorType::AxIdentifier);
        assert!(matches!(
            chain.primary.selector_type,
            SelectorType::TextContent | SelectorType::CssSelector
        ));
    }

    #[test]
    fn test_generate_selectors_empty_semantic() {
        let ranker = SelectorRanker::new();

        let semantic = SemanticContext {
            ax_role: None,
            title: None,
            identifier: None,
            ..Default::default()
        };

        let chain = ranker.generate_selectors(&semantic);

        // Should return placeholder selector
        assert_eq!(chain.primary.value, "unknown");
        assert!(chain.fallbacks.is_empty());
    }

    #[test]
    fn test_generate_selectors_with_parent() {
        let ranker = SelectorRanker::new();

        let semantic = SemanticContext {
            ax_role: Some("AXButton".to_string()),
            title: Some("Submit".to_string()),
            parent_title: Some("Form Container".to_string()),
            ..Default::default()
        };

        let chain = ranker.generate_selectors(&semantic);

        // Should include relative position selector
        let has_relative = chain.all_selectors().any(|s| {
            s.selector_type == SelectorType::RelativePosition
        });
        assert!(has_relative);
    }

    #[test]
    fn test_text_specificity() {
        let ranker = SelectorRanker::new();

        let short = ranker.calculate_text_specificity("OK");
        let long = ranker.calculate_text_specificity("Submit Order #12345");

        assert!(long > short);
    }

    #[test]
    fn test_text_specificity_with_numbers() {
        let ranker = SelectorRanker::new();

        let without_numbers = ranker.calculate_text_specificity("Submit Order");
        let with_numbers = ranker.calculate_text_specificity("Submit Order #12345");

        // Text with numbers should have higher specificity
        assert!(with_numbers > without_numbers);
    }

    #[test]
    fn test_text_specificity_edge_cases() {
        let ranker = SelectorRanker::new();

        // Empty string
        let empty = ranker.calculate_text_specificity("");
        assert_eq!(empty, 0.5); // Base score

        // Single character
        let single = ranker.calculate_text_specificity("X");
        assert!(single > 0.5);

        // Very long text (should cap at 1.0)
        let very_long = ranker.calculate_text_specificity(&"a".repeat(100));
        assert!(very_long <= 1.0);

        // All numbers
        let all_numbers = ranker.calculate_text_specificity("123456789");
        assert!(all_numbers > 0.6); // Has number bonus
    }

    #[test]
    fn test_calculate_rank() {
        let ranker = SelectorRanker::new();

        let selector = RankedSelector {
            selector_type: SelectorType::AxIdentifier,
            value: "test-id".to_string(),
            stability: 0.95,
            specificity: 1.0,
            rank_score: 0.0,
        };

        let rank = ranker.calculate_rank(&selector);

        // Should be weighted average: 0.7 * 0.95 + 0.3 * 1.0 = 0.965
        assert!((rank - 0.965).abs() < 0.001);
    }

    #[test]
    fn test_calculate_rank_custom_weights() {
        let ranker = SelectorRanker {
            stability_weight: 0.5,
            specificity_weight: 0.5,
        };

        let selector = RankedSelector {
            selector_type: SelectorType::TextContent,
            value: "Test".to_string(),
            stability: 0.8,
            specificity: 0.6,
            rank_score: 0.0,
        };

        let rank = ranker.calculate_rank(&selector);

        // Should be equal weighted: 0.5 * 0.8 + 0.5 * 0.6 = 0.7
        assert!((rank - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_generate_css_selector_with_identifier() {
        let ranker = SelectorRanker::new();

        let semantic = SemanticContext {
            ax_role: Some("AXButton".to_string()),
            identifier: Some("submit-btn".to_string()),
            ..Default::default()
        };

        let css = ranker.generate_css_selector(&semantic, "AXButton");

        // Should include role and ID
        assert!(css.contains("AXButton"));
        assert!(css.contains("submit-btn"));
        assert!(css.contains("#"));
    }

    #[test]
    fn test_generate_css_selector_with_title() {
        let ranker = SelectorRanker::new();

        let semantic = SemanticContext {
            ax_role: Some("AXButton".to_string()),
            title: Some("Submit".to_string()),
            identifier: None,
            ..Default::default()
        };

        let css = ranker.generate_css_selector(&semantic, "AXButton");

        // Should include role and title attribute
        assert!(css.contains("AXButton"));
        assert!(css.contains("[title='Submit']"));
    }

    #[test]
    fn test_generate_css_selector_with_special_characters() {
        let ranker = SelectorRanker::new();

        let semantic = SemanticContext {
            ax_role: Some("AXButton".to_string()),
            title: Some("Submit & Continue".to_string()),
            identifier: None,
            ..Default::default()
        };

        let css = ranker.generate_css_selector(&semantic, "AXButton");

        // Should still generate selector with special characters
        assert!(css.contains("AXButton"));
        assert!(css.contains("Submit & Continue"));
    }

    #[test]
    fn test_selector_chain_all_selectors() {
        let ranker = SelectorRanker::new();

        let semantic = SemanticContext {
            ax_role: Some("AXButton".to_string()),
            title: Some("Submit".to_string()),
            identifier: Some("btn-submit".to_string()),
            ..Default::default()
        };

        let chain = ranker.generate_selectors(&semantic);
        let all: Vec<_> = chain.all_selectors().collect();

        // Should include primary + fallbacks
        assert!(all.len() >= 2);
        assert_eq!(all[0].selector_type, chain.primary.selector_type);
    }

    #[test]
    fn test_selector_ranking_order() {
        let ranker = SelectorRanker::new();

        let semantic = SemanticContext {
            ax_role: Some("AXButton".to_string()),
            title: Some("Submit Order #12345".to_string()),
            identifier: Some("btn-submit".to_string()),
            parent_title: Some("Form".to_string()),
            ..Default::default()
        };

        let chain = ranker.generate_selectors(&semantic);

        // Primary should have highest rank score
        for fallback in &chain.fallbacks {
            assert!(chain.primary.rank_score >= fallback.rank_score);
        }

        // Fallbacks should be ordered by rank
        for i in 0..(chain.fallbacks.len() - 1) {
            assert!(chain.fallbacks[i].rank_score >= chain.fallbacks[i + 1].rank_score);
        }
    }

    #[test]
    fn test_fallback_limit() {
        let ranker = SelectorRanker::new();

        let semantic = SemanticContext {
            ax_role: Some("AXButton".to_string()),
            title: Some("Submit".to_string()),
            identifier: Some("btn-submit".to_string()),
            parent_title: Some("Form".to_string()),
            ..Default::default()
        };

        let chain = ranker.generate_selectors(&semantic);

        // Should keep max 2 fallbacks
        assert!(chain.fallbacks.len() <= 2);
    }

    #[test]
    fn test_selector_chain_with_minimal_context() {
        let ranker = SelectorRanker::new();

        let semantic = SemanticContext {
            title: Some("OK".to_string()),
            ..Default::default()
        };

        let chain = ranker.generate_selectors(&semantic);

        // Should still generate a valid selector
        assert_eq!(chain.primary.selector_type, SelectorType::TextContent);
        assert_eq!(chain.primary.value, "OK");
    }
}
