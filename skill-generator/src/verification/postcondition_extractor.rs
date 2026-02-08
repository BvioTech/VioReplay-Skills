//! State Transition Analysis and Postcondition Extraction

use crate::capture::types::EnrichedEvent;

/// State signal type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalType {
    /// AX Role (highest stability)
    AxRole,
    /// Window title change
    WindowTitle,
    /// Element appearance/visibility
    ElementAppearance,
    /// Cursor shape change
    CursorShape,
}

impl SignalType {
    /// Get stability rank (higher = more stable)
    pub fn stability_rank(&self) -> u8 {
        match self {
            SignalType::AxRole => 4,
            SignalType::WindowTitle => 3,
            SignalType::ElementAppearance => 2,
            SignalType::CursorShape => 1,
        }
    }
}

/// Extracted postcondition
#[derive(Debug, Clone)]
pub struct Postcondition {
    /// Signal type
    pub signal_type: SignalType,
    /// Attribute being checked
    pub attribute: String,
    /// Expected value
    pub expected_value: String,
    /// Pre-state value (if different)
    pub pre_value: Option<String>,
    /// Stability score
    pub stability: f32,
}

/// Postcondition extractor
pub struct PostconditionExtractor {
    /// Minimum stability threshold
    pub min_stability: f32,
}

impl PostconditionExtractor {
    /// Create with default settings
    pub fn new() -> Self {
        Self { min_stability: 0.5 }
    }

    /// Extract postconditions from pre/post state
    pub fn extract(
        &self,
        pre_event: Option<&EnrichedEvent>,
        post_event: &EnrichedEvent,
    ) -> Vec<Postcondition> {
        let mut conditions = Vec::new();

        let pre_semantic = pre_event.and_then(|e| e.semantic.as_ref());
        let post_semantic = post_event.semantic.as_ref();

        // Extract role stability
        if let Some(post) = post_semantic {
            if let Some(role) = &post.ax_role {
                conditions.push(Postcondition {
                    signal_type: SignalType::AxRole,
                    attribute: "AXRole".to_string(),
                    expected_value: role.clone(),
                    pre_value: pre_semantic.and_then(|s| s.ax_role.clone()),
                    stability: SignalType::AxRole.stability_rank() as f32 / 4.0,
                });
            }
        }

        // Extract window title changes
        if let (Some(pre), Some(post)) = (pre_semantic, post_semantic) {
            if pre.window_title != post.window_title {
                if let Some(new_title) = &post.window_title {
                    conditions.push(Postcondition {
                        signal_type: SignalType::WindowTitle,
                        attribute: "WindowTitle".to_string(),
                        expected_value: new_title.clone(),
                        pre_value: pre.window_title.clone(),
                        stability: SignalType::WindowTitle.stability_rank() as f32 / 4.0,
                    });
                }
            }
        }

        // Extract cursor shape changes
        let pre_cursor = pre_event.map(|e| e.raw.cursor_state);
        let post_cursor = post_event.raw.cursor_state;

        if pre_cursor.map(|c| c != post_cursor).unwrap_or(false) {
            conditions.push(Postcondition {
                signal_type: SignalType::CursorShape,
                attribute: "CursorState".to_string(),
                expected_value: format!("{:?}", post_cursor),
                pre_value: pre_cursor.map(|c| format!("{:?}", c)),
                stability: SignalType::CursorShape.stability_rank() as f32 / 4.0,
            });
        }

        // Filter by stability
        conditions
            .into_iter()
            .filter(|c| c.stability >= self.min_stability)
            .collect()
    }

    /// Rank postconditions by stability (descending)
    pub fn rank_postconditions(&self, conditions: &mut [Postcondition]) {
        conditions.sort_by(|a, b| b.stability.total_cmp(&a.stability));
    }

    /// Select best postcondition for verification
    pub fn select_best<'a>(&self, conditions: &'a [Postcondition]) -> Option<&'a Postcondition> {
        conditions.iter().max_by(|a, b| a.stability.total_cmp(&b.stability))
    }
}

impl Default for PostconditionExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capture::types::{CursorState, EventType, ModifierFlags, RawEvent, SemanticContext, SemanticSource};
    use crate::time::timebase::Timestamp;

    fn make_test_event(
        timestamp: u64,
        ax_role: Option<&str>,
        window_title: Option<&str>,
        cursor_state: CursorState,
    ) -> EnrichedEvent {
        let raw = RawEvent::mouse(
            Timestamp::from_ticks(timestamp),
            EventType::LeftMouseDown,
            100.0,
            100.0,
            cursor_state,
            ModifierFlags::default(),
            1,
        );

        let mut event = EnrichedEvent::new(raw, 0);

        if ax_role.is_some() || window_title.is_some() {
            let semantic = SemanticContext {
                ax_role: ax_role.map(|s| s.to_string()),
                window_title: window_title.map(|s| s.to_string()),
                source: SemanticSource::Accessibility,
                confidence: 1.0,
                ..Default::default()
            };
            event.semantic = Some(semantic);
        }

        event
    }

    #[test]
    fn test_signal_type_stability() {
        assert!(SignalType::AxRole.stability_rank() > SignalType::CursorShape.stability_rank());
    }

    #[test]
    fn test_extractor_creation() {
        let extractor = PostconditionExtractor::new();
        assert!(extractor.min_stability > 0.0);
    }

    #[test]
    fn test_signal_type_stability_ranking() {
        assert_eq!(SignalType::AxRole.stability_rank(), 4);
        assert_eq!(SignalType::WindowTitle.stability_rank(), 3);
        assert_eq!(SignalType::ElementAppearance.stability_rank(), 2);
        assert_eq!(SignalType::CursorShape.stability_rank(), 1);
    }

    #[test]
    fn test_extract_ax_role_postcondition() {
        let extractor = PostconditionExtractor::new();
        let post_event = make_test_event(1000, Some("AXButton"), None, CursorState::Arrow);

        let conditions = extractor.extract(None, &post_event);

        assert_eq!(conditions.len(), 1);
        assert_eq!(conditions[0].signal_type, SignalType::AxRole);
        assert_eq!(conditions[0].attribute, "AXRole");
        assert_eq!(conditions[0].expected_value, "AXButton");
        assert!(conditions[0].pre_value.is_none());
        assert_eq!(conditions[0].stability, 1.0);
    }

    #[test]
    fn test_extract_window_title_change() {
        let extractor = PostconditionExtractor::new();
        let pre_event = make_test_event(1000, None, Some("Old Title"), CursorState::Arrow);
        let post_event = make_test_event(2000, None, Some("New Title"), CursorState::Arrow);

        let conditions = extractor.extract(Some(&pre_event), &post_event);

        let title_condition = conditions.iter().find(|c| c.signal_type == SignalType::WindowTitle);
        assert!(title_condition.is_some());

        let cond = title_condition.unwrap();
        assert_eq!(cond.attribute, "WindowTitle");
        assert_eq!(cond.expected_value, "New Title");
        assert_eq!(cond.pre_value, Some("Old Title".to_string()));
        assert_eq!(cond.stability, 0.75);
    }

    #[test]
    fn test_extract_cursor_state_change() {
        let mut extractor = PostconditionExtractor::new();
        // Lower stability threshold to allow cursor state changes
        extractor.min_stability = 0.2;

        let pre_event = make_test_event(1000, None, None, CursorState::Arrow);
        let post_event = make_test_event(2000, None, None, CursorState::IBeam);

        let conditions = extractor.extract(Some(&pre_event), &post_event);

        let cursor_condition = conditions.iter().find(|c| c.signal_type == SignalType::CursorShape);
        assert!(cursor_condition.is_some());

        let cond = cursor_condition.unwrap();
        assert_eq!(cond.attribute, "CursorState");
        assert_eq!(cond.expected_value, "IBeam");
        assert!(cond.pre_value.is_some());
        assert_eq!(cond.stability, 0.25);
    }

    #[test]
    fn test_filter_by_stability_threshold() {
        let mut extractor = PostconditionExtractor::new();
        extractor.min_stability = 0.5;

        let pre_event = make_test_event(1000, None, None, CursorState::Arrow);
        let post_event = make_test_event(2000, None, None, CursorState::IBeam);

        let conditions = extractor.extract(Some(&pre_event), &post_event);

        // Cursor shape has stability 0.25, should be filtered out
        assert!(conditions.is_empty());
    }

    #[test]
    fn test_multiple_postconditions() {
        let extractor = PostconditionExtractor::new();
        let pre_event = make_test_event(1000, Some("AXTextField"), Some("Login"), CursorState::Arrow);
        let post_event = make_test_event(2000, Some("AXButton"), Some("Dashboard"), CursorState::PointingHand);

        let conditions = extractor.extract(Some(&pre_event), &post_event);

        // Should have: AXRole, WindowTitle, CursorState
        assert!(conditions.len() >= 2);
        assert!(conditions.iter().any(|c| c.signal_type == SignalType::AxRole));
        assert!(conditions.iter().any(|c| c.signal_type == SignalType::WindowTitle));
    }

    #[test]
    fn test_rank_postconditions() {
        let extractor = PostconditionExtractor::new();

        let mut conditions = vec![
            Postcondition {
                signal_type: SignalType::CursorShape,
                attribute: "CursorState".to_string(),
                expected_value: "IBeam".to_string(),
                pre_value: None,
                stability: 0.25,
            },
            Postcondition {
                signal_type: SignalType::AxRole,
                attribute: "AXRole".to_string(),
                expected_value: "AXButton".to_string(),
                pre_value: None,
                stability: 1.0,
            },
            Postcondition {
                signal_type: SignalType::WindowTitle,
                attribute: "WindowTitle".to_string(),
                expected_value: "New Window".to_string(),
                pre_value: None,
                stability: 0.75,
            },
        ];

        extractor.rank_postconditions(&mut conditions);

        assert_eq!(conditions[0].signal_type, SignalType::AxRole);
        assert_eq!(conditions[1].signal_type, SignalType::WindowTitle);
        assert_eq!(conditions[2].signal_type, SignalType::CursorShape);
    }

    #[test]
    fn test_select_best_postcondition() {
        let extractor = PostconditionExtractor::new();

        let conditions = vec![
            Postcondition {
                signal_type: SignalType::CursorShape,
                attribute: "CursorState".to_string(),
                expected_value: "IBeam".to_string(),
                pre_value: None,
                stability: 0.25,
            },
            Postcondition {
                signal_type: SignalType::AxRole,
                attribute: "AXRole".to_string(),
                expected_value: "AXButton".to_string(),
                pre_value: None,
                stability: 1.0,
            },
        ];

        let best = extractor.select_best(&conditions);
        assert!(best.is_some());
        assert_eq!(best.unwrap().signal_type, SignalType::AxRole);
    }

    #[test]
    fn test_select_best_empty_conditions() {
        let extractor = PostconditionExtractor::new();
        let conditions: Vec<Postcondition> = vec![];

        let best = extractor.select_best(&conditions);
        assert!(best.is_none());
    }

    #[test]
    fn test_no_window_title_change_when_same() {
        let extractor = PostconditionExtractor::new();
        let pre_event = make_test_event(1000, None, Some("Same Title"), CursorState::Arrow);
        let post_event = make_test_event(2000, None, Some("Same Title"), CursorState::Arrow);

        let conditions = extractor.extract(Some(&pre_event), &post_event);

        let title_condition = conditions.iter().find(|c| c.signal_type == SignalType::WindowTitle);
        assert!(title_condition.is_none());
    }

    #[test]
    fn test_default_extractor() {
        let extractor = PostconditionExtractor::default();
        assert_eq!(extractor.min_stability, 0.5);
    }
}
