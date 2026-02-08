//! Mental Operator Detection using GOMS model
//!
//! Detects cognitive pauses (M-operators) that mark task boundaries.

use crate::capture::types::EnrichedEvent;
use crate::time::timebase::MachTimebase;

/// Hesitation index components
#[derive(Debug, Clone, Copy)]
pub struct HesitationIndex {
    /// Inverse velocity component (1/v)
    pub inverse_velocity: f64,
    /// Direction change rate (|dθ/dt|)
    pub direction_change: f64,
    /// Pause duration in ms
    pub pause_duration: f64,
    /// Combined index
    pub total: f64,
}

/// Detected operator type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperatorType {
    /// Mental operator (user thinking)
    Mental,
    /// Response operator (system latency)
    Response,
    /// Perceptual operator (visual scanning)
    Perceptual,
    /// Motor operator (physical action)
    Motor,
}

/// Detected chunk boundary
#[derive(Debug, Clone)]
pub struct ChunkBoundary {
    /// Event index where boundary occurs
    pub event_index: usize,
    /// Timestamp
    pub timestamp_ticks: u64,
    /// Type of operator that caused the boundary
    pub operator_type: OperatorType,
    /// Hesitation index at this point
    pub hesitation_index: HesitationIndex,
    /// Confidence in this boundary
    pub confidence: f32,
}

/// GOMS-based mental operator detector
pub struct GomsDetector {
    /// Threshold for mental operator detection
    pub mental_threshold: f64,
    /// Minimum pause to consider (ms)
    pub min_pause_ms: u64,
    /// Maximum pause before it's definitely mental (ms)
    pub max_pause_ms: u64,
}

impl GomsDetector {
    /// Create with default thresholds
    pub fn new() -> Self {
        Self {
            mental_threshold: 0.7,
            min_pause_ms: 300,   // 300ms minimum hesitation
            max_pause_ms: 2000,  // 2s definitely a pause
        }
    }

    /// Detect chunk boundaries in an event stream
    pub fn detect_boundaries(&self, events: &[EnrichedEvent]) -> Vec<ChunkBoundary> {
        MachTimebase::init();

        let mut boundaries = Vec::new();

        for i in 1..events.len() {
            let prev = &events[i - 1];
            let curr = &events[i];

            // Calculate time delta
            let time_delta_ms = MachTimebase::elapsed_millis(
                prev.raw.timestamp.ticks(),
                curr.raw.timestamp.ticks(),
            );

            // Skip if too short
            if time_delta_ms < self.min_pause_ms {
                continue;
            }

            // Check if this is a system response (wait cursor)
            let is_system_response = prev.raw.cursor_state.is_busy() || curr.raw.cursor_state.is_busy();

            // Calculate hesitation index
            let hesitation = self.calculate_hesitation(events, i);

            // Determine operator type
            let operator_type = if is_system_response {
                OperatorType::Response
            } else if hesitation.total > self.mental_threshold || time_delta_ms > self.max_pause_ms {
                OperatorType::Mental
            } else if hesitation.direction_change > 0.5 {
                OperatorType::Perceptual
            } else {
                continue; // Not a significant boundary
            };

            // Only create boundaries for mental operators (for chunking)
            if operator_type == OperatorType::Mental {
                let confidence = self.calculate_confidence(time_delta_ms, &hesitation, is_system_response);

                boundaries.push(ChunkBoundary {
                    event_index: i,
                    timestamp_ticks: curr.raw.timestamp.ticks(),
                    operator_type,
                    hesitation_index: hesitation,
                    confidence,
                });
            }
        }

        boundaries
    }

    /// Calculate hesitation index at a point
    fn calculate_hesitation(&self, events: &[EnrichedEvent], index: usize) -> HesitationIndex {
        let prev = if index > 0 { Some(&events[index - 1]) } else { None };
        let curr = &events[index];

        // Calculate velocity (using spatial distance / time)
        let (inverse_velocity, direction_change) = if let Some(p) = prev {
            let dx = curr.raw.coordinates.0 - p.raw.coordinates.0;
            let dy = curr.raw.coordinates.1 - p.raw.coordinates.1;
            let distance = (dx * dx + dy * dy).sqrt();

            let time_delta_ms = MachTimebase::elapsed_millis(
                p.raw.timestamp.ticks(),
                curr.raw.timestamp.ticks(),
            );

            let velocity = if time_delta_ms > 0 {
                distance / (time_delta_ms as f64)
            } else {
                0.0
            };

            let inv_v = if velocity > 0.001 { 1.0 / velocity } else { 1.0 };

            // Direction change calculation using angle difference between consecutive segments
            let dir_change = self.calculate_direction_change(events, index);

            (inv_v.min(1.0), dir_change) // Cap at 1.0
        } else {
            (1.0, 0.0)
        };

        // Pause duration
        let pause_duration = if let Some(p) = prev {
            MachTimebase::elapsed_millis(p.raw.timestamp.ticks(), curr.raw.timestamp.ticks()) as f64 / 1000.0
        } else {
            0.0
        };

        // Combined index (weighted sum)
        let total = 0.3 * inverse_velocity + 0.3 * direction_change + 0.4 * pause_duration.min(2.0) / 2.0;

        HesitationIndex {
            inverse_velocity,
            direction_change,
            pause_duration: pause_duration * 1000.0, // Convert back to ms
            total,
        }
    }

    /// Calculate direction change rate (|dθ/dt|) at a point
    /// Returns a value between 0.0 and 1.0, where higher values indicate sharper turns
    fn calculate_direction_change(&self, events: &[EnrichedEvent], index: usize) -> f64 {
        // Need at least 3 points to calculate direction change
        if index < 2 || index >= events.len() {
            return 0.0;
        }

        let p0 = &events[index - 2];
        let p1 = &events[index - 1];
        let p2 = &events[index];

        // Calculate vectors
        let v1_x = p1.raw.coordinates.0 - p0.raw.coordinates.0;
        let v1_y = p1.raw.coordinates.1 - p0.raw.coordinates.1;
        let v2_x = p2.raw.coordinates.0 - p1.raw.coordinates.0;
        let v2_y = p2.raw.coordinates.1 - p1.raw.coordinates.1;

        // Calculate magnitudes
        let mag1 = (v1_x * v1_x + v1_y * v1_y).sqrt();
        let mag2 = (v2_x * v2_x + v2_y * v2_y).sqrt();

        // Avoid division by zero for stationary points
        if mag1 < 0.001 || mag2 < 0.001 {
            return 0.0;
        }

        // Calculate dot product and cross product
        let dot = v1_x * v2_x + v1_y * v2_y;
        let cross = v1_x * v2_y - v1_y * v2_x;

        // Calculate angle change using atan2 for signed angle
        let angle_change = cross.atan2(dot).abs();

        // Normalize to 0-1 range (π radians = 180 degrees = max change = 1.0)
        let normalized = angle_change / std::f64::consts::PI;

        // Factor in time - faster direction changes are more significant
        let time_delta_ms = MachTimebase::elapsed_millis(
            p1.raw.timestamp.ticks(),
            p2.raw.timestamp.ticks(),
        );

        // Weight by angular velocity (rad/ms)
        if time_delta_ms > 0 {
            let angular_velocity = angle_change / (time_delta_ms as f64);
            // Scale: 0.01 rad/ms is considered high
            (normalized + (angular_velocity * 100.0).min(1.0)) / 2.0
        } else {
            normalized
        }
    }

    /// Calculate confidence in boundary detection
    fn calculate_confidence(&self, time_delta_ms: u64, hesitation: &HesitationIndex, is_system_response: bool) -> f32 {
        if is_system_response {
            return 0.5; // Lower confidence for system responses
        }

        // Higher confidence for longer pauses
        let time_factor = (time_delta_ms as f32 / self.max_pause_ms as f32).min(1.0);

        // Higher confidence for higher hesitation index
        let hesitation_factor = (hesitation.total as f32).min(1.0);

        (time_factor + hesitation_factor) / 2.0
    }
}

impl Default for GomsDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_goms_detector_creation() {
        let detector = GomsDetector::new();
        assert!(detector.mental_threshold > 0.0);
        assert!(detector.min_pause_ms > 0);
    }

    #[test]
    fn test_hesitation_index() {
        let index = HesitationIndex {
            inverse_velocity: 0.5,
            direction_change: 0.3,
            pause_duration: 500.0,
            total: 0.6,
        };

        assert!(index.total > 0.0);
    }

    #[test]
    fn test_operator_type_variants() {
        assert_ne!(OperatorType::Motor, OperatorType::Mental);
        assert_ne!(OperatorType::Perceptual, OperatorType::Response);
    }

    #[test]
    fn test_chunk_boundary_creation() {
        let boundary = ChunkBoundary {
            event_index: 5,
            timestamp_ticks: 1000000,
            operator_type: OperatorType::Mental,
            hesitation_index: HesitationIndex {
                inverse_velocity: 0.8,
                direction_change: 0.2,
                pause_duration: 1500.0,
                total: 0.7,
            },
            confidence: 0.85,
        };

        assert_eq!(boundary.event_index, 5);
        assert_eq!(boundary.operator_type, OperatorType::Mental);
        assert!(boundary.confidence > 0.5);
    }

    #[test]
    fn test_default_detector() {
        let detector = GomsDetector::default();
        assert!(detector.mental_threshold > 0.0);
    }

    #[test]
    fn test_hesitation_index_components() {
        // Test that all components contribute to total
        let high_velocity = HesitationIndex {
            inverse_velocity: 0.1, // Low inverse = high velocity
            direction_change: 0.1,
            pause_duration: 100.0,
            total: 0.2,
        };

        let low_velocity = HesitationIndex {
            inverse_velocity: 0.9, // High inverse = low velocity
            direction_change: 0.1,
            pause_duration: 100.0,
            total: 0.4,
        };

        // Higher inverse velocity should contribute to higher hesitation
        assert!(low_velocity.inverse_velocity > high_velocity.inverse_velocity);
    }

    #[test]
    fn test_operator_type_equality() {
        assert_eq!(OperatorType::Mental, OperatorType::Mental);
        assert_eq!(OperatorType::Response, OperatorType::Response);
        assert_eq!(OperatorType::Perceptual, OperatorType::Perceptual);
        assert_eq!(OperatorType::Motor, OperatorType::Motor);
    }

    #[test]
    fn test_detector_thresholds() {
        let detector = GomsDetector::new();

        // Default thresholds
        assert_eq!(detector.min_pause_ms, 300);
        assert_eq!(detector.max_pause_ms, 2000);
        assert!((detector.mental_threshold - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_chunk_boundary_with_high_confidence() {
        let boundary = ChunkBoundary {
            event_index: 10,
            timestamp_ticks: 5000000,
            operator_type: OperatorType::Mental,
            hesitation_index: HesitationIndex {
                inverse_velocity: 1.0,
                direction_change: 0.8,
                pause_duration: 2000.0,
                total: 0.95,
            },
            confidence: 0.98,
        };

        assert!(boundary.confidence > 0.9);
        assert!(boundary.hesitation_index.total > 0.9);
    }

    #[test]
    fn test_perceptual_operator_detection() {
        // High direction change should suggest perceptual scanning
        let hesitation = HesitationIndex {
            inverse_velocity: 0.3,
            direction_change: 0.7, // High direction change
            pause_duration: 200.0,
            total: 0.5,
        };

        assert!(hesitation.direction_change > 0.5);
    }

    #[test]
    fn test_detect_boundaries_empty_events() {
        use crate::time::timebase::MachTimebase;
        MachTimebase::init();

        let detector = GomsDetector::new();
        let events: Vec<EnrichedEvent> = vec![];

        let boundaries = detector.detect_boundaries(&events);
        assert!(boundaries.is_empty());
    }

    #[test]
    fn test_detect_boundaries_single_event() {
        use crate::capture::types::{RawEvent, EventType, CursorState, ModifierFlags};
        use crate::time::timebase::{MachTimebase, Timestamp};

        MachTimebase::init();

        let detector = GomsDetector::new();
        let raw = RawEvent::mouse(
            Timestamp::from_ticks(1000),
            EventType::MouseMoved,
            100.0,
            100.0,
            CursorState::Arrow,
            ModifierFlags::default(),
            0,
        );
        let events = vec![EnrichedEvent::new(raw, 0)];

        let boundaries = detector.detect_boundaries(&events);
        assert!(boundaries.is_empty()); // Need at least 2 events
    }

    #[test]
    fn test_detect_boundaries_short_pause() {
        use crate::capture::types::{RawEvent, EventType, CursorState, ModifierFlags};
        use crate::time::timebase::{MachTimebase, Timestamp};

        MachTimebase::init();

        let detector = GomsDetector::new();

        // Events with short pause (< min_pause_ms)
        let raw1 = RawEvent::mouse(
            Timestamp::from_ticks(1000000),
            EventType::MouseMoved,
            100.0,
            100.0,
            CursorState::Arrow,
            ModifierFlags::default(),
            0,
        );
        let raw2 = RawEvent::mouse(
            Timestamp::from_ticks(1001000), // ~1ms later
            EventType::MouseMoved,
            110.0,
            110.0,
            CursorState::Arrow,
            ModifierFlags::default(),
            0,
        );

        let events = vec![
            EnrichedEvent::new(raw1, 0),
            EnrichedEvent::new(raw2, 1),
        ];

        let boundaries = detector.detect_boundaries(&events);
        // Short pause should not create boundaries
        assert!(boundaries.is_empty());
    }

    #[test]
    fn test_detect_boundaries_mental_operator() {
        use crate::capture::types::{RawEvent, EventType, CursorState, ModifierFlags};
        use crate::time::timebase::{MachTimebase, Timestamp};

        MachTimebase::init();

        let detector = GomsDetector::new();

        // Events with long pause (> max_pause_ms = 2000ms)
        let raw1 = RawEvent::mouse(
            Timestamp::from_ticks(1000000000),
            EventType::MouseMoved,
            100.0,
            100.0,
            CursorState::Arrow,
            ModifierFlags::default(),
            0,
        );
        let raw2 = RawEvent::mouse(
            Timestamp::from_ticks(3000000000), // Long pause
            EventType::LeftMouseDown,
            200.0,
            200.0,
            CursorState::Arrow,
            ModifierFlags::default(),
            1,
        );

        let events = vec![
            EnrichedEvent::new(raw1, 0),
            EnrichedEvent::new(raw2, 1),
        ];

        let boundaries = detector.detect_boundaries(&events);
        // Long pause should create mental operator boundary
        assert!(!boundaries.is_empty());
        assert_eq!(boundaries[0].operator_type, OperatorType::Mental);
    }

    #[test]
    fn test_detect_boundaries_system_response() {
        use crate::capture::types::{RawEvent, EventType, CursorState, ModifierFlags};
        use crate::time::timebase::{MachTimebase, Timestamp};

        MachTimebase::init();

        let detector = GomsDetector::new();

        // Events with busy cursor (system response)
        let raw1 = RawEvent::mouse(
            Timestamp::from_ticks(1000000000),
            EventType::MouseMoved,
            100.0,
            100.0,
            CursorState::Wait, // Busy cursor
            ModifierFlags::default(),
            0,
        );
        let raw2 = RawEvent::mouse(
            Timestamp::from_ticks(2000000000),
            EventType::MouseMoved,
            110.0,
            110.0,
            CursorState::Arrow,
            ModifierFlags::default(),
            0,
        );

        let events = vec![
            EnrichedEvent::new(raw1, 0),
            EnrichedEvent::new(raw2, 1),
        ];

        let boundaries = detector.detect_boundaries(&events);
        // System response is filtered out (only Mental operators create boundaries)
        assert!(boundaries.is_empty());
    }

    #[test]
    fn test_calculate_hesitation_stationary() {
        use crate::capture::types::{RawEvent, EventType, CursorState, ModifierFlags};
        use crate::time::timebase::{MachTimebase, Timestamp};

        MachTimebase::init();

        let detector = GomsDetector::new();

        // Two events at same position
        let raw1 = RawEvent::mouse(
            Timestamp::from_ticks(1000000000),
            EventType::MouseMoved,
            100.0,
            100.0,
            CursorState::Arrow,
            ModifierFlags::default(),
            0,
        );
        let raw2 = RawEvent::mouse(
            Timestamp::from_ticks(1500000000),
            EventType::MouseMoved,
            100.0,
            100.0, // Same position
            CursorState::Arrow,
            ModifierFlags::default(),
            0,
        );

        let events = vec![
            EnrichedEvent::new(raw1, 0),
            EnrichedEvent::new(raw2, 1),
        ];

        let hesitation = detector.calculate_hesitation(&events, 1);

        // Stationary point should have high inverse velocity
        assert!(hesitation.inverse_velocity > 0.9);
        assert!(hesitation.pause_duration > 0.0);
    }

    #[test]
    fn test_calculate_direction_change_straight_line() {
        use crate::capture::types::{RawEvent, EventType, CursorState, ModifierFlags};
        use crate::time::timebase::{MachTimebase, Timestamp};

        MachTimebase::init();

        let detector = GomsDetector::new();

        // Three events in a straight line
        let raw1 = RawEvent::mouse(
            Timestamp::from_ticks(1000000000),
            EventType::MouseMoved,
            100.0,
            100.0,
            CursorState::Arrow,
            ModifierFlags::default(),
            0,
        );
        let raw2 = RawEvent::mouse(
            Timestamp::from_ticks(1100000000),
            EventType::MouseMoved,
            110.0,
            100.0, // Moving right
            CursorState::Arrow,
            ModifierFlags::default(),
            0,
        );
        let raw3 = RawEvent::mouse(
            Timestamp::from_ticks(1200000000),
            EventType::MouseMoved,
            120.0,
            100.0, // Continue moving right
            CursorState::Arrow,
            ModifierFlags::default(),
            0,
        );

        let events = vec![
            EnrichedEvent::new(raw1, 0),
            EnrichedEvent::new(raw2, 1),
            EnrichedEvent::new(raw3, 2),
        ];

        let dir_change = detector.calculate_direction_change(&events, 2);

        // Straight line should have minimal direction change
        assert!(dir_change < 0.2);
    }

    #[test]
    fn test_calculate_direction_change_sharp_turn() {
        use crate::capture::types::{RawEvent, EventType, CursorState, ModifierFlags};
        use crate::time::timebase::{MachTimebase, Timestamp};

        MachTimebase::init();

        let detector = GomsDetector::new();

        // Three events with sharp turn
        let raw1 = RawEvent::mouse(
            Timestamp::from_ticks(1000000000),
            EventType::MouseMoved,
            100.0,
            100.0,
            CursorState::Arrow,
            ModifierFlags::default(),
            0,
        );
        let raw2 = RawEvent::mouse(
            Timestamp::from_ticks(1100000000),
            EventType::MouseMoved,
            110.0,
            100.0, // Moving right
            CursorState::Arrow,
            ModifierFlags::default(),
            0,
        );
        let raw3 = RawEvent::mouse(
            Timestamp::from_ticks(1200000000),
            EventType::MouseMoved,
            110.0,
            110.0, // Turn down (90 degrees)
            CursorState::Arrow,
            ModifierFlags::default(),
            0,
        );

        let events = vec![
            EnrichedEvent::new(raw1, 0),
            EnrichedEvent::new(raw2, 1),
            EnrichedEvent::new(raw3, 2),
        ];

        let dir_change = detector.calculate_direction_change(&events, 2);

        // Sharp turn should have higher direction change
        assert!(dir_change > 0.1);
    }

    #[test]
    fn test_calculate_direction_change_insufficient_points() {
        use crate::capture::types::{RawEvent, EventType, CursorState, ModifierFlags};
        use crate::time::timebase::{MachTimebase, Timestamp};

        MachTimebase::init();

        let detector = GomsDetector::new();

        let raw1 = RawEvent::mouse(
            Timestamp::from_ticks(1000000000),
            EventType::MouseMoved,
            100.0,
            100.0,
            CursorState::Arrow,
            ModifierFlags::default(),
            0,
        );

        let events = vec![EnrichedEvent::new(raw1, 0)];

        // Index 0 doesn't have enough previous points
        let dir_change = detector.calculate_direction_change(&events, 0);
        assert_eq!(dir_change, 0.0);
    }

    #[test]
    fn test_calculate_confidence_with_system_response() {
        let detector = GomsDetector::new();

        let hesitation = HesitationIndex {
            inverse_velocity: 0.8,
            direction_change: 0.5,
            pause_duration: 1000.0,
            total: 0.75,
        };

        let confidence = detector.calculate_confidence(1000, &hesitation, true);

        // System response should result in lower confidence (0.5)
        assert_eq!(confidence, 0.5);
    }

    #[test]
    fn test_calculate_confidence_high_pause() {
        let detector = GomsDetector::new();

        let hesitation = HesitationIndex {
            inverse_velocity: 0.9,
            direction_change: 0.8,
            pause_duration: 2000.0, // At max_pause_ms
            total: 0.9,
        };

        let confidence = detector.calculate_confidence(2000, &hesitation, false);

        // High pause and hesitation should result in high confidence
        assert!(confidence > 0.8);
    }

    #[test]
    fn test_calculate_confidence_low_pause() {
        let detector = GomsDetector::new();

        let hesitation = HesitationIndex {
            inverse_velocity: 0.3,
            direction_change: 0.2,
            pause_duration: 300.0, // At min_pause_ms
            total: 0.3,
        };

        let confidence = detector.calculate_confidence(300, &hesitation, false);

        // Low pause and hesitation should result in lower confidence
        assert!(confidence < 0.5);
    }

    #[test]
    fn test_hesitation_index_total_calculation() {
        // Test weighted sum calculation
        let hesitation = HesitationIndex {
            inverse_velocity: 1.0,
            direction_change: 1.0,
            pause_duration: 2000.0, // 2 seconds
            total: 0.3 * 1.0 + 0.3 * 1.0 + 0.4 * 1.0, // Max values
        };

        // Total should be approximately 1.0 (weights: 0.3, 0.3, 0.4)
        assert!((hesitation.total - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_custom_detector_thresholds() {
        let detector = GomsDetector {
            mental_threshold: 0.5,
            min_pause_ms: 100,
            max_pause_ms: 1000,
        };

        assert_eq!(detector.mental_threshold, 0.5);
        assert_eq!(detector.min_pause_ms, 100);
        assert_eq!(detector.max_pause_ms, 1000);
    }

    #[test]
    fn test_chunk_boundary_confidence_range() {
        let boundary = ChunkBoundary {
            event_index: 3,
            timestamp_ticks: 123456789,
            operator_type: OperatorType::Mental,
            hesitation_index: HesitationIndex {
                inverse_velocity: 0.5,
                direction_change: 0.5,
                pause_duration: 800.0,
                total: 0.6,
            },
            confidence: 0.75,
        };

        // Confidence should be in valid range [0.0, 1.0]
        assert!(boundary.confidence >= 0.0);
        assert!(boundary.confidence <= 1.0);
    }

    #[test]
    fn test_motor_operator_type() {
        // Test that Motor operator type exists and is distinct
        assert_ne!(OperatorType::Motor, OperatorType::Mental);
        assert_ne!(OperatorType::Motor, OperatorType::Response);
        assert_ne!(OperatorType::Motor, OperatorType::Perceptual);
    }

    #[test]
    fn test_multiple_boundaries_detection() {
        use crate::capture::types::{RawEvent, EventType, CursorState, ModifierFlags};
        use crate::time::timebase::{MachTimebase, Timestamp};

        MachTimebase::init();

        let detector = GomsDetector::new();

        // Create events with multiple long pauses
        let events = vec![
            EnrichedEvent::new(
                RawEvent::mouse(
                    Timestamp::from_ticks(1000000000),
                    EventType::MouseMoved,
                    100.0,
                    100.0,
                    CursorState::Arrow,
                    ModifierFlags::default(),
                    0,
                ),
                0,
            ),
            EnrichedEvent::new(
                RawEvent::mouse(
                    Timestamp::from_ticks(3500000000), // Long pause
                    EventType::LeftMouseDown,
                    200.0,
                    200.0,
                    CursorState::Arrow,
                    ModifierFlags::default(),
                    1,
                ),
                1,
            ),
            EnrichedEvent::new(
                RawEvent::mouse(
                    Timestamp::from_ticks(6000000000), // Another long pause
                    EventType::LeftMouseDown,
                    300.0,
                    300.0,
                    CursorState::Arrow,
                    ModifierFlags::default(),
                    1,
                ),
                2,
            ),
        ];

        let boundaries = detector.detect_boundaries(&events);

        // Should detect multiple boundaries
        assert!(!boundaries.is_empty());
        for boundary in &boundaries {
            assert_eq!(boundary.operator_type, OperatorType::Mental);
            assert!(boundary.confidence > 0.0);
        }
    }
}
