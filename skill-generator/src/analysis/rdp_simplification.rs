//! Ramer-Douglas-Peucker Polyline Simplification
//!
//! Simplifies mouse trajectories by removing points that don't significantly
//! contribute to the overall shape. This reduces noise while preserving
//! the intentional movement pattern.

use crate::capture::types::RawEvent;

/// Default epsilon for RDP simplification (pixels)
pub const DEFAULT_EPSILON: f64 = 3.0;

/// Point in 2D space with timestamp
#[derive(Debug, Clone, Copy)]
pub struct TrajectoryPoint {
    pub x: f64,
    pub y: f64,
    pub timestamp_ticks: u64,
}

impl TrajectoryPoint {
    pub fn new(x: f64, y: f64, timestamp_ticks: u64) -> Self {
        Self { x, y, timestamp_ticks }
    }

    pub fn from_event(event: &RawEvent) -> Self {
        Self {
            x: event.coordinates.0,
            y: event.coordinates.1,
            timestamp_ticks: event.timestamp.ticks(),
        }
    }

    /// Calculate Euclidean distance to another point
    pub fn distance_to(&self, other: &TrajectoryPoint) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }

    /// Calculate perpendicular distance to a line segment
    pub fn perpendicular_distance(&self, line_start: &TrajectoryPoint, line_end: &TrajectoryPoint) -> f64 {
        let dx = line_end.x - line_start.x;
        let dy = line_end.y - line_start.y;

        let line_length_sq = dx * dx + dy * dy;

        if line_length_sq < 1e-10 {
            // Line segment is actually a point
            return self.distance_to(line_start);
        }

        // Calculate perpendicular distance using cross product
        let numerator = ((self.x - line_start.x) * dy - (self.y - line_start.y) * dx).abs();
        numerator / line_length_sq.sqrt()
    }
}

/// RDP polyline simplification algorithm
pub struct RdpSimplifier {
    /// Epsilon threshold (pixels)
    pub epsilon: f64,
}

impl RdpSimplifier {
    /// Create a new simplifier with default epsilon
    pub fn new() -> Self {
        Self {
            epsilon: DEFAULT_EPSILON,
        }
    }

    /// Create a new simplifier with custom epsilon.
    ///
    /// Clamps epsilon to the range \[0.001, 10000.0\] to prevent degenerate behavior.
    pub fn with_epsilon(epsilon: f64) -> Self {
        Self {
            epsilon: epsilon.clamp(0.001, 10_000.0),
        }
    }

    /// Simplify a trajectory of mouse events
    pub fn simplify_events(&self, events: &[RawEvent]) -> Vec<TrajectoryPoint> {
        let points: Vec<TrajectoryPoint> = events
            .iter()
            .filter(|e| e.event_type.is_mouse_move())
            .map(TrajectoryPoint::from_event)
            .collect();

        self.simplify(&points)
    }

    /// Simplify a trajectory of points
    pub fn simplify(&self, points: &[TrajectoryPoint]) -> Vec<TrajectoryPoint> {
        if points.len() <= 2 {
            return points.to_vec();
        }

        // Find the point with maximum distance from the line between start and end
        let (max_dist, max_index) = self.find_max_distance(points);

        if max_dist > self.epsilon {
            // Recursively simplify both halves
            let mut left = self.simplify(&points[..=max_index]);
            let right = self.simplify(&points[max_index..]);

            // Remove the duplicate point at the junction
            left.pop();
            left.extend(right);
            left
        } else {
            // All points are within epsilon of the line, keep only endpoints
            // Safety: points.len() > 2 is guaranteed by the early return above
            match (points.first(), points.last()) {
                (Some(&first), Some(&last)) => vec![first, last],
                _ => points.to_vec(),
            }
        }
    }

    /// Find the point with maximum perpendicular distance from the line
    fn find_max_distance(&self, points: &[TrajectoryPoint]) -> (f64, usize) {
        let start = &points[0];
        let end = match points.last() {
            Some(p) => p,
            None => return (0.0, 0),
        };

        let mut max_dist = 0.0;
        let mut max_index = 0;

        for (i, point) in points.iter().enumerate().skip(1).take(points.len() - 2) {
            let dist = point.perpendicular_distance(start, end);
            if dist > max_dist {
                max_dist = dist;
                max_index = i;
            }
        }

        (max_dist, max_index)
    }

    /// Calculate the total path length of simplified trajectory
    pub fn path_length(points: &[TrajectoryPoint]) -> f64 {
        points
            .windows(2)
            .map(|w| w[0].distance_to(&w[1]))
            .sum()
    }

    /// Calculate compression ratio
    pub fn compression_ratio(&self, original: &[TrajectoryPoint], simplified: &[TrajectoryPoint]) -> f64 {
        if original.is_empty() {
            return 1.0;
        }
        1.0 - (simplified.len() as f64 / original.len() as f64)
    }
}

impl Default for RdpSimplifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Simplified trajectory with metadata
#[derive(Debug, Clone)]
pub struct SimplifiedTrajectory {
    /// Simplified points
    pub points: Vec<TrajectoryPoint>,
    /// Original point count
    pub original_count: usize,
    /// Total path length (pixels)
    pub path_length: f64,
    /// Compression ratio (0-1)
    pub compression_ratio: f64,
    /// Start timestamp
    pub start_time: u64,
    /// End timestamp
    pub end_time: u64,
}

impl SimplifiedTrajectory {
    /// Create from raw events
    pub fn from_events(events: &[RawEvent], epsilon: f64) -> Self {
        let simplifier = RdpSimplifier::with_epsilon(epsilon);
        let points: Vec<TrajectoryPoint> = events
            .iter()
            .filter(|e| e.event_type.is_mouse_move())
            .map(TrajectoryPoint::from_event)
            .collect();

        let simplified = simplifier.simplify(&points);
        let path_length = RdpSimplifier::path_length(&simplified);
        let compression = simplifier.compression_ratio(&points, &simplified);

        Self {
            start_time: simplified.first().map(|p| p.timestamp_ticks).unwrap_or(0),
            end_time: simplified.last().map(|p| p.timestamp_ticks).unwrap_or(0),
            original_count: points.len(),
            path_length,
            compression_ratio: compression,
            points: simplified,
        }
    }

    /// Get trajectory direction vector (start to end)
    pub fn direction_vector(&self) -> Option<(f64, f64)> {
        if self.points.len() < 2 {
            return None;
        }

        let start = &self.points[0];
        let end = self.points.last()?;

        let dx = end.x - start.x;
        let dy = end.y - start.y;
        let length = (dx * dx + dy * dy).sqrt();

        if length < 1e-10 {
            return None;
        }

        Some((dx / length, dy / length))
    }

    /// Get start position
    pub fn start_position(&self) -> Option<(f64, f64)> {
        self.points.first().map(|p| (p.x, p.y))
    }

    /// Get end position
    pub fn end_position(&self) -> Option<(f64, f64)> {
        self.points.last().map(|p| (p.x, p.y))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_point(x: f64, y: f64) -> TrajectoryPoint {
        TrajectoryPoint::new(x, y, 0)
    }

    #[test]
    fn test_perpendicular_distance() {
        let point = make_point(1.0, 1.0);
        let start = make_point(0.0, 0.0);
        let end = make_point(2.0, 0.0);

        let dist = point.perpendicular_distance(&start, &end);
        assert!((dist - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_simplify_straight_line() {
        let simplifier = RdpSimplifier::with_epsilon(1.0);

        // Points along a straight line
        let points: Vec<TrajectoryPoint> = (0..10)
            .map(|i| make_point(i as f64, 0.0))
            .collect();

        let simplified = simplifier.simplify(&points);

        // Should reduce to just start and end
        assert_eq!(simplified.len(), 2);
        assert!((simplified[0].x - 0.0).abs() < 0.001);
        assert!((simplified[1].x - 9.0).abs() < 0.001);
    }

    #[test]
    fn test_simplify_preserves_corners() {
        let simplifier = RdpSimplifier::with_epsilon(1.0);

        // L-shaped path
        let points = vec![
            make_point(0.0, 0.0),
            make_point(1.0, 0.0),
            make_point(2.0, 0.0),
            make_point(3.0, 0.0),
            make_point(4.0, 0.0),
            make_point(5.0, 0.0), // Corner
            make_point(5.0, 1.0),
            make_point(5.0, 2.0),
            make_point(5.0, 3.0),
            make_point(5.0, 4.0),
            make_point(5.0, 5.0),
        ];

        let simplified = simplifier.simplify(&points);

        // Should keep start, corner, and end
        assert_eq!(simplified.len(), 3);
        assert!((simplified[0].x - 0.0).abs() < 0.001);
        assert!((simplified[1].x - 5.0).abs() < 0.001);
        assert!((simplified[1].y - 0.0).abs() < 0.001);
        assert!((simplified[2].y - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_simplify_circle() {
        let simplifier = RdpSimplifier::with_epsilon(2.0);

        // Generate points along a circle
        let points: Vec<TrajectoryPoint> = (0..200)
            .map(|i| {
                let angle = (i as f64 / 200.0) * 2.0 * std::f64::consts::PI;
                make_point(
                    50.0 + 30.0 * angle.cos(),
                    50.0 + 30.0 * angle.sin(),
                )
            })
            .collect();

        let simplified = simplifier.simplify(&points);

        // Should significantly reduce points but keep circular shape
        assert!(simplified.len() < 50);
        assert!(simplified.len() > 3);

        // Verify compression
        let ratio = simplifier.compression_ratio(&points, &simplified);
        assert!(ratio > 0.7); // At least 70% compression
    }

    #[test]
    fn test_path_length() {
        let points = vec![
            make_point(0.0, 0.0),
            make_point(3.0, 0.0),
            make_point(3.0, 4.0),
        ];

        let length = RdpSimplifier::path_length(&points);
        assert!((length - 7.0).abs() < 0.001); // 3 + 4 = 7
    }

    #[test]
    fn test_empty_input() {
        let simplifier = RdpSimplifier::new();
        let points: Vec<TrajectoryPoint> = vec![];
        let simplified = simplifier.simplify(&points);
        assert!(simplified.is_empty());
    }

    #[test]
    fn test_single_point() {
        let simplifier = RdpSimplifier::new();
        let points = vec![make_point(1.0, 2.0)];
        let simplified = simplifier.simplify(&points);
        assert_eq!(simplified.len(), 1);
    }

    #[test]
    fn test_two_points() {
        let simplifier = RdpSimplifier::new();
        let points = vec![make_point(0.0, 0.0), make_point(10.0, 10.0)];
        let simplified = simplifier.simplify(&points);
        assert_eq!(simplified.len(), 2);
        assert!((simplified[0].x - 0.0).abs() < 0.001);
        assert!((simplified[1].x - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_perpendicular_distance_zero_length_line() {
        // Test when line segment is actually a point
        let point = make_point(5.0, 5.0);
        let start = make_point(3.0, 3.0);
        let end = make_point(3.0, 3.0); // Same as start

        let dist = point.perpendicular_distance(&start, &end);
        // Should return direct distance since line length is zero
        let expected = ((5.0_f64 - 3.0).powi(2) + (5.0_f64 - 3.0).powi(2)).sqrt();
        assert!((dist - expected).abs() < 0.001);
    }

    #[test]
    fn test_perpendicular_distance_on_line() {
        // Point that lies exactly on the line
        let point = make_point(1.0, 0.0);
        let start = make_point(0.0, 0.0);
        let end = make_point(2.0, 0.0);

        let dist = point.perpendicular_distance(&start, &end);
        assert!(dist < 0.001); // Should be approximately zero
    }

    #[test]
    fn test_distance_to() {
        let p1 = make_point(0.0, 0.0);
        let p2 = make_point(3.0, 4.0);

        let dist = p1.distance_to(&p2);
        assert!((dist - 5.0).abs() < 0.001); // 3-4-5 triangle
    }

    #[test]
    fn test_compression_ratio() {
        let simplifier = RdpSimplifier::with_epsilon(2.0);
        let original = vec![
            make_point(0.0, 0.0),
            make_point(1.0, 0.0),
            make_point(2.0, 0.0),
            make_point(3.0, 0.0),
            make_point(4.0, 0.0),
        ];
        let simplified = simplifier.simplify(&original);

        let ratio = simplifier.compression_ratio(&original, &simplified);
        // 5 points reduced to 2 = 3/5 = 0.6 compression
        assert!((ratio - 0.6).abs() < 0.001);
    }

    #[test]
    fn test_compression_ratio_empty() {
        let simplifier = RdpSimplifier::new();
        let original: Vec<TrajectoryPoint> = vec![];
        let simplified: Vec<TrajectoryPoint> = vec![];

        let ratio = simplifier.compression_ratio(&original, &simplified);
        assert!((ratio - 1.0).abs() < 0.001); // Should return 1.0 for empty
    }

    #[test]
    fn test_simplify_zigzag() {
        let simplifier = RdpSimplifier::with_epsilon(0.5);

        // Zigzag pattern with larger amplitude
        let points = vec![
            make_point(0.0, 0.0),
            make_point(1.0, 3.0),
            make_point(2.0, 0.0),
            make_point(3.0, 3.0),
            make_point(4.0, 0.0),
        ];

        let simplified = simplifier.simplify(&points);
        // Should keep more points due to zigzag pattern (at least 3)
        assert!(simplified.len() >= 3, "Expected at least 3 points, got {}", simplified.len());
    }

    #[test]
    fn test_path_length_empty() {
        let points: Vec<TrajectoryPoint> = vec![];
        let length = RdpSimplifier::path_length(&points);
        assert_eq!(length, 0.0);
    }

    #[test]
    fn test_path_length_single_point() {
        let points = vec![make_point(5.0, 5.0)];
        let length = RdpSimplifier::path_length(&points);
        assert_eq!(length, 0.0);
    }

    #[test]
    fn test_simplified_trajectory_direction_vector() {
        use crate::capture::types::{RawEvent, EventType, CursorState, ModifierFlags};
        use crate::time::timebase::Timestamp;

        let events = vec![
            RawEvent::mouse(
                Timestamp::from_ticks(0),
                EventType::MouseMoved,
                0.0,
                0.0,
                CursorState::Arrow,
                ModifierFlags::default(),
                0,
            ),
            RawEvent::mouse(
                Timestamp::from_ticks(1000),
                EventType::MouseMoved,
                3.0,
                4.0,
                CursorState::Arrow,
                ModifierFlags::default(),
                0,
            ),
        ];

        let trajectory = SimplifiedTrajectory::from_events(&events, 1.0);
        let dir = trajectory.direction_vector();

        assert!(dir.is_some());
        if let Some((dx, dy)) = dir {
            let length = (dx * dx + dy * dy).sqrt();
            assert!((length - 1.0).abs() < 0.001); // Should be normalized
        }
    }

    #[test]
    fn test_simplified_trajectory_positions() {
        use crate::capture::types::{RawEvent, EventType, CursorState, ModifierFlags};
        use crate::time::timebase::Timestamp;

        let events = vec![
            RawEvent::mouse(
                Timestamp::from_ticks(0),
                EventType::MouseMoved,
                10.0,
                20.0,
                CursorState::Arrow,
                ModifierFlags::default(),
                0,
            ),
            RawEvent::mouse(
                Timestamp::from_ticks(1000),
                EventType::MouseMoved,
                30.0,
                40.0,
                CursorState::Arrow,
                ModifierFlags::default(),
                0,
            ),
        ];

        let trajectory = SimplifiedTrajectory::from_events(&events, 1.0);

        assert_eq!(trajectory.start_position(), Some((10.0, 20.0)));
        assert_eq!(trajectory.end_position(), Some((30.0, 40.0)));
    }

    #[test]
    fn test_simplified_trajectory_empty_direction() {
        use crate::capture::types::{RawEvent, EventType, CursorState, ModifierFlags};
        use crate::time::timebase::Timestamp;

        let events = vec![
            RawEvent::mouse(
                Timestamp::from_ticks(0),
                EventType::MouseMoved,
                5.0,
                5.0,
                CursorState::Arrow,
                ModifierFlags::default(),
                0,
            ),
        ];

        let trajectory = SimplifiedTrajectory::from_events(&events, 1.0);
        assert!(trajectory.direction_vector().is_none());
    }

    #[test]
    fn test_with_epsilon_clamps_negative() {
        let simplifier = RdpSimplifier::with_epsilon(-5.0);
        assert!((simplifier.epsilon - 0.001).abs() < 1e-6);
    }

    #[test]
    fn test_with_epsilon_clamps_zero() {
        let simplifier = RdpSimplifier::with_epsilon(0.0);
        assert!((simplifier.epsilon - 0.001).abs() < 1e-6);
    }

    #[test]
    fn test_with_epsilon_clamps_too_large() {
        let simplifier = RdpSimplifier::with_epsilon(99_999.0);
        assert!((simplifier.epsilon - 10_000.0).abs() < 1e-6);
    }

    #[test]
    fn test_with_epsilon_valid_value_unchanged() {
        let simplifier = RdpSimplifier::with_epsilon(5.0);
        assert!((simplifier.epsilon - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_with_epsilon_boundary_values() {
        let min = RdpSimplifier::with_epsilon(0.001);
        assert!((min.epsilon - 0.001).abs() < 1e-6);

        let max = RdpSimplifier::with_epsilon(10_000.0);
        assert!((max.epsilon - 10_000.0).abs() < 1e-6);
    }

    #[test]
    fn test_simplified_trajectory_stationary() {
        use crate::capture::types::{RawEvent, EventType, CursorState, ModifierFlags};
        use crate::time::timebase::Timestamp;

        // Multiple events at same position
        let events = vec![
            RawEvent::mouse(
                Timestamp::from_ticks(0),
                EventType::MouseMoved,
                100.0,
                100.0,
                CursorState::Arrow,
                ModifierFlags::default(),
                0,
            ),
            RawEvent::mouse(
                Timestamp::from_ticks(1000),
                EventType::MouseMoved,
                100.0,
                100.0,
                CursorState::Arrow,
                ModifierFlags::default(),
                0,
            ),
        ];

        let trajectory = SimplifiedTrajectory::from_events(&events, 1.0);
        assert!(trajectory.direction_vector().is_none()); // No direction for stationary
        assert_eq!(trajectory.path_length, 0.0);
    }
}
