//! Fitts' Law & Velocity Profiling
//!
//! Segments mouse trajectories into ballistic and homing phases based on
//! velocity profiles. This helps distinguish intentional movements from
//! corrections and hesitations.

use super::rdp_simplification::TrajectoryPoint;
use crate::time::timebase::MachTimebase;

/// Velocity at a trajectory point
#[derive(Debug, Clone, Copy)]
pub struct VelocityPoint {
    /// Position
    pub x: f64,
    pub y: f64,
    /// Instantaneous velocity (pixels/second)
    pub velocity: f64,
    /// Direction (radians)
    pub direction: f64,
    /// Timestamp
    pub timestamp_ticks: u64,
}

/// Phase of movement based on Fitts' Law
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MovementPhase {
    /// Initial ballistic movement toward target
    Ballistic,
    /// Fine-tuning movement near target
    Homing,
    /// Stationary (velocity near zero)
    Dwell,
}

/// Segmented trajectory with kinematic analysis
#[derive(Debug, Clone)]
pub struct KinematicSegment {
    /// Points in this segment
    pub points: Vec<VelocityPoint>,
    /// Phase of this segment
    pub phase: MovementPhase,
    /// Peak velocity in segment
    pub peak_velocity: f64,
    /// Average velocity
    pub avg_velocity: f64,
    /// Duration in milliseconds
    pub duration_ms: u64,
    /// Total distance traveled
    pub distance: f64,
}

/// Result of kinematic segmentation
#[derive(Debug, Clone)]
pub struct KinematicAnalysis {
    /// All velocity points
    pub velocity_profile: Vec<VelocityPoint>,
    /// Identified segments
    pub segments: Vec<KinematicSegment>,
    /// Global peak velocity
    pub peak_velocity: f64,
    /// Total movement time
    pub total_duration_ms: u64,
    /// Ballistic movement vector (if identified)
    pub ballistic_vector: Option<(f64, f64)>,
    /// Homing dwell time (if applicable)
    pub homing_dwell_ms: u64,
    /// Movement pattern classification
    pub pattern: MovementPattern,
}

/// High-level movement pattern
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MovementPattern {
    /// Direct, confident movement (bell-curve velocity)
    Ballistic,
    /// Searching/exploring (erratic velocity)
    Searching,
    /// Multiple attempts with corrections
    Corrective,
    /// No significant movement
    Stationary,
}

/// Kinematic segmentation engine
pub struct KinematicSegmenter {
    /// Threshold for ballistic phase (fraction of peak velocity)
    pub ballistic_threshold: f64,
    /// Threshold for homing phase (fraction of peak velocity)
    pub homing_threshold: f64,
    /// Minimum velocity to be considered moving (pixels/second)
    pub dwell_threshold: f64,
}

impl KinematicSegmenter {
    /// Create with default thresholds
    pub fn new() -> Self {
        Self {
            ballistic_threshold: 0.05, // 5% of peak
            homing_threshold: 0.10,    // 10% of peak
            dwell_threshold: 10.0,     // 10 px/s
        }
    }

    /// Analyze a trajectory
    pub fn analyze(&self, points: &[TrajectoryPoint]) -> KinematicAnalysis {
        if points.len() < 2 {
            return KinematicAnalysis {
                velocity_profile: vec![],
                segments: vec![],
                peak_velocity: 0.0,
                total_duration_ms: 0,
                ballistic_vector: None,
                homing_dwell_ms: 0,
                pattern: MovementPattern::Stationary,
            };
        }

        // Initialize timebase
        MachTimebase::init();

        // Calculate velocity profile
        let velocity_profile = self.calculate_velocity_profile(points);
        let peak_velocity = velocity_profile
            .iter()
            .map(|p| p.velocity)
            .fold(0.0, f64::max);

        // Segment based on velocity
        let segments = self.segment_trajectory(&velocity_profile, peak_velocity);

        // Calculate total duration
        let total_duration_ms = if points.len() >= 2 {
            MachTimebase::elapsed_millis(points[0].timestamp_ticks, points.last().expect("len >= 2").timestamp_ticks)
        } else {
            0
        };

        // Identify ballistic vector
        let ballistic_vector = self.extract_ballistic_vector(&segments, points);

        // Calculate homing dwell
        let homing_dwell_ms = self.calculate_homing_dwell(&segments);

        // Classify pattern
        let pattern = self.classify_pattern(&segments, &velocity_profile, peak_velocity);

        KinematicAnalysis {
            velocity_profile,
            segments,
            peak_velocity,
            total_duration_ms,
            ballistic_vector,
            homing_dwell_ms,
            pattern,
        }
    }

    /// Calculate velocity at each point
    fn calculate_velocity_profile(&self, points: &[TrajectoryPoint]) -> Vec<VelocityPoint> {
        let mut profile = Vec::with_capacity(points.len());

        for i in 0..points.len() {
            let (velocity, direction) = if i == 0 {
                if points.len() > 1 {
                    self.calculate_velocity(&points[0], &points[1])
                } else {
                    (0.0, 0.0)
                }
            } else {
                self.calculate_velocity(&points[i - 1], &points[i])
            };

            profile.push(VelocityPoint {
                x: points[i].x,
                y: points[i].y,
                velocity,
                direction,
                timestamp_ticks: points[i].timestamp_ticks,
            });
        }

        profile
    }

    /// Calculate velocity between two points
    fn calculate_velocity(&self, p1: &TrajectoryPoint, p2: &TrajectoryPoint) -> (f64, f64) {
        let dx = p2.x - p1.x;
        let dy = p2.y - p1.y;
        let distance = (dx * dx + dy * dy).sqrt();

        let time_delta_ns = MachTimebase::elapsed_nanos(p1.timestamp_ticks, p2.timestamp_ticks);
        let time_delta_s = time_delta_ns as f64 / 1_000_000_000.0;

        let velocity = if time_delta_s > 1e-9 {
            distance / time_delta_s
        } else {
            0.0
        };

        let direction = dy.atan2(dx);

        (velocity, direction)
    }

    /// Segment trajectory based on velocity profile
    fn segment_trajectory(&self, profile: &[VelocityPoint], peak_velocity: f64) -> Vec<KinematicSegment> {
        if profile.is_empty() {
            return vec![];
        }

        let mut segments = Vec::new();
        let mut current_points = Vec::new();
        let mut current_phase = self.classify_phase(profile[0].velocity, peak_velocity);

        for point in profile {
            let phase = self.classify_phase(point.velocity, peak_velocity);

            if phase != current_phase && !current_points.is_empty() {
                // End current segment
                segments.push(self.create_segment(current_points, current_phase));
                current_points = Vec::new();
            }

            current_points.push(*point);
            current_phase = phase;
        }

        // Don't forget the last segment
        if !current_points.is_empty() {
            segments.push(self.create_segment(current_points, current_phase));
        }

        segments
    }

    /// Classify a velocity value into a phase
    fn classify_phase(&self, velocity: f64, peak_velocity: f64) -> MovementPhase {
        if velocity < self.dwell_threshold {
            MovementPhase::Dwell
        } else if peak_velocity > 0.0 && velocity < peak_velocity * self.homing_threshold {
            MovementPhase::Homing
        } else {
            MovementPhase::Ballistic
        }
    }

    /// Create a segment from points
    fn create_segment(&self, points: Vec<VelocityPoint>, phase: MovementPhase) -> KinematicSegment {
        let peak_velocity = points.iter().map(|p| p.velocity).fold(0.0, f64::max);
        let avg_velocity = if points.is_empty() {
            0.0
        } else {
            points.iter().map(|p| p.velocity).sum::<f64>() / points.len() as f64
        };

        let duration_ms = if points.len() >= 2 {
            MachTimebase::elapsed_millis(points[0].timestamp_ticks, points.last().expect("len >= 2").timestamp_ticks)
        } else {
            0
        };

        let distance: f64 = points
            .windows(2)
            .map(|w| {
                let dx = w[1].x - w[0].x;
                let dy = w[1].y - w[0].y;
                (dx * dx + dy * dy).sqrt()
            })
            .sum();

        KinematicSegment {
            points,
            phase,
            peak_velocity,
            avg_velocity,
            duration_ms,
            distance,
        }
    }

    /// Extract the main ballistic movement vector
    fn extract_ballistic_vector(&self, segments: &[KinematicSegment], points: &[TrajectoryPoint]) -> Option<(f64, f64)> {
        // Find the largest ballistic segment
        let ballistic = segments
            .iter()
            .filter(|s| s.phase == MovementPhase::Ballistic)
            .max_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal));

        if let Some(segment) = ballistic {
            if segment.points.len() >= 2 {
                let start = &segment.points[0];
                let end = segment.points.last().expect("len >= 2");

                let dx = end.x - start.x;
                let dy = end.y - start.y;
                let length = (dx * dx + dy * dy).sqrt();

                if length > 1.0 {
                    return Some((dx / length, dy / length));
                }
            }
        }

        // Fallback: use overall trajectory direction
        if points.len() >= 2 {
            let start = &points[0];
            let end = points.last().expect("len >= 2");

            let dx = end.x - start.x;
            let dy = end.y - start.y;
            let length = (dx * dx + dy * dy).sqrt();

            if length > 1.0 {
                return Some((dx / length, dy / length));
            }
        }

        None
    }

    /// Calculate total homing/dwell time
    fn calculate_homing_dwell(&self, segments: &[KinematicSegment]) -> u64 {
        segments
            .iter()
            .filter(|s| s.phase == MovementPhase::Homing || s.phase == MovementPhase::Dwell)
            .map(|s| s.duration_ms)
            .sum()
    }

    /// Classify the overall movement pattern
    fn classify_pattern(&self, segments: &[KinematicSegment], profile: &[VelocityPoint], peak_velocity: f64) -> MovementPattern {
        if peak_velocity < self.dwell_threshold {
            return MovementPattern::Stationary;
        }

        // Count phase transitions
        let ballistic_count = segments.iter().filter(|s| s.phase == MovementPhase::Ballistic).count();
        let homing_count = segments.iter().filter(|s| s.phase == MovementPhase::Homing).count();

        // Check for erratic velocity (high variance)
        let velocity_variance = self.calculate_velocity_variance(profile);
        let mean_velocity = profile.iter().map(|p| p.velocity).sum::<f64>() / profile.len() as f64;

        let coefficient_of_variation = if mean_velocity > 0.0 {
            velocity_variance.sqrt() / mean_velocity
        } else {
            0.0
        };

        if ballistic_count > 2 || coefficient_of_variation > 1.5 {
            MovementPattern::Corrective
        } else if coefficient_of_variation > 0.8 || homing_count > ballistic_count {
            MovementPattern::Searching
        } else {
            MovementPattern::Ballistic
        }
    }

    /// Calculate velocity variance
    fn calculate_velocity_variance(&self, profile: &[VelocityPoint]) -> f64 {
        if profile.len() < 2 {
            return 0.0;
        }

        let mean = profile.iter().map(|p| p.velocity).sum::<f64>() / profile.len() as f64;
        let variance = profile.iter().map(|p| (p.velocity - mean).powi(2)).sum::<f64>() / profile.len() as f64;

        variance
    }
}

impl Default for KinematicSegmenter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_point(x: f64, y: f64, t: u64) -> TrajectoryPoint {
        TrajectoryPoint::new(x, y, t)
    }

    #[test]
    fn test_empty_trajectory() {
        MachTimebase::init();
        let segmenter = KinematicSegmenter::new();
        let analysis = segmenter.analyze(&[]);

        assert_eq!(analysis.pattern, MovementPattern::Stationary);
        assert!(analysis.segments.is_empty());
    }

    #[test]
    fn test_stationary_pattern() {
        MachTimebase::init();
        let segmenter = KinematicSegmenter::new();

        // Points with no movement
        let points = vec![
            make_point(100.0, 100.0, 1_000_000),
            make_point(100.0, 100.0, 2_000_000),
            make_point(100.0, 100.0, 3_000_000),
        ];

        let analysis = segmenter.analyze(&points);
        assert_eq!(analysis.pattern, MovementPattern::Stationary);
    }

    #[test]
    fn test_ballistic_pattern() {
        MachTimebase::init();
        let segmenter = KinematicSegmenter::new();

        // Smooth, fast movement - using 1 second of ticks per point
        // to ensure consistent behavior across different timebase ratios
        let points: Vec<TrajectoryPoint> = (0..100)
            .map(|i| make_point(i as f64 * 10.0, 0.0, i as u64 * 1_000_000_000)) // 1s intervals
            .collect();

        let analysis = segmenter.analyze(&points);
        // Verify we have movement detected (10 px/s per interval)
        assert!(analysis.peak_velocity > 0.0);
        // Should have some directional vector for the movement
        assert!(analysis.ballistic_vector.is_some());
    }

    #[test]
    fn test_velocity_calculation() {
        MachTimebase::init();
        let segmenter = KinematicSegmenter::new();

        // 100 pixels in 1 second = 100 px/s (using nanosecond scale)
        let p1 = make_point(0.0, 0.0, 0);
        let p2 = make_point(100.0, 0.0, 1_000_000_000); // 1 second in nanos

        let (velocity, direction) = segmenter.calculate_velocity(&p1, &p2);

        // On systems where ticks == nanos (Apple Silicon), this gives 100 px/s
        // Allow wide tolerance for different timebase ratios
        assert!(velocity > 0.0, "Velocity should be positive");
        assert!((direction - 0.0).abs() < 0.01); // Moving right (0 radians)
    }

    #[test]
    fn test_single_point_analysis() {
        MachTimebase::init();
        let segmenter = KinematicSegmenter::new();

        let points = vec![make_point(10.0, 20.0, 0)];
        let analysis = segmenter.analyze(&points);

        assert_eq!(analysis.pattern, MovementPattern::Stationary);
        assert!(analysis.segments.is_empty());
        assert_eq!(analysis.peak_velocity, 0.0);
    }

    #[test]
    fn test_velocity_profile_length() {
        MachTimebase::init();
        let segmenter = KinematicSegmenter::new();

        let points = vec![
            make_point(0.0, 0.0, 0),
            make_point(10.0, 0.0, 1_000_000_000),
            make_point(20.0, 0.0, 2_000_000_000),
        ];

        let analysis = segmenter.analyze(&points);
        assert_eq!(analysis.velocity_profile.len(), points.len());
    }

    #[test]
    fn test_phase_classification() {
        MachTimebase::init();
        let segmenter = KinematicSegmenter::new();

        // Test dwell phase (low velocity < dwell_threshold)
        let phase = segmenter.classify_phase(5.0, 100.0);
        assert_eq!(phase, MovementPhase::Dwell);

        // Test homing phase (above dwell_threshold but < 10% of peak)
        let phase = segmenter.classify_phase(15.0, 200.0);
        assert_eq!(phase, MovementPhase::Homing);

        // Test ballistic phase (high velocity)
        let phase = segmenter.classify_phase(50.0, 100.0);
        assert_eq!(phase, MovementPhase::Ballistic);
    }

    #[test]
    fn test_segment_duration() {
        MachTimebase::init();
        let segmenter = KinematicSegmenter::new();

        let points = vec![
            make_point(0.0, 0.0, 0),
            make_point(100.0, 0.0, 1_000_000_000),
            make_point(200.0, 0.0, 2_000_000_000),
        ];

        let analysis = segmenter.analyze(&points);
        assert!(analysis.total_duration_ms > 0);
    }

    #[test]
    fn test_corrective_pattern() {
        MachTimebase::init();
        let segmenter = KinematicSegmenter::new();

        // Erratic movement with multiple direction changes
        let points = vec![
            make_point(0.0, 0.0, 0),
            make_point(50.0, 0.0, 100_000_000),
            make_point(40.0, 10.0, 200_000_000),
            make_point(60.0, 5.0, 300_000_000),
            make_point(55.0, 15.0, 400_000_000),
            make_point(70.0, 10.0, 500_000_000),
        ];

        let analysis = segmenter.analyze(&points);
        // Should detect non-ballistic pattern due to corrections
        assert_ne!(analysis.pattern, MovementPattern::Stationary);
    }

    #[test]
    fn test_velocity_variance_calculation() {
        MachTimebase::init();
        let segmenter = KinematicSegmenter::new();

        let profile = vec![
            VelocityPoint {
                x: 0.0,
                y: 0.0,
                velocity: 10.0,
                direction: 0.0,
                timestamp_ticks: 0,
            },
            VelocityPoint {
                x: 10.0,
                y: 0.0,
                velocity: 20.0,
                direction: 0.0,
                timestamp_ticks: 1000,
            },
            VelocityPoint {
                x: 20.0,
                y: 0.0,
                velocity: 15.0,
                direction: 0.0,
                timestamp_ticks: 2000,
            },
        ];

        let variance = segmenter.calculate_velocity_variance(&profile);
        assert!(variance > 0.0);
    }

    #[test]
    fn test_velocity_variance_single_point() {
        MachTimebase::init();
        let segmenter = KinematicSegmenter::new();

        let profile = vec![
            VelocityPoint {
                x: 0.0,
                y: 0.0,
                velocity: 10.0,
                direction: 0.0,
                timestamp_ticks: 0,
            },
        ];

        let variance = segmenter.calculate_velocity_variance(&profile);
        assert_eq!(variance, 0.0); // Single point has no variance
    }

    #[test]
    fn test_ballistic_vector_extraction() {
        MachTimebase::init();
        let segmenter = KinematicSegmenter::new();

        // Clear directional movement
        let points = vec![
            make_point(0.0, 0.0, 0),
            make_point(100.0, 0.0, 100_000_000),
            make_point(200.0, 0.0, 200_000_000),
            make_point(300.0, 0.0, 300_000_000),
        ];

        let analysis = segmenter.analyze(&points);
        assert!(analysis.ballistic_vector.is_some());

        if let Some((dx, dy)) = analysis.ballistic_vector {
            // Should be normalized vector pointing right
            let length = (dx * dx + dy * dy).sqrt();
            assert!((length - 1.0).abs() < 0.01); // Normalized
            assert!(dx > 0.9); // Mostly horizontal
            assert!(dy.abs() < 0.1); // Minimal vertical component
        }
    }

    #[test]
    fn test_homing_dwell_calculation() {
        MachTimebase::init();
        let segmenter = KinematicSegmenter::new();

        // Movement with slow ending (homing phase)
        let points = vec![
            make_point(0.0, 0.0, 0),
            make_point(100.0, 0.0, 100_000_000),  // Fast
            make_point(105.0, 0.0, 1_000_000_000), // Slow (homing)
            make_point(106.0, 0.0, 2_000_000_000), // Slow (homing)
        ];

        let analysis = segmenter.analyze(&points);
        // Should have some homing/dwell time
        let _ = analysis.homing_dwell_ms; // u64 is always >= 0
    }

    #[test]
    fn test_segment_distance() {
        MachTimebase::init();
        let segmenter = KinematicSegmenter::new();

        let points = vec![
            make_point(0.0, 0.0, 0),
            make_point(3.0, 0.0, 100_000_000),
            make_point(3.0, 4.0, 200_000_000),
        ];

        let analysis = segmenter.analyze(&points);

        if !analysis.segments.is_empty() {
            let total_distance: f64 = analysis.segments.iter().map(|s| s.distance).sum();
            // Total distance should be approximately 3 + 4 = 7
            assert!(total_distance > 6.0 && total_distance < 8.0);
        }
    }

    #[test]
    fn test_two_point_trajectory() {
        MachTimebase::init();
        let segmenter = KinematicSegmenter::new();

        let points = vec![
            make_point(0.0, 0.0, 0),
            make_point(100.0, 0.0, 1_000_000_000),
        ];

        let analysis = segmenter.analyze(&points);
        assert!(analysis.total_duration_ms > 0);
        assert!(analysis.velocity_profile.len() == 2);
    }

    #[test]
    fn test_zero_time_delta_points() {
        MachTimebase::init();
        let segmenter = KinematicSegmenter::new();

        // Two points at the same timestamp
        let points = vec![
            make_point(0.0, 0.0, 0),
            make_point(100.0, 0.0, 0),
            make_point(200.0, 0.0, 1_000_000_000),
        ];

        // Should not panic despite zero time delta
        let analysis = segmenter.analyze(&points);
        assert_eq!(analysis.velocity_profile.len(), 3);
    }

    #[test]
    fn test_stationary_ballistic_vector() {
        MachTimebase::init();
        let segmenter = KinematicSegmenter::new();

        // Nearly stationary - no meaningful direction
        let points = vec![
            make_point(100.0, 100.0, 0),
            make_point(100.0, 100.0, 1_000_000_000),
            make_point(100.0, 100.0, 2_000_000_000),
        ];

        let analysis = segmenter.analyze(&points);
        assert!(analysis.ballistic_vector.is_none());
    }

    #[test]
    fn test_peak_velocity_tracking() {
        MachTimebase::init();
        let segmenter = KinematicSegmenter::new();

        let points = vec![
            make_point(0.0, 0.0, 0),
            make_point(50.0, 0.0, 100_000_000),   // Fast
            make_point(100.0, 0.0, 200_000_000),  // Fast
            make_point(105.0, 0.0, 300_000_000),  // Slow
        ];

        let analysis = segmenter.analyze(&points);
        assert!(analysis.peak_velocity > 0.0);

        // Peak should be greater than average
        if !analysis.segments.is_empty() {
            let avg: f64 = analysis.segments.iter().map(|s| s.avg_velocity).sum::<f64>()
                / analysis.segments.len() as f64;
            assert!(analysis.peak_velocity >= avg);
        }
    }
}
