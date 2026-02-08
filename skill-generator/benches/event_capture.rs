//! Criterion benchmarks for performance-critical hot paths
//!
//! Covers: ring buffer push/pop, RDP trajectory simplification,
//! kinematic segmentation, and GOMS boundary detection.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use skill_generator::capture::ring_buffer::EventRingBuffer;
use skill_generator::capture::types::{
    CursorState, EnrichedEvent, EventType, ModifierFlags, RawEvent,
};
use skill_generator::time::timebase::{MachTimebase, Timestamp};

use skill_generator::analysis::rdp_simplification::{RdpSimplifier, TrajectoryPoint};
use skill_generator::analysis::kinematic_segmentation::KinematicSegmenter;
use skill_generator::chunking::goms_detector::GomsDetector;

fn make_mouse_event(tick: u64) -> RawEvent {
    RawEvent::mouse(
        Timestamp::from_ticks(tick),
        EventType::MouseMoved,
        100.0,
        200.0,
        CursorState::Arrow,
        ModifierFlags::default(),
        0,
    )
}

fn make_enriched_event(tick: u64, seq: u64) -> EnrichedEvent {
    EnrichedEvent::new(make_mouse_event(tick), seq)
}

fn make_trajectory_point(x: f64, y: f64, tick: u64) -> TrajectoryPoint {
    TrajectoryPoint::new(x, y, tick)
}

// ---------------------------------------------------------------------------
// Ring buffer benchmarks
// ---------------------------------------------------------------------------

fn bench_ring_buffer_push(c: &mut Criterion) {
    MachTimebase::init();

    c.bench_function("ring_buffer_push", |b| {
        let buffer = EventRingBuffer::with_capacity(8192);
        let (mut producer, mut consumer) = buffer.split();
        let event = make_mouse_event(1000);

        b.iter(|| {
            if producer.push(black_box(event.clone())) {
                // Drain periodically to avoid filling up
            } else {
                consumer.pop_batch(4096);
                producer.push(black_box(event.clone()));
            }
        });
    });
}

fn bench_ring_buffer_pop(c: &mut Criterion) {
    MachTimebase::init();

    c.bench_function("ring_buffer_pop", |b| {
        let buffer = EventRingBuffer::with_capacity(8192);
        let (mut producer, mut consumer) = buffer.split();

        // Pre-fill buffer
        for i in 0..8192 {
            producer.push(make_mouse_event(i));
        }

        b.iter(|| {
            if let Some(slot) = consumer.pop() {
                black_box(slot);
                // Refill so we always have data
                producer.push(make_mouse_event(0));
            }
        });
    });
}

fn bench_ring_buffer_pop_batch(c: &mut Criterion) {
    MachTimebase::init();

    let mut group = c.benchmark_group("ring_buffer_pop_batch");
    for batch_size in [16, 64, 256, 1024] {
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &batch_size,
            |b, &size| {
                let buffer = EventRingBuffer::with_capacity(8192);
                let (mut producer, mut consumer) = buffer.split();

                b.iter(|| {
                    // Refill
                    for i in 0..size {
                        producer.push(make_mouse_event(i as u64));
                    }
                    let batch = consumer.pop_batch(black_box(size));
                    black_box(batch);
                });
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// RDP simplification benchmarks
// ---------------------------------------------------------------------------

fn generate_circle_points(n: usize) -> Vec<TrajectoryPoint> {
    (0..n)
        .map(|i| {
            let angle = (i as f64 / n as f64) * 2.0 * std::f64::consts::PI;
            make_trajectory_point(
                50.0 + 30.0 * angle.cos(),
                50.0 + 30.0 * angle.sin(),
                i as u64 * 1000,
            )
        })
        .collect()
}

fn generate_zigzag_points(n: usize) -> Vec<TrajectoryPoint> {
    (0..n)
        .map(|i| {
            let x = i as f64;
            let y = if i % 2 == 0 { 0.0 } else { 10.0 };
            make_trajectory_point(x, y, i as u64 * 1000)
        })
        .collect()
}

fn bench_rdp_simplify(c: &mut Criterion) {
    MachTimebase::init();
    let simplifier = RdpSimplifier::with_epsilon(2.0);

    let mut group = c.benchmark_group("rdp_simplify");

    for count in [50, 200, 1000, 5000] {
        let circle = generate_circle_points(count);
        group.bench_with_input(
            BenchmarkId::new("circle", count),
            &circle,
            |b, points| {
                b.iter(|| {
                    let result = simplifier.simplify(black_box(points));
                    black_box(result);
                });
            },
        );

        let zigzag = generate_zigzag_points(count);
        group.bench_with_input(
            BenchmarkId::new("zigzag", count),
            &zigzag,
            |b, points| {
                b.iter(|| {
                    let result = simplifier.simplify(black_box(points));
                    black_box(result);
                });
            },
        );
    }

    // Straight line (best case: collapses to 2 points)
    for count in [50, 200, 1000, 5000] {
        let straight: Vec<TrajectoryPoint> = (0..count)
            .map(|i| make_trajectory_point(i as f64, 0.0, i as u64 * 1000))
            .collect();
        group.bench_with_input(
            BenchmarkId::new("straight", count),
            &straight,
            |b, points| {
                b.iter(|| {
                    let result = simplifier.simplify(black_box(points));
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

fn bench_rdp_path_length(c: &mut Criterion) {
    MachTimebase::init();

    let points = generate_circle_points(1000);
    c.bench_function("rdp_path_length_1000", |b| {
        b.iter(|| {
            let len = RdpSimplifier::path_length(black_box(&points));
            black_box(len);
        });
    });
}

// ---------------------------------------------------------------------------
// Kinematic segmentation benchmarks
// ---------------------------------------------------------------------------

fn generate_ballistic_trajectory(n: usize) -> Vec<TrajectoryPoint> {
    // Simulate a ballistic move: accelerate, plateau, decelerate
    (0..n)
        .map(|i| {
            let t = i as f64 / n as f64;
            // S-curve position: smooth acceleration/deceleration
            let s = 3.0 * t * t - 2.0 * t * t * t;
            make_trajectory_point(s * 500.0, s * 300.0, i as u64 * 16_000_000) // ~16ms per sample (60Hz)
        })
        .collect()
}

fn bench_kinematic_segmentation(c: &mut Criterion) {
    MachTimebase::init();
    let segmenter = KinematicSegmenter::new();

    let mut group = c.benchmark_group("kinematic_segmentation");

    for count in [50, 200, 1000] {
        let trajectory = generate_ballistic_trajectory(count);
        group.bench_with_input(
            BenchmarkId::from_parameter(count),
            &trajectory,
            |b, points| {
                b.iter(|| {
                    let analysis = segmenter.analyze(black_box(points));
                    black_box(analysis);
                });
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// GOMS boundary detection benchmarks
// ---------------------------------------------------------------------------

fn generate_enriched_events_with_pauses(n: usize) -> Vec<EnrichedEvent> {
    // Generate events with periodic pauses to trigger boundary detection
    let mut events = Vec::with_capacity(n);
    let mut tick: u64 = 0;
    for i in 0..n {
        events.push(make_enriched_event(tick, i as u64));
        // Insert a longer pause every 20 events (simulates M-operator)
        if i % 20 == 19 {
            tick += 500_000_000; // ~500ms pause (in mach ticks ~= ns on Apple Silicon)
        } else {
            tick += 16_000_000; // ~16ms normal gap
        }
    }
    events
}

fn bench_goms_detection(c: &mut Criterion) {
    MachTimebase::init();
    let detector = GomsDetector::new();

    let mut group = c.benchmark_group("goms_detection");

    for count in [50, 200, 1000] {
        let events = generate_enriched_events_with_pauses(count);
        group.bench_with_input(
            BenchmarkId::from_parameter(count),
            &events,
            |b, evts| {
                b.iter(|| {
                    let boundaries = detector.detect_boundaries(black_box(evts));
                    black_box(boundaries);
                });
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Point geometry micro-benchmarks
// ---------------------------------------------------------------------------

fn bench_perpendicular_distance(c: &mut Criterion) {
    let point = make_trajectory_point(50.0, 50.0, 0);
    let start = make_trajectory_point(0.0, 0.0, 0);
    let end = make_trajectory_point(100.0, 0.0, 0);

    c.bench_function("perpendicular_distance", |b| {
        b.iter(|| {
            let d = black_box(&point).perpendicular_distance(black_box(&start), black_box(&end));
            black_box(d);
        });
    });
}

criterion_group!(
    benches,
    bench_ring_buffer_push,
    bench_ring_buffer_pop,
    bench_ring_buffer_pop_batch,
    bench_rdp_simplify,
    bench_rdp_path_length,
    bench_kinematic_segmentation,
    bench_goms_detection,
    bench_perpendicular_distance,
);
criterion_main!(benches);
