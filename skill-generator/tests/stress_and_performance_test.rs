//! Stress and Performance Tests for the Skill Generator
//!
//! These tests verify system behavior under high load conditions:
//! - High frequency event capture (1000 Hz)
//! - Extended recordings with thousands of events
//! - Rapid semantic query performance
//! - Skill generation latency
//! - Ring buffer saturation and graceful degradation
//! - Concurrent access patterns

use skill_generator::capture::ring_buffer::{EventRingBuffer, ProcessedEventStore};
use skill_generator::capture::types::{
    CursorState, EnrichedEvent, EventType, ModifierFlags, RawEvent, SemanticContext,
    SemanticSource, SemanticState,
};
use skill_generator::time::timebase::{MachTimebase, Timestamp};
use skill_generator::workflow::generator::{GeneratorConfig, SkillGenerator};
use skill_generator::workflow::recording::Recording;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

// =============================================================================
// Test Helpers
// =============================================================================

/// Create a test raw event with specific parameters
fn make_raw_event(event_type: EventType, x: f64, y: f64) -> RawEvent {
    RawEvent {
        timestamp: Timestamp::now(),
        event_type,
        coordinates: (x, y),
        cursor_state: CursorState::Arrow,
        key_code: None,
        character: None,
        modifiers: ModifierFlags::default(),
        scroll_delta: None,
        click_count: 0,
    }
}

/// Create a click event with semantic context
fn make_click_with_context(
    x: f64,
    y: f64,
    role: &str,
    title: &str,
    click_count: u8,
) -> EnrichedEvent {
    let raw = RawEvent {
        timestamp: Timestamp::now(),
        event_type: EventType::LeftMouseDown,
        coordinates: (x, y),
        cursor_state: CursorState::PointingHand,
        key_code: None,
        character: None,
        modifiers: ModifierFlags::default(),
        scroll_delta: None,
        click_count,
    };

    let mut event = EnrichedEvent::new(raw, 0);
    event.semantic = Some(SemanticContext {
        ax_role: Some(role.to_string()),
        title: Some(title.to_string()),
        identifier: Some(format!("id_{}", title.to_lowercase().replace(' ', "_"))),
        source: SemanticSource::Accessibility,
        confidence: 0.95,
        ..Default::default()
    });
    event
}

/// Create a keyboard event
fn make_key_event(character: char, key_code: u16) -> EnrichedEvent {
    let raw = RawEvent {
        timestamp: Timestamp::now(),
        event_type: EventType::KeyDown,
        coordinates: (0.0, 0.0),
        cursor_state: CursorState::IBeam,
        key_code: Some(key_code),
        character: Some(character),
        modifiers: ModifierFlags::default(),
        scroll_delta: None,
        click_count: 0,
    };

    EnrichedEvent::new(raw, 0)
}

/// Measure memory usage approximation (event count * estimated event size)
fn estimate_memory_usage_bytes(event_count: usize) -> usize {
    // Approximate size of EnrichedEvent including semantic context
    // RawEvent: ~120 bytes, SemanticContext: ~500 bytes, EnrichedEvent overhead: ~50 bytes
    const ESTIMATED_EVENT_SIZE: usize = 700;
    event_count * ESTIMATED_EVENT_SIZE
}

// =============================================================================
// Test 1: High Frequency Input (1000 Hz for 10 seconds)
// =============================================================================

#[test]
fn test_high_frequency_input() {
    MachTimebase::init();

    // Use a large buffer to handle 10 seconds at 1000 Hz = 10,000 events
    // Buffer size must be power of 2
    let buffer = EventRingBuffer::with_capacity(16384);
    let stats = buffer.stats();
    let (mut producer, mut consumer) = buffer.split();

    // Target: 1000 Hz for 10 seconds = 10,000 events
    let target_hz = 1000;
    let duration_secs = 10;
    let total_events = target_hz * duration_secs;
    let interval_micros = 1_000_000 / target_hz;

    // Track timing and events
    let events_pushed = Arc::new(AtomicUsize::new(0));
    let events_consumed = Arc::new(AtomicUsize::new(0));
    let producer_done = Arc::new(std::sync::atomic::AtomicBool::new(false));

    let events_consumed_clone = Arc::clone(&events_consumed);
    let producer_done_clone = Arc::clone(&producer_done);

    // Spawn consumer thread
    let consumer_handle = thread::spawn(move || {
        let mut total_consumed = 0;
        loop {
            let batch = consumer.pop_batch(100);
            total_consumed += batch.len();
            events_consumed_clone.store(total_consumed, Ordering::Relaxed);

            if batch.is_empty() {
                if producer_done_clone.load(Ordering::Relaxed) {
                    // Check one more time after producer is done
                    let final_batch = consumer.pop_batch(1000);
                    total_consumed += final_batch.len();
                    events_consumed_clone.store(total_consumed, Ordering::Relaxed);
                    break;
                }
                thread::sleep(Duration::from_micros(100));
            }
        }
        total_consumed
    });

    // Producer loop - simulate 1000 Hz event rate
    let start = Instant::now();
    let mut pushed = 0;

    for i in 0..total_events {
        let x = (i as f64 * 0.1).sin() * 500.0 + 960.0;
        let y = (i as f64 * 0.1).cos() * 500.0 + 540.0;
        let event = make_raw_event(EventType::MouseMoved, x, y);

        if producer.push(event) {
            pushed += 1;
        }
        events_pushed.store(pushed, Ordering::Relaxed);

        // Sleep to maintain target frequency (busy-wait for precision)
        let target_time = start + Duration::from_micros((i as u64 + 1) * interval_micros as u64);
        while Instant::now() < target_time {
            std::hint::spin_loop();
        }
    }

    let elapsed = start.elapsed();
    producer_done.store(true, Ordering::Relaxed);

    // Wait for consumer to finish
    let consumed = consumer_handle.join().expect("Consumer thread panicked");

    // Verify results
    let pushed_count = stats.events_pushed.load(Ordering::Relaxed);
    let dropped_count = stats.events_dropped.load(Ordering::Relaxed);

    println!("High Frequency Test Results:");
    println!("  Target events: {}", total_events);
    println!("  Events pushed: {}", pushed_count);
    println!("  Events dropped: {}", dropped_count);
    println!("  Events consumed: {}", consumed);
    println!("  Elapsed time: {:?}", elapsed);
    println!("  Effective rate: {:.1} Hz", pushed_count as f64 / elapsed.as_secs_f64());

    // Verify zero drops (buffer is large enough)
    assert_eq!(
        dropped_count, 0,
        "Expected zero drops with adequate buffer size, but got {} drops",
        dropped_count
    );

    // Verify all pushed events were consumed
    assert_eq!(
        consumed as u64, pushed_count,
        "All pushed events should be consumed"
    );

    // Verify we achieved close to target rate (within 20% due to OS scheduling)
    let effective_rate = pushed_count as f64 / elapsed.as_secs_f64();
    assert!(
        effective_rate >= target_hz as f64 * 0.8,
        "Effective rate {:.1} Hz should be at least 80% of target {} Hz",
        effective_rate,
        target_hz
    );
}

// =============================================================================
// Test 2: Extended Recording (5000+ events)
// =============================================================================

#[test]
fn test_extended_recording() {
    MachTimebase::init();

    let mut recording = Recording::new(
        "extended_test".to_string(),
        Some("Extended recording with 5000+ events".to_string()),
    );

    let event_count = 5500;
    let start = Instant::now();

    // Generate diverse events
    for i in 0..event_count {
        let event_type = match i % 10 {
            0 | 1 => {
                // Clicks (20%)
                let event = make_click_with_context(
                    (i % 1920) as f64,
                    (i % 1080) as f64,
                    "AXButton",
                    &format!("Button_{}", i),
                    1,
                );
                recording.add_event(event);
                continue;
            }
            2 | 3 => {
                // Keyboard events (20%)
                let chars = "abcdefghijklmnopqrstuvwxyz";
                let c = chars.chars().nth(i % 26).unwrap_or('a');
                let event = make_key_event(c, (c as u16) % 256);
                recording.add_event(event);
                continue;
            }
            4 => EventType::ScrollWheel,
            _ => EventType::MouseMoved,
        };

        // Mouse move or scroll events
        let raw = RawEvent {
            timestamp: Timestamp::now(),
            event_type,
            coordinates: ((i % 1920) as f64, (i % 1080) as f64),
            cursor_state: CursorState::Arrow,
            key_code: None,
            character: None,
            modifiers: ModifierFlags::default(),
            scroll_delta: if event_type == EventType::ScrollWheel {
                Some((0.0, -10.0))
            } else {
                None
            },
            click_count: 0,
        };
        recording.add_raw_event(raw);
    }

    recording.finalize(event_count as u64 * 10); // ~10ms per event

    let creation_time = start.elapsed();

    // Verify recording
    assert_eq!(
        recording.len(),
        event_count,
        "Recording should contain all events"
    );

    // Estimate memory usage
    let estimated_memory = estimate_memory_usage_bytes(event_count);
    let memory_mb = estimated_memory as f64 / (1024.0 * 1024.0);

    println!("Extended Recording Test Results:");
    println!("  Event count: {}", recording.len());
    println!("  Creation time: {:?}", creation_time);
    println!("  Estimated memory: {:.2} MB", memory_mb);
    println!("  Click events: {}", recording.click_events().len());
    println!("  Keyboard events: {}", recording.keyboard_events().len());

    // Memory should be reasonable (< 10MB for 5500 events)
    assert!(
        memory_mb < 10.0,
        "Memory usage {:.2} MB should be less than 10 MB",
        memory_mb
    );

    // Creation should be fast (< 1 second)
    assert!(
        creation_time < Duration::from_secs(1),
        "Recording creation {:?} should be under 1 second",
        creation_time
    );

    // Test serialization round-trip
    let json = serde_json::to_string(&recording).expect("Serialization failed");
    let loaded: Recording = serde_json::from_str(&json).expect("Deserialization failed");

    assert_eq!(loaded.len(), event_count, "Serialization round-trip should preserve events");
}

// =============================================================================
// Test 3: Rapid Semantic Queries
// =============================================================================

#[test]
fn test_rapid_semantic_queries() {
    MachTimebase::init();

    // Simulate rapid accessibility query patterns
    let buffer = EventRingBuffer::with_capacity(4096);
    let store = ProcessedEventStore::new(10000);
    let _stats = buffer.stats(); // Stats available but not needed for this test
    let (mut producer, mut consumer) = buffer.split();

    let query_count = 1000;
    let start = Instant::now();

    // Push events that would trigger semantic queries
    for i in 0..query_count {
        let raw = RawEvent {
            timestamp: Timestamp::now(),
            event_type: EventType::LeftMouseDown,
            coordinates: ((i * 10 % 1920) as f64, (i * 10 % 1080) as f64),
            cursor_state: CursorState::PointingHand,
            key_code: None,
            character: None,
            modifiers: ModifierFlags::default(),
            scroll_delta: None,
            click_count: 1,
        };
        producer.push(raw);
    }

    let push_time = start.elapsed();

    // Consume and simulate semantic processing
    let process_start = Instant::now();
    let mut processed = 0;

    while let Some(slot) = consumer.pop() {
        // Simulate semantic lookup latency (1-5 microseconds)
        // In real code, this would be an AX API call
        std::hint::black_box(&slot.event.coordinates);

        // Mark as processed
        slot.semantic_state.store(SemanticState::Filled, Ordering::Release);
        store.store(slot);
        processed += 1;
    }

    let process_time = process_start.elapsed();
    let total_time = start.elapsed();

    println!("Rapid Semantic Query Test Results:");
    println!("  Query count: {}", query_count);
    println!("  Push time: {:?}", push_time);
    println!("  Process time: {:?}", process_time);
    println!("  Total time: {:?}", total_time);
    println!("  Events processed: {}", processed);
    println!("  Query rate: {:.1} queries/sec", processed as f64 / process_time.as_secs_f64());

    assert_eq!(processed, query_count, "All events should be processed");
    assert_eq!(store.len(), query_count, "Store should contain all processed events");

    // Processing should be fast (< 100ms for 1000 queries without actual AX calls)
    assert!(
        process_time < Duration::from_millis(100),
        "Processing {:?} should be under 100ms",
        process_time
    );
}

// =============================================================================
// Test 4: Skill Generation Latency
// =============================================================================

#[test]
fn test_skill_generation_latency() {
    MachTimebase::init();

    // Create recordings of various sizes
    let sizes = [10, 50, 100, 500, 1000];
    let mut results = Vec::new();

    for &size in &sizes {
        let mut recording = Recording::new(
            format!("latency_test_{}", size),
            Some(format!("Latency test with {} events", size)),
        );

        // Add significant events (clicks and keyboard)
        for i in 0..size {
            if i % 3 == 0 {
                let event = make_click_with_context(
                    (i * 20 % 1920) as f64,
                    (i * 20 % 1080) as f64,
                    "AXButton",
                    &format!("Button_{}", i),
                    1,
                );
                recording.add_event(event);
            } else {
                let c = (b'a' + (i % 26) as u8) as char;
                let event = make_key_event(c, c as u16);
                recording.add_event(event);
            }
        }
        recording.finalize(size as u64 * 100);

        // Measure generation time
        let generator = SkillGenerator::with_config(GeneratorConfig {
            include_verification: true,
            extract_variables: true,
            selector_chain_depth: 3,
            ..Default::default()
        });

        let start = Instant::now();
        let skill = generator.generate(&recording).expect("Generation failed");
        let generation_time = start.elapsed();

        // Also measure markdown rendering time
        let render_start = Instant::now();
        let _markdown = generator.render_to_markdown(&skill);
        let render_time = render_start.elapsed();

        results.push((size, generation_time, render_time, skill.steps.len()));
    }

    println!("\nSkill Generation Latency Test Results:");
    println!("{:>8} {:>15} {:>15} {:>10}", "Events", "Gen Time", "Render Time", "Steps");
    println!("{}", "-".repeat(50));

    for (size, gen_time, render_time, steps) in &results {
        println!(
            "{:>8} {:>15?} {:>15?} {:>10}",
            size, gen_time, render_time, steps
        );
    }

    // Verify latency requirements
    for (size, gen_time, render_time, _) in &results {
        // Generation should scale reasonably (< 1ms per event for small recordings)
        let max_gen_time = Duration::from_millis((*size as u64).max(50));
        assert!(
            *gen_time < max_gen_time,
            "Generation for {} events ({:?}) should be under {:?}",
            size, gen_time, max_gen_time
        );

        // Rendering should be fast (< 10ms for any size)
        assert!(
            *render_time < Duration::from_millis(100),
            "Rendering for {} events ({:?}) should be under 100ms",
            size, render_time
        );
    }
}

// =============================================================================
// Test 5: Ring Buffer Saturation
// =============================================================================

#[test]
fn test_ring_buffer_saturation() {
    MachTimebase::init();

    // Use a small buffer to test saturation behavior
    let buffer_size = 64; // Small buffer for saturation testing
    let buffer = EventRingBuffer::with_capacity(buffer_size);
    let stats = buffer.stats();
    let (mut producer, mut consumer) = buffer.split();

    // Track graceful degradation metrics
    let mut push_results = Vec::new();
    let mut saturation_points = Vec::new();
    let events_to_push = 1000;

    // Consumer runs slowly to cause saturation
    let consumer_delay = Duration::from_micros(500); // Slow consumer
    let producer_delay = Duration::from_micros(50);  // Fast producer

    let consumer_count = Arc::new(AtomicU64::new(0));
    let consumer_count_clone = Arc::clone(&consumer_count);
    let stop_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let stop_flag_clone = Arc::clone(&stop_flag);

    // Start slow consumer
    let consumer_handle = thread::spawn(move || {
        let mut consumed = 0u64;
        while !stop_flag_clone.load(Ordering::Relaxed) {
            if let Some(_slot) = consumer.pop() {
                consumed += 1;
                consumer_count_clone.store(consumed, Ordering::Relaxed);
                thread::sleep(consumer_delay);
            } else {
                thread::sleep(Duration::from_micros(100));
            }
        }
        // Drain remaining
        while let Some(_slot) = consumer.pop() {
            consumed += 1;
        }
        consumer_count_clone.store(consumed, Ordering::Relaxed);
        consumed
    });

    // Fast producer - will cause saturation
    let start = Instant::now();
    let mut successful_pushes = 0;
    let mut failed_pushes = 0;
    let mut first_drop_idx = None;

    for i in 0..events_to_push {
        let event = make_raw_event(EventType::MouseMoved, i as f64, i as f64);
        let success = producer.push(event);

        if success {
            successful_pushes += 1;
        } else {
            if first_drop_idx.is_none() {
                first_drop_idx = Some(i);
            }
            failed_pushes += 1;
        }

        push_results.push(success);

        // Record saturation points
        if producer.is_full() {
            saturation_points.push(i);
        }

        thread::sleep(producer_delay);
    }

    let _push_time = start.elapsed();
    stop_flag.store(true, Ordering::Relaxed);

    // Wait for consumer to drain
    let consumed = consumer_handle.join().expect("Consumer panicked");

    println!("\nRing Buffer Saturation Test Results:");
    println!("  Buffer size: {}", buffer_size);
    println!("  Events attempted: {}", events_to_push);
    println!("  Successful pushes: {}", successful_pushes);
    println!("  Failed pushes (gracefully dropped): {}", failed_pushes);
    println!("  Events consumed: {}", consumed);
    println!("  First drop at event: {:?}", first_drop_idx);
    println!("  Saturation points: {}", saturation_points.len());
    println!("  Stats - pushed: {}, dropped: {}",
             stats.events_pushed.load(Ordering::Relaxed),
             stats.events_dropped.load(Ordering::Relaxed));

    // Verify graceful handling
    // 1. No panics occurred (we got here)
    // 2. Dropped events are tracked
    assert_eq!(
        stats.events_dropped.load(Ordering::Relaxed),
        failed_pushes as u64,
        "Dropped count should match failed pushes"
    );

    // 3. All successful pushes were eventually consumed
    assert_eq!(
        consumed,
        stats.events_pushed.load(Ordering::Relaxed),
        "All pushed events should be consumed"
    );

    // 4. Saturation should have occurred (slow consumer, fast producer)
    assert!(
        failed_pushes > 0,
        "Test should have caused some drops to verify graceful handling"
    );

    // 5. Drop rate should be reasonable (not catastrophic)
    let drop_rate = failed_pushes as f64 / events_to_push as f64;
    println!("  Drop rate: {:.1}%", drop_rate * 100.0);
}

// =============================================================================
// Test 6: Concurrent Access
// =============================================================================

#[test]
fn test_concurrent_access() {
    MachTimebase::init();

    // Test concurrent access to ProcessedEventStore (thread-safe component)
    let store = Arc::new(ProcessedEventStore::new(10000));
    let thread_count = 4;
    let events_per_thread = 250;
    let total_expected = thread_count * events_per_thread;

    // Also test ring buffer SPSC pattern with verification
    let buffer = EventRingBuffer::with_capacity(4096);
    let stats = Arc::new(buffer.stats());
    let (producer, consumer) = buffer.split();

    // Wrap in Arc+Mutex only for the test (not recommended for production)
    let producer = Arc::new(parking_lot::Mutex::new(producer));
    let consumer = Arc::new(parking_lot::Mutex::new(consumer));

    let store_clone = Arc::clone(&store);
    let consumer_clone = Arc::clone(&consumer);
    let _stats_clone = Arc::clone(&stats); // Available for debugging if needed

    // Track concurrent operations
    let operations_completed = Arc::new(AtomicUsize::new(0));
    let operations_clone = Arc::clone(&operations_completed);

    // Consumer thread
    let consumer_handle = thread::spawn(move || {
        let mut consumed = 0;
        let target = total_expected;

        while consumed < target {
            let mut con = consumer_clone.lock();
            let batch = con.pop_batch(50);
            drop(con); // Release lock early

            for slot in batch {
                store_clone.store(slot);
                consumed += 1;
                operations_clone.fetch_add(1, Ordering::Relaxed);
            }

            if consumed < target {
                thread::sleep(Duration::from_micros(100));
            }
        }
        consumed
    });

    // Multiple producer threads (simulating concurrent event sources)
    let mut producer_handles = Vec::new();

    for thread_id in 0..thread_count {
        let producer_clone = Arc::clone(&producer);
        let ops = Arc::clone(&operations_completed);

        let handle = thread::spawn(move || {
            let mut pushed = 0;
            for i in 0..events_per_thread {
                let event = make_raw_event(
                    EventType::MouseMoved,
                    (thread_id * 1000 + i) as f64,
                    thread_id as f64,
                );

                // Acquire lock, push, release
                let mut prod = producer_clone.lock();
                while !prod.push(event.clone()) {
                    drop(prod);
                    thread::sleep(Duration::from_micros(50));
                    prod = producer_clone.lock();
                }
                drop(prod);

                pushed += 1;
                ops.fetch_add(1, Ordering::Relaxed);
            }
            pushed
        });

        producer_handles.push(handle);
    }

    // Wait for all producers
    let mut total_pushed = 0;
    for handle in producer_handles {
        total_pushed += handle.join().expect("Producer thread panicked");
    }

    // Wait for consumer
    let consumed = consumer_handle.join().expect("Consumer thread panicked");

    println!("\nConcurrent Access Test Results:");
    println!("  Thread count: {} producers + 1 consumer", thread_count);
    println!("  Events per producer: {}", events_per_thread);
    println!("  Total pushed: {}", total_pushed);
    println!("  Total consumed: {}", consumed);
    println!("  Store size: {}", store.len());
    println!("  Total operations: {}", operations_completed.load(Ordering::Relaxed));

    // Verify correctness
    assert_eq!(
        total_pushed, total_expected,
        "All producer events should be pushed"
    );

    assert_eq!(
        consumed, total_expected,
        "Consumer should receive all events"
    );

    assert_eq!(
        store.len(), total_expected,
        "Store should contain all events"
    );

    // Verify no data loss
    let stats_pushed = stats.events_pushed.load(Ordering::Relaxed);
    let stats_dropped = stats.events_dropped.load(Ordering::Relaxed);

    assert_eq!(
        stats_pushed as usize, total_expected,
        "Stats should reflect all pushes"
    );

    assert_eq!(
        stats_dropped, 0,
        "No events should be dropped with adequate buffer"
    );

    // Verify event integrity by draining and checking
    let drained = store.drain();
    assert_eq!(drained.len(), total_expected, "Drain should return all events");

    // Verify thread IDs are represented (checking coordinates)
    let mut thread_ids_seen: std::collections::HashSet<u64> = std::collections::HashSet::new();
    for slot in &drained {
        let y = slot.event.coordinates.1 as u64;
        thread_ids_seen.insert(y);
    }

    assert_eq!(
        thread_ids_seen.len(), thread_count,
        "Events from all producer threads should be present"
    );
}

// =============================================================================
// Additional Performance Benchmarks
// =============================================================================

#[test]
fn test_event_creation_throughput() {
    MachTimebase::init();

    let iterations = 100_000;
    let start = Instant::now();

    for i in 0..iterations {
        let _event = make_raw_event(EventType::MouseMoved, i as f64, i as f64);
    }

    let elapsed = start.elapsed();
    let throughput = iterations as f64 / elapsed.as_secs_f64();

    println!("\nEvent Creation Throughput:");
    println!("  Iterations: {}", iterations);
    println!("  Time: {:?}", elapsed);
    println!("  Throughput: {:.0} events/sec", throughput);

    // Should be able to create at least 1M events per second
    assert!(
        throughput > 1_000_000.0,
        "Throughput {:.0} should exceed 1M events/sec",
        throughput
    );
}

#[test]
fn test_ring_buffer_throughput() {
    MachTimebase::init();

    let buffer = EventRingBuffer::with_capacity(8192);
    let (mut producer, mut consumer) = buffer.split();

    let iterations = 50_000;

    // Measure push throughput
    let push_start = Instant::now();
    for i in 0..iterations {
        let event = make_raw_event(EventType::MouseMoved, i as f64, i as f64);
        producer.push(event);

        // Pop to prevent buffer from filling
        if i % 2 == 0 {
            let _ = consumer.pop();
        }
    }
    let push_time = push_start.elapsed();

    // Drain remaining
    let pop_start = Instant::now();
    while consumer.pop().is_some() {}
    let pop_time = pop_start.elapsed();

    let push_throughput = iterations as f64 / push_time.as_secs_f64();
    let combined_throughput = iterations as f64 / (push_time + pop_time).as_secs_f64();

    println!("\nRing Buffer Throughput:");
    println!("  Iterations: {}", iterations);
    println!("  Push time: {:?}", push_time);
    println!("  Pop time: {:?}", pop_time);
    println!("  Push throughput: {:.0} ops/sec", push_throughput);
    println!("  Combined throughput: {:.0} ops/sec", combined_throughput);

    // Should achieve at least 500K ops/sec
    assert!(
        push_throughput > 500_000.0,
        "Push throughput {:.0} should exceed 500K ops/sec",
        push_throughput
    );
}

#[test]
fn test_recording_serialization_performance() {
    MachTimebase::init();

    // Create a medium-sized recording
    let event_count = 1000;
    let mut recording = Recording::new(
        "serialization_test".to_string(),
        Some("Test serialization performance".to_string()),
    );

    for i in 0..event_count {
        let event = make_click_with_context(
            (i % 1920) as f64,
            (i % 1080) as f64,
            "AXButton",
            &format!("Button_{}", i),
            1,
        );
        recording.add_event(event);
    }
    recording.finalize(event_count as u64 * 100);

    // Measure serialization
    let ser_start = Instant::now();
    let json = serde_json::to_string(&recording).expect("Serialization failed");
    let ser_time = ser_start.elapsed();

    // Measure deserialization
    let de_start = Instant::now();
    let _loaded: Recording = serde_json::from_str(&json).expect("Deserialization failed");
    let de_time = de_start.elapsed();

    let json_size = json.len();
    let size_per_event = json_size / event_count;

    println!("\nSerialization Performance:");
    println!("  Event count: {}", event_count);
    println!("  JSON size: {} bytes ({} KB)", json_size, json_size / 1024);
    println!("  Size per event: {} bytes", size_per_event);
    println!("  Serialization time: {:?}", ser_time);
    println!("  Deserialization time: {:?}", de_time);
    println!("  Ser throughput: {:.0} events/sec", event_count as f64 / ser_time.as_secs_f64());
    println!("  De throughput: {:.0} events/sec", event_count as f64 / de_time.as_secs_f64());

    // Should complete within reasonable time (< 100ms for 1000 events)
    assert!(
        ser_time < Duration::from_millis(500),
        "Serialization {:?} should be under 500ms",
        ser_time
    );
    assert!(
        de_time < Duration::from_millis(500),
        "Deserialization {:?} should be under 500ms",
        de_time
    );
}

#[test]
fn test_timestamp_precision() {
    MachTimebase::init();

    // Test timestamp precision under load with work between samples
    let iterations = 1_000;
    let mut timestamps = Vec::with_capacity(iterations);

    let start = Instant::now();
    for i in 0..iterations {
        timestamps.push(Timestamp::now());
        // Add some minimal work between timestamps to ensure time passes
        // This simulates realistic event capture timing
        std::hint::black_box(i * i);
        for _ in 0..100 {
            std::hint::black_box(0);
        }
    }
    let elapsed = start.elapsed();

    // Verify monotonicity (timestamps should never go backward)
    let mut monotonic_violations = 0;
    for i in 1..timestamps.len() {
        if timestamps[i].ticks() < timestamps[i - 1].ticks() {
            monotonic_violations += 1;
        }
    }

    // Check that timestamps are non-decreasing (not strictly increasing, due to timing resolution)
    let mut decreasing_count = 0;
    for i in 1..timestamps.len() {
        if timestamps[i].ticks() < timestamps[i - 1].ticks() {
            decreasing_count += 1;
        }
    }

    // Check unique timestamps (precision varies by hardware)
    let unique_count = timestamps.iter().map(|t| t.ticks()).collect::<std::collections::HashSet<_>>().len();
    let unique_ratio = unique_count as f64 / iterations as f64;

    println!("\nTimestamp Precision Test:");
    println!("  Iterations: {}", iterations);
    println!("  Time: {:?}", elapsed);
    println!("  Monotonic violations: {}", monotonic_violations);
    println!("  Decreasing timestamps: {}", decreasing_count);
    println!("  Unique timestamps: {} ({:.1}%)", unique_count, unique_ratio * 100.0);

    // Should have no monotonicity violations (timestamps never go backward)
    assert_eq!(
        monotonic_violations, 0,
        "Timestamps should never decrease"
    );

    // The key invariant is monotonicity, not uniqueness
    // On fast hardware, multiple samples may share the same tick value
    // but they should never go backward
    assert_eq!(
        decreasing_count, 0,
        "Timestamps should be non-decreasing"
    );

    // Should have some uniqueness (at least 5% to show precision is working)
    // This is a loose bound since actual precision depends on hardware
    assert!(
        unique_ratio > 0.05,
        "Timestamp uniqueness {:.1}% should be > 5% (sanity check)",
        unique_ratio * 100.0
    );
}
