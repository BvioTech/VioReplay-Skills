//! Integration tests for the capture pipeline
//!
//! These tests verify the complete event capture pipeline:
//! Event generation -> Ring buffer -> Semantic backfill -> Processed store

use skill_generator::capture::ring_buffer::{
    EventRingBuffer, ProcessedEventStore, RingBufferStats,
};
use skill_generator::capture::types::{
    CursorState, EnrichedEvent, EventType, ModifierFlags, RawEvent, SemanticContext,
    SemanticSource, SemanticState,
};
use skill_generator::time::timebase::{MachTimebase, Timestamp};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

/// Create a test raw event
fn make_test_event(event_type: EventType, x: f64, y: f64) -> RawEvent {
    MachTimebase::init();
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

/// Create a test event with click count
fn make_click_event(x: f64, y: f64, click_count: u8) -> RawEvent {
    MachTimebase::init();
    RawEvent {
        timestamp: Timestamp::now(),
        event_type: EventType::LeftMouseDown,
        coordinates: (x, y),
        cursor_state: CursorState::PointingHand,
        key_code: None,
        character: None,
        modifiers: ModifierFlags::default(),
        scroll_delta: None,
        click_count,
    }
}

/// Create a test keyboard event
fn make_key_event(key_code: u16, character: char, modifiers: ModifierFlags) -> RawEvent {
    MachTimebase::init();
    RawEvent {
        timestamp: Timestamp::now(),
        event_type: EventType::KeyDown,
        coordinates: (0.0, 0.0),
        cursor_state: CursorState::IBeam,
        key_code: Some(key_code),
        character: Some(character),
        modifiers,
        scroll_delta: None,
        click_count: 0,
    }
}

#[test]
fn test_ring_buffer_spsc_flow() {
    MachTimebase::init();
    let buffer = EventRingBuffer::with_capacity(64);
    let stats = buffer.stats();
    let (mut producer, mut consumer) = buffer.split();

    // Producer pushes events
    for i in 0..10 {
        let event = make_test_event(EventType::MouseMoved, i as f64 * 10.0, i as f64 * 20.0);
        assert!(producer.push(event));
    }

    // Verify stats
    assert_eq!(stats.events_pushed.load(Ordering::Relaxed), 10);
    assert_eq!(stats.events_dropped.load(Ordering::Relaxed), 0);

    // Consumer pops events
    let batch = consumer.pop_batch(10);
    assert_eq!(batch.len(), 10);

    // Verify sequence numbers are in order
    for (i, slot) in batch.iter().enumerate() {
        assert_eq!(slot.sequence, i as u64);
        assert_eq!(slot.event.coordinates.0, i as f64 * 10.0);
    }

    // Verify consumed stats
    assert_eq!(stats.events_consumed.load(Ordering::Relaxed), 10);
}

#[test]
fn test_concurrent_producer_consumer() {
    MachTimebase::init();
    let buffer = EventRingBuffer::with_capacity(256);
    let stats = buffer.stats();
    let (mut producer, mut consumer) = buffer.split();

    let event_count = 100;
    let consumed_events = Arc::new(std::sync::Mutex::new(Vec::new()));
    let consumed_events_clone = Arc::clone(&consumed_events);

    // Spawn consumer thread
    let consumer_handle = thread::spawn(move || {
        let mut total_consumed = 0;
        while total_consumed < event_count {
            let batch = consumer.pop_batch(10);
            for slot in batch {
                consumed_events_clone.lock().unwrap().push(slot.sequence);
                total_consumed += 1;
            }
            if total_consumed < event_count {
                thread::sleep(Duration::from_micros(100));
            }
        }
    });

    // Producer pushes events
    for i in 0..event_count {
        let event = make_test_event(EventType::MouseMoved, i as f64, 0.0);
        while !producer.push(event.clone()) {
            thread::sleep(Duration::from_micros(50));
        }
    }

    // Wait for consumer to finish
    consumer_handle.join().unwrap();

    // Verify all events were consumed in order
    let consumed = consumed_events.lock().unwrap();
    assert_eq!(consumed.len(), event_count);
    for (i, &seq) in consumed.iter().enumerate() {
        assert_eq!(seq, i as u64);
    }
}

#[test]
fn test_buffer_overflow_handling() {
    MachTimebase::init();
    let buffer = EventRingBuffer::with_capacity(8);
    let stats = buffer.stats();
    let (mut producer, _consumer) = buffer.split();

    // Fill the buffer completely
    for _ in 0..8 {
        let event = make_test_event(EventType::MouseMoved, 0.0, 0.0);
        assert!(producer.push(event));
    }

    // Buffer should be full
    assert!(producer.is_full());

    // Try to push more - should fail
    for _ in 0..5 {
        let event = make_test_event(EventType::MouseMoved, 0.0, 0.0);
        assert!(!producer.push(event));
    }

    // Verify drop stats
    assert_eq!(stats.events_pushed.load(Ordering::Relaxed), 8);
    assert_eq!(stats.events_dropped.load(Ordering::Relaxed), 5);
}

#[test]
fn test_processed_event_store() {
    MachTimebase::init();
    let store = ProcessedEventStore::new(100);
    let buffer = EventRingBuffer::with_capacity(64);
    let (mut producer, mut consumer) = buffer.split();

    // Push and pop events
    for i in 0..10 {
        let event = make_test_event(EventType::LeftMouseDown, i as f64 * 50.0, i as f64 * 50.0);
        producer.push(event);
    }

    // Process and store
    let batch = consumer.pop_batch(10);
    for slot in batch {
        store.store(slot);
    }

    assert_eq!(store.len(), 10);

    // Drain and verify
    let events = store.drain();
    assert_eq!(events.len(), 10);
    assert!(store.is_empty());
}

#[test]
fn test_semantic_state_transitions() {
    MachTimebase::init();
    let buffer = EventRingBuffer::with_capacity(64);
    let (mut producer, mut consumer) = buffer.split();

    let event = make_click_event(100.0, 200.0, 1);
    producer.push(event);

    let slot = consumer.pop().expect("Should have event");

    // Initial state should be Pending
    assert_eq!(
        slot.semantic_state.load(Ordering::Acquire),
        SemanticState::Pending
    );

    // Mark as failed
    slot.mark_failed();
    assert_eq!(
        slot.semantic_state.load(Ordering::Acquire),
        SemanticState::Failed
    );
}

#[test]
fn test_event_type_categories() {
    // Test mouse movement events
    assert!(EventType::MouseMoved.is_mouse_move());
    assert!(EventType::LeftMouseDragged.is_mouse_move());
    assert!(EventType::RightMouseDragged.is_mouse_move());
    assert!(EventType::OtherMouseDragged.is_mouse_move());
    assert!(!EventType::LeftMouseDown.is_mouse_move());

    // Test click events
    assert!(EventType::LeftMouseDown.is_click());
    assert!(EventType::LeftMouseUp.is_click());
    assert!(EventType::RightMouseDown.is_click());
    assert!(EventType::OtherMouseUp.is_click());
    assert!(!EventType::MouseMoved.is_click());

    // Test keyboard events
    assert!(EventType::KeyDown.is_keyboard());
    assert!(EventType::KeyUp.is_keyboard());
    assert!(EventType::FlagsChanged.is_keyboard());
    assert!(!EventType::LeftMouseDown.is_keyboard());
}

#[test]
fn test_modifier_flags() {
    let mut mods = ModifierFlags::default();
    assert!(!mods.shift);
    assert!(!mods.command);

    mods.shift = true;
    mods.command = true;
    mods.option = true;

    assert!(mods.shift);
    assert!(mods.command);
    assert!(mods.option);
    assert!(!mods.control);
}

#[test]
fn test_cursor_state_context() {
    // Test text input context
    assert!(CursorState::IBeam.is_text_input());
    assert!(!CursorState::Arrow.is_text_input());

    // Test clickable context
    assert!(CursorState::PointingHand.is_clickable());
    assert!(!CursorState::Arrow.is_clickable());

    // Test busy context
    assert!(CursorState::Wait.is_busy());
    assert!(CursorState::Progress.is_busy());
    assert!(!CursorState::Arrow.is_busy());
}

#[test]
fn test_semantic_context_default() {
    let ctx = SemanticContext::default();
    assert!(ctx.ax_role.is_none());
    assert!(ctx.title.is_none());
    assert_eq!(ctx.confidence, 1.0);
    assert_eq!(ctx.source, SemanticSource::Accessibility);
}

#[test]
fn test_enriched_event_creation() {
    MachTimebase::init();
    let raw = make_click_event(100.0, 200.0, 2);
    let mut enriched = EnrichedEvent::new(raw.clone(), 0);
    enriched.semantic = Some(SemanticContext {
        ax_role: Some("AXButton".to_string()),
        title: Some("Submit".to_string()),
        ..Default::default()
    });

    assert_eq!(enriched.raw.event_type, EventType::LeftMouseDown);
    assert_eq!(enriched.raw.click_count, 2);
    assert!(enriched.semantic.is_some());
    assert_eq!(
        enriched.semantic.as_ref().unwrap().ax_role,
        Some("AXButton".to_string())
    );
}

#[test]
fn test_high_frequency_event_capture() {
    MachTimebase::init();
    let buffer = EventRingBuffer::with_capacity(4096);
    let stats = buffer.stats();
    let (mut producer, mut consumer) = buffer.split();

    // Simulate high-frequency mouse movement (1000 events)
    let event_count = 1000;

    for i in 0..event_count {
        let x = (i as f64).sin() * 100.0 + 500.0;
        let y = (i as f64).cos() * 100.0 + 500.0;
        let event = make_test_event(EventType::MouseMoved, x, y);
        producer.push(event);
    }

    // Verify all events were captured
    assert_eq!(
        stats.events_pushed.load(Ordering::Relaxed),
        event_count as u64
    );
    assert_eq!(stats.events_dropped.load(Ordering::Relaxed), 0);

    // Consume all events
    let mut consumed = 0;
    while let Some(_) = consumer.pop() {
        consumed += 1;
    }
    assert_eq!(consumed, event_count);
}

#[test]
fn test_timestamp_ordering() {
    MachTimebase::init();
    let buffer = EventRingBuffer::with_capacity(64);
    let (mut producer, mut consumer) = buffer.split();

    // Push events with small delays to ensure different timestamps
    let mut timestamps = Vec::new();
    for i in 0..5 {
        let event = make_test_event(EventType::MouseMoved, i as f64, 0.0);
        timestamps.push(event.timestamp.ticks());
        producer.push(event);
        thread::sleep(Duration::from_micros(10));
    }

    // Verify timestamps are monotonically increasing
    let batch = consumer.pop_batch(5);
    for (i, slot) in batch.iter().enumerate() {
        if i > 0 {
            assert!(
                slot.event.timestamp.ticks() >= batch[i - 1].event.timestamp.ticks(),
                "Timestamps should be monotonically increasing"
            );
        }
    }
}
