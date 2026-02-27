//! Lock-Free Ring Buffer for Event Capture
//!
//! This module implements a high-performance, lock-free SPSC (Single Producer,
//! Single Consumer) ring buffer optimized for the event capture pipeline.
//!
//! Architecture:
//! - Producer (Event Tap callback): Never blocks, pushes events at up to 1000 Hz
//! - Consumer (Semantic backfill thread): Can block, processes events asynchronously
//!
//! The design uses the `rtrb` crate for the core ring buffer implementation,
//! with additional event slot management for semantic backfill.

use super::types::{AtomicSemanticState, RawEvent, SemanticContext, SemanticState};
use parking_lot::RwLock;
use rtrb::{Consumer, Producer, RingBuffer};
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Default ring buffer capacity (must be power of 2)
pub const DEFAULT_CAPACITY: usize = 8192;

/// Event slot in the ring buffer with semantic backfill support
///
/// This struct is cache-line aligned (64 bytes) to prevent false sharing
/// between concurrent readers and writers.
#[repr(align(64))]
pub struct EventSlot {
    /// The raw event data
    pub event: RawEvent,
    /// Semantic state (atomic for lock-free access)
    pub semantic_state: AtomicSemanticState,
    /// Semantic data (written by consumer thread only)
    semantic_data: UnsafeCell<Option<SemanticContext>>,
    /// Sequence number for ordering verification
    pub sequence: u64,
}

// Safety: EventSlot is Send because:
// - RawEvent is Send
// - AtomicSemanticState is atomic
// - semantic_data is only written by consumer after state check
unsafe impl Send for EventSlot {}

// Safety: EventSlot is Sync because:
// - All fields are either atomic or protected by state machine
unsafe impl Sync for EventSlot {}

impl EventSlot {
    /// Create a new event slot
    pub fn new(event: RawEvent, sequence: u64) -> Self {
        Self {
            event,
            semantic_state: AtomicSemanticState::new(SemanticState::Pending),
            semantic_data: UnsafeCell::new(None),
            sequence,
        }
    }

    /// Get semantic data reference (only valid when state is Filled)
    ///
    /// # Safety
    /// Caller must ensure state is Filled before calling
    pub fn get_semantic(&self) -> Option<&SemanticContext> {
        if self.semantic_state.load(Ordering::Acquire) == SemanticState::Filled {
            // Safety: We only read after verifying state is Filled,
            // and only the consumer writes semantic_data
            unsafe { (*self.semantic_data.get()).as_ref() }
        } else {
            None
        }
    }

    /// Set semantic data (consumer thread only)
    ///
    /// # Safety
    /// Must only be called from consumer thread while state is Pending
    pub unsafe fn set_semantic(&self, context: SemanticContext) {
        *self.semantic_data.get() = Some(context);
        self.semantic_state
            .store(SemanticState::Filled, Ordering::Release);
    }

    /// Mark semantic lookup as failed
    pub fn mark_failed(&self) {
        self.semantic_state
            .store(SemanticState::Failed, Ordering::Release);
    }
}

/// Lock-free ring buffer for events
///
/// This is the core data structure connecting the event tap (producer)
/// to the semantic backfill system (consumer).
pub struct EventRingBuffer {
    /// The underlying rtrb producer (for event tap thread)
    producer: Option<Producer<EventSlot>>,
    /// The underlying rtrb consumer (for backfill thread)
    consumer: Option<Consumer<EventSlot>>,
    /// Sequence counter for event ordering
    sequence: AtomicU64,
    /// Statistics
    stats: Arc<RingBufferStats>,
    /// Capacity
    capacity: usize,
}

/// Ring buffer statistics for monitoring
#[derive(Debug, Default)]
pub struct RingBufferStats {
    /// Total events pushed
    pub events_pushed: AtomicU64,
    /// Events dropped due to full buffer
    pub events_dropped: AtomicU64,
    /// Events successfully consumed
    pub events_consumed: AtomicU64,
    /// Peak buffer occupancy
    pub peak_occupancy: AtomicU64,
}

impl EventRingBuffer {
    /// Create a new ring buffer with default capacity
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_CAPACITY)
    }

    /// Create a new ring buffer with specified capacity
    ///
    /// # Panics
    /// Panics if capacity is not a power of 2
    pub fn with_capacity(capacity: usize) -> Self {
        assert!(
            capacity.is_power_of_two(),
            "Ring buffer capacity must be a power of 2"
        );

        let (producer, consumer) = RingBuffer::new(capacity);

        Self {
            producer: Some(producer),
            consumer: Some(consumer),
            sequence: AtomicU64::new(0),
            stats: Arc::new(RingBufferStats::default()),
            capacity,
        }
    }

    /// Split the ring buffer into producer and consumer halves
    ///
    /// This must be called once to separate the producer (for event tap)
    /// from the consumer (for semantic backfill).
    pub fn split(mut self) -> (EventProducer, EventConsumer) {
        let producer = self.producer.take().expect("Producer already taken");
        let consumer = self.consumer.take().expect("Consumer already taken");

        (
            EventProducer {
                inner: producer,
                sequence: Arc::new(self.sequence),
                stats: Arc::clone(&self.stats),
                capacity: self.capacity,
            },
            EventConsumer {
                inner: consumer,
                stats: Arc::clone(&self.stats),
            },
        )
    }

    /// Get statistics
    pub fn stats(&self) -> Arc<RingBufferStats> {
        Arc::clone(&self.stats)
    }
}

impl Default for EventRingBuffer {
    fn default() -> Self {
        Self::new()
    }
}

/// Producer half of the ring buffer (for event tap thread)
pub struct EventProducer {
    inner: Producer<EventSlot>,
    sequence: Arc<AtomicU64>,
    stats: Arc<RingBufferStats>,
    capacity: usize,
}

impl EventProducer {
    /// Push an event into the ring buffer.
    ///
    /// This method is lock-free and will never block. If the buffer is full,
    /// the event is dropped and the drop counter is incremented.
    ///
    /// Returns true if the event was successfully pushed, false if dropped.
    #[inline]
    pub fn push(&mut self, event: RawEvent) -> bool {
        let sequence = self.sequence.fetch_add(1, Ordering::Relaxed);
        let slot = EventSlot::new(event, sequence);

        match self.inner.push(slot) {
            Ok(()) => {
                self.stats.events_pushed.fetch_add(1, Ordering::Relaxed);

                // Update peak occupancy
                let current = self.inner.slots();
                let occupied = self.capacity - current;
                let mut peak = self.stats.peak_occupancy.load(Ordering::Relaxed);
                while occupied as u64 > peak {
                    match self.stats.peak_occupancy.compare_exchange_weak(
                        peak,
                        occupied as u64,
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    ) {
                        Ok(_) => break,
                        Err(p) => peak = p,
                    }
                }

                true
            }
            Err(_) => {
                self.stats.events_dropped.fetch_add(1, Ordering::Relaxed);
                // Roll back sequence number on failure (consistent with try_push)
                self.sequence.fetch_sub(1, Ordering::Relaxed);
                false
            }
        }
    }

    /// Try to push an event, returning the slot for inspection if successful
    #[inline]
    pub fn try_push(&mut self, event: RawEvent) -> Result<u64, RawEvent> {
        let sequence = self.sequence.fetch_add(1, Ordering::Relaxed);
        let slot = EventSlot::new(event.clone(), sequence);

        match self.inner.push(slot) {
            Ok(()) => {
                self.stats.events_pushed.fetch_add(1, Ordering::Relaxed);
                Ok(sequence)
            }
            Err(_) => {
                self.stats.events_dropped.fetch_add(1, Ordering::Relaxed);
                // Roll back sequence number on failure
                self.sequence.fetch_sub(1, Ordering::Relaxed);
                Err(event)
            }
        }
    }

    /// Check available slots without pushing
    #[inline]
    pub fn available_slots(&self) -> usize {
        self.inner.slots()
    }

    /// Check if buffer is full
    #[inline]
    pub fn is_full(&self) -> bool {
        self.inner.is_full()
    }

    /// Get current sequence number
    #[inline]
    pub fn sequence(&self) -> u64 {
        self.sequence.load(Ordering::Relaxed)
    }
}

// Safety: EventProducer is Send because it only accesses its own data
unsafe impl Send for EventProducer {}

/// Consumer half of the ring buffer (for semantic backfill thread)
pub struct EventConsumer {
    inner: Consumer<EventSlot>,
    stats: Arc<RingBufferStats>,
}

impl EventConsumer {
    /// Pop an event from the ring buffer.
    ///
    /// This method may block if used with condition variables,
    /// but the pop itself is lock-free.
    #[inline]
    pub fn pop(&mut self) -> Option<EventSlot> {
        match self.inner.pop() {
            Ok(slot) => {
                self.stats.events_consumed.fetch_add(1, Ordering::Relaxed);
                Some(slot)
            }
            Err(_) => None,
        }
    }

    /// Try to peek at the next event without removing it
    #[inline]
    pub fn peek(&self) -> Option<&EventSlot> {
        self.inner.peek().ok()
    }

    /// Check if there are events available
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Get number of available events
    #[inline]
    pub fn available(&self) -> usize {
        self.inner.slots()
    }

    /// Pop multiple events at once (batch processing)
    pub fn pop_batch(&mut self, max_count: usize) -> Vec<EventSlot> {
        let mut batch = Vec::with_capacity(max_count);
        for _ in 0..max_count {
            if let Some(slot) = self.pop() {
                batch.push(slot);
            } else {
                break;
            }
        }
        batch
    }
}

// Safety: EventConsumer is Send because it only accesses its own data
unsafe impl Send for EventConsumer {}

/// Thread-safe storage for processed events
///
/// Events are moved here after semantic backfill is complete.
/// This allows the ring buffer to be drained continuously.
pub struct ProcessedEventStore {
    events: RwLock<Vec<EventSlot>>,
    capacity: usize,
}

impl ProcessedEventStore {
    /// Create a new processed event store
    pub fn new(capacity: usize) -> Self {
        Self {
            events: RwLock::new(Vec::with_capacity(capacity)),
            capacity,
        }
    }

    /// Store a processed event
    pub fn store(&self, slot: EventSlot) -> bool {
        let mut events = self.events.write();
        if events.len() < self.capacity {
            events.push(slot);
            true
        } else {
            false
        }
    }

    /// Get all stored events
    pub fn drain(&self) -> Vec<EventSlot> {
        let mut events = self.events.write();
        std::mem::take(&mut *events)
    }

    /// Get event count
    pub fn len(&self) -> usize {
        self.events.read().len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.events.read().is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time::timebase::{MachTimebase, Timestamp};
    use crate::capture::types::{CursorState, EventType, ModifierFlags};

    fn make_test_event() -> RawEvent {
        RawEvent::mouse(
            Timestamp::from_ticks(1000),
            EventType::MouseMoved,
            100.0,
            200.0,
            CursorState::Arrow,
            ModifierFlags::default(),
            0,
        )
    }

    #[test]
    fn test_ring_buffer_creation() {
        let buffer = EventRingBuffer::new();
        assert_eq!(buffer.capacity, DEFAULT_CAPACITY);
    }

    #[test]
    fn test_ring_buffer_split() {
        let buffer = EventRingBuffer::with_capacity(64);
        let (producer, consumer) = buffer.split();

        assert!(!producer.is_full());
        assert!(consumer.is_empty());
    }

    #[test]
    fn test_push_and_pop() {
        MachTimebase::init();
        let buffer = EventRingBuffer::with_capacity(64);
        let (mut producer, mut consumer) = buffer.split();

        // Push an event
        let event = make_test_event();
        assert!(producer.push(event));

        // Pop the event
        let slot = consumer.pop().expect("Should have event");
        assert_eq!(slot.event.event_type, EventType::MouseMoved);
        assert_eq!(slot.sequence, 0);
    }

    #[test]
    fn test_buffer_full() {
        MachTimebase::init();
        let buffer = EventRingBuffer::with_capacity(4);
        let (mut producer, _consumer) = buffer.split();

        // Fill the buffer
        for _ in 0..4 {
            assert!(producer.push(make_test_event()));
        }

        // Buffer should be full now
        assert!(producer.is_full());
        assert!(!producer.push(make_test_event())); // Should fail
    }

    #[test]
    fn test_sequence_numbers() {
        MachTimebase::init();
        let buffer = EventRingBuffer::with_capacity(64);
        let (mut producer, mut consumer) = buffer.split();

        // Push multiple events
        for _ in 0..10 {
            producer.push(make_test_event());
        }

        // Verify sequence numbers
        for i in 0..10 {
            let slot = consumer.pop().expect("Should have event");
            assert_eq!(slot.sequence, i);
        }
    }

    #[test]
    fn test_statistics() {
        MachTimebase::init();
        let buffer = EventRingBuffer::with_capacity(4);
        let stats = buffer.stats();
        let (mut producer, mut consumer) = buffer.split();

        // Push events
        for _ in 0..6 {
            producer.push(make_test_event());
        }

        // Check stats
        assert_eq!(stats.events_pushed.load(Ordering::Relaxed), 4);
        assert_eq!(stats.events_dropped.load(Ordering::Relaxed), 2);

        // Pop events
        for _ in 0..4 {
            consumer.pop();
        }

        assert_eq!(stats.events_consumed.load(Ordering::Relaxed), 4);
    }

    #[test]
    fn test_batch_pop() {
        MachTimebase::init();
        let buffer = EventRingBuffer::with_capacity(64);
        let (mut producer, mut consumer) = buffer.split();

        // Push 10 events
        for _ in 0..10 {
            producer.push(make_test_event());
        }

        // Pop batch of 5
        let batch = consumer.pop_batch(5);
        assert_eq!(batch.len(), 5);

        // 5 should remain
        assert_eq!(consumer.available(), 5);
    }

    #[test]
    fn test_semantic_state_transition() {
        MachTimebase::init();
        let event = make_test_event();
        let slot = EventSlot::new(event, 0);

        // Initial state should be Pending
        assert_eq!(
            slot.semantic_state.load(Ordering::SeqCst),
            SemanticState::Pending
        );

        // Transition to Failed
        slot.mark_failed();
        assert_eq!(
            slot.semantic_state.load(Ordering::SeqCst),
            SemanticState::Failed
        );
    }

    #[test]
    fn test_processed_event_store() {
        MachTimebase::init();
        let store = ProcessedEventStore::new(100);

        let slot = EventSlot::new(make_test_event(), 0);
        assert!(store.store(slot));
        assert_eq!(store.len(), 1);

        let events = store.drain();
        assert_eq!(events.len(), 1);
        assert!(store.is_empty());
    }

    #[test]
    fn test_event_slot_semantic_lifecycle() {
        MachTimebase::init();
        let event = make_test_event();
        let slot = EventSlot::new(event, 42);

        // Initially pending
        assert_eq!(slot.semantic_state.load(Ordering::SeqCst), SemanticState::Pending);
        assert_eq!(slot.sequence, 42);

        // No semantic data initially
        assert!(slot.get_semantic().is_none());

        // Mark as failed
        slot.mark_failed();
        assert_eq!(slot.semantic_state.load(Ordering::SeqCst), SemanticState::Failed);
        assert!(slot.get_semantic().is_none());
    }

    #[test]
    fn test_event_slot_set_semantic() {
        MachTimebase::init();
        use crate::capture::types::SemanticSource;

        let event = make_test_event();
        let slot = EventSlot::new(event, 0);

        let context = SemanticContext {
            ax_role: Some("AXButton".to_string()),
            title: Some("Click Me".to_string()),
            source: SemanticSource::Accessibility,
            confidence: 1.0,
            ..Default::default()
        };

        // Set semantic data
        unsafe {
            slot.set_semantic(context);
        }

        assert_eq!(slot.semantic_state.load(Ordering::SeqCst), SemanticState::Filled);

        let retrieved = slot.get_semantic();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().ax_role, Some("AXButton".to_string()));
        assert_eq!(retrieved.unwrap().title, Some("Click Me".to_string()));
    }

    #[test]
    fn test_ring_buffer_capacity_validation() {
        MachTimebase::init();
        // Valid capacities (powers of 2)
        let _ = EventRingBuffer::with_capacity(64);
        let _ = EventRingBuffer::with_capacity(128);
        let _ = EventRingBuffer::with_capacity(256);
    }

    #[test]
    #[should_panic(expected = "Ring buffer capacity must be a power of 2")]
    fn test_ring_buffer_invalid_capacity() {
        let _ = EventRingBuffer::with_capacity(100); // Not a power of 2
    }

    #[test]
    fn test_producer_try_push() {
        MachTimebase::init();
        let buffer = EventRingBuffer::with_capacity(4);
        let (mut producer, _consumer) = buffer.split();

        let event = make_test_event();

        // Try push should succeed
        let result = producer.try_push(event.clone());
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0); // First sequence number

        // Try push again
        let result = producer.try_push(event.clone());
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 1); // Second sequence number
    }

    #[test]
    fn test_producer_try_push_full_buffer() {
        MachTimebase::init();
        let buffer = EventRingBuffer::with_capacity(4);
        let (mut producer, _consumer) = buffer.split();

        // Fill the buffer
        for _ in 0..4 {
            assert!(producer.try_push(make_test_event()).is_ok());
        }

        // Next push should fail
        let event = make_test_event();
        let result = producer.try_push(event.clone());
        assert!(result.is_err());

        // Event should be returned
        let returned_event = result.unwrap_err();
        assert_eq!(returned_event.event_type, event.event_type);
    }

    #[test]
    fn test_producer_available_slots() {
        MachTimebase::init();
        let buffer = EventRingBuffer::with_capacity(8);
        let (mut producer, _consumer) = buffer.split();

        assert_eq!(producer.available_slots(), 8);
        assert!(!producer.is_full());

        // Push some events
        for _ in 0..3 {
            producer.push(make_test_event());
        }

        assert_eq!(producer.available_slots(), 5);
        assert!(!producer.is_full());

        // Fill remaining slots
        for _ in 0..5 {
            producer.push(make_test_event());
        }

        assert!(producer.is_full());
    }

    #[test]
    fn test_consumer_peek() {
        MachTimebase::init();
        let buffer = EventRingBuffer::with_capacity(64);
        let (mut producer, consumer) = buffer.split();

        // Initially empty
        assert!(consumer.peek().is_none());

        // Push an event
        producer.push(make_test_event());

        // Peek should return the event without removing it
        let peeked = consumer.peek();
        assert!(peeked.is_some());
        assert_eq!(peeked.unwrap().sequence, 0);

        // Peek again - should still be there
        let peeked_again = consumer.peek();
        assert!(peeked_again.is_some());
        assert_eq!(peeked_again.unwrap().sequence, 0);
    }

    #[test]
    fn test_consumer_available() {
        MachTimebase::init();
        let buffer = EventRingBuffer::with_capacity(64);
        let (mut producer, consumer) = buffer.split();

        assert_eq!(consumer.available(), 0);
        assert!(consumer.is_empty());

        // Push events
        for _ in 0..5 {
            producer.push(make_test_event());
        }

        assert_eq!(consumer.available(), 5);
        assert!(!consumer.is_empty());
    }

    #[test]
    fn test_ring_buffer_peak_occupancy() {
        MachTimebase::init();
        let buffer = EventRingBuffer::with_capacity(16);
        let stats = buffer.stats();
        let (mut producer, mut consumer) = buffer.split();

        // Push 10 events
        for _ in 0..10 {
            producer.push(make_test_event());
        }

        let peak = stats.peak_occupancy.load(Ordering::Relaxed);
        assert!(peak >= 10);

        // Consume 5 events
        for _ in 0..5 {
            consumer.pop();
        }

        // Peak should remain at or above 10
        let peak_after = stats.peak_occupancy.load(Ordering::Relaxed);
        assert!(peak_after >= 10);

        // Push 8 more (total occupancy: 5 + 8 = 13)
        for _ in 0..8 {
            producer.push(make_test_event());
        }

        let peak_final = stats.peak_occupancy.load(Ordering::Relaxed);
        assert!(peak_final >= 13);
    }

    #[test]
    fn test_processed_event_store_capacity_limit() {
        MachTimebase::init();
        let store = ProcessedEventStore::new(5);

        // Store 5 events - should succeed
        for i in 0..5 {
            let slot = EventSlot::new(make_test_event(), i);
            assert!(store.store(slot));
        }

        assert_eq!(store.len(), 5);

        // Try to store one more - should fail
        let slot = EventSlot::new(make_test_event(), 999);
        assert!(!store.store(slot));
        assert_eq!(store.len(), 5);
    }

    #[test]
    fn test_processed_event_store_drain_multiple_times() {
        MachTimebase::init();
        let store = ProcessedEventStore::new(100);

        // Store some events
        for i in 0..3 {
            store.store(EventSlot::new(make_test_event(), i));
        }

        // First drain
        let events1 = store.drain();
        assert_eq!(events1.len(), 3);
        assert!(store.is_empty());

        // Second drain should be empty
        let events2 = store.drain();
        assert_eq!(events2.len(), 0);
        assert!(store.is_empty());
    }

    #[test]
    fn test_concurrent_producer_consumer_simulation() {
        use std::thread;

        MachTimebase::init();
        let buffer = EventRingBuffer::with_capacity(256);
        let stats = buffer.stats();
        let (mut producer, mut consumer) = buffer.split();

        // Spawn producer thread
        let producer_handle = thread::spawn(move || {
            for _ in 0..100 {
                producer.push(make_test_event());
                std::thread::sleep(std::time::Duration::from_micros(10));
            }
        });

        // Spawn consumer thread
        let consumer_handle = thread::spawn(move || {
            let mut consumed = 0;
            while consumed < 100 {
                if let Some(_slot) = consumer.pop() {
                    consumed += 1;
                }
                std::thread::sleep(std::time::Duration::from_micros(10));
            }
            consumed
        });

        // Wait for both threads
        producer_handle.join().unwrap();
        let consumed_count = consumer_handle.join().unwrap();

        assert_eq!(consumed_count, 100);
        assert_eq!(stats.events_pushed.load(Ordering::Relaxed), 100);
        assert_eq!(stats.events_consumed.load(Ordering::Relaxed), 100);
    }

    #[test]
    fn test_event_slot_alignment() {
        // Verify EventSlot is properly aligned to prevent false sharing
        use std::mem;
        assert_eq!(mem::align_of::<EventSlot>(), 64, "EventSlot should be cache-line aligned");
    }

    #[test]
    fn test_ring_buffer_default() {
        let buffer = EventRingBuffer::default();
        assert_eq!(buffer.capacity, DEFAULT_CAPACITY);
    }

    #[test]
    fn test_stats_default() {
        let stats = RingBufferStats::default();
        assert_eq!(stats.events_pushed.load(Ordering::Relaxed), 0);
        assert_eq!(stats.events_dropped.load(Ordering::Relaxed), 0);
        assert_eq!(stats.events_consumed.load(Ordering::Relaxed), 0);
        assert_eq!(stats.peak_occupancy.load(Ordering::Relaxed), 0);
    }
}
