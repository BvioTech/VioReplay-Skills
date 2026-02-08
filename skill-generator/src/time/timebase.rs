//! mach_absolute_time Bridge
//!
//! Provides microsecond-precision timing using macOS mach_absolute_time.
//! This is the only reliable way to get monotonic, high-resolution timestamps
//! that are consistent across Apple Silicon and Intel Macs.

use std::sync::OnceLock;

/// Global timebase info, initialized once at startup
static TIMEBASE_INFO: OnceLock<TimebaseInfo> = OnceLock::new();

/// Cached timebase conversion factors
#[derive(Debug, Clone, Copy)]
struct TimebaseInfo {
    numer: u32,
    denom: u32,
}

/// High-precision timebase using mach_absolute_time
///
/// This struct provides:
/// - Microsecond precision timestamps
/// - Monotonic guarantees (time never goes backward)
/// - Consistent behavior on M1/M2/Intel
/// - Zero-overhead in hot path (raw ticks stored, converted lazily)
#[derive(Debug, Clone, Copy)]
pub struct MachTimebase;

impl MachTimebase {
    /// Initialize the timebase. Call once at startup.
    /// This fetches mach_timebase_info() and caches the conversion factors.
    pub fn init() {
        TIMEBASE_INFO.get_or_init(|| {
            let mut info = mach2::mach_time::mach_timebase_info_data_t {
                numer: 0,
                denom: 0,
            };
            // Safety: mach_timebase_info is always safe to call
            unsafe {
                mach2::mach_time::mach_timebase_info(&mut info);
            }
            TimebaseInfo {
                numer: info.numer,
                denom: info.denom,
            }
        });
    }

    /// Get current mach_absolute_time ticks.
    /// This is the raw hardware counter value - extremely fast to obtain.
    #[inline(always)]
    pub fn now_ticks() -> u64 {
        // Safety: mach_absolute_time is always safe to call
        unsafe { mach2::mach_time::mach_absolute_time() }
    }

    /// Convert raw mach ticks to nanoseconds.
    ///
    /// Note: On Apple Silicon, numer/denom is typically 1/1 (ticks == nanos).
    /// On Intel, it varies based on CPU frequency.
    #[inline]
    pub fn ticks_to_nanos(ticks: u64) -> u64 {
        let info = TIMEBASE_INFO.get().expect("MachTimebase::init() not called");
        // Use u128 to prevent overflow on large tick counts
        ((ticks as u128 * info.numer as u128) / info.denom as u128) as u64
    }

    /// Convert raw mach ticks to microseconds.
    #[inline]
    pub fn ticks_to_micros(ticks: u64) -> u64 {
        Self::ticks_to_nanos(ticks) / 1_000
    }

    /// Convert raw mach ticks to milliseconds.
    #[inline]
    pub fn ticks_to_millis(ticks: u64) -> u64 {
        Self::ticks_to_nanos(ticks) / 1_000_000
    }

    /// Get current time in nanoseconds since boot.
    /// Prefer `now_ticks()` in hot paths and convert later.
    #[inline]
    pub fn now_nanos() -> u64 {
        Self::ticks_to_nanos(Self::now_ticks())
    }

    /// Get current time in microseconds since boot.
    #[inline]
    pub fn now_micros() -> u64 {
        Self::now_nanos() / 1_000
    }

    /// Calculate elapsed time between two tick values in nanoseconds.
    /// Returns 0 if end < start (clock wraparound, though unlikely).
    #[inline]
    pub fn elapsed_nanos(start_ticks: u64, end_ticks: u64) -> u64 {
        if end_ticks >= start_ticks {
            Self::ticks_to_nanos(end_ticks - start_ticks)
        } else {
            0
        }
    }

    /// Calculate elapsed time in microseconds.
    #[inline]
    pub fn elapsed_micros(start_ticks: u64, end_ticks: u64) -> u64 {
        Self::elapsed_nanos(start_ticks, end_ticks) / 1_000
    }

    /// Calculate elapsed time in milliseconds.
    #[inline]
    pub fn elapsed_millis(start_ticks: u64, end_ticks: u64) -> u64 {
        Self::elapsed_nanos(start_ticks, end_ticks) / 1_000_000
    }

    /// Get the timebase info for debugging/logging.
    pub fn get_timebase_info() -> (u32, u32) {
        let info = TIMEBASE_INFO.get().expect("MachTimebase::init() not called");
        (info.numer, info.denom)
    }

    /// Check if two tick values maintain monotonicity.
    /// Returns true if t2 >= t1.
    #[inline]
    pub fn is_monotonic(t1: u64, t2: u64) -> bool {
        t2 >= t1
    }
}

/// A timestamp wrapper that stores raw mach ticks.
/// Conversion to human-readable units is deferred until needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Default)]
pub struct Timestamp(u64);

impl Timestamp {
    /// Create a new timestamp from raw mach ticks.
    #[inline]
    pub const fn from_ticks(ticks: u64) -> Self {
        Self(ticks)
    }

    /// Alias for `from_ticks` for compatibility.
    #[inline]
    pub const fn from_raw(ticks: u64) -> Self {
        Self::from_ticks(ticks)
    }

    /// Capture current timestamp.
    #[inline]
    pub fn now() -> Self {
        Self(MachTimebase::now_ticks())
    }

    /// Get the raw tick value.
    #[inline]
    pub const fn ticks(&self) -> u64 {
        self.0
    }

    /// Convert to nanoseconds.
    #[inline]
    pub fn as_nanos(&self) -> u64 {
        MachTimebase::ticks_to_nanos(self.0)
    }

    /// Convert to microseconds.
    #[inline]
    pub fn as_micros(&self) -> u64 {
        MachTimebase::ticks_to_micros(self.0)
    }

    /// Convert to milliseconds.
    #[inline]
    pub fn as_millis(&self) -> u64 {
        MachTimebase::ticks_to_millis(self.0)
    }

    /// Calculate duration since another timestamp.
    #[inline]
    pub fn duration_since(&self, earlier: Timestamp) -> Duration {
        Duration::from_ticks(self.0.saturating_sub(earlier.0))
    }

    /// Check if this timestamp is after another.
    #[inline]
    pub fn is_after(&self, other: Timestamp) -> bool {
        self.0 > other.0
    }
}


impl serde::Serialize for Timestamp {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        // Serialize as raw ticks for maximum precision
        serializer.serialize_u64(self.0)
    }
}

impl<'de> serde::Deserialize<'de> for Timestamp {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let ticks = u64::deserialize(deserializer)?;
        Ok(Timestamp(ticks))
    }
}

/// A duration wrapper using raw mach ticks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Duration(u64);

impl Duration {
    /// Create a duration from raw ticks.
    #[inline]
    pub const fn from_ticks(ticks: u64) -> Self {
        Self(ticks)
    }

    /// Create a duration from nanoseconds.
    #[inline]
    pub fn from_nanos(nanos: u64) -> Self {
        let info = TIMEBASE_INFO.get().expect("MachTimebase::init() not called");
        let ticks = ((nanos as u128 * info.denom as u128) / info.numer as u128) as u64;
        Self(ticks)
    }

    /// Create a duration from microseconds.
    #[inline]
    pub fn from_micros(micros: u64) -> Self {
        Self::from_nanos(micros * 1_000)
    }

    /// Create a duration from milliseconds.
    #[inline]
    pub fn from_millis(millis: u64) -> Self {
        Self::from_nanos(millis * 1_000_000)
    }

    /// Get raw tick count.
    #[inline]
    pub const fn ticks(&self) -> u64 {
        self.0
    }

    /// Convert to nanoseconds.
    #[inline]
    pub fn as_nanos(&self) -> u64 {
        MachTimebase::ticks_to_nanos(self.0)
    }

    /// Convert to microseconds.
    #[inline]
    pub fn as_micros(&self) -> u64 {
        MachTimebase::ticks_to_micros(self.0)
    }

    /// Convert to milliseconds.
    #[inline]
    pub fn as_millis(&self) -> u64 {
        MachTimebase::ticks_to_millis(self.0)
    }

    /// Convert to seconds as f64.
    #[inline]
    pub fn as_secs_f64(&self) -> f64 {
        self.as_nanos() as f64 / 1_000_000_000.0
    }

    /// Zero duration.
    pub const ZERO: Duration = Duration(0);
}

impl std::ops::Add for Duration {
    type Output = Duration;

    fn add(self, rhs: Self) -> Self::Output {
        Duration(self.0.saturating_add(rhs.0))
    }
}

impl std::ops::Sub for Duration {
    type Output = Duration;

    fn sub(self, rhs: Self) -> Self::Output {
        Duration(self.0.saturating_sub(rhs.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timebase_init() {
        MachTimebase::init();
        let (numer, denom) = MachTimebase::get_timebase_info();
        assert!(numer > 0, "numerator should be positive");
        assert!(denom > 0, "denominator should be positive");
    }

    #[test]
    fn test_monotonicity() {
        MachTimebase::init();
        let t1 = MachTimebase::now_ticks();
        // Small busy loop to ensure time passes
        for _ in 0..1000 {
            std::hint::black_box(0);
        }
        let t2 = MachTimebase::now_ticks();
        assert!(
            MachTimebase::is_monotonic(t1, t2),
            "timestamps must be monotonic"
        );
        assert!(t2 > t1, "time must advance");
    }

    #[test]
    fn test_conversion_consistency() {
        MachTimebase::init();
        let ticks = MachTimebase::now_ticks();
        let nanos = MachTimebase::ticks_to_nanos(ticks);
        let micros = MachTimebase::ticks_to_micros(ticks);
        let millis = MachTimebase::ticks_to_millis(ticks);

        assert_eq!(micros, nanos / 1_000);
        assert_eq!(millis, nanos / 1_000_000);
    }

    #[test]
    fn test_timestamp_ordering() {
        MachTimebase::init();
        let t1 = Timestamp::now();
        std::thread::sleep(std::time::Duration::from_micros(100));
        let t2 = Timestamp::now();

        assert!(t2.is_after(t1));
        assert!(t2 > t1);

        let duration = t2.duration_since(t1);
        assert!(duration.as_micros() >= 100);
    }

    #[test]
    fn test_duration_arithmetic() {
        MachTimebase::init();
        let d1 = Duration::from_millis(100);
        let d2 = Duration::from_millis(50);

        let sum = d1 + d2;
        let diff = d1 - d2;

        assert_eq!(sum.as_millis(), 150);
        assert_eq!(diff.as_millis(), 50);
    }

    #[test]
    fn test_elapsed_calculation() {
        MachTimebase::init();
        let start = MachTimebase::now_ticks();
        std::thread::sleep(std::time::Duration::from_millis(10));
        let end = MachTimebase::now_ticks();

        let elapsed = MachTimebase::elapsed_millis(start, end);
        assert!(elapsed >= 10, "elapsed should be at least 10ms");
        assert!(elapsed < 100, "elapsed should be less than 100ms");
    }

    #[test]
    fn test_elapsed_with_wraparound() {
        MachTimebase::init();
        // Test when end < start (unlikely but handled)
        let start_ticks = 1000;
        let end_ticks = 500;

        let elapsed = MachTimebase::elapsed_nanos(start_ticks, end_ticks);
        assert_eq!(elapsed, 0, "elapsed should be 0 on wraparound");

        let elapsed_micros = MachTimebase::elapsed_micros(start_ticks, end_ticks);
        assert_eq!(elapsed_micros, 0);

        let elapsed_millis = MachTimebase::elapsed_millis(start_ticks, end_ticks);
        assert_eq!(elapsed_millis, 0);
    }

    #[test]
    fn test_timestamp_from_ticks() {
        MachTimebase::init();
        let ticks = 123456789u64;
        let ts = Timestamp::from_ticks(ticks);
        assert_eq!(ts.ticks(), ticks);
    }

    #[test]
    fn test_timestamp_conversions() {
        MachTimebase::init();
        let ts = Timestamp::now();

        let nanos = ts.as_nanos();
        let micros = ts.as_micros();
        let millis = ts.as_millis();

        // Verify conversion consistency
        assert_eq!(micros, nanos / 1_000);
        assert_eq!(millis, nanos / 1_000_000);
    }

    #[test]
    fn test_timestamp_default() {
        let ts = Timestamp::default();
        assert_eq!(ts.ticks(), 0);
    }

    #[test]
    fn test_timestamp_comparison() {
        MachTimebase::init();
        let t1 = Timestamp::from_ticks(1000);
        let t2 = Timestamp::from_ticks(2000);
        let t3 = Timestamp::from_ticks(1000);

        assert!(t2 > t1);
        assert!(t1 < t2);
        assert_eq!(t1, t3);
        assert!(t2.is_after(t1));
        assert!(!t1.is_after(t2));
    }

    #[test]
    fn test_duration_from_units() {
        MachTimebase::init();

        let d_nanos = Duration::from_nanos(1_000_000);
        let d_micros = Duration::from_micros(1_000);
        let d_millis = Duration::from_millis(1);

        // All should be approximately equal (1 millisecond)
        let diff1 = d_nanos.as_millis().abs_diff(d_micros.as_millis());
        let diff2 = d_micros.as_millis().abs_diff(d_millis.as_millis());

        assert!(diff1 <= 1, "nano and micro conversions should be close");
        assert!(diff2 <= 1, "micro and milli conversions should be close");
    }

    #[test]
    fn test_duration_zero() {
        assert_eq!(Duration::ZERO.ticks(), 0);
        assert_eq!(Duration::ZERO.as_nanos(), 0);
        assert_eq!(Duration::ZERO.as_micros(), 0);
        assert_eq!(Duration::ZERO.as_millis(), 0);
    }

    #[test]
    fn test_duration_as_secs_f64() {
        MachTimebase::init();
        let d = Duration::from_millis(1500);
        let secs = d.as_secs_f64();

        // Should be approximately 1.5 seconds
        assert!(secs >= 1.49 && secs <= 1.51, "expected ~1.5s, got {}", secs);
    }

    #[test]
    fn test_duration_saturating_arithmetic() {
        MachTimebase::init();
        let d1 = Duration::from_ticks(u64::MAX);
        let d2 = Duration::from_ticks(100);

        // Addition should saturate at u64::MAX
        let sum = d1 + d2;
        assert_eq!(sum.ticks(), u64::MAX);

        // Subtraction should saturate at 0
        let small = Duration::from_ticks(10);
        let large = Duration::from_ticks(100);
        let diff = small - large;
        assert_eq!(diff.ticks(), 0);
    }

    #[test]
    fn test_timestamp_serialization() {
        MachTimebase::init();
        let ts = Timestamp::from_ticks(123456789);

        // Serialize to JSON
        let json = serde_json::to_string(&ts).unwrap();
        assert_eq!(json, "123456789");

        // Deserialize back
        let deserialized: Timestamp = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.ticks(), ts.ticks());
    }

    #[test]
    fn test_now_functions_all_units() {
        MachTimebase::init();

        let nanos1 = MachTimebase::now_nanos();
        std::thread::sleep(std::time::Duration::from_micros(10));
        let nanos2 = MachTimebase::now_nanos();
        assert!(nanos2 > nanos1);

        let micros1 = MachTimebase::now_micros();
        std::thread::sleep(std::time::Duration::from_micros(10));
        let micros2 = MachTimebase::now_micros();
        assert!(micros2 >= micros1);
    }

    #[test]
    fn test_is_monotonic_edge_cases() {
        assert!(MachTimebase::is_monotonic(100, 100), "equal values should be monotonic");
        assert!(MachTimebase::is_monotonic(100, 200), "increasing should be monotonic");
        assert!(!MachTimebase::is_monotonic(200, 100), "decreasing should not be monotonic");
        assert!(MachTimebase::is_monotonic(0, 0), "zero should be monotonic with itself");
    }

    #[test]
    fn test_timestamp_duration_since_saturating() {
        MachTimebase::init();
        let t1 = Timestamp::from_ticks(1000);
        let t2 = Timestamp::from_ticks(500);

        // When t2 < t1, duration should saturate to 0
        let duration = t2.duration_since(t1);
        assert_eq!(duration.ticks(), 0);
    }
}
