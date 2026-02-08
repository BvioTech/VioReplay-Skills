//! High-precision timing module using mach_absolute_time
//!
//! This module provides microsecond-precision timing that is:
//! - Monotonic (never goes backward)
//! - Consistent across Apple Silicon and Intel
//! - Zero-overhead in the hot path

pub mod timebase;

pub use timebase::MachTimebase;
