mod micros;
mod nanos;
mod picos;
mod scalar;

pub(crate) const S_PER_MIN: f64 = 60.0;
pub(crate) const S_PER_HOUR: f64 = 3_600.0;
pub(crate) const MS_PER_S: f64 = 1_000.0;
pub(crate) const NS_PER_US: u64 = 1_000;
pub(crate) const NS_PER_S: f64 = 1_000_000_000.0;
pub(crate) const PS_PER_NS: u64 = 1_000;
pub(crate) const PS_PER_S: f64 = 1_000_000_000_000.0;

pub use micros::{TimeMicros, TimeMicrosDelta};
pub use nanos::{TimeNanos, TimeNanosDelta};
pub use picos::{TimePicos, TimePicosDelta};
pub use scalar::{TimeHours, TimeMillis, TimeMinutes, TimeSeconds};

#[cfg(test)]
mod tests;
