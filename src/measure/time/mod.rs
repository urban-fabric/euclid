mod micros;
mod scalar;

pub(crate) const S_PER_MIN: f64 = 60.0;
pub(crate) const S_PER_HOUR: f64 = 3_600.0;
pub(crate) const MS_PER_S: f64 = 1_000.0;

pub use micros::{TimeMicros, TimeMicrosDelta};
pub use scalar::{TimeHours, TimeMillis, TimeMinutes, TimeSeconds};

#[cfg(test)]
mod tests;
