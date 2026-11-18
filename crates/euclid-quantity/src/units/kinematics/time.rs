mod time_hours;
mod time_micros;
mod time_millis;
mod time_minutes;
mod time_nanos;
mod time_picos;
mod time_seconds;

pub use time_hours::TimeHours;
pub use time_micros::{TimeMicros, TimeMicrosDelta};
pub use time_millis::TimeMillis;
pub use time_minutes::TimeMinutes;
pub use time_nanos::{TimeNanos, TimeNanosDelta};
pub use time_picos::{TimePicos, TimePicosDelta};
pub use time_seconds::TimeSeconds;

const S_PER_MIN: f64 = 60.0;
const S_PER_HOUR: f64 = 3_600.0;
const MS_PER_S: f64 = 1_000.0;
const NS_PER_US: u64 = 1_000;
const NS_PER_S: f64 = 1_000_000_000.0;
const PS_PER_NS: u64 = 1_000;
const PS_PER_S: f64 = 1_000_000_000_000.0;
