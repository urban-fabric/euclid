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

// The f64 scalar units (`TimeSeconds`, `TimeMillis`, ...) get `Display` from the
// `impl_unit!` macro in `core::units`; the integer-repr sub-second units do not,
// so give them the same unit-suffixed `Display` here (e.g. `"100 ns"`).
macro_rules! impl_time_display {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl core::fmt::Display for $ty {
                fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                    write!(
                        f,
                        "{} {}",
                        <$ty as $crate::Quantity>::raw(*self),
                        <$ty as $crate::Quantity>::UNIT_NAME,
                    )
                }
            }
        )+
    };
}

impl_time_display!(
    TimeNanos,
    TimeNanosDelta,
    TimePicos,
    TimePicosDelta,
    TimeMicros,
    TimeMicrosDelta,
);

#[cfg(test)]
mod tests;
