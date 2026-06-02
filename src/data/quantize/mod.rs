//! Fallible quantization for lossy quantity boundaries.
//!
//! The infallible constructors in [`crate::weights`] (`TimeWeight::from_seconds`,
//! `DistanceWeight::from_meters`) silently fold non-finite or out-of-range inputs
//! onto the `u32::MAX` infinity sentinel and only `debug_assert!` on negatives.
//! That is convenient for hot routing paths but hides precision and range loss at
//! the boundary `euclid` owns: the float quantity to compact integer storage.
//!
//! The `try_*` conversions here are the explicit counterpart. They never panic
//! and never silently saturate; instead they report exactly which contract was
//! broken via [`ConversionError`], so callers handling untrusted or derived
//! values can decide what to do.
//!
//! "Checked" means they validate finiteness, sign, and target range — *not* that
//! the value lands on an exact sub-unit boundary. A fractional centisecond or
//! decimeter is still rounded to the nearest whole storage unit, exactly as the
//! infallible path does, and that ordinary rounding is not reported as an error.

use core::fmt;

use crate::{DistanceWeight, LengthMeters, QuantityError, TimeSeconds, TimeWeight};

/// Why a checked quantity conversion was rejected: the value failed the
/// finiteness, sign, or range contract. Sub-unit rounding is not an error.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ConversionError {
    /// The source value was `NaN` or infinite, so it has no finite encoding.
    NotFinite,
    /// The source value was negative, which the target unit cannot represent.
    Negative,
    /// The source value was finite and non-negative but exceeds the target range.
    Overflow,
}

impl fmt::Display for ConversionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let msg = match self {
            Self::NotFinite => "value is not finite",
            Self::Negative => "value is negative",
            Self::Overflow => "value exceeds the target range",
        };
        f.write_str(msg)
    }
}

impl std::error::Error for ConversionError {}

impl From<QuantityError> for ConversionError {
    fn from(value: QuantityError) -> Self {
        match value {
            QuantityError::NotFinite => Self::NotFinite,
        }
    }
}

/// Validate a finite, non-negative scalar before quantizing it, returning the
/// rounded integer in the requested fixed-point scale.
fn checked_fixed_point(value: f64, units_per_whole: f64) -> Result<u32, ConversionError> {
    if !value.is_finite() {
        return Err(ConversionError::NotFinite);
    }
    if value < 0.0 {
        return Err(ConversionError::Negative);
    }
    let scaled = (value * units_per_whole).round();
    if scaled >= f64::from(u32::MAX) {
        return Err(ConversionError::Overflow);
    }
    Ok(scaled as u32)
}

impl TimeWeight {
    /// Quantize seconds into a centisecond [`TimeWeight`], reporting loss instead
    /// of folding onto [`TimeWeight::INF`].
    ///
    /// Note that `u32::MAX` centiseconds is reserved as the infinity sentinel, so
    /// a value that rounds to it is reported as [`ConversionError::Overflow`]
    /// rather than silently becoming `INF`.
    pub fn try_from_seconds(seconds: f64) -> Result<Self, ConversionError> {
        checked_fixed_point(seconds, 100.0).map(Self::from_centiseconds)
    }
}

impl DistanceWeight {
    /// Quantize meters into a decimeter [`DistanceWeight`], reporting loss instead
    /// of folding onto [`DistanceWeight::INF`].
    pub fn try_from_meters(meters: f64) -> Result<Self, ConversionError> {
        checked_fixed_point(meters, 10.0).map(Self::from_decimeters)
    }
}

impl LengthMeters {
    /// Checked counterpart of [`LengthMeters::to_distance_weight`].
    pub fn try_to_distance_weight(self) -> Result<DistanceWeight, ConversionError> {
        DistanceWeight::try_from_meters(self.as_meters())
    }
}

impl TimeSeconds {
    /// Checked counterpart of [`TimeSeconds::to_time_weight`].
    pub fn try_to_time_weight(self) -> Result<TimeWeight, ConversionError> {
        TimeWeight::try_from_seconds(self.as_seconds())
    }
}

#[cfg(test)]
mod tests;
