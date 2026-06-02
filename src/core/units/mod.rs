//! A uniform unit-of-measure protocol: dimension-checked conversions and symbols.
//!
//! euclid models each unit as its own newtype (`LengthMeters`, `LengthKm`,
//! `LengthFeet`, ...). That keeps the unit of a value visible in its type, but
//! without a shared protocol every conversion is a bespoke method and nothing
//! stops a `LengthMeters` from being handed where a `TimeSeconds` was meant.
//!
//! [`UnitOfMeasure`] gives the unit newtypes a common backbone, modelled on
//! `squants`: each scalar unit declares the [dimension](dim) it measures and how
//! to express it in that dimension's *primary* unit (meters, seconds, m/s, kelvin,
//! or a bare factor). From that, one generic [`convert`](UnitOfMeasure::convert)
//! works across every unit, and the compiler rejects cross-dimension conversions
//! because the `Dim` associated types would not match.
//!
//! Each scalar unit also gains a [`Display`](std::fmt::Display) that prints the
//! value with its unit symbol (`5 m`, `90 km/h`, `50 %`), so a logged or printed
//! quantity always states what unit it is in.
//!
//! ```
//! use euclid::{LengthKm, LengthFeet, UnitOfMeasure};
//!
//! let one_km = LengthKm::new(1.0);
//! let feet: LengthFeet = one_km.convert();
//! assert!((feet.as_feet() - 3280.84).abs() < 0.01);
//! assert_eq!(one_km.to_string(), "1 km");
//! ```
//!
//! Cross-dimension conversions do not compile:
//!
//! ```compile_fail
//! use euclid::{LengthMeters, TimeSeconds, UnitOfMeasure};
//! // error[E0271]: type mismatch — `Length` is not `Time`
//! let _: TimeSeconds = LengthMeters::new(1.0).convert();
//! ```

use crate::ScalarQuantity;

/// Type-level markers identifying a physical dimension.
///
/// Two units convert into one another only when they share the same marker, which
/// the compiler enforces. These are the type-level counterparts of the runtime
/// [`crate::Dimension`] enum.
pub mod dim {
    /// Acceleration.
    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    pub struct Acceleration;
    /// Plane angle.
    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    pub struct Angle;
    /// Area.
    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    pub struct Area;
    /// Length / distance.
    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    pub struct Length;
    /// Mass.
    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    pub struct Mass;
    /// Elapsed time / duration.
    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    pub struct Time;
    /// Speed / velocity magnitude.
    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    pub struct Speed;
    /// Absolute temperature.
    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    pub struct Temperature;
    /// Dimensionless ratios (factors, percentages).
    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    pub struct Dimensionless;
}

/// A scalar unit of measure that belongs to a dimension and has a known
/// scale and offset to that dimension's primary unit.
///
/// Implemented for every `f64`-backed unit newtype. The provided methods give a
/// single, dimension-checked conversion path that replaces ad-hoc `to_*` methods
/// with one uniform protocol.
pub trait UnitOfMeasure: ScalarQuantity {
    /// The dimension this unit measures. Conversions are only possible between
    /// units that share the same `Dim`.
    type Dim;

    /// Multiply a value in this unit by `TO_PRIMARY` and add `PRIMARY_OFFSET` to
    /// express it in the dimension's primary unit. The primary unit therefore has
    /// `TO_PRIMARY == 1.0` and `PRIMARY_OFFSET == 0.0`.
    const TO_PRIMARY: f64;

    /// Offset added after scaling into the primary unit.
    const PRIMARY_OFFSET: f64 = 0.0;

    /// This quantity's value expressed in the dimension's primary unit.
    fn in_primary(self) -> f64 {
        self.raw() * Self::TO_PRIMARY + Self::PRIMARY_OFFSET
    }

    /// Convert a primary-unit value into this unit's raw numeric value.
    fn value_from_primary(primary: f64) -> f64 {
        (primary - Self::PRIMARY_OFFSET) / Self::TO_PRIMARY
    }

    /// This quantity's numeric value expressed in the unit `To`.
    fn value_in<To: UnitOfMeasure<Dim = Self::Dim>>(self) -> f64 {
        To::value_from_primary(self.in_primary())
    }

    /// Convert to another unit of the same dimension.
    fn convert<To: UnitOfMeasure<Dim = Self::Dim>>(self) -> To {
        To::from_raw(self.value_in::<To>())
    }
}

mod impls;

#[cfg(test)]
mod tests;
