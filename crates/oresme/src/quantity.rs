/// The physical dimension a quantity measures.
///
/// The single source of truth is [`euclid_units::Dimension`]; it is re-exported here so the concrete
/// (`Copy`/`Pod`/serde) units in this crate share the exact same `Dimension` type — and the same
/// runtime dimension tags — as euclid's carrier-generic unit contracts.
pub use euclid_units::Dimension;

pub trait Quantity: Copy + PartialOrd {
    type Repr: Copy;
    const DIMENSION: Dimension;
    const UNIT_NAME: &'static str;
    fn raw(self) -> Self::Repr;
    fn from_raw(v: Self::Repr) -> Self;
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum QuantityError {
    /// The source value was `NaN` or infinite, so it has no finite encoding.
    NotFinite,
}

impl core::fmt::Display for QuantityError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotFinite => f.write_str("value is not finite"),
        }
    }
}

impl std::error::Error for QuantityError {}

/// A [`Quantity`] backed by an `f64`.
///
/// Every `f64`-backed quantity is a `ScalarQuantity`, and gets uniform finite
/// construction, finiteness checks, ordering helpers, rounding, and approximate
/// equality. Vector-like operations live on [`LinearQuantity`] so absolute scalar
/// values such as temperatures are not accidentally treated as displacements.
pub trait ScalarQuantity: Quantity<Repr = f64> {
    /// Construct from a raw value, rejecting non-finite inputs.
    ///
    /// The plain [`from_raw`](Quantity::from_raw) constructor validates nothing
    /// and the inherent `new` constructors only `debug_assert!` against `NaN`, so
    /// a `NaN` or infinity can slip through silently in release builds. This is
    /// the checked counterpart for untrusted or computed inputs.
    fn from_raw_finite(value: f64) -> Result<Self, QuantityError> {
        if value.is_finite() {
            Ok(Self::from_raw(value))
        } else {
            Err(QuantityError::NotFinite)
        }
    }

    /// The lesser of two values.
    fn min(self, other: Self) -> Self {
        if other.raw() < self.raw() {
            other
        } else {
            self
        }
    }

    /// The greater of two values.
    fn max(self, other: Self) -> Self {
        if other.raw() > self.raw() {
            other
        } else {
            self
        }
    }

    /// Clamp into the inclusive range `[lo, hi]`, which must satisfy `lo <= hi`.
    fn clamp(self, lo: Self, hi: Self) -> Self {
        let v = self.raw();
        let clamped = if v < lo.raw() {
            lo.raw()
        } else if v > hi.raw() {
            hi.raw()
        } else {
            v
        };
        Self::from_raw(clamped)
    }

    /// Round toward negative infinity to a whole number of this unit.
    fn floor(self) -> Self {
        Self::from_raw(self.raw().floor())
    }

    /// Round toward positive infinity to a whole number of this unit.
    fn ceil(self) -> Self {
        Self::from_raw(self.raw().ceil())
    }

    /// Round to the nearest whole number of this unit (ties away from zero).
    fn round(self) -> Self {
        Self::from_raw(self.raw().round())
    }

    /// Truncate toward zero to a whole number of this unit.
    fn trunc(self) -> Self {
        Self::from_raw(self.raw().trunc())
    }

    /// Whether the value is finite (neither infinite nor `NaN`).
    fn is_finite(self) -> bool {
        self.raw().is_finite()
    }

    /// Whether the value is positive or negative infinity.
    fn is_infinite(self) -> bool {
        self.raw().is_infinite()
    }

    /// Whether the value is `NaN`.
    fn is_nan(self) -> bool {
        self.raw().is_nan()
    }

    /// Approximate equality within a tolerance expressed in the same unit.
    ///
    /// The tolerance is itself a quantity, not a bare `f64`, so its dimension
    /// cannot be confused with some other unit's.
    fn approx_eq(self, other: Self, tolerance: Self) -> bool {
        (self.raw() - other.raw()).abs() <= tolerance.raw().abs()
    }
}

impl<T> ScalarQuantity for T where T: Quantity<Repr = f64> {}

/// An `f64` quantity that behaves like a vector/displacement.
///
/// Linear quantities can be negated, scaled, added, subtracted, interpolated,
/// and used in interval arithmetic. Absolute scalar values such as Celsius or
/// Fahrenheit intentionally do not implement this trait.
///
/// ```compile_fail
/// use oresme::{Interval, TemperatureCelsius};
///
/// let _ = Interval::point(TemperatureCelsius::new(20.0));
/// ```
pub trait LinearQuantity: ScalarQuantity {
    /// The absolute value.
    fn abs(self) -> Self {
        Self::from_raw(self.raw().abs())
    }

    /// The sign of the raw value, following [`f64::signum`]: `1.0` for positive
    /// values and `+0.0`, `-1.0` for negative values and `-0.0`, and `NaN` for
    /// `NaN`.
    fn signum(self) -> f64 {
        self.raw().signum()
    }

    /// The dimensionless ratio `self / other`.
    fn ratio(self, other: Self) -> f64 {
        self.raw() / other.raw()
    }

    /// Linear interpolation: `t = 0.0` yields `self`, `t = 1.0` yields `other`.
    fn lerp(self, other: Self, t: f64) -> Self {
        Self::from_raw(self.raw() * (1.0 - t) + other.raw() * t)
    }

    /// Apply a function to the raw value, staying in the same unit.
    fn map(self, f: impl Fn(f64) -> f64) -> Self {
        Self::from_raw(f(self.raw()))
    }
}

pub(crate) fn scalar_add<T: LinearQuantity>(lhs: T, rhs: T) -> T {
    T::from_raw(lhs.raw() + rhs.raw())
}

pub(crate) fn scalar_sub<T: LinearQuantity>(lhs: T, rhs: T) -> T {
    T::from_raw(lhs.raw() - rhs.raw())
}

pub(crate) fn scalar_neg<T: LinearQuantity>(value: T) -> T {
    T::from_raw(-value.raw())
}

/// Scale a linear quantity by a dimensionless scalar while preserving its unit
/// type.
pub fn scalar_mul<T: LinearQuantity>(value: T, factor: f64) -> T {
    T::from_raw(value.raw() * factor)
}

pub(crate) fn scalar_div<T: LinearQuantity>(value: T, divisor: f64) -> T {
    T::from_raw(value.raw() / divisor)
}

pub(crate) fn scalar_sum<T, I>(iter: I) -> T
where
    T: LinearQuantity,
    I: Iterator<Item = T>,
{
    T::from_raw(iter.map(Quantity::raw).sum())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    //! Tests for the uniform quantity APIs. These use
    //! a spread of unit types to confirm the provided methods behave identically
    //! regardless of which quantity they are called on.

    use crate::{
        Factor, LengthMeters, LinearQuantity, Percentage, Quantity, QuantityError, ScalarQuantity,
        SpeedMs, TimeSeconds,
    };

    #[test]
    fn from_raw_finite_accepts_finite_and_rejects_the_rest() {
        assert_eq!(
            LengthMeters::from_raw_finite(3.5).unwrap(),
            LengthMeters::new(3.5)
        );
        assert_eq!(
            LengthMeters::from_raw_finite(f64::NAN),
            Err(QuantityError::NotFinite)
        );
        assert_eq!(
            TimeSeconds::from_raw_finite(f64::INFINITY),
            Err(QuantityError::NotFinite)
        );
    }

    #[test]
    fn abs_is_available_uniformly_even_without_an_inherent_method() {
        assert_eq!(SpeedMs::new(-4.0).abs(), SpeedMs::new(4.0));
        assert_eq!(Factor::new(-2.0).abs(), Factor::new(2.0));
    }

    #[test]
    fn signum_follows_sign() {
        assert_eq!(LengthMeters::new(-7.0).signum(), -1.0);
        assert_eq!(LengthMeters::new(7.0).signum(), 1.0);
    }

    #[test]
    fn min_max_and_clamp() {
        let a = LengthMeters::new(2.0);
        let b = LengthMeters::new(5.0);
        assert_eq!(a.min(b), a);
        assert_eq!(a.max(b), b);
        assert_eq!(LengthMeters::new(7.0).clamp(a, b), b);
        assert_eq!(LengthMeters::new(1.0).clamp(a, b), a);
        assert_eq!(LengthMeters::new(3.0).clamp(a, b), LengthMeters::new(3.0));
    }

    #[test]
    fn rounding_family_stays_in_unit() {
        let x = TimeSeconds::new(2.6);
        assert_eq!(x.floor(), TimeSeconds::new(2.0));
        assert_eq!(x.ceil(), TimeSeconds::new(3.0));
        assert_eq!(x.round(), TimeSeconds::new(3.0));
        assert_eq!(TimeSeconds::new(-2.6).trunc(), TimeSeconds::new(-2.0));
    }

    #[test]
    fn finiteness_predicates() {
        assert!(Percentage::new(50.0).is_finite());
        let inf = Percentage::from_raw(f64::INFINITY);
        let nan = Percentage::from_raw(f64::NAN);
        assert!(!inf.is_finite());
        assert!(inf.is_infinite());
        assert!(nan.is_nan());
    }

    #[test]
    fn ratio_is_dimensionless() {
        assert_eq!(LengthMeters::new(10.0).ratio(LengthMeters::new(4.0)), 2.5);
    }

    #[test]
    fn lerp_and_map() {
        let a = LengthMeters::new(0.0);
        let b = LengthMeters::new(10.0);
        assert_eq!(a.lerp(b, 0.25), LengthMeters::new(2.5));
        assert_eq!(b.map(|m| m * 2.0), LengthMeters::new(20.0));
    }

    #[test]
    fn approx_eq_uses_a_typed_tolerance() {
        let a = LengthMeters::new(1.000_1);
        let b = LengthMeters::new(1.000_2);
        assert!(a.approx_eq(b, LengthMeters::new(0.001)));
        assert!(!a.approx_eq(b, LengthMeters::new(0.000_01)));
        assert!(a.approx_eq(b, LengthMeters::new(-0.001)));
    }
}
