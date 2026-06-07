#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum Dimension {
    Acceleration,
    Area,
    Length,
    Mass,
    Temperature,
    Time,
    Speed,
    Angle,
    Dimensionless,
}

impl Dimension {
    /// Const-evaluable equality. The derived `PartialEq` cannot run in `const`
    /// context, where the [`UnitOfMeasure`](crate::UnitOfMeasure) impls assert
    /// that a unit's runtime dimension matches its type-level marker.
    pub(crate) const fn same(self, other: Self) -> bool {
        self as u8 == other as u8
    }
}

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
/// use euclid::{Interval, TemperatureCelsius};
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

#[cfg(test)]
mod tests;

pub fn scalar_add<T: LinearQuantity>(lhs: T, rhs: T) -> T {
    T::from_raw(lhs.raw() + rhs.raw())
}

pub fn scalar_sub<T: LinearQuantity>(lhs: T, rhs: T) -> T {
    T::from_raw(lhs.raw() - rhs.raw())
}

pub fn scalar_neg<T: LinearQuantity>(value: T) -> T {
    T::from_raw(-value.raw())
}

pub fn scalar_mul<T: LinearQuantity>(value: T, factor: f64) -> T {
    T::from_raw(value.raw() * factor)
}

pub fn scalar_div<T: LinearQuantity>(value: T, divisor: f64) -> T {
    T::from_raw(value.raw() / divisor)
}

pub fn scalar_sum<T, I>(iter: I) -> T
where
    T: LinearQuantity,
    I: Iterator<Item = T>,
{
    T::from_raw(iter.map(Quantity::raw).sum())
}
