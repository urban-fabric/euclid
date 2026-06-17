//! Closed intervals over scalar quantities, for uncertainty propagation.
//!
//! A measured length is rarely an exact `f64`; it is a value with a tolerance. An
//! [`Interval<Q>`] carries the `[lo, hi]` bounds of a scalar quantity and threads
//! them through arithmetic so that error propagates automatically. The dimensional
//! cross-products mirror [`crate::units::ops::dimensional`]: dividing an `Interval<LengthMeters>`
//! by an `Interval<TimeSeconds>` yields an `Interval<SpeedMs>`, the band of speeds
//! consistent with the distance and time bands.
//!
//! Bounds are kept ordered (`lo <= hi`) at construction. Arithmetic uses the
//! interval extension of each operation: addition adds endpoints, subtraction
//! crosses them, and multiplication takes the extremes over the four corner
//! combinations. Division does the same when the divisor excludes zero; when the
//! divisor straddles zero the quotient is unbounded, so division returns the
//! fully unbounded interval (`-inf, +inf`) rather than a misleadingly finite band.

use core::cmp::Ordering;
use core::ops::{Add, Div, Mul, Neg, Sub};

use crate::{LengthMeters, LinearQuantity, SpeedMs, TimeSeconds};

/// A closed interval `[lo, hi]` of a scalar quantity `Q`.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Interval<Q: LinearQuantity> {
    lo: Q,
    hi: Q,
}

impl<Q: LinearQuantity> Interval<Q> {
    /// Build an interval from two bounds in any order.
    pub fn new(a: Q, b: Q) -> Self {
        match a.partial_cmp(&b) {
            Some(Ordering::Greater) => Self { lo: b, hi: a },
            _ => Self { lo: a, hi: b },
        }
    }

    /// A degenerate (zero-width) interval containing a single value.
    pub fn point(value: Q) -> Self {
        Self {
            lo: value,
            hi: value,
        }
    }

    /// An interval centred on `midpoint` extending `radius` either side. The
    /// magnitude of `radius` is used, so a negative radius is treated as positive.
    pub fn from_midpoint_radius(midpoint: Q, radius: Q) -> Self {
        let r = radius.raw().abs();
        Self::from_raw_bounds(midpoint.raw() - r, midpoint.raw() + r)
    }

    /// Construct directly from already-ordered raw bounds (`lo <= hi`).
    fn from_raw_bounds(lo: f64, hi: f64) -> Self {
        Self {
            lo: Q::from_raw(lo),
            hi: Q::from_raw(hi),
        }
    }

    fn raw_bounds(self) -> (f64, f64) {
        (self.lo.raw(), self.hi.raw())
    }

    /// The lower bound.
    pub fn lo(self) -> Q {
        self.lo
    }

    /// The upper bound.
    pub fn hi(self) -> Q {
        self.hi
    }

    /// The centre of the interval.
    pub fn midpoint(self) -> Q {
        Q::from_raw(0.5 * (self.lo.raw() + self.hi.raw()))
    }

    /// Half the width: the symmetric `±` tolerance around the midpoint.
    pub fn radius(self) -> Q {
        Q::from_raw(0.5 * (self.hi.raw() - self.lo.raw()))
    }

    /// The full width `hi - lo`.
    pub fn width(self) -> Q {
        Q::from_raw(self.hi.raw() - self.lo.raw())
    }

    /// Whether `value` lies within the closed interval.
    pub fn contains(self, value: Q) -> bool {
        self.lo.raw() <= value.raw() && value.raw() <= self.hi.raw()
    }

    /// Whether the interval has zero width.
    pub fn is_degenerate(self) -> bool {
        self.lo.raw() == self.hi.raw()
    }

    /// Clamp a value into the closed interval.
    pub fn clamp(self, value: Q) -> Q {
        let v = value.raw();
        let clamped = if v < self.lo.raw() {
            self.lo.raw()
        } else if v > self.hi.raw() {
            self.hi.raw()
        } else {
            v
        };
        Q::from_raw(clamped)
    }

    /// Whether `other` lies entirely within this interval.
    pub fn contains_interval(self, other: Self) -> bool {
        self.lo.raw() <= other.lo.raw() && other.hi.raw() <= self.hi.raw()
    }

    /// Whether the two intervals share at least one point.
    pub fn intersects(self, other: Self) -> bool {
        self.lo.raw() <= other.hi.raw() && other.lo.raw() <= self.hi.raw()
    }

    /// The overlap of the two intervals, or `None` if they are disjoint.
    pub fn intersection(self, other: Self) -> Option<Self> {
        if self.intersects(other) {
            let lo = self.lo.raw().max(other.lo.raw());
            let hi = self.hi.raw().min(other.hi.raw());
            Some(Self::from_raw_bounds(lo, hi))
        } else {
            None
        }
    }

    /// The smallest interval containing both intervals (their convex hull).
    pub fn hull(self, other: Self) -> Self {
        let lo = self.lo.raw().min(other.lo.raw());
        let hi = self.hi.raw().max(other.hi.raw());
        Self::from_raw_bounds(lo, hi)
    }
}

/// Take the extremes over the four corner combinations of `op(a, b)`.
///
/// This is exact for monotone-per-argument operations such as multiplication,
/// where the extremes of the product always sit at endpoint combinations.
fn corner_bounds(a: (f64, f64), b: (f64, f64), op: impl Fn(f64, f64) -> f64) -> (f64, f64) {
    let corners = [op(a.0, b.0), op(a.0, b.1), op(a.1, b.0), op(a.1, b.1)];
    let mut lo = corners[0];
    let mut hi = corners[0];
    for &c in &corners[1..] {
        if c < lo {
            lo = c;
        }
        if c > hi {
            hi = c;
        }
    }
    (lo, hi)
}

/// Interval division `num / den`.
///
/// Corner evaluation alone is unsound when the divisor contains zero: it only
/// samples the endpoints and misses the singularity at `den = 0`, where the
/// quotient blows up. When the divisor straddles zero we therefore return the
/// fully unbounded interval, which is the sound (if not tightest) result.
fn div_bounds(num: (f64, f64), den: (f64, f64)) -> (f64, f64) {
    let (c, d) = den;
    if c <= 0.0 && d >= 0.0 {
        return (f64::NEG_INFINITY, f64::INFINITY);
    }
    corner_bounds(num, den, |a, b| a / b)
}

impl<Q: LinearQuantity> Add for Interval<Q> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::from_raw_bounds(self.lo.raw() + rhs.lo.raw(), self.hi.raw() + rhs.hi.raw())
    }
}

impl<Q: LinearQuantity> Sub for Interval<Q> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::from_raw_bounds(self.lo.raw() - rhs.hi.raw(), self.hi.raw() - rhs.lo.raw())
    }
}

impl<Q: LinearQuantity> Neg for Interval<Q> {
    type Output = Self;
    fn neg(self) -> Self {
        Self::from_raw_bounds(-self.hi.raw(), -self.lo.raw())
    }
}

impl<Q: LinearQuantity> Mul<f64> for Interval<Q> {
    type Output = Self;
    fn mul(self, k: f64) -> Self {
        let (a, b) = (self.lo.raw() * k, self.hi.raw() * k);
        if a <= b {
            Self::from_raw_bounds(a, b)
        } else {
            Self::from_raw_bounds(b, a)
        }
    }
}

impl<Q: LinearQuantity> Mul<Interval<Q>> for f64 {
    type Output = Interval<Q>;
    fn mul(self, x: Interval<Q>) -> Interval<Q> {
        x * self
    }
}

impl<Q: LinearQuantity> Div<f64> for Interval<Q> {
    type Output = Self;
    fn div(self, k: f64) -> Self {
        let (a, b) = (self.lo.raw() / k, self.hi.raw() / k);
        if a <= b {
            Self::from_raw_bounds(a, b)
        } else {
            Self::from_raw_bounds(b, a)
        }
    }
}

impl Div<Interval<TimeSeconds>> for Interval<LengthMeters> {
    type Output = Interval<SpeedMs>;
    fn div(self, t: Interval<TimeSeconds>) -> Interval<SpeedMs> {
        let (lo, hi) = div_bounds(self.raw_bounds(), t.raw_bounds());
        Interval::from_raw_bounds(lo, hi)
    }
}

impl Div<Interval<SpeedMs>> for Interval<LengthMeters> {
    type Output = Interval<TimeSeconds>;
    fn div(self, v: Interval<SpeedMs>) -> Interval<TimeSeconds> {
        let (lo, hi) = div_bounds(self.raw_bounds(), v.raw_bounds());
        Interval::from_raw_bounds(lo, hi)
    }
}

impl Mul<Interval<TimeSeconds>> for Interval<SpeedMs> {
    type Output = Interval<LengthMeters>;
    fn mul(self, t: Interval<TimeSeconds>) -> Interval<LengthMeters> {
        let (lo, hi) = corner_bounds(self.raw_bounds(), t.raw_bounds(), |ms, s| ms * s);
        Interval::from_raw_bounds(lo, hi)
    }
}

impl Mul<Interval<SpeedMs>> for Interval<TimeSeconds> {
    type Output = Interval<LengthMeters>;
    fn mul(self, v: Interval<SpeedMs>) -> Interval<LengthMeters> {
        let (lo, hi) = corner_bounds(self.raw_bounds(), v.raw_bounds(), |s, ms| s * ms);
        Interval::from_raw_bounds(lo, hi)
    }
}

#[cfg(test)]
mod tests;
