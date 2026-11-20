//! Closed intervals over scalar quantities, for uncertainty propagation.
//!
//! A measured length is rarely an exact `f64`; it is a value with a tolerance. An
//! [`Interval<Q>`] carries the `[lo, hi]` bounds of a scalar quantity and threads
//! them through arithmetic so that error propagates automatically. A few dimensional
//! cross-products are provided directly on intervals: dividing an `Interval<LengthMeters>`
//! by an `Interval<TimeSeconds>` yields an `Interval<SpeedMs>`, the band of speeds
//! consistent with the distance and time bands.
//!
//! Bounds are kept ordered (`lo <= hi`) at construction. Arithmetic uses the
//! interval extension of each operation: addition adds endpoints, subtraction
//! crosses them, and multiplication takes the extremes over the four corner
//! combinations. Division does the same when the divisor excludes zero; when the
//! divisor straddles zero the quotient is unbounded, so division returns the
//! fully unbounded interval (`-inf, +inf`) rather than a misleadingly finite band.
//!
//! The predicates and set operations are **not** implemented here: they delegate to
//! [`euclid_scalar::interval`], the carrier-generic kernels, so `[lo, hi]` means the
//! same thing in euclid-quantity as it does everywhere else in euclid. This type's job is the
//! part the kernels deliberately leave out — holding the bounds as typed quantities
//! rather than a loose `(lo, hi)` pair, and carrying the dimensional cross-products.
//! Note that the kernels treat `lo > hi` as the empty interval, a state [`new`]
//! cannot produce, so the empty-interval branches in `intersects` and
//! `contains_interval` are unreachable from here and only `intersection` can return
//! empty bounds.
//!
//! [`new`]: Interval::new

use core::ops::{Add, Div, Mul, Neg, Sub};

use euclid_scalar::interval as kernels;

use crate::{LengthMeters, LinearQuantity, SpeedMs, TimeSeconds};

/// A closed interval `[lo, hi]` of a scalar quantity `Q`.
///
/// `Q` must be a [`LinearQuantity`], which is what keeps absolute scalars out. An interval of
/// temperatures *in Celsius* is not an interval in the vector-space sense — its arithmetic would
/// claim that 20 °C + 20 °C is meaningful — so `TemperatureCelsius` declines
/// `LinearQuantity` and this does not compile:
///
/// ```compile_fail
/// use euclid_quantity::{Interval, TemperatureCelsius};
///
/// let _ = Interval::point(TemperatureCelsius::new(20.0));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Interval<Q: LinearQuantity<Repr = f64> + Copy> {
    lo: Q,
    hi: Q,
}

impl<Q: LinearQuantity<Repr = f64> + Copy> Interval<Q> {
    /// Build an interval from two bounds in any order.
    pub fn new(a: Q, b: Q) -> Self {
        let (lo, hi) = kernels::from_point_pair(a.raw(), b.raw());
        Self::from_raw_bounds(lo, hi)
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
        Q::from_raw(self.lo.raw().midpoint(self.hi.raw()))
    }

    /// Half the width: the symmetric `±` tolerance around the midpoint.
    pub fn radius(self) -> Q {
        Q::from_raw(self.hi.raw().midpoint(-self.lo.raw()))
    }

    /// The full width `hi - lo`.
    pub fn width(self) -> Q {
        Q::from_raw(self.hi.raw() - self.lo.raw())
    }

    /// Whether `value` lies within the closed interval.
    pub fn contains(self, value: Q) -> bool {
        let (lo, hi) = self.raw_bounds();
        kernels::contains(lo, hi, value.raw())
    }

    /// Whether the interval has zero width.
    pub fn is_degenerate(self) -> bool {
        self.lo.raw() == self.hi.raw()
    }

    /// Clamp a value into the closed interval.
    pub fn clamp(self, value: Q) -> Q {
        let (lo, hi) = self.raw_bounds();
        Q::from_raw(kernels::clamp_point(lo, hi, value.raw()))
    }

    /// Whether `other` lies entirely within this interval.
    pub fn contains_interval(self, other: Self) -> bool {
        let (lo, hi) = self.raw_bounds();
        let (other_lo, other_hi) = other.raw_bounds();
        kernels::contains_interval(lo, hi, other_lo, other_hi)
    }

    /// Whether the two intervals share at least one point.
    pub fn intersects(self, other: Self) -> bool {
        let (lo, hi) = self.raw_bounds();
        let (other_lo, other_hi) = other.raw_bounds();
        kernels::intersects(lo, hi, other_lo, other_hi)
    }

    /// The overlap of the two intervals, or `None` if they are disjoint.
    pub fn intersection(self, other: Self) -> Option<Self> {
        let (lo, hi) = self.raw_bounds();
        let (other_lo, other_hi) = other.raw_bounds();
        let (lo, hi) = kernels::intersection(lo, hi, other_lo, other_hi);
        // A disjoint pair comes back as empty bounds (`lo > hi`) rather than as an
        // error, so the emptiness check is what turns it into `None`.
        if kernels::is_empty(lo, hi) {
            None
        } else {
            Some(Self::from_raw_bounds(lo, hi))
        }
    }

    /// The smallest interval containing both intervals (their convex hull).
    pub fn hull(self, other: Self) -> Self {
        let (lo, hi) = self.raw_bounds();
        let (other_lo, other_hi) = other.raw_bounds();
        let (lo, hi) = kernels::union(lo, hi, other_lo, other_hi);
        Self::from_raw_bounds(lo, hi)
    }
}

/// Take the extremes over the four corner combinations of `op(a, b)`.
///
/// This is exact for monotone-per-argument operations such as multiplication,
/// where the extremes of the product always sit at endpoint combinations.
/// Equal corners retain the first value's signed zero.
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

impl<Q: LinearQuantity<Repr = f64> + Copy> Add for Interval<Q> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::from_raw_bounds(self.lo.raw() + rhs.lo.raw(), self.hi.raw() + rhs.hi.raw())
    }
}

impl<Q: LinearQuantity<Repr = f64> + Copy> Sub for Interval<Q> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::from_raw_bounds(self.lo.raw() - rhs.hi.raw(), self.hi.raw() - rhs.lo.raw())
    }
}

impl<Q: LinearQuantity<Repr = f64> + Copy> Neg for Interval<Q> {
    type Output = Self;
    fn neg(self) -> Self {
        Self::from_raw_bounds(-self.hi.raw(), -self.lo.raw())
    }
}

impl<Q: LinearQuantity<Repr = f64> + Copy> Mul<f64> for Interval<Q> {
    type Output = Self;
    fn mul(self, k: f64) -> Self {
        // A negative `k` swaps the endpoints, so the bounds are re-ordered rather
        // than assumed.
        let (lo, hi) = kernels::from_point_pair(self.lo.raw() * k, self.hi.raw() * k);
        Self::from_raw_bounds(lo, hi)
    }
}

impl<Q: LinearQuantity<Repr = f64> + Copy> Mul<Interval<Q>> for f64 {
    type Output = Interval<Q>;
    fn mul(self, x: Interval<Q>) -> Interval<Q> {
        x * self
    }
}

impl<Q: LinearQuantity<Repr = f64> + Copy> Div<f64> for Interval<Q> {
    type Output = Self;
    fn div(self, k: f64) -> Self {
        let (lo, hi) = kernels::from_point_pair(self.lo.raw() / k, self.hi.raw() / k);
        Self::from_raw_bounds(lo, hi)
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
