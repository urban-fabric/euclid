//! Physical-dimension contracts over any [`Real`] carrier.
//!
//! Like the rest of euclid, `units` ships *contracts*, not concrete types. A quantity is any
//! [`Real`]-backed carrier that implements [`Quantity`]; the dimension traits ([`Length`],
//! [`Time`], [`Speed`], [`Area`], [`Angle`]) refine it with the operations that dimension
//! supports, including the cross-dimensional algebra (`length × length = area`,
//! `length / time = speed`, …) expressed through associated quantity types. euclid does not
//! ship a `Meters`/`Seconds` struct; a consumer defines its own unit carriers and implements
//! these traits for them.
//!
//! Quantities are scalar-valued: a length carries one `T`, where `T` can be a host scalar,
//! expression, or graph scalar. Values are `Clone`, comparisons return masks through
//! [`Compare`], and host literals are lifted through
//! [`Real::scalar`]/[`scalar_like`](Real::scalar_like).

use crate::{Compare, Element, Real};

/// The physical dimension a quantity measures.
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
    /// Const-evaluable equality. The derived `PartialEq` cannot run in `const` context,
    /// where a unit's [`Quantity`] impl can assert that its runtime dimension matches its
    /// type-level marker.
    pub const fn same(self, other: Self) -> bool {
        self as u8 == other as u8
    }
}

/// The mask a comparison over `Q`'s carrier yields — `bool` for host reals and a carrier mask
/// for expression/tensor implementations.
pub type Mask<Q> = <<Q as Quantity>::Repr as Compare>::Mask;

/// A scalar quantity backed by a [`Real`] carrier. `Clone`, not `Copy`: expression and graph
/// handles may not be cheaply copyable.
pub trait Quantity: Clone {
    /// The numeric carrier, such as `f64` or an expression/tensor scalar.
    type Repr: Real;
    const DIMENSION: Dimension;
    const UNIT_NAME: &'static str;
    fn raw(&self) -> Self::Repr;
    fn from_raw(v: Self::Repr) -> Self;
}

/// Uniform scalar helpers for every quantity, written once over the [`Real`] surface.
/// Blanket-implemented.
pub trait ScalarQuantity: Quantity {
    /// The comparison-free lesser (`Real::min`, NaN-aware on the reals).
    fn min(self, other: Self) -> Self {
        Self::from_raw(self.raw().min(other.raw()))
    }
    /// The comparison-free greater (`Real::max`).
    fn max(self, other: Self) -> Self {
        Self::from_raw(self.raw().max(other.raw()))
    }
    /// Clamp into `[lo, hi]` (`Real::clamp`).
    fn clamp(self, lo: Self, hi: Self) -> Self {
        Self::from_raw(self.raw().clamp(lo.raw(), hi.raw()))
    }
    fn floor(self) -> Self {
        Self::from_raw(self.raw().floor())
    }
    fn ceil(self) -> Self {
        Self::from_raw(self.raw().ceil())
    }
    fn round(self) -> Self {
        Self::from_raw(self.raw().round())
    }
    fn trunc(self) -> Self {
        Self::from_raw(self.raw().trunc())
    }

    /// Finiteness as a [`Mask`]. Compose it with `select` rather than branching on it.
    fn is_finite(&self) -> Mask<Self> {
        self.raw().is_finite()
    }
    fn is_nan(&self) -> Mask<Self> {
        self.raw().is_nan()
    }
    fn is_infinite(&self) -> Mask<Self> {
        self.raw().is_infinite()
    }

    /// Approximate equality within a same-unit tolerance, as a [`Mask`].
    fn approx_eq(&self, other: &Self, tolerance: &Self) -> Mask<Self> {
        (self.raw() - other.raw()).abs().le(&tolerance.raw().abs())
    }
}

impl<Q: Quantity> ScalarQuantity for Q {}

/// A quantity that behaves like a displacement/vector — addition, negation, scaling,
/// interpolation. Opt-in per unit (an absolute scalar such as a temperature does **not**
/// implement it). The `+`/`-`/scale operations are provided over the carrier, so a consumer's
/// unit type gets them for free without hand-writing `core::ops` impls.
pub trait LinearQuantity: ScalarQuantity {
    /// Same-unit sum.
    fn add(self, other: Self) -> Self {
        Self::from_raw(self.raw() + other.raw())
    }

    /// Same-unit difference.
    fn sub(self, other: Self) -> Self {
        Self::from_raw(self.raw() - other.raw())
    }

    /// Negation.
    fn neg(self) -> Self {
        Self::from_raw(-self.raw())
    }

    fn abs(self) -> Self {
        Self::from_raw(self.raw().abs())
    }

    /// The sign of the carrier (`Real::signum`).
    fn signum(self) -> Self::Repr {
        self.raw().signum()
    }

    /// The dimensionless ratio `self / other`.
    fn ratio(self, other: Self) -> Self::Repr {
        self.raw() / other.raw()
    }

    /// Linear interpolation with a carrier-valued `t`: `t = 0` → `self`, `t = 1` → `other`.
    fn lerp(self, other: Self, t: Self::Repr) -> Self {
        let one = <Self::Repr as Element>::one();
        Self::from_raw(self.raw() * (one - t.clone()) + other.raw() * t)
    }

    /// Scale by a carrier-valued dimensionless factor (e.g. a graph-computed gain).
    fn scale(self, factor: Self::Repr) -> Self {
        Self::from_raw(self.raw() * factor)
    }

    /// Scale by a host literal, lifted into the carrier via
    /// [`scalar_like`](Real::scalar_like) so the carrier can preserve any needed context.
    fn scaled(self, k: f64) -> Self {
        let factor = self.raw().scalar_like(k);
        Self::from_raw(self.raw() * factor)
    }
}

/// A length-dimensioned quantity. Carries the cross-dimensional algebra
/// `length × length = area`, `length / time = speed`, and `length / speed = time` through its
/// associated [`Area`], [`Speed`], and [`Time`] partner quantities.
pub trait Length: LinearQuantity {
    /// The area produced by `length × length`.
    type Area: Area<Repr = Self::Repr>;
    /// The time that pairs with this length and speed.
    type Time: Time<Repr = Self::Repr>;
    /// The speed produced by `length / time`.
    type Speed: Speed<Repr = Self::Repr>;

    /// `length × length = area`.
    fn times(self, width: Self) -> Self::Area;
    /// `length / time = speed`.
    fn per(self, elapsed: Self::Time) -> Self::Speed;
    /// `length / speed = time`.
    fn over(self, speed: Self::Speed) -> Self::Time;

    /// `length²` — [`times`](Self::times) with a copy of `self`.
    fn squared(self) -> Self::Area {
        self.clone().times(self)
    }
}

/// A time-dimensioned quantity. `time × speed = length`.
pub trait Time: LinearQuantity {
    /// The length produced by `time × speed`.
    type Length: Length<Repr = Self::Repr>;
    /// The speed that pairs with this time.
    type Speed: Speed<Repr = Self::Repr>;

    /// `time × speed = length`.
    fn times(self, speed: Self::Speed) -> Self::Length;
}

/// A speed-dimensioned quantity. `speed × time = length`.
pub trait Speed: LinearQuantity {
    /// The length produced by `speed × time`.
    type Length: Length<Repr = Self::Repr>;
    /// The time that pairs with this speed.
    type Time: Time<Repr = Self::Repr>;

    /// `speed × time = length`.
    fn times(self, elapsed: Self::Time) -> Self::Length;
}

/// An area-dimensioned quantity. `area / length = length`.
pub trait Area: LinearQuantity {
    /// The length produced by `area / length`.
    type Length: Length<Repr = Self::Repr>;

    /// `area / length = length`.
    fn over(self, side: Self::Length) -> Self::Length;
}

/// An angle-dimensioned quantity, carried in radians. Every operation is provided over the
/// [`Real`] carrier, so a consumer's angle type only implements [`Quantity`] and
/// [`LinearQuantity`].
pub trait Angle: LinearQuantity {
    /// Wrap a carrier value already in radians.
    fn from_radians(radians: Self::Repr) -> Self {
        Self::from_raw(radians)
    }

    /// Wrap a carrier value in degrees, converting to radians.
    fn from_degrees(degrees: Self::Repr) -> Self {
        Self::from_raw(degrees.to_radians())
    }

    /// `π` radians (a half turn).
    fn half_turn() -> Self {
        Self::from_raw(Self::Repr::pi())
    }

    /// `2π` radians (a full turn).
    fn full_turn() -> Self {
        Self::from_raw(Self::Repr::tau())
    }

    /// The angle in radians.
    fn radians(&self) -> Self::Repr {
        self.raw()
    }

    /// The angle in degrees.
    fn degrees(&self) -> Self::Repr {
        self.raw().to_degrees()
    }

    /// Sine of the angle.
    fn sin(&self) -> Self::Repr {
        self.raw().sin()
    }

    /// Cosine of the angle.
    fn cos(&self) -> Self::Repr {
        self.raw().cos()
    }

    /// Tangent of the angle.
    fn tan(&self) -> Self::Repr {
        self.raw().tan()
    }

    /// Normalize into `[0, 2π)`.
    fn normalized_unsigned(self) -> Self {
        Self::from_raw(self.raw().rem_euclid(Self::Repr::tau()))
    }

    /// Normalize into `(-π, π]`.
    fn normalized_signed(self) -> Self {
        let tau = Self::Repr::tau();
        let pi = Self::Repr::pi();
        let r = self.raw().rem_euclid(tau.clone());
        let shifted = r.clone() - tau;
        let over_pi = r.gt(&pi);
        Self::from_raw(<Self::Repr as Compare>::select(&over_pi, shifted, r))
    }
}

#[cfg(test)]
mod tests;
