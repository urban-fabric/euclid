//! Physical quantities over any [`Real`] carrier.
//!
//! The unit wrappers are scalar-valued: a `Meters<T>` contains one `T`, where `T` can be
//! a host scalar, expression, or graph scalar. They are not vector or matrix wrappers.
//!
//! The wrappers are generic over the [`Real`] surface: values are `Clone`, comparisons return
//! masks through [`Compare`], and host literals are lifted through
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
    /// where a unit's `UnitOfMeasure` impl asserts that its runtime dimension matches its
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

/// A quantity that behaves like a displacement/vector — negation, scaling, interpolation.
/// Opt-in per unit (an absolute scalar such as a temperature does **not** implement it).
pub trait LinearQuantity: ScalarQuantity {
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

mod dimensional;
mod measures;

#[cfg(test)]
mod tests;

pub use measures::{Angle, Km, Meters, Seconds, SpeedMs, SquareMeters};
