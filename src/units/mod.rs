//! Lazy physical quantities over any [`LazyReal`] carrier.
//!
//! The unit wrappers are scalar-valued: a `LazyMeters<T>` contains one `T`, where `T` can be
//! a host scalar, expression, or graph scalar. They are not vector or matrix wrappers.
//!
//! These are the lazy peers of the concrete `Copy`/`Pod`/serde unit newtypes in `oresme`.
//! euclid keeps the dependency-free form here because it only needs the [`LazyReal`]
//! surface: `Clone` instead of `Copy`, mask-returning comparisons through
//! [`LazyCompare`], and host literals lifted through
//! [`LazyReal::scalar`]/[`scalar_like`](LazyReal::scalar_like).

use crate::{LazyCompare, LazyReal};

/// The physical dimension a quantity measures. The single source of truth shared by the
/// lazy units here and the concrete units in `oresme` (which re-exports this type).
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

/// The mask a comparison over `Q`'s carrier yields — `bool` when the carrier is
/// `f64`/`f32`, a `pred` column/tensor on a backend.
pub type Mask<Q> = <<Q as LazyQuantity>::Repr as LazyCompare>::Mask;

/// A scalar quantity backed by a [`LazyReal`] carrier. `Clone`, not `Copy`: a backend
/// carrier (an expression / graph handle) is not `Copy`.
pub trait LazyQuantity: Clone {
    /// The numeric carrier — `f64`, a polars `Expr`, a backend `Tensor` scalar.
    type Repr: LazyReal;
    const DIMENSION: Dimension;
    const UNIT_NAME: &'static str;
    fn raw(&self) -> Self::Repr;
    fn from_raw(v: Self::Repr) -> Self;
}

/// Uniform scalar helpers for every lazy quantity, written once over the [`LazyReal`]
/// surface so they lower to host branches on `f64` and to `Select`/`Max`/`Min` on a
/// column/tensor backend. Blanket-implemented.
pub trait LazyScalarQuantity: LazyQuantity {
    /// The comparison-free lesser (`LazyReal::min`, NaN-aware on the reals).
    fn min(self, other: Self) -> Self {
        Self::from_raw(self.raw().min(other.raw()))
    }
    /// The comparison-free greater (`LazyReal::max`).
    fn max(self, other: Self) -> Self {
        Self::from_raw(self.raw().max(other.raw()))
    }
    /// Clamp into `[lo, hi]` (`LazyReal::clamp`).
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

    /// Finiteness as a [`Mask`] — `bool` for `f64`, a `pred` column/tensor on a backend
    /// (compose it with `select` rather than branching).
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

impl<Q: LazyQuantity> LazyScalarQuantity for Q {}

/// A lazy quantity that behaves like a displacement/vector — negation, scaling,
/// interpolation. Opt-in per unit (an absolute scalar such as a temperature does **not**
/// implement it).
pub trait LazyLinearQuantity: LazyScalarQuantity {
    fn abs(self) -> Self {
        Self::from_raw(self.raw().abs())
    }

    /// The sign of the carrier (`LazyReal::signum`).
    fn signum(self) -> Self::Repr {
        self.raw().signum()
    }

    /// The dimensionless ratio `self / other`.
    fn ratio(self, other: Self) -> Self::Repr {
        self.raw() / other.raw()
    }

    /// Linear interpolation with a carrier-valued `t`: `t = 0` → `self`, `t = 1` → `other`.
    fn lerp(self, other: Self, t: Self::Repr) -> Self {
        let one = <Self::Repr as LazyReal>::one();
        Self::from_raw(self.raw() * (one - t.clone()) + other.raw() * t)
    }

    /// Scale by a carrier-valued dimensionless factor (e.g. a graph-computed gain).
    fn scale(self, factor: Self::Repr) -> Self {
        Self::from_raw(self.raw() * factor)
    }

    /// Scale by a host literal, lifted into the carrier via
    /// [`scalar_like`](LazyReal::scalar_like) so a backend keeps device/arena context.
    fn scaled(self, k: f64) -> Self {
        let factor = self.raw().scalar_like(k);
        Self::from_raw(self.raw() * factor)
    }
}

/// Generate a lazy linear-unit newtype and its full operator set. One name per
/// invocation.
macro_rules! lazy_linear_unit {
    ($(#[$meta:meta])* $name:ident, $dim:expr, $unit:literal) => {
        $(#[$meta])*
        #[derive(Clone, Debug, PartialEq)]
        pub struct $name<T = f64>(pub(crate) T);

        impl<T: $crate::LazyReal> $name<T> {
            /// Wrap a carrier value.
            pub fn new(value: T) -> Self {
                Self(value)
            }
            /// Unwrap to the carrier — the "lower"/evaluate boundary (extract the
            /// `Expr`/`Tensor`, or the plain `f64`).
            pub fn value(self) -> T {
                self.0
            }
        }

        impl<T: $crate::LazyReal> $crate::units::LazyQuantity for $name<T> {
            type Repr = T;
            const DIMENSION: $crate::units::Dimension = $dim;
            const UNIT_NAME: &'static str = $unit;
            fn raw(&self) -> T {
                self.0.clone()
            }
            fn from_raw(v: T) -> Self {
                Self(v)
            }
        }

        impl<T: $crate::LazyReal> $crate::units::LazyLinearQuantity for $name<T> {}

        impl<T: $crate::LazyReal> ::core::ops::Add for $name<T> {
            type Output = Self;
            fn add(self, rhs: Self) -> Self {
                Self(self.0 + rhs.0)
            }
        }

        impl<T: $crate::LazyReal> ::core::ops::Sub for $name<T> {
            type Output = Self;
            fn sub(self, rhs: Self) -> Self {
                Self(self.0 - rhs.0)
            }
        }

        impl<T: $crate::LazyReal> ::core::ops::Neg for $name<T> {
            type Output = Self;
            fn neg(self) -> Self {
                Self(-self.0)
            }
        }

        impl<T: $crate::LazyReal> ::core::ops::AddAssign for $name<T> {
            fn add_assign(&mut self, rhs: Self) {
                self.0 = self.0.clone() + rhs.0;
            }
        }

        impl<T: $crate::LazyReal> ::core::ops::SubAssign for $name<T> {
            fn sub_assign(&mut self, rhs: Self) {
                self.0 = self.0.clone() - rhs.0;
            }
        }

        /// Scale by a bare carrier factor (`length * gain`).
        impl<T: $crate::LazyReal> ::core::ops::Mul<T> for $name<T> {
            type Output = Self;
            fn mul(self, k: T) -> Self {
                Self(self.0 * k)
            }
        }
    };
}

mod dimensional;
mod measures;

#[cfg(test)]
mod tests;

pub use measures::{LazyKm, LazyMeters, LazySeconds, LazySpeedMs, LazySquareMeters};
