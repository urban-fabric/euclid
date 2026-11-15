//! The quantity surface — re-exported from `euclid-units` rather than redefined here.
//!
//! This module used to declare its own `Quantity`, `ScalarQuantity` and `LinearQuantity`
//! traits, named identically to euclid-units' and shadowing them at the crate root. Every unit
//! type then implemented *both* families, with a hand-written bridge delegating one to the
//! other — 329 impl blocks of it. `use euclid_quantity::Quantity` and
//! `use euclid_units::Quantity` were different traits with the same method names, which is
//! the kind of thing that reads as a typo for years before anyone checks.
//!
//! There is now one family. These are euclid-units' traits, and a unit here implements
//! [`Measure`] plus whichever of the opt-in contracts it actually satisfies. Because
//! `<f64 as Compare>::Mask` is `bool`, the mask-returning predicates collapse to plain `bool`
//! for every carrier in this crate, so most of the surface a caller sees is unchanged.
//!
//! What euclid-units deliberately cannot express is anything that needs a *host* `f64` — to
//! branch on, or to hand to a host closure. A carrier that might be an expression handle can do
//! neither. Those two conveniences stay here, on [`HostQuantity`].

/// The physical dimension a quantity measures.
///
/// The single source of truth is [`euclid_units::Dimension`]; it is re-exported here so the
/// concrete (`Copy`/`Pod`/serde) units in this crate share the exact same `Dimension` type — and
/// the same runtime dimension tags — as euclid's carrier-generic unit contracts.
pub use euclid_units::Dimension;

/// euclid's quantity contracts, re-exported as this crate's own surface.
///
/// [`Measure`] is the one a unit implements by hand: carrier type, dimension, unit symbol, and
/// raw access. [`Quantity`] follows from it by blanket impl for any `Real`-backed measure.
/// [`ScalarQuantity`] and [`LinearQuantity`] are opt-in empty impls over euclid-units' default
/// bodies, and [`DiscreteQuantity`] is the integer-carrier counterpart for counts and sizes.
pub use euclid_units::{DiscreteQuantity, LinearQuantity, Measure, Quantity, ScalarQuantity};

/// Why a quantity could not be converted into a representation that cannot hold it.
///
/// Returned by the fallible conversions on [`crate::HostQuantity`], where the
/// target is an integer or a bounded encoding and the source is an unconstrained `f64`.
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

/// The two conveniences that require a value to be a real host `f64`.
///
/// Both are absent from euclid-units for the same structural reason, not by omission:
///
/// - [`from_raw_finite`](Self::from_raw_finite) returns a `Result`, so it has to branch on the
///   value at construction time.
/// - [`map`](Self::map) applies a host closure `impl Fn(f64) -> f64` to the carrier.
///
/// A carrier that might be an expression handle or a graph node can do neither — it has no host
/// value to inspect at that point, which is exactly why euclid-units stops at mask-returning
/// predicates and carrier-valued scaling. Every carrier in this crate *is* a concrete `f64`, so
/// both are available here.
///
/// Blanket-implemented for every `f64`-backed [`Measure`], so a unit gets it without opting in.
pub trait HostQuantity: Measure<Repr = f64> {
    /// Construct from a raw value, rejecting non-finite inputs.
    ///
    /// The plain [`from_raw`](Measure::from_raw) constructor validates nothing and the inherent
    /// `new` constructors only `debug_assert!` against `NaN`, so a `NaN` or infinity can slip
    /// through silently in release builds. This is the checked counterpart for untrusted or
    /// computed inputs.
    fn from_raw_finite(value: f64) -> Result<Self, QuantityError> {
        if value.is_finite() {
            Ok(Self::from_raw(value))
        } else {
            Err(QuantityError::NotFinite)
        }
    }

    /// Apply a function to the raw value, staying in the same unit.
    fn map(self, f: impl Fn(f64) -> f64) -> Self {
        Self::from_raw(f(self.raw()))
    }
}

impl<T> HostQuantity for T where T: Measure<Repr = f64> {}

// --- operator helpers ----------------------------------------------------------------------
//
// The per-unit `Add`/`Sub`/`Neg`/`Mul<f64>`/`Div<f64>`/`Sum` impls call these rather than
// `LinearQuantity` directly, which keeps one place to look when the lowering changes.
//
// Each call is fully qualified. `LinearQuantity::add` and `core::ops::Add::add` have the same
// name and both are in scope at every one of these call sites, so method syntax would be
// ambiguous — and the ambiguity would resolve to the operator impl, which calls back into here.

pub(crate) fn scalar_add<T: LinearQuantity>(lhs: T, rhs: T) -> T {
    LinearQuantity::add(lhs, rhs)
}

pub(crate) fn scalar_sub<T: LinearQuantity>(lhs: T, rhs: T) -> T {
    LinearQuantity::sub(lhs, rhs)
}

pub(crate) fn scalar_neg<T: LinearQuantity>(value: T) -> T {
    LinearQuantity::neg(value)
}

/// Scale a linear quantity by a dimensionless scalar while preserving its unit type.
pub fn scalar_mul<T: LinearQuantity<Repr = f64>>(value: T, factor: f64) -> T {
    LinearQuantity::scale(value, factor)
}

pub(crate) fn scalar_div<T: LinearQuantity<Repr = f64>>(value: T, divisor: f64) -> T {
    // Not `scale(1.0 / divisor)`: the reciprocal rounds once before multiplying, which is a
    // different result from dividing directly.
    T::from_raw(value.raw() / divisor)
}

pub(crate) fn scalar_sum<T, I>(iter: I) -> T
where
    T: LinearQuantity<Repr = f64>,
    I: Iterator<Item = T>,
{
    T::from_raw(iter.map(|v| v.raw()).sum())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    //! Tests for the uniform quantity APIs. These use
    //! a spread of unit types to confirm the provided methods behave identically
    //! regardless of which quantity they are called on.

    use crate::{
        Factor, HostQuantity, LengthMeters, LinearQuantity, Measure, Percentage, QuantityError,
        ScalarQuantity, SpeedMs, TimeSeconds,
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
        assert_eq!(LinearQuantity::abs(SpeedMs::new(-4.0)), SpeedMs::new(4.0));
        assert_eq!(LinearQuantity::abs(Factor::new(-2.0)), Factor::new(2.0));
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
        assert!(!ScalarQuantity::is_finite(&inf));
        assert!(ScalarQuantity::is_infinite(&inf));
        assert!(ScalarQuantity::is_nan(&nan));
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
        assert!(a.approx_eq(&b, &LengthMeters::new(0.001)));
        assert!(!a.approx_eq(&b, &LengthMeters::new(0.000_01)));
        assert!(a.approx_eq(&b, &LengthMeters::new(-0.001)));
    }
}
