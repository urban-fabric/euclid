//! A property search over the law suites, for locating *where* a carrier fails them.
//!
//! Behind a second feature, `conformance-generated`, which enables the parent's and additionally
//! pulls the property engine. The division is deliberate: the suites in [`super`] are the
//! contract and resolve no engine at all, so a consumer binding the laws pays nothing; this is
//! tooling, and costs a dependency.
//!
//! A file of its own since the split, rather than an inline `pub mod` — it was the largest item
//! in a module that had passed the 400-line budget, and it shares none of the others' imports.

use euclid_scalar::Real;
use euclid_stability::{Config, Gen, check, classify, generate};

use super::{Mask, linear_laws, measure_laws, quantity_laws, scalar_laws};
use crate::{LinearQuantity, Measure, Quantity, ScalarQuantity};

/// A raw carrier value, weighted toward what the default bodies branch on.
///
/// The half-integers and the small negatives are not decoration: they are the inputs that
/// separate `round` from a truncation and `trunc` from a `floor`.
fn raw() -> Gen<f64> {
    generate::weighted(vec![
        (2, generate::just(0.0)),
        (2, generate::just(0.5)),
        (2, generate::just(-0.5)),
        (2, generate::just(-1.5)),
        (2, generate::just(1.0)),
        (12, generate::float::magnitude(-10, 20)),
    ])
}

/// Three is the minimum that exercises a three-operand law like `clamp`; six matches what a
/// fixed sample typically uses, and the laws are `O(n³)`, so wider buys repetition.
fn carriers<Q: Measure<Repr = f64> + 'static>() -> Gen<Vec<Q>> {
    generate::vec(raw().map(Q::from_raw), 3, 6)
}

fn signs(xs: &[f64]) -> &'static str {
    match (xs.iter().any(|v| *v < 0.0), xs.iter().any(|v| *v > 0.0)) {
        (true, true) => "mixed signs",
        (true, false) => "all non-positive",
        (false, true) => "all non-negative",
        (false, false) => "all zero",
    }
}

fn settings() -> Config {
    Config::default()
        .cases(400)
        .require("mixed signs", 1.0)
        .require("all non-negative", 1.0)
}

/// [`measure_laws`] and [`scalar_laws`] over generated carriers.
///
/// The pair a carrier that stops short of linearity can run — an affine scale such as a
/// temperature, where addition is not defined and [`LinearQuantity`] is deliberately
/// withheld.
pub fn measure_and_scalar_laws_generated<Q>(tolerance: &Q, what: &str)
where
    Q: Quantity + ScalarQuantity + Measure<Repr = f64> + PartialEq + core::fmt::Debug + 'static,
    Q::Repr: PartialEq + core::fmt::Debug + Real,
    Mask<Q>: PartialEq + core::fmt::Debug,
{
    check(what, &settings(), &carriers::<Q>(), |xs| {
        classify!(signs(&xs.iter().map(Measure::raw).collect::<Vec<_>>()));
        measure_laws(&xs, what);
        scalar_laws(&xs, tolerance, what);
        Ok(())
    });
}

/// [`quantity_laws`] over generated values for a linear carrier.
///
/// The associativity check uses the absolute tolerance in the carrier's own unit.
/// Account for rounding at the largest generated magnitudes. The reference
/// `f64` length carrier uses a tolerance of `1e-9`.
pub fn quantity_laws_generated<Q>(tolerance: &Q, what: &str)
where
    Q: Quantity + LinearQuantity + Measure<Repr = f64> + PartialEq + core::fmt::Debug + 'static,
    Q::Repr: PartialEq + core::fmt::Debug + Real,
    Mask<Q>: PartialEq + core::fmt::Debug,
{
    check(what, &settings(), &carriers::<Q>(), |xs| {
        classify!(signs(&xs.iter().map(Measure::raw).collect::<Vec<_>>()));
        quantity_laws(&xs, tolerance, what);
        Ok(())
    });
}

/// [`linear_laws`] alone, for a carrier whose scalar half is already covered.
pub fn linear_laws_generated<Q>(tolerance: &Q, what: &str)
where
    Q: Quantity + LinearQuantity + Measure<Repr = f64> + PartialEq + core::fmt::Debug + 'static,
    Q::Repr: PartialEq + core::fmt::Debug + Real,
    Mask<Q>: PartialEq + core::fmt::Debug,
{
    check(what, &settings(), &carriers::<Q>(), |xs| {
        classify!(signs(&xs.iter().map(Measure::raw).collect::<Vec<_>>()));
        linear_laws(&xs, tolerance, what);
        Ok(())
    });
}
