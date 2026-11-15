//! The reference carriers, checked against the public conformance suites.
//!
//! The laws are [`crate::conformance`] — public API a carrier author in another repository
//! calls. Running them against `f64`, `f32` and `bool` proves the suites are satisfiable and
//! that the reference impls satisfy them.
//!
//! Running them at **both float widths** is the part that matters: the suites take a tolerance,
//! and a bound that is right for `f64` and silently too loose for `f32` would hide exactly the
//! single-precision regression this crate's `f32` suite exists to catch. The tolerances below
//! are set from the measured error at each width, not shared.

use crate::conformance::{self as laws, ConformanceSample};

// Each carrier declares its own sample and its own tolerance. The tolerances are the point:
// sharing one between f64 and f32 would either fail at f32 or stop catching anything at f64.

impl ConformanceSample for f64 {
    /// Strictly positive and inside `(0, π/2)`: the logarithm laws need `x > 0`, and
    /// `atan(tan(x))` stays on the principal branch. `1.0` is included because several series
    /// implementations cross over there.
    fn sample() -> Vec<Self> {
        vec![0.25, 0.5, 1.0, 1.25, 1.5, 0.125]
    }
    /// A few ulps of relative error: `exp(ln(x))` loses about one per operation and the
    /// identity chains two.
    fn tolerance() -> f64 {
        1e-14
    }
}

impl ConformanceSample for f32 {
    fn sample() -> Vec<Self> {
        vec![0.25, 0.5, 1.0, 1.25, 1.5, 0.125]
    }
    fn tolerance() -> f64 {
        1e-6
    }
}

impl ConformanceSample for bool {
    /// The whole domain, so the boolean laws are exhaustive rather than sampled.
    fn sample() -> Vec<Self> {
        vec![false, true]
    }
}

#[test]
fn both_float_widths_satisfy_the_real_laws_at_their_own_precision() {
    // Each width brings its own tolerance from `ConformanceSample`, which is what lets one
    // call site serve both.
    laws::real_laws_for::<f64>("f64");
    laws::real_laws_for::<f32>("f32");
}

#[test]
fn both_float_widths_satisfy_the_mask_laws() {
    // Masks are discrete, so these are exact at both widths. Includes a negative and a zero,
    // where a comparison implemented by subtraction-and-sign-test goes wrong.
    laws::mask_laws(&[0.0_f64, 1.0, -2.5, 4.0], "f64");
    laws::mask_laws(&[0.0_f32, 1.0, -2.5, 4.0], "f32");
}

#[test]
fn integer_and_natural_carriers_satisfy_the_mask_laws() {
    // The comparison tier is not float-specific, and the integer carriers reach it too.
    laws::mask_laws(&[0_i64, 1, -1, 7], "i64");
    laws::mask_laws(&[0_i32, 1, -1, 7], "i32");
    laws::mask_laws(&[0_u64, 1, 7, 100], "u64");
}

#[test]
fn bool_satisfies_the_boolean_laws() {
    // The whole domain, so this is exhaustive rather than sampled.
    laws::boolean_laws_for::<bool>("bool");
}

// ===== The exported generated laws, run against the reference carriers ==========
//
// `GeneratedSample` is public API now, so euclid is its first caller. These impls double as the
// worked examples a carrier author copies, which is why the ranges are justified rather than
// merely picked.

use crate::conformance::generated::{self, GeneratedSample};
use euclid_stability::{Gen, generate};

impl GeneratedSample for f64 {
    /// The domain the fixed sample states and `real_laws` requires: **strictly positive and
    /// inside `(0, π/2)`**, drawn log-uniformly so the decades below `1` are as well covered as
    /// the one above.
    ///
    /// `magnitude` rather than `in_range` for that reason — a uniform draw over `(0, 1.5)` puts
    /// almost nothing near `1e-3`, and the logarithm and `ln_1p` crossovers live down there.
    ///
    /// # The `abs` and the `clamp` are both load-bearing
    ///
    /// Writing this generator got the domain wrong twice, and both times the law caught it rather
    /// than passing — which is the argument for shipping the laws rather than describing them:
    ///
    /// - without `abs`, `magnitude` is **signed**, and `-0.015625` made `exp(ln(x))` return `NaN`;
    /// - without the `clamp`, `magnitude(-6, 0)` reaches past `1`, and `1.5707963267948968` —
    ///   one ulp above `π/2` — made `atan(tan(x))` come back negated, because `tan` had crossed
    ///   the pole and `atan` returned the other branch.
    ///
    /// Neither is a defect in the law. `real_laws` documents its sample as finite, strictly
    /// positive and inside the principal branch, and **a generator is as much part of that
    /// contract as the list it replaces.**
    fn generator() -> Gen<Self> {
        generate::weighted(vec![
            (3, generate::just(1.0)),
            (
                6,
                generate::float::magnitude(-6, 0).map(|v| v.abs().clamp(1e-6, 1.5)),
            ),
            (3, generate::float::in_range(0.5, 1.5)),
        ])
    }
}

impl GeneratedSample for bool {
    /// The whole domain, so the boolean laws stay exhaustive rather than sampled even under a
    /// generator.
    fn generator() -> Gen<Self> {
        generate::boolean()
    }
}

#[test]
fn f64_satisfies_the_generated_real_laws() {
    // The same tolerance the fixed suite uses at this width — the generated search is a wider
    // domain at the same bound, not a looser one.
    generated::real_laws_generated::<f64>(1e-14, "f64");
}

#[test]
fn f64_satisfies_the_generated_mask_laws() {
    generated::mask_laws_generated::<f64>("f64");
}

#[test]
fn bool_satisfies_the_generated_boolean_laws() {
    generated::boolean_laws_generated::<bool>("bool");
}
