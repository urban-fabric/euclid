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
