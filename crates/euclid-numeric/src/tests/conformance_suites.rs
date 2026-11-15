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

use crate::conformance as laws;

/// Samples for the real laws: strictly positive and finite, since the logarithm laws need
/// `x > 0`, and inside `(0, π/2)` so `atan(tan(x))` stays on the principal branch.
///
/// `1.0` is included deliberately — it is where several series implementations cross over, and
/// where `ln_1p`/`exp_m1` stop being the accurate choice and start merely agreeing with the
/// naive form.
const REALS_F64: [f64; 6] = [0.25, 0.5, 1.0, 1.25, 1.5, 0.125];

#[test]
fn f64_satisfies_the_real_laws() {
    // 1e-14 is a few ulps of relative error, which the inverse pairs need: `exp(ln(x))` loses
    // about one ulp per operation and the identity chains two.
    laws::real_laws(&REALS_F64, 1e-14, "f64");
}

#[test]
fn f32_satisfies_the_real_laws_at_its_own_precision() {
    let xs: Vec<f32> = REALS_F64.iter().map(|&x| x as f32).collect();
    // ~1e-6 relative, the f32 counterpart. Deliberately not the f64 bound: passing 1e-14 here
    // would fail, and loosening the f64 bound to match would stop it catching anything.
    laws::real_laws(&xs, 1e-6, "f32");
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
    laws::boolean_laws(&[false, true], "bool");
}
