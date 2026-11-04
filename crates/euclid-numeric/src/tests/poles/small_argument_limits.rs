//! Functions with a linear limit at the origin, checked where their *internals* leave the
//! format long before their arguments do.
//!
//! Part of the [poles suite](super). Several functions here are built on a squared or halved
//! argument, and `x²` leaves `f64` at `1e-154` while `x` survives to `1e-308`. That gap is
//! invisible from the outside and was where three defects lived:
//!
//! - **`erf(x) = gamma_lr(½, x²)`.** At `x = 1e-154` the square is already subnormal and
//!   carries a handful of digits, so `erf(1e-160)` was wrong by `5.6e-6`; by `x = 1.6e-162`
//!   the square was exactly zero and `erf` returned **zero**, where the true `1.13e-162` is an
//!   ordinary number.
//! - **`erf_inv`** finishes with one Newton step through `erf`, so it inherited the failure —
//!   returning a constant `√π·p` instead of `(√π/2)·p` for every `p` below `1.5e-150`, a
//!   factor of two. Fixing `erf` repaired both, which is what a shared-dependency bug looks
//!   like from the outside.
//! - **`I₁(x)`** reached its answer through `exp(ln(x/2) + ⋯)`, and `ln(x/2) = −691` at
//!   `x = 1e-300` carries `ε·691` of absolute error that the `exp` converts into *relative*
//!   error: `3.5e-14`, against a documented `1.7e-14`.
//!
//! What the three have in common is that the limit itself is perfectly conditioned — `erf(x)`
//! really is `2x/√π`, `I₁(x)` really is `x/2` — so the right answer is not merely close but
//! *exact to rounding*, and that is what these tests demand. A relative comparison is
//! essential: all three defects produced large *relative* error on tiny *absolute* values,
//! which is precisely what the mixed criterion used elsewhere forgives.

use crate::Real;

/// Arguments from where the next series term first falls below half an ulp, down to the
/// bottom of the format. The interesting ones are `1e-154` (where `x²` turns subnormal) and
/// `1e-162` (where it reaches zero).
const FROM_CUBIC_CUTOFF: &[f64] = &[
    1e-8,
    1e-20,
    1e-100,
    1e-150,
    1e-154,
    1e-160,
    1e-162,
    1e-200,
    1e-300,
    f64::MIN_POSITIVE,
];

/// Arguments for the functions whose second term is *linear* rather than quadratic — `W(x)` is
/// `x − x²` and `I₁ᵉ(x)` is `(x/2)(1 − x)` — so they only reach their limit once `x` itself is
/// below half an ulp.
const BELOW_HALF_ULP: &[f64] = &[1e-20, 1e-100, 1e-162, 1e-300, f64::MIN_POSITIVE];

/// Assert `f(x) == slope·x` to within a couple of ulps across `args`.
#[track_caller]
fn linear_at_the_origin(name: &str, f: impl Fn(f64) -> f64, slope: f64, args: &[f64]) {
    for &x in args {
        let expected = slope * x;
        let got = f(x);
        let err = ((got - expected) / expected).abs();
        assert!(
            err <= 2.0 * f64::EPSILON,
            "{name}({x:e}) = {got:e}, expected {expected:e} (rel {err:e})"
        );
    }
}

#[test]
fn erf_is_exactly_its_linear_term_at_the_origin() {
    // erf(x) = (2/√π)(x − x³/3 + ⋯); below 1e-8 the cubic term is under half an ulp.
    let slope = 2.0 / core::f64::consts::PI.sqrt();
    linear_at_the_origin("erf", Real::erf, slope, FROM_CUBIC_CUTOFF);
    // Odd, including the sign of a zero — which the old `if x == 0.0 { 0.0 }` branch dropped.
    for &x in FROM_CUBIC_CUTOFF {
        assert_eq!(Real::erf(-x), -Real::erf(x), "erf must be odd at {x:e}");
    }
    assert_eq!(Real::erf(0.0f64), 0.0);
    assert!(
        Real::erf(-0.0f64).is_sign_negative(),
        "erf(-0) must be -0, since erf is odd"
    );
    // And the complement saturates to exactly 1 once erf is below half an ulp of it — which
    // needs x below eps/2, not merely below the cubic cutoff.
    for &x in BELOW_HALF_ULP {
        assert_eq!(Real::erfc(x), 1.0, "erfc({x:e}) should round to exactly 1");
        assert_eq!(Real::erfcx(x), 1.0, "erfcx({x:e}) likewise");
    }
}

#[test]
fn erf_inv_is_exactly_its_linear_term_at_the_origin() {
    // erf_inv(p) = (√π/2)(p + πp³/12 + ⋯). The inverse is the harder direction, because its
    // final Newton step runs through `erf` — so this test passes only while `erf` itself is
    // exact down here.
    linear_at_the_origin(
        "erf_inv",
        Real::erf_inv,
        core::f64::consts::PI.sqrt() * 0.5,
        FROM_CUBIC_CUTOFF,
    );
    for &x in FROM_CUBIC_CUTOFF {
        assert_eq!(Real::erf_inv(-x), -Real::erf_inv(x), "erf_inv must be odd");
    }
    // The round trip closes to a single rounding in each direction.
    for &p in FROM_CUBIC_CUTOFF {
        let back = Real::erf(Real::erf_inv(p));
        assert!(
            ((back - p) / p).abs() <= 2.0 * f64::EPSILON,
            "erf(erf_inv({p:e})) = {back:e} should return p"
        );
    }
}

#[test]
fn the_other_odd_functions_are_exact_at_the_origin() {
    // Each has an exactly-conditioned linear limit, so each should be exact — and each is
    // reached by a different mechanism, so this is five independent statements rather than one.
    linear_at_the_origin("Si", Real::sin_int, 1.0, FROM_CUBIC_CUTOFF);
    linear_at_the_origin("I1", Real::bessel_i1, 0.5, FROM_CUBIC_CUTOFF);
    linear_at_the_origin("J1", Real::bessel_j1, 0.5, FROM_CUBIC_CUTOFF);
    // `W(x) = x − x² + ⋯` and `I₁ᵉ(x) = (x/2)(1 − x + ⋯)` have a *linear* correction, so they
    // need `x` itself below half an ulp before the leading term is the whole answer.
    linear_at_the_origin("W", Real::lambert_w, 1.0, BELOW_HALF_ULP);
    linear_at_the_origin("I1e", Real::bessel_i1e, 0.5, BELOW_HALF_ULP);
    // The even functions saturate to their own limits instead.
    for &x in BELOW_HALF_ULP {
        assert_eq!(Real::bessel_j0(x), 1.0, "J0 -> 1");
        assert_eq!(Real::bessel_i0(x), 1.0, "I0 -> 1");
        assert_eq!(Real::sinc(x), 1.0, "sinc -> 1");
        assert_eq!(Real::exprel(x), 1.0, "exprel -> 1");
    }
    // `cos_m1` is the exception, and deliberately so: it must *not* saturate to zero, because
    // resolving `cos(x) − 1 = −x²/2` where the naive subtraction gives zero is the entire
    // reason it exists. So the statement here is the quadratic limit, not a limit of 1.
    for &x in BELOW_HALF_ULP {
        let expected = -(x * x) * 0.5;
        let got = Real::cos_m1(x);
        assert!(
            got == expected || ((got - expected) / expected).abs() <= 2.0 * f64::EPSILON,
            "cos_m1({x:e}) = {got:e}, expected -x^2/2 = {expected:e}"
        );
    }
}

#[test]
fn the_f32_carrier_has_the_same_limits() {
    // The same statement one carrier down, where the window in which a squared argument is
    // lost is *wider*, not narrower: `x²` turns subnormal at `1e-19` rather than `1e-154`.
    let slope = 2.0 / core::f32::consts::PI.sqrt();
    for &x in &[
        1e-5f32,
        1e-10,
        1e-19,
        1e-22,
        1e-30,
        1e-38,
        f32::MIN_POSITIVE,
    ] {
        let expected = slope * x;
        let got = Real::erf(x);
        assert!(
            ((got - expected) / expected).abs() <= 2.0 * f32::EPSILON,
            "f32 erf({x:e}) = {got:e}, expected {expected:e}"
        );
        assert_eq!(Real::erf(-x), -Real::erf(x), "f32 erf must be odd");

        let expected_inv = core::f32::consts::PI.sqrt() * 0.5 * x;
        let got_inv = Real::erf_inv(x);
        assert!(
            ((got_inv - expected_inv) / expected_inv).abs() <= 4.0 * f32::EPSILON,
            "f32 erf_inv({x:e}) = {got_inv:e}, expected {expected_inv:e}"
        );
    }
    // f32's eps/2 is 6e-8, so the complement saturates from about 5e-8 downward.
    for &x in &[1e-10f32, 1e-20, 1e-30, 1e-38] {
        assert_eq!(Real::erfc(x), 1.0, "f32 erfc({x:e}) should be exactly 1");
        assert_eq!(Real::bessel_i0(x), 1.0);
        assert!(Real::bessel_i1(x) <= x && Real::bessel_i1(x) >= 0.0);
    }
}
