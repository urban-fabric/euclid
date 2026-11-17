//! **Are the lifted constants right?** Every named constant on [`Real`], on both reference
//! carriers.
//!
//! These are hand-transcribed digits or small expressions, so each one is an independent
//! opportunity for a typo that no formula test would localize — a wrong `half_ln_tau` shows up
//! as a slightly-off log-density, not as an obvious failure. Where `core` has the constant it
//! is the reference; where it does not (`euler_gamma`, `ln_pi`, `ln_tau`, the turn constants)
//! the value is checked against a definition computed here.

use crate::{Element, Real};
use euclid_utils::{EULER_GAMMA, assert_close, assert_exact};

/// Tolerance for a constant reached through arithmetic rather than a literal: a few ulp.
const NEAR: f64 = 4.0 * f64::EPSILON;

/// [`assert_close`] at [`NEAR`] — the bar for a derived constant.
#[track_caller]
fn near(got: f64, want: f64, name: &str) {
    assert_close(got, want, NEAR, name);
}

#[test]
fn core_constants_match_exactly() {
    use core::f64::consts;
    assert_exact(<f64 as Real>::pi(), consts::PI, "pi");
    assert_exact(<f64 as Real>::tau(), consts::TAU, "tau");
    assert_exact(<f64 as Real>::frac_pi_2(), consts::FRAC_PI_2, "frac_pi_2");
    assert_exact(<f64 as Real>::frac_pi_4(), consts::FRAC_PI_4, "frac_pi_4");
    assert_exact(<f64 as Real>::sqrt_2(), consts::SQRT_2, "sqrt_2");
    assert_exact(
        <f64 as Real>::frac_1_sqrt_2(),
        consts::FRAC_1_SQRT_2,
        "frac_1_sqrt_2",
    );
    assert_exact(<f64 as Real>::ln_2(), consts::LN_2, "ln_2");
    assert_exact(<f64 as Real>::ln_10(), consts::LN_10, "ln_10");
    assert_exact(<f64 as Real>::e(), consts::E, "e");
    assert_exact(<f64 as Real>::log2_e(), consts::LOG2_E, "log2_e");
    assert_exact(<f64 as Real>::log10_e(), consts::LOG10_E, "log10_e");
    assert_exact(<f64 as Real>::log2_10(), consts::LOG2_10, "log2_10");
    assert_exact(<f64 as Real>::log10_2(), consts::LOG10_2, "log10_2");
}

#[test]
fn simple_constants_are_assert_exact() {
    assert_exact(<f64 as Element>::zero(), 0.0, "zero");
    assert_exact(<f64 as Element>::one(), 1.0, "one");
    assert_exact(<f64 as Real>::two(), 2.0, "two");
    assert_exact(<f64 as Real>::half(), 0.5, "half");
    assert_exact(<f64 as Real>::neg_one(), -1.0, "neg_one");
    assert!(<f64 as Real>::infinity().is_infinite());
    assert!(<f64 as Real>::infinity() > 0.0);
    assert!(<f64 as Real>::neg_infinity().is_infinite());
    assert!(<f64 as Real>::neg_infinity() < 0.0);
    assert!(<f64 as Real>::nan().is_nan());
}

#[test]
fn derived_log_constants_match_their_definitions() {
    use core::f64::consts;
    near(<f64 as Real>::ln_pi(), consts::PI.ln(), "ln_pi");
    near(<f64 as Real>::ln_tau(), consts::TAU.ln(), "ln_tau");
    near(
        <f64 as Real>::half_ln_tau(),
        0.5 * consts::TAU.ln(),
        "half_ln_tau",
    );
    // The normal-density normalizer is the motivating use: ln(1/√(2π)) = −½ ln τ.
    near(
        -<f64 as Real>::half_ln_tau(),
        (1.0 / consts::TAU.sqrt()).ln(),
        "half_ln_tau as normalizer",
    );
}

#[test]
fn euler_gamma_matches_its_known_digits_and_negative_digamma_at_one() {
    // γ = 0.5772156649015328606065120900824024310421…
    near(
        <f64 as Real>::euler_gamma(),
        EULER_GAMMA,
        "euler_gamma digits",
    );
    // The defining relation on this trait: γ = −ψ(1). digamma carries ~1.3e-12 error, so this
    // is the loose end of the comparison, not the constant.
    let err = (<f64 as Real>::euler_gamma() + Real::digamma(1.0)).abs();
    assert!(err < 1e-11, "euler_gamma != -digamma(1): err {err:.3e}");
}

#[test]
fn turn_constants_are_exact_and_consistent() {
    assert_exact(<f64 as Real>::full_turn_degrees(), 360.0, "full turn");
    assert_exact(<f64 as Real>::half_turn_degrees(), 180.0, "half turn");
    assert_exact(<f64 as Real>::quarter_turn_degrees(), 90.0, "quarter turn");
    assert_exact(
        <f64 as Real>::three_half_turn_degrees(),
        540.0,
        "three-half turn",
    );
    // The conversion factors must agree with the turn constants and with `to_radians`.
    near(
        <f64 as Real>::radians_per_degree(),
        core::f64::consts::PI / 180.0,
        "radians_per_degree",
    );
    near(
        <f64 as Real>::degrees_per_radian(),
        180.0 / core::f64::consts::PI,
        "degrees_per_radian",
    );
    // And they are reciprocals.
    near(
        <f64 as Real>::radians_per_degree() * <f64 as Real>::degrees_per_radian(),
        1.0,
        "conversion factors are reciprocal",
    );
}

#[test]
fn constants_satisfy_their_defining_relations() {
    use core::f64::consts;
    // tau = 2 pi
    near(<f64 as Real>::tau(), 2.0 * consts::PI, "tau = 2pi");
    // sqrt_2 squared is 2
    near(
        <f64 as Real>::sqrt_2() * <f64 as Real>::sqrt_2(),
        2.0,
        "sqrt_2^2 = 2",
    );
    // frac_1_sqrt_2 is the reciprocal
    near(
        <f64 as Real>::sqrt_2() * <f64 as Real>::frac_1_sqrt_2(),
        1.0,
        "sqrt_2 * 1/sqrt_2 = 1",
    );
    // ln e = 1
    near(Real::ln(&<f64 as Real>::e()), 1.0, "ln e = 1");
    // log2_e * ln_2 = 1
    near(
        <f64 as Real>::log2_e() * <f64 as Real>::ln_2(),
        1.0,
        "log2_e * ln_2 = 1",
    );
    // log10_e * ln_10 = 1
    near(
        <f64 as Real>::log10_e() * <f64 as Real>::ln_10(),
        1.0,
        "log10_e * ln_10 = 1",
    );
    // log2_10 * log10_2 = 1
    near(
        <f64 as Real>::log2_10() * <f64 as Real>::log10_2(),
        1.0,
        "log2_10 * log10_2 = 1",
    );
}

#[test]
fn scalar_lifting_round_trips_the_constants() {
    // `scalar` is how a carrier lifts an arbitrary host literal; on f64 it must be the
    // identity, so every constant should be reproducible through it.
    for want in [
        core::f64::consts::PI,
        core::f64::consts::E,
        0.5,
        -1.0,
        1e-300,
        1e300,
    ] {
        assert_exact(
            <f64 as Real>::scalar(want),
            want,
            "scalar is identity on f64",
        );
    }
    // `scalar_like` takes context from an operand but must produce the same value.
    let ctx = 42.0f64;
    assert_exact(
        Real::scalar_like(&ctx, core::f64::consts::PI),
        core::f64::consts::PI,
        "scalar_like on f64",
    );
}

// --- the f32 carrier ----------------------------------------------------

#[test]
fn f32_constants_match_core_exactly() {
    use core::f32::consts;
    assert_eq!(<f32 as Real>::pi(), consts::PI);
    assert_eq!(<f32 as Real>::tau(), consts::TAU);
    assert_eq!(<f32 as Real>::frac_pi_2(), consts::FRAC_PI_2);
    assert_eq!(<f32 as Real>::frac_pi_4(), consts::FRAC_PI_4);
    assert_eq!(<f32 as Real>::sqrt_2(), consts::SQRT_2);
    assert_eq!(<f32 as Real>::frac_1_sqrt_2(), consts::FRAC_1_SQRT_2);
    assert_eq!(<f32 as Real>::ln_2(), consts::LN_2);
    assert_eq!(<f32 as Real>::ln_10(), consts::LN_10);
    assert_eq!(<f32 as Real>::e(), consts::E);
    assert_eq!(<f32 as Real>::log2_e(), consts::LOG2_E);
    assert_eq!(<f32 as Real>::log10_e(), consts::LOG10_E);
    assert_eq!(<f32 as Real>::two(), 2.0);
    assert_eq!(<f32 as Real>::half(), 0.5);
}

#[test]
fn f32_and_f64_constants_agree_to_single_precision() {
    // Each f32 constant must be the f64 one rounded, not an independently-typed literal that
    // drifted. Checked through the trait so a divergent f32 impl is caught.
    let pairs: [(f32, f64); 14] = [
        (<f32 as Real>::pi(), <f64 as Real>::pi()),
        (<f32 as Real>::tau(), <f64 as Real>::tau()),
        (<f32 as Real>::e(), <f64 as Real>::e()),
        (<f32 as Real>::ln_2(), <f64 as Real>::ln_2()),
        (<f32 as Real>::ln_10(), <f64 as Real>::ln_10()),
        (<f32 as Real>::ln_pi(), <f64 as Real>::ln_pi()),
        (<f32 as Real>::ln_tau(), <f64 as Real>::ln_tau()),
        (<f32 as Real>::half_ln_tau(), <f64 as Real>::half_ln_tau()),
        (<f32 as Real>::euler_gamma(), <f64 as Real>::euler_gamma()),
        (<f32 as Real>::sqrt_2(), <f64 as Real>::sqrt_2()),
        (<f32 as Real>::log2_e(), <f64 as Real>::log2_e()),
        (<f32 as Real>::log10_e(), <f64 as Real>::log10_e()),
        (<f32 as Real>::log2_10(), <f64 as Real>::log2_10()),
        (<f32 as Real>::log10_2(), <f64 as Real>::log10_2()),
    ];
    for (idx, (narrow, wide)) in pairs.iter().enumerate() {
        // The f32 value should be within one f32 ulp of the f64 value.
        let err = (f64::from(*narrow) - wide).abs();
        let tol = f64::from(f32::EPSILON) * wide.abs().max(1.0);
        assert!(
            err <= tol,
            "constant {idx}: f32 {narrow} vs f64 {wide} (err {err:.3e} > {tol:.3e})"
        );
    }
}

#[test]
fn f32_non_finite_sentinels_are_available() {
    assert!(<f32 as Real>::infinity().is_infinite());
    assert!(<f32 as Real>::neg_infinity() < 0.0);
    assert!(<f32 as Real>::nan().is_nan());
    assert_eq!(<f32 as Element>::zero(), 0.0);
    assert_eq!(<f32 as Element>::one(), 1.0);
}
