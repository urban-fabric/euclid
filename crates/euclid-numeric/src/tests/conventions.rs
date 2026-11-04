//! **Which convention does it follow?** Pins the semantics of the std-mirroring elementary
//! methods, where more than one defensible answer exists.
//!
//! These 24 methods (`sin`, `floor`, `round`, `min`, `signum`, `atan2`, …) share their names
//! with `f64`'s inherent methods and currently carry no doc comments on the trait, on the
//! reasonable assumption that "it does what `f64` does". That assumption is safe for a
//! consumer *calling* them and unsafe for a consumer *implementing* them: an expression or
//! graph carrier has to choose a rounding mode, a NaN policy, and a branch convention, and
//! nothing in the contract currently says which.
//!
//! An unstated convention is the one failure mode that defeats the whole library — two
//! backends resolving it differently means the same formula yields different answers on
//! different interfaces. So this suite fixes the intended behavior as executable
//! specification, and doubles as the list of what the trait docs should say.

use crate::{Compare, Real};

// --- rounding family ----------------------------------------------------

#[test]
fn round_is_half_away_from_zero_not_half_to_even() {
    // The genuinely ambiguous one. IEEE's default is half-to-even (`2.5 -> 2`); Rust's
    // `f64::round` is half-away-from-zero (`2.5 -> 3`). A carrier that lowers to a
    // banker's-rounding primitive would silently disagree.
    assert_eq!(Real::round(0.5f64), 1.0);
    assert_eq!(Real::round(1.5f64), 2.0);
    assert_eq!(
        Real::round(2.5f64),
        3.0,
        "half-away-from-zero, not half-to-even"
    );
    assert_eq!(Real::round(-0.5f64), -1.0);
    assert_eq!(Real::round(-2.5f64), -3.0);
    assert_eq!(Real::round(2.4f64), 2.0);
    assert_eq!(Real::round(2.6f64), 3.0);
}

#[test]
fn trunc_and_floor_differ_on_negatives() {
    // `trunc` rounds toward zero; `floor` toward −∞. They agree only for non-negatives.
    assert_eq!(Real::trunc(2.7f64), 2.0);
    assert_eq!(Real::floor(2.7f64), 2.0);
    assert_eq!(Real::trunc(-2.7f64), -2.0, "trunc goes toward zero");
    assert_eq!(Real::floor(-2.7f64), -3.0, "floor goes toward -inf");
    assert_eq!(Real::ceil(-2.7f64), -2.0);
    assert_eq!(Real::ceil(2.1f64), 3.0);
}

#[test]
fn fract_keeps_the_sign_of_its_argument() {
    // `fract` is `self − trunc(self)`, so it is negative for negative inputs. The other
    // common definition (`self − floor(self)`, always in `[0,1)`) would disagree on sign.
    assert!((Real::fract(2.75f64) - 0.75).abs() < 1e-15);
    assert!(
        (Real::fract(-2.75f64) + 0.75).abs() < 1e-15,
        "fract(-2.75) is -0.75, not +0.25"
    );
    // The defining identity, across signs.
    for i in -40..40 {
        let x = f64::from(i) * 0.37;
        let err = (Real::trunc(x) + Real::fract(x) - x).abs();
        assert!(err < 1e-14, "trunc + fract != x at {x}");
    }
}

#[test]
fn floor_and_ceil_bracket_every_value() {
    for i in -100..100 {
        let x = f64::from(i) * 0.41;
        let (f, c) = (Real::floor(x), Real::ceil(x));
        assert!(f <= x && x <= c, "floor <= x <= ceil failed at {x}");
        assert!(c - f <= 1.0, "floor and ceil differ by at most 1 at {x}");
        // Integral inputs are fixed points.
        assert_eq!(Real::floor(f), f);
        assert_eq!(Real::ceil(c), c);
    }
}

#[test]
fn rounding_is_idempotent_and_preserves_integers() {
    for i in -20..20 {
        let n = f64::from(i);
        assert_eq!(Real::round(n), n);
        assert_eq!(Real::floor(n), n);
        assert_eq!(Real::ceil(n), n);
        assert_eq!(Real::trunc(n), n);
        assert_eq!(Real::fract(n), 0.0);
    }
}

// --- sign family --------------------------------------------------------

#[test]
fn signum_returns_plus_or_minus_one_and_never_zero() {
    // The ambiguity: some languages define sign(0) = 0. Rust's `f64::signum` returns ±1.0
    // even at zero, carrying the sign bit.
    assert_eq!(Real::signum(3.0f64), 1.0);
    assert_eq!(Real::signum(-3.0f64), -1.0);
    assert_eq!(Real::signum(0.0f64), 1.0, "signum(+0) is +1, not 0");
    assert_eq!(Real::signum(-0.0f64), -1.0, "signum(-0) is -1, not 0");
    assert!(Real::signum(f64::NAN).is_nan());
}

#[test]
fn copysign_transfers_only_the_sign_bit() {
    assert_eq!(Real::copysign(3.0f64, 1.0), 3.0);
    assert_eq!(Real::copysign(3.0f64, -1.0), -3.0);
    assert_eq!(Real::copysign(-3.0f64, 1.0), 3.0);
    // Signed zero is a valid sign source — this is how the sign of a zero is observable.
    assert_eq!(Real::copysign(1.0f64, -0.0), -1.0);
    assert_eq!(Real::copysign(1.0f64, 0.0), 1.0);
    // Magnitude is preserved exactly, including for infinities.
    assert_eq!(Real::copysign(f64::INFINITY, -1.0), f64::NEG_INFINITY);
    // abs is the |·| half of the same decomposition.
    for i in -30..30 {
        let x = f64::from(i) * 0.7;
        assert_eq!(Real::copysign(Real::abs(&x), x), x, "copysign(|x|, x) = x");
    }
}

// --- min / max NaN policy -----------------------------------------------

#[test]
fn min_and_max_return_the_non_nan_operand() {
    // This is the sharpest divergence risk on the trait. `f64::min`/`max` implement IEEE
    // minNum/maxNum: a NaN operand is *ignored*. That is the opposite of NaN-propagating
    // arithmetic, and the opposite of what a mask-based `select(a < b, a, b)` would give
    // (a comparison with NaN is false, so the branch would pick `b` unconditionally).
    let n = f64::NAN;
    assert_eq!(Real::min(n, 1.0), 1.0, "NaN is ignored, not propagated");
    assert_eq!(Real::max(n, 1.0), 1.0);
    assert_eq!(Real::min(1.0, n), 1.0);
    assert_eq!(Real::max(1.0, n), 1.0);
    // Both NaN gives NaN, there being nothing else to return.
    assert!(Real::min(n, n).is_nan());

    // Ordinary operands behave as expected.
    assert_eq!(Real::min(2.0f64, 5.0), 2.0);
    assert_eq!(Real::max(2.0f64, 5.0), 5.0);
}

#[test]
fn min_and_max_differ_from_the_mask_based_selection_on_nan() {
    // Spelled out, because a carrier that implements `min` as `select(lt, a, b)` would *not*
    // match: with a NaN operand the comparison is false and the result differs.
    let n = f64::NAN;
    let mask_based = Compare::select(&Compare::lt(&n, &1.0), n, 1.0);
    assert_eq!(mask_based, 1.0);
    assert_eq!(Real::min(n, 1.0), 1.0);
    // Here they agree; with the operands swapped they do not.
    let swapped = Compare::select(&Compare::lt(&1.0, &n), 1.0, n);
    assert!(swapped.is_nan(), "mask-based selection propagates the NaN");
    assert_eq!(Real::min(1.0, n), 1.0, "min does not");
}

#[test]
fn clamp_composes_min_and_max() {
    for i in -50..50 {
        let x = f64::from(i) * 0.3;
        let (lo, hi) = (-2.0f64, 3.0f64);
        assert_eq!(Real::clamp(x, lo, hi), Real::min(Real::max(x, lo), hi));
    }
}

// --- atan2 branch conventions -------------------------------------------

#[test]
fn atan2_follows_the_standard_quadrant_convention() {
    use core::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};
    let close = |a: f64, b: f64, what: &str| {
        assert!((a - b).abs() < 1e-15, "{what}: {a} vs {b}");
    };
    // atan2(y, x) with `self` = y.
    close(Real::atan2(0.0f64, 1.0), 0.0, "atan2(0, 1)");
    close(Real::atan2(1.0f64, 1.0), FRAC_PI_4, "atan2(1, 1)");
    close(Real::atan2(1.0f64, 0.0), FRAC_PI_2, "atan2(1, 0)");
    close(Real::atan2(0.0f64, -1.0), PI, "atan2(0, -1) is +pi");
    close(Real::atan2(-1.0f64, 0.0), -FRAC_PI_2, "atan2(-1, 0)");
    close(
        Real::atan2(-1.0f64, -1.0),
        -3.0 * FRAC_PI_4,
        "third quadrant",
    );
    // Range is (-pi, pi].
    for i in -50..50 {
        for j in -50..50 {
            if i == 0 && j == 0 {
                continue;
            }
            let a = Real::atan2(f64::from(i), f64::from(j));
            assert!(
                a > -PI - 1e-15 && a <= PI + 1e-15,
                "atan2 out of range: {a}"
            );
        }
    }
}

#[test]
fn atan2_at_the_origin_is_defined_not_nan() {
    // Both arguments zero has no mathematical answer; IEEE still defines it by sign.
    assert_eq!(Real::atan2(0.0f64, 0.0), 0.0);
    assert_eq!(Real::atan2(-0.0f64, 0.0), -0.0);
    assert!((Real::atan2(0.0f64, -0.0) - core::f64::consts::PI).abs() < 1e-15);
}

#[test]
fn atan2_recovers_the_angle_of_a_vector() {
    // The motivating use: converting a vector back to its polar angle.
    for i in 1..72 {
        let theta = f64::from(i) * 0.08 - 2.8; // within (-pi, pi)
        let (s, c) = Real::sin_cos(theta);
        let recovered = Real::atan2(s, c);
        assert!(
            (recovered - theta).abs() < 1e-14,
            "atan2(sin t, cos t) != t at {theta}"
        );
    }
}

// --- composite consistency ----------------------------------------------

#[test]
fn square_recip_and_cbrt_agree_with_their_definitions() {
    for i in 1..80 {
        let x = f64::from(i) * 0.37;
        assert!((Real::square(&x) - x * x).abs() < 1e-13 * x * x);
        assert!((Real::recip(x) - 1.0 / x).abs() < 1e-15 / x);
        let c = Real::cbrt(x);
        assert!((c * c * c - x).abs() < 1e-12 * x, "cbrt^3 != x at {x}");
    }
    // cbrt is defined for negative inputs, unlike sqrt.
    assert!((Real::cbrt(-8.0f64) + 2.0).abs() < 1e-14);
}

#[test]
fn log2_agrees_with_the_natural_log_scaled() {
    for i in 1..100 {
        let x = f64::from(i) * 0.5;
        let want = Real::ln(&x) * <f64 as Real>::log2_e();
        assert!(
            (Real::log2(x) - want).abs() < 1e-13 * want.abs().max(1.0),
            "log2 disagrees at {x}"
        );
    }
    // Exact at powers of two.
    for p in -20..20 {
        assert_eq!(Real::log2(2.0f64.powi(p)), f64::from(p));
    }
}

#[test]
fn mul_add_matches_the_unfused_expression_to_within_one_rounding() {
    // `mul_add` may be a single fused instruction, so it can be *more* accurate than
    // `a*b + c`. The contract is that it is not less accurate.
    for i in 1..40 {
        let (a, b, c) = (f64::from(i) * 0.3, 1.7, -0.9);
        let fused = Real::mul_add(a, b, c);
        let unfused = a * b + c;
        assert!(
            (fused - unfused).abs() <= 4.0 * f64::EPSILON * fused.abs().max(1.0),
            "mul_add diverged at {a}"
        );
    }
}

#[test]
fn powi_matches_repeated_multiplication() {
    for base in [-3.0f64, -0.5, 0.5, 2.0, 7.0] {
        for n in 0..10i64 {
            let mut want = 1.0f64;
            for _ in 0..n {
                want *= base;
            }
            let got = Real::powi(base, n);
            assert!(
                (got - want).abs() <= 1e-12 * want.abs().max(1.0),
                "powi({base}, {n}) = {got}, want {want}"
            );
        }
        // Negative exponents are the reciprocal.
        let got = Real::powi(base, -3);
        let want = 1.0 / (base * base * base);
        assert!((got - want).abs() <= 1e-12 * want.abs().max(1.0));
    }
    // powi is defined for negative bases where powf is not.
    assert_eq!(Real::powi(-2.0f64, 2), 4.0);
    assert!(
        Real::powf(&-2.0f64, 0.5).is_nan(),
        "powf of a negative base"
    );
}

#[test]
fn rem_euclid_is_always_non_negative() {
    // The convention that matters for angle wrapping: `rem_euclid` returns a value in
    // `[0, |m|)`, unlike `%` which keeps the sign of the dividend.
    for i in -60..60 {
        let x = f64::from(i) * 0.7;
        let m = 3.0f64;
        let r = Real::rem_euclid(x, m);
        assert!(
            (0.0..m).contains(&r),
            "rem_euclid({x}, {m}) = {r} escaped [0, {m})"
        );
        // And it is a true remainder: x = q*m + r for some integer q.
        let q = Real::round((x - r) / m);
        assert!((q * m + r - x).abs() < 1e-12, "not a remainder at {x}");
    }
    // Contrast with `%`, which keeps the dividend's sign. Read through variables so the
    // comparison is not constant-folded away.
    let (dividend, modulus) = (-1.0f64, 3.0f64);
    assert!(dividend % modulus < 0.0, "`%` keeps the dividend's sign");
    assert_eq!(Real::rem_euclid(dividend, modulus), 2.0);
}

#[test]
fn angle_wrapping_through_rem_euclid_is_stable_over_many_turns() {
    // The practical use: normalizing an accumulated angle. Must stay in range even after
    // many turns, where naive subtraction loops would drift or run long.
    let tau = <f64 as Real>::tau();
    for turns in [1.0f64, 10.0, 1e3, 1e6] {
        let angle = core::f64::consts::FRAC_PI_4 + turns * tau;
        let wrapped = Real::rem_euclid(angle, tau);
        assert!((0.0..tau).contains(&wrapped));
        // At 1e6 turns the input has ~6 digits of the fraction left, so compare loosely.
        assert!(
            (wrapped - core::f64::consts::FRAC_PI_4).abs() < 1e-8,
            "wrapped {wrapped} after {turns} turns"
        );
    }
}

#[test]
fn degree_radian_conversions_match_the_turn_constants() {
    assert_eq!(Real::to_radians(0.0f64), 0.0);
    let half = <f64 as Real>::half_turn_degrees();
    assert!((Real::to_radians(half) - <f64 as Real>::pi()).abs() < 1e-15);
    assert!(
        (Real::to_degrees(<f64 as Real>::pi()) - half).abs() < 1e-13,
        "to_degrees(pi) should be 180"
    );
    // Quarter turn is pi/2.
    assert!(
        (Real::to_radians(<f64 as Real>::quarter_turn_degrees()) - <f64 as Real>::frac_pi_2())
            .abs()
            < 1e-15
    );
}

// --- trigonometric conventions ------------------------------------------

#[test]
fn trig_functions_take_radians_and_hit_their_exact_points() {
    use core::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};
    let close = |a: f64, b: f64, what: &str| assert!((a - b).abs() < 1e-15, "{what}: {a}");
    close(Real::sin(0.0f64), 0.0, "sin 0");
    close(Real::cos(0.0f64), 1.0, "cos 0");
    close(Real::sin(FRAC_PI_2), 1.0, "sin pi/2");
    close(Real::cos(PI), -1.0, "cos pi");
    close(Real::tan(FRAC_PI_4), 1.0, "tan pi/4");
    // Inverse functions return the principal branch.
    close(Real::asin(1.0f64), FRAC_PI_2, "asin 1");
    close(Real::acos(1.0f64), 0.0, "acos 1");
    close(Real::atan(1.0f64), FRAC_PI_4, "atan 1");
    // asin/acos ranges.
    for i in -20..=20 {
        let v = f64::from(i) * 0.05;
        let a = Real::asin(v);
        assert!((-FRAC_PI_2..=FRAC_PI_2).contains(&a), "asin range");
        let c = Real::acos(v);
        assert!((0.0..=PI).contains(&c), "acos range");
    }
}

#[test]
fn hyperbolic_functions_satisfy_their_identity() {
    for i in -30..30 {
        let x = f64::from(i) * 0.2;
        // cosh² − sinh² = 1. The identity is itself cancellation-prone: by |x| = 6 both
        // squares are ~4e4, so the difference loses ~5 digits before it is compared to 1.
        // The tolerance therefore scales with the magnitude being cancelled, not with 1.
        let (s, c) = (Real::sinh(x), Real::cosh(x));
        let cancelled = c * c;
        assert!(
            (c * c - s * s - 1.0).abs() < 1e-14 * cancelled.max(1.0),
            "cosh²-sinh²=1 at {x}"
        );
        // tanh = sinh / cosh
        assert!((Real::tanh(x) - s / c).abs() < 1e-14, "tanh at {x}");
    }
}

// --- the same conventions on f32 ----------------------------------------

#[test]
fn f32_follows_the_same_conventions() {
    // A divergent f32 code path would show up here even though the values differ in precision.
    assert_eq!(Real::round(2.5f32), 3.0);
    assert_eq!(Real::round(-2.5f32), -3.0);
    assert_eq!(Real::trunc(-2.7f32), -2.0);
    assert_eq!(Real::floor(-2.7f32), -3.0);
    assert_eq!(Real::signum(-0.0f32), -1.0);
    assert_eq!(Real::copysign(1.0f32, -0.0), -1.0);
    assert_eq!(Real::min(f32::NAN, 1.0), 1.0);
    assert_eq!(Real::max(f32::NAN, 1.0), 1.0);
    assert_eq!(Real::rem_euclid(-1.0f32, 3.0), 2.0);
    assert!((Real::fract(-2.75f32) + 0.75).abs() < 1e-6);
}
