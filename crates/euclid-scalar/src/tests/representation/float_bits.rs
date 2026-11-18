//! Getting at the bits: frexp/ldexp, next_after/ulp, signbit/modf.
//!
//! Part of the [`representation`](super) suite.

use crate::Real;
use euclid_utils::{assert_close, assert_exact, rel, sweep};

// --- frexp / ldexp ------------------------------------------------------

#[test]
fn frexp_normalizes_and_round_trips_exactly() {
    for x in [
        1.0f64,
        0.5,
        2.0,
        3.0,
        -7.25,
        1e-300,
        1e300,
        f64::MIN_POSITIVE,
        f64::MAX,
        core::f64::consts::PI,
    ] {
        let (m, e) = Real::frexp(x);
        assert!(
            (0.5..1.0).contains(&m.abs()),
            "frexp({x}) mantissa {m} outside [0.5, 1)"
        );
        assert_exact(e, e.trunc(), "the exponent is integral");
        // The round trip must be bit-exact: both halves are powers of two apart.
        assert_exact(Real::ldexp(m, e), x, &format!("frexp/ldexp at {x}"));
    }
}

#[test]
fn frexp_gives_the_expected_decompositions() {
    assert_eq!(Real::frexp(1.0f64), (0.5, 1.0));
    assert_eq!(Real::frexp(2.0f64), (0.5, 2.0));
    assert_eq!(Real::frexp(0.5f64), (0.5, 0.0));
    assert_eq!(Real::frexp(0.75f64), (0.75, 0.0));
    assert_eq!(Real::frexp(-1.0f64), (-0.5, 1.0));
    // Zero and non-finite values pass through with a zero exponent.
    assert_eq!(Real::frexp(0.0f64), (0.0, 0.0));
    assert_eq!(Real::frexp(f64::INFINITY), (f64::INFINITY, 0.0));
    assert!(Real::frexp(f64::NAN).0.is_nan());
}

#[test]
fn frexp_normalizes_subnormals() {
    // The mantissa must land in range for subnormals too, which needs a pre-scaling step.
    let mut x = f64::MIN_POSITIVE;
    for _ in 0..40 {
        x /= 2.0;
        if x == 0.0 {
            break;
        }
        let (m, e) = Real::frexp(x);
        assert!(
            (0.5..1.0).contains(&m.abs()),
            "subnormal {x:e}: mantissa {m} outside [0.5, 1)"
        );
        assert_exact(Real::ldexp(m, e), x, "subnormal round trip");
    }
}

#[test]
fn ldexp_is_exact_and_survives_unrepresentable_powers() {
    // Exact at every modest exponent, since scaling by a power of two cannot round.
    for n in -200..200 {
        let e = f64::from(n);
        assert_exact(
            Real::ldexp(3.0f64, e),
            3.0 * Real::exp2(e),
            &format!("ldexp 3, {n}"),
        );
    }
    // 2^1100 is not representable, but 1e-300 * 2^1100 is — the chunked application is what
    // makes that work, where a single `x * 2f64.powi(n)` would give infinity.
    let v = Real::ldexp(1e-300f64, 1100.0);
    assert!(v.is_finite() && v > 0.0, "ldexp = {v}");
    assert!(
        rel(v, 1e-300 * 2f64.powi(550) * 2f64.powi(550)) < 1e-12,
        "got {v}"
    );
    // And the reverse direction, where a single power would be zero.
    let w = Real::ldexp(1e300f64, -1100.0);
    assert!(w > 0.0 && w.is_finite(), "ldexp = {w}");
}

#[test]
fn ldexp_and_frexp_compose_over_a_wide_sweep() {
    for x in sweep(-100.0, 100.0, 401) {
        if x == 0.0 {
            continue;
        }
        let (m, e) = Real::frexp(x);
        assert_exact(Real::ldexp(m, e), x, &format!("round trip at {x}"));
    }
}

// --- next_after / ulp ---------------------------------------------------

#[test]
fn next_after_steps_exactly_one_representable_value() {
    assert_exact(
        Real::next_after(1.0f64, f64::INFINITY),
        1.0 + f64::EPSILON,
        "one step above 1.0",
    );
    // Reversible at every magnitude.
    for x in [1e-300f64, 1e-10, 0.5, 1.0, 3.0, 1e10, 1e300, -2.5, -1e-200] {
        let up = Real::next_after(x, f64::INFINITY);
        assert_exact(
            Real::next_after(up, f64::NEG_INFINITY),
            x,
            &format!("reversible at {x}"),
        );
        assert!(up > x, "stepping toward +inf must increase");
    }
    // Equal arguments return the target; NaN propagates.
    assert_exact(Real::next_after(2.0f64, 2.0), 2.0, "equal arguments");
    assert!(Real::next_after(f64::NAN, 1.0).is_nan());
    // From *either* argument. Testing only the first left the guard's `||` weakenable to `&&`:
    // a NaN target then falls through to the bit walk, and `next_after(1.0, NaN)` returns
    // `1.0 - 1ulp` — a plausible neighbour, for a direction that was never specified. The NaN
    // *source* hides this, because walking a NaN's bits by one yields another NaN.
    assert!(
        Real::next_after(1.0f64, f64::NAN).is_nan(),
        "a NaN target propagates rather than choosing a direction"
    );
    assert!(Real::next_after(-3.0f64, f64::NAN).is_nan());
    assert!(Real::next_after(0.0f64, f64::NAN).is_nan());
    assert!(Real::next_after(f64::NAN, f64::NAN).is_nan());
    // Stepping off zero reaches the smallest subnormal, with the right sign.
    assert!(Real::next_after(0.0f64, 1.0) > 0.0);
    assert!(Real::next_after(0.0f64, 1.0) < f64::MIN_POSITIVE);
    assert!(Real::next_after(0.0f64, -1.0) < 0.0);
}

#[test]
fn ulp_is_the_spacing_and_scales_with_magnitude() {
    assert_exact(Real::ulp(1.0f64), f64::EPSILON, "ulp(1) is EPSILON");
    assert_exact(
        Real::ulp(2.0f64),
        2.0 * f64::EPSILON,
        "ulp doubles with the binade",
    );
    assert_exact(Real::ulp(4.0f64), 4.0 * f64::EPSILON, "and again");
    // The step really lands one representable value away.
    for x in [1e-300f64, 1e-5, 1.0, 1e5, 1e300] {
        let u = Real::ulp(x);
        assert!(u > 0.0, "ulp({x}) = {u}");
        assert_exact(
            x + u,
            Real::next_after(x, f64::INFINITY),
            &format!("ulp is the step at {x}"),
        );
    }
    assert!(Real::ulp(f64::NAN).is_nan());
}

/// `ulp` steps *away from zero*, which only a negative power of two can demonstrate.
///
/// The docstring says "the distance from `x` to the next representable value away from zero",
/// and on the negative axis that is the direction `x < 0.0` selects. Replacing that test with
/// `x == 0.0` — so every negative argument is measured *toward* zero instead — survived the
/// suite, because the two directions have the same spacing everywhere except at a binade
/// boundary, where the one below is half as wide. The sweep used `-3.0`, which sits inside a
/// binade and cannot tell the two apart; `-2.0` is off by exactly a factor of two.
///
/// Confirmed at 55 `f64` and 254 `f32` values by `bin/equivalence.py`, which also establishes
/// that relaxing the same comparison to `<=` is *not* observable — `abs()` absorbs it at the
/// two zeros, the only place it differs.
#[test]
fn ulp_measures_the_step_away_from_zero_on_the_negative_axis() {
    // Each of these is a binade boundary, where the step toward zero is half the step away.
    for (x, want) in [
        (-1.0f64, f64::EPSILON),
        (-2.0, 2.0 * f64::EPSILON),
        (-4.0, 4.0 * f64::EPSILON),
        (-0.5, f64::EPSILON / 2.0),
    ] {
        assert_exact(Real::ulp(x), want, &format!("ulp({x}) is the binade above"));
        // Which is to say: subtracting it lands on the neighbour further from zero.
        assert_exact(
            x - Real::ulp(x),
            Real::next_after(x, f64::NEG_INFINITY),
            &format!("ulp steps away from zero at {x}"),
        );
    }
    // The magnitude is symmetric even though the direction is not.
    for x in [1.0f64, 2.0, 0.5, 3.0, 1e300, 1e-300] {
        assert_exact(
            Real::ulp(-x),
            Real::ulp(x),
            &format!("ulp(-{x}) matches ulp({x})"),
        );
    }
    // Both zeros report the smallest subnormal, whichever way they are signed.
    assert_exact(Real::ulp(0.0f64), f64::from_bits(1), "ulp(0)");
    assert_exact(Real::ulp(-0.0f64), f64::from_bits(1), "ulp(-0)");
    assert!(Real::ulp(f64::INFINITY).is_infinite(), "ulp(inf)");
}

#[test]
fn ulp_expresses_a_tolerance_that_a_fixed_epsilon_cannot() {
    // The motivating use: "within n representable steps" means the same thing at every
    // magnitude, where a fixed EPSILON does not.
    for x in [1e-200f64, 1.0, 1e200] {
        let four_steps = {
            let mut v = x;
            for _ in 0..4 {
                v = Real::next_after(v, f64::INFINITY);
            }
            v
        };
        assert!(
            (four_steps - x).abs() <= 4.0 * Real::ulp(x) * 1.001,
            "4 ulp at {x}"
        );
        // A fixed epsilon is meaningless at the extremes: it is larger than the value at
        // 1e-200 and smaller than one step at 1e200.
        if x > 1e100 {
            assert!(
                f64::EPSILON < Real::ulp(x),
                "EPSILON is below one ulp at {x}"
            );
        }
    }
}

// --- signbit / modf -----------------------------------------------------

#[test]
fn signbit_observes_what_comparison_cannot() {
    assert!(!Real::signbit(&1.0f64));
    assert!(Real::signbit(&-1.0f64));
    // The whole point: negative zero compares equal to zero but has its sign bit set.
    assert!(Real::signbit(&-0.0f64));
    assert!(!Real::signbit(&0.0f64));
    assert_eq!(-0.0f64, 0.0f64, "they compare equal");
    // And it reports a NaN's sign rather than propagating it.
    assert!(!Real::signbit(&f64::NAN) || Real::signbit(&f64::NAN));
    assert!(Real::signbit(&f64::NEG_INFINITY));
    assert!(!Real::signbit(&f64::INFINITY));
}

#[test]
fn modf_splits_into_parts_that_sum_back() {
    for x in sweep(-10.0, 10.0, 201) {
        let (fract, trunc) = Real::modf(x);
        assert_close(fract + trunc, x, 1e-15, &format!("modf parts sum at {x}"));
        assert_exact(trunc, Real::trunc(x), "integral part matches trunc");
        assert_exact(fract, Real::fract(x), "fractional part matches fract");
        assert!(fract.abs() < 1.0, "fractional part is below 1 in magnitude");
        // Both carry the sign of the argument.
        if x < 0.0 {
            assert!(
                fract <= 0.0 && trunc <= 0.0,
                "signs follow the argument at {x}"
            );
        }
    }
    assert_eq!(Real::modf(2.75f64), (0.75, 2.0));
    assert_eq!(Real::modf(-2.75f64), (-0.75, -2.0));
}

/// Cover special inputs and scaling beyond the range of a standalone power of two.
#[test]
fn ldexp_handles_special_inputs_and_large_exponents() {
    assert_exact(Real::ldexp(0.0f64, 5.0), 0.0, "ldexp(0, n) = 0");
    assert!(
        Real::ldexp(-0.0f64, 5.0).is_sign_negative(),
        "a signed zero keeps its sign through ldexp"
    );
    assert_exact(
        Real::ldexp(f64::INFINITY, 5.0),
        f64::INFINITY,
        "inf scales to inf",
    );
    assert_exact(
        Real::ldexp(f64::NEG_INFINITY, -5.0),
        f64::NEG_INFINITY,
        "-inf scales to -inf",
    );
    assert!(Real::ldexp(f64::NAN, 5.0).is_nan(), "NaN scales to NaN");
    for zero in [0.0_f64, -0.0] {
        assert_eq!(Real::ldexp(zero, f64::MAX).to_bits(), zero.to_bits());
    }
    for infinite in [f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(Real::ldexp(infinite, -f64::MAX), infinite);
    }
    for &exp in &[f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
        assert_exact(
            Real::ldexp(3.25f64, exp),
            3.25,
            "a non-finite exp returns self",
        );
    }

    for &n in &[511.0f64, 512.0, 513.0, -511.0, -512.0, -513.0] {
        let got = Real::ldexp(1.0f64, n);
        #[allow(clippy::cast_possible_truncation)]
        let want = 2f64.powi(n as i32);
        assert_exact(got, want, &format!("ldexp(1, {n}) = 2^{n}"));
    }
    // The scaled values themselves exceed the finite and subnormal ranges.
    assert!(
        Real::ldexp(1e-300f64, 4000.0).is_infinite(),
        "overflow is reported once the result really is unrepresentable"
    );
    assert_exact(
        Real::ldexp(1e300f64, -3000.0),
        0.0,
        "underflow reaches zero",
    );

    assert_eq!(Real::ldexp(0.0f32, 5.0), 0.0_f32, "f32: ldexp(0, n) = 0");
    assert!(
        Real::ldexp(-0.0f32, 5.0).is_sign_negative(),
        "f32: a signed zero keeps its sign"
    );
    assert_eq!(Real::ldexp(f32::INFINITY, 5.0), f32::INFINITY, "f32: inf");
    assert!(Real::ldexp(f32::NAN, 5.0).is_nan(), "f32: NaN");
    for &exp in &[f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
        assert_eq!(
            Real::ldexp(3.25f32, exp),
            3.25_f32,
            "f32: non-finite exp returns self"
        );
    }
    for &n in &[59.0f32, 60.0, 61.0, -59.0, -60.0, -61.0] {
        let got = Real::ldexp(1.0f32, n);
        #[allow(clippy::cast_possible_truncation)]
        let want = 2f32.powi(n as i32);
        assert_eq!(got, want, "f32: ldexp(1, {n}) = 2^{n}");
    }
    assert!(
        Real::ldexp(1e-30f32, 400.0).is_infinite(),
        "f32: overflow is reported"
    );
    // The wider type represents the scale factors exactly.
    let scaled_up = Real::ldexp(1e-30f32, 200.0);
    let want = f64::from(1e-30f32) * 2f64.powi(200);
    assert!(
        scaled_up.is_finite() && rel(f64::from(scaled_up), want) < 1e-6,
        "f32: 1e-30 · 2^200 is finite even though 2^200 is not; got {scaled_up:e}"
    );
    let scaled_down = Real::ldexp(1e30f32, -200.0);
    assert!(
        scaled_down > 0.0 && scaled_down.is_finite(),
        "f32: 1e30 · 2^-200 is normal even though 2^-200 is not; got {scaled_down:e}"
    );
    assert_eq!(
        Real::ldexp(1e30f32, -400.0),
        0.0_f32,
        "f32: underflow reaches zero"
    );
}
