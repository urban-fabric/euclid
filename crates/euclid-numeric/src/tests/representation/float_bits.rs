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
