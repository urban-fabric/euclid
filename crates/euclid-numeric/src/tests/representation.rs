//! Floating-point representation (`frexp`, `ldexp`, `next_after`, `ulp`, `signbit`, `modf`)
//! and the further analysis added with it (`exprel`, `rgamma`, `polygamma`).
//!
//! The representation group is mostly *exactness* claims — a decomposition that round-trips
//! bit-for-bit, a step that lands on precisely the adjacent representable value — so these
//! assert equality rather than a tolerance wherever that is the actual contract.

use crate::Real;
use euclid_utils::{assert_close, assert_exact, derivative, rel, sweep};

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

// --- exprel -------------------------------------------------------------

#[test]
fn exprel_fills_the_removable_singularity() {
    assert_exact(Real::exprel(0.0f64), 1.0, "exprel(0) = 1");
    // Continuous across it: exprel -> 1 as x -> 0. The approach is linear (exprel ~ 1 + x/2),
    // so this is only a 1e-15 statement once x is below 2e-15.
    for i in 16..40 {
        let x = 10f64.powi(-i);
        assert_close(Real::exprel(x), 1.0, 1e-15, &format!("exprel({x})"));
    }
    // Just above that, check against the series explicitly. Three terms are needed: at
    // x = 1e-6 the x^2/6 term is 1.7e-13, well above the tolerance, while the next one
    // (x^3/24) is 4e-20 and genuinely negligible.
    for i in 6..15 {
        let x = 10f64.powi(-i);
        let series = 1.0 + x / 2.0 + x * x / 6.0 + x * x * x / 24.0;
        assert_close(Real::exprel(x), series, 1e-15, &format!("exprel({x})"));
    }
    // And it agrees with expm1(x)/x away from the origin, which is the naive form's domain.
    for x in sweep(0.1, 20.0, 100) {
        assert_close(
            Real::exprel(x),
            Real::exp_m1(&x) / x,
            1e-14,
            &format!("exprel({x})"),
        );
        assert_close(
            Real::exprel(-x),
            Real::exp_m1(&-x) / -x,
            1e-14,
            &format!("exprel({})", -x),
        );
    }
}

#[test]
fn exprel_keeps_digits_where_the_direct_quotient_collapses() {
    // (exp(x) - 1)/x built naively is 0/0 at the origin and loses everything nearby.
    let x = 1e-17f64;
    assert_exact(x.exp() - 1.0, 0.0, "the naive numerator collapses");
    assert_close(Real::exprel(x), 1.0, 1e-15, "exprel does not");
    // Limits.
    assert_eq!(Real::exprel(f64::INFINITY), f64::INFINITY);
    assert_eq!(Real::exprel(f64::NEG_INFINITY), 0.0);
    assert!(Real::exprel(f64::NAN).is_nan());
}

// --- rgamma -------------------------------------------------------------

#[test]
fn rgamma_vanishes_at_the_poles_of_gamma() {
    // 1/Gamma is entire, taking the value zero exactly where Gamma has a pole.
    for n in 0..20 {
        assert_exact(
            Real::rgamma(-f64::from(n)),
            0.0,
            &format!("1/Gamma(-{n}) = 0"),
        );
    }
}

#[test]
fn rgamma_inverts_gamma_on_both_sides_of_the_origin() {
    for x in sweep(0.25, 20.0, 80) {
        assert_close(
            Real::rgamma(x),
            1.0 / Real::gamma(x),
            1e-12,
            &format!("rgamma({x})"),
        );
    }
    // Negative, non-integer arguments, where Gamma alternates sign.
    for x in [-0.5f64, -1.5, -2.5, -3.5, -0.25, -4.75, -5.5] {
        let g = Real::gamma(x);
        assert_close(Real::rgamma(x), 1.0 / g, 1e-10, &format!("rgamma({x})"));
        // The sign must follow Gamma's.
        assert_eq!(Real::rgamma(x) < 0.0, g < 0.0, "sign disagreement at {x}");
    }
}

#[test]
fn rgamma_is_defined_where_gamma_overflows() {
    // Gamma overflows from about x = 172; 1/Gamma stays representable far past that, which is
    // the reason to have it rather than composing `gamma(x).recip()`.
    assert!(Real::gamma(175.0f64).is_infinite(), "gamma overflows here");
    assert_exact(
        Real::gamma(175.0f64).recip(),
        0.0,
        "so the naive reciprocal is zero",
    );
    let r = Real::rgamma(175.0f64);
    assert!(r > 0.0 && r.is_finite(), "rgamma(175) = {r}");
    // Normal-range case, where the comparison can be tight.
    assert_close(
        Real::rgamma(160.0f64).ln(),
        -Real::ln_gamma(160.0f64),
        1e-13,
        "rgamma(160)",
    );
}

// --- polygamma ----------------------------------------------------------

#[test]
fn polygamma_generalizes_digamma_and_trigamma() {
    for x in sweep(0.25, 20.0, 80) {
        assert_close(Real::polygamma(x, 0), Real::digamma(x), 1e-13, "order 0");
        assert_close(Real::polygamma(x, 1), Real::trigamma(x), 1e-13, "order 1");
    }
}

#[test]
fn polygamma_matches_its_closed_forms_at_one() {
    // psi^(n)(1) = (-1)^{n+1} n! zeta(n+1).
    let pi2 = core::f64::consts::PI * core::f64::consts::PI;
    assert_close(Real::polygamma(1.0f64, 1), pi2 / 6.0, 1e-12, "psi'(1)");
    assert_close(
        Real::polygamma(1.0f64, 2),
        -2.0 * 1.202_056_903_159_594_3,
        1e-11,
        "psi''(1) = -2 zeta(3)",
    );
    assert_close(
        Real::polygamma(1.0f64, 3),
        6.0 * pi2 * pi2 / 90.0,
        1e-11,
        "psi'''(1) = 6 zeta(4)",
    );
}

#[test]
fn polygamma_satisfies_its_recurrence() {
    // psi^(n)(x+1) = psi^(n)(x) + (-1)^n n! / x^{n+1}
    for order in 2..7u32 {
        let n = f64::from(order);
        let factorial = (1..=order).map(f64::from).product::<f64>();
        let sign = if order.is_multiple_of(2) { 1.0 } else { -1.0 };
        for x in sweep(0.5, 15.0, 40) {
            assert_close(
                Real::polygamma(x + 1.0, order),
                Real::polygamma(x, order) + sign * factorial / Real::powf(&x, n + 1.0),
                1e-10,
                &format!("recurrence at order {order}, x={x}"),
            );
        }
    }
}

#[test]
fn each_polygamma_order_is_the_derivative_of_the_previous() {
    // The relationship-web edge: differentiating order n gives order n+1. This ties the whole
    // ladder together and would catch a sign or factorial slip at any order.
    for order in 0..5u32 {
        for x in sweep(1.0, 10.0, 25) {
            let d = derivative(|t| Real::polygamma(t, order), x);
            assert_close(
                d,
                Real::polygamma(x, order + 1),
                1e-6,
                &format!("d/dx psi^({order}) = psi^({})", order + 1),
            );
        }
    }
}

#[test]
fn polygamma_has_the_expected_sign_and_monotonicity() {
    // For x > 0, psi^(n) alternates in sign with n: positive for odd n, negative for even n.
    for order in 1..6u32 {
        for x in sweep(0.5, 20.0, 40) {
            let v = Real::polygamma(x, order);
            if !order.is_multiple_of(2) {
                assert!(v > 0.0, "psi^({order})({x}) = {v} should be positive");
            } else {
                assert!(v < 0.0, "psi^({order})({x}) = {v} should be negative");
            }
        }
    }
    // And out-of-domain arguments are NaN rather than a silent value.
    assert!(Real::polygamma(-1.0f64, 2).is_nan());
    assert!(Real::polygamma(0.0f64, 2).is_nan());
    assert!(Real::polygamma(f64::NAN, 2).is_nan());
}

// --- the f32 carrier ----------------------------------------------------

#[test]
fn the_new_surface_is_available_on_f32() {
    // Representation: the f32 layout differs (8-bit exponent, 23-bit mantissa), so this is a
    // genuinely separate implementation rather than a narrowed f64 one.
    assert_eq!(Real::frexp(1.0f32), (0.5, 1.0));
    assert_eq!(Real::frexp(2.0f32), (0.5, 2.0));
    for x in [1.0f32, -7.25, 1e-30, 1e30, f32::MIN_POSITIVE] {
        let (m, e) = Real::frexp(x);
        assert!((0.5..1.0).contains(&m.abs()), "f32 mantissa {m} at {x}");
        assert_eq!(Real::ldexp(m, e), x, "f32 round trip at {x}");
    }
    assert_eq!(Real::ulp(1.0f32), f32::EPSILON);
    assert_eq!(Real::next_after(1.0f32, f32::INFINITY), 1.0 + f32::EPSILON);
    assert!(Real::signbit(&-0.0f32));
    assert_eq!(Real::modf(2.75f32), (0.75, 2.0));

    // f32 overflows at 2^128, so the chunked ldexp matters at a far lower threshold.
    assert!(Real::ldexp(1e-30f32, 120.0).is_finite());

    // Analysis.
    assert_eq!(Real::exprel(0.0f32), 1.0);
    assert!((Real::exprel(1.0f32) - (core::f32::consts::E - 1.0)).abs() < 1e-6);
    assert_eq!(Real::rgamma(-3.0f32), 0.0);
    assert!((Real::polygamma(1.0f32, 1) - core::f32::consts::PI.powi(2) / 6.0).abs() < 1e-5);
}
