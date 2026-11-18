//! The further-analysis surface: exprel, rgamma, polygamma.
//!
//! Part of the [`representation`](super) suite.

use crate::Real;
use euclid_utils::{assert_close, assert_exact, derivative, sweep};

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

/// The two infinities, which differ — and which nothing reached in either precision.
///
/// Six mutants lived in this one comparison, three per precision, because no test ever passed
/// `rgamma` a non-finite argument. The arms are genuinely different and the asymmetry is the
/// mathematical content: `Γ` grows without bound as `x → +∞`, so `1/Γ → 0`; in the negative
/// direction `Γ` alternates sign across every pole and has no limit at all, so `1/Γ` has none
/// either and `NaN` is the honest answer rather than a refusal to compute.
///
/// Swapping the comparison therefore reports a definite zero where the limit does not exist,
/// which is the worse of the two directions to get wrong: a caller sees a plausible number.
#[test]
fn rgamma_distinguishes_the_two_infinities() {
    assert_exact(Real::rgamma(f64::INFINITY), 0.0, "1/Gamma(+inf) = 0");
    assert!(
        Real::rgamma(f64::NEG_INFINITY).is_nan(),
        "1/Gamma(-inf) has no limit, so it must be NaN"
    );
    assert!(Real::rgamma(f64::NAN).is_nan(), "NaN in, NaN out");

    // The f32 peer carries its own copy of this branch, so it needs its own check.
    assert_eq!(
        Real::rgamma(f32::INFINITY),
        0.0_f32,
        "f32: 1/Gamma(+inf) = 0"
    );
    assert!(
        Real::rgamma(f32::NEG_INFINITY).is_nan(),
        "f32: 1/Gamma(-inf) must be NaN"
    );
    assert!(Real::rgamma(f32::NAN).is_nan(), "f32: NaN in, NaN out");
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
    f32_next_after_steps_off_zero_with_the_right_sign();
    f32_frexp_guards_its_edge_cases_and_ulp_steps_away_from_zero();
    assert!(Real::signbit(&-0.0f32));
    assert_eq!(Real::modf(2.75f32), (0.75, 2.0));

    assert!(Real::ldexp(1e-30f32, 120.0).is_finite());

    // Analysis.
    assert_eq!(Real::exprel(0.0f32), 1.0);
    assert!((Real::exprel(1.0f32) - (core::f32::consts::E - 1.0)).abs() < 1e-6);
    assert_eq!(Real::rgamma(-3.0f32), 0.0);
    assert!((Real::polygamma(1.0f32, 1) - core::f32::consts::PI.powi(2) / 6.0).abs() < 1e-5);
}

/// Stepping off zero must carry the sign of the *target*, in `f32` as in `f64`.
///
/// A deleted negation survived here: `next_after(0.0, -1.0)` returning `+MIN_SUBNORMAL` instead
/// of `-MIN_SUBNORMAL`. The `f64` suite asserts this sign and the f32 peer did not — the two
/// layouts are genuinely separate implementations (8-bit exponent against 11), so this is the
/// same inherited-nothing situation as `Interpolate` and the `NormedDivisionAlgebra` tier.
///
/// It matters for the use `ulp` and `next_after` exist to serve: a bracketing search that steps
/// off zero in the wrong direction walks away from its root rather than toward it, and the
/// magnitude is right so nothing downstream looks wrong.
fn f32_next_after_steps_off_zero_with_the_right_sign() {
    let up = Real::next_after(0.0f32, 1.0);
    let down = Real::next_after(0.0f32, -1.0);
    assert!(
        up > 0.0,
        "stepping off zero toward +1 must be positive, got {up:e}"
    );
    assert!(
        down < 0.0,
        "stepping off zero toward -1 must be negative, got {down:e}"
    );
    assert_eq!(up, f32::from_bits(1), "and it is the smallest subnormal");
    assert_eq!(down, -f32::from_bits(1), "with the sign bit set");
    assert!(
        up < f32::MIN_POSITIVE,
        "a subnormal is below the smallest normal"
    );

    // Reversible at every magnitude, which is what makes it usable as a step.
    for x in [1e-30f32, 0.5, 1.0, 3.0, 1e30, -2.5] {
        let stepped = Real::next_after(x, f32::INFINITY);
        assert!(stepped > x, "stepping toward +inf must increase at {x:e}");
        assert_eq!(
            Real::next_after(stepped, f32::NEG_INFINITY),
            x,
            "reversible at {x:e}"
        );
        // `ulp` is that same step, which ties the two together.
        assert_eq!(x + Real::ulp(x), stepped, "ulp is the step at {x:e}");
    }
    // Equal arguments return the target, and NaN propagates — from either argument. A NaN
    // *target* is the one that matters: with the guard's `||` weakened to `&&` it falls through
    // to the bit walk and returns a neighbour, while a NaN *source* still yields a NaN.
    assert_eq!(Real::next_after(2.0f32, 2.0), 2.0);
    assert!(Real::next_after(f32::NAN, 1.0).is_nan());
    assert!(
        Real::next_after(1.0f32, f32::NAN).is_nan(),
        "f32: a NaN target propagates"
    );
    assert!(Real::next_after(-3.0f32, f32::NAN).is_nan());
    assert!(Real::next_after(0.0f32, f32::NAN).is_nan());
    assert!(Real::ulp(f32::NAN).is_nan());
}

/// `frexp`'s guard and `ulp`'s direction at `f32`, both of which the f64 suite alone pinned.
///
/// Two survivors, and the same cause behind each: an assertion the f64 file makes about a
/// separately-implemented peer.
///
/// **`frexp`'s guard was never entered.** Weakening `if x == 0.0 || !x.is_finite()` to `&&`
/// makes the condition unsatisfiable — nothing is both zero and non-finite — so every
/// argument falls through to the decomposition. For zero that is **unbounded recursion**: the
/// exponent field reads as `0`, which is the subnormal path, so it rescales zero and asks
/// again forever. The f64 peer catches this in `frexp_gives_the_expected_decompositions`; the
/// f32 surface check tested only `1.0` and `2.0` and never passed a zero.
///
/// **`ulp` measured the wrong direction.** See the f64 peer,
/// `ulp_measures_the_step_away_from_zero_on_the_negative_axis`, for why only a negative power
/// of two shows it.
fn f32_frexp_guards_its_edge_cases_and_ulp_steps_away_from_zero() {
    // Zero and the non-finite values pass through with a zero exponent rather than being
    // decomposed. Reaching this at all is the point: the mutated guard does not return here.
    assert_eq!(Real::frexp(0.0f32), (0.0, 0.0), "frexp(0)");
    assert!(
        Real::signbit(&Real::frexp(-0.0f32).0),
        "frexp(-0) keeps the sign bit"
    );
    assert_eq!(
        Real::frexp(f32::INFINITY),
        (f32::INFINITY, 0.0),
        "frexp(inf)"
    );
    assert_eq!(
        Real::frexp(f32::NEG_INFINITY),
        (f32::NEG_INFINITY, 0.0),
        "frexp(-inf)"
    );
    let (m, e) = Real::frexp(f32::NAN);
    assert!(m.is_nan() && e == 0.0, "frexp(NaN) propagates");

    // Subnormals *do* take the rescaling path, which is what makes the zero case delicate.
    let subnormal = f32::MIN_POSITIVE / 4.0;
    let (m, e) = Real::frexp(subnormal);
    assert!((0.5..1.0).contains(&m.abs()), "subnormal mantissa {m}");
    assert_eq!(Real::ldexp(m, e), subnormal, "and it round trips");

    // `ulp` away from zero: at a binade boundary the step toward zero is half as wide.
    for (x, want) in [
        (-1.0f32, f32::EPSILON),
        (-2.0, 2.0 * f32::EPSILON),
        (-0.5, f32::EPSILON / 2.0),
    ] {
        assert_eq!(Real::ulp(x), want, "f32: ulp({x}) is the binade above");
        assert_eq!(
            x - Real::ulp(x),
            Real::next_after(x, f32::NEG_INFINITY),
            "f32: ulp steps away from zero at {x}"
        );
    }
    assert_eq!(Real::ulp(-3.0f32), Real::ulp(3.0f32), "symmetric magnitude");
    assert_eq!(Real::ulp(0.0f32), f32::from_bits(1), "f32: ulp(0)");
    assert!(Real::ulp(f32::INFINITY).is_infinite(), "f32: ulp(inf)");
}
