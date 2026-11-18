//! Conditioning checks for the numeric module.

use super::*;

#[test]
fn condition_estimation_checks_inputs_before_calling_the_function() {
    let seen = core::cell::Cell::new(0);
    for x in [0.0, -0.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(
            condition_at(
                |value| {
                    seen.set(seen.get() + 1);
                    value
                },
                x
            ),
            None
        );
    }
    assert_eq!(seen.get(), 0);
    assert_eq!(super::super::central_difference(5.0, 1.0, 0.5), Some(4.0));
    assert_eq!(super::super::central_difference(f64::NAN, 1.0, 0.5), None);
    assert_eq!(
        super::super::central_difference(1.0, f64::INFINITY, 0.5),
        None
    );
}

/// `κ(x) = |x f'(x)/f(x)|`. For `f(x) = x^n` that is exactly `|n|`, whatever `x` is — a closed
/// form to check the estimator against.
#[test]
fn the_condition_number_of_a_power_is_its_exponent() {
    for (power, expected) in [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (-1.0, 1.0)] {
        let condition =
            condition_at(|x: f64| x.powf(power), 2.5).expect("a power is well conditioned at 2.5");
        assert!(
            (condition - expected).abs() < 0.01,
            "kappa for x^{power} came out {condition}, expected {expected}"
        );
    }
}

/// `exp` is ill-conditioned for large arguments — `κ(x) = |x|` — which is why `exp(1e3)` cannot
/// be computed to full relative accuracy from an `f64` argument even in principle.
#[test]
fn exp_is_ill_conditioned_at_large_arguments() {
    let near_one = condition_at(f64::exp, 1.0).expect("defined at 1");
    let far_out = condition_at(f64::exp, 500.0).expect("defined at 500");

    assert!((near_one - 1.0).abs() < 0.01, "kappa(exp, 1) = {near_one}");
    assert!(
        (far_out - 500.0).abs() / 500.0 < 0.01,
        "kappa(exp, 500) = {far_out}, expected about 500"
    );
}

/// The classic ill-conditioned point: `ln` near 1, where the result approaches zero while the
/// input does not, so the relative condition number diverges.
#[test]
fn ln_is_ill_conditioned_near_one() {
    let condition = condition_at(f64::ln, 1.0 + 1e-10).expect("defined just above 1");
    assert!(
        condition > 1e9,
        "ln near 1 should be severely ill-conditioned, kappa = {condition:e}"
    );
}

#[test]
fn the_condition_number_declines_to_answer_where_undefined() {
    assert_eq!(
        condition_at(f64::sin, 0.0),
        None,
        "kappa is undefined at x = 0"
    );
    assert_eq!(
        condition_at(|_| 0.0, 1.0),
        None,
        "kappa is undefined at f(x) = 0"
    );
    assert_eq!(condition_at(|_| f64::NAN, 1.0), None);
}

/// The bound must scale with the conditioning, and must never fall below plain round-off.
#[test]
fn the_forward_error_bound_tracks_the_conditioning() {
    let benign = forward_error_bound(f64::sqrt, 2.0, 8.0).expect("sqrt is well conditioned");
    let severe = forward_error_bound(f64::exp, 500.0, 8.0).expect("exp is not");

    assert!(
        benign >= 8.0 * UNIT_ROUNDOFF,
        "the bound must never fall below round-off, got {benign:e}"
    );
    assert!(
        severe > benign * 100.0,
        "an ill-conditioned point must get a looser bound: {severe:e} vs {benign:e}"
    );
}

/// And the bound is *useful*: a correctly-rounded kernel fits inside it. `sqrt` is required by
/// IEEE 754 to be correctly rounded, so it is the one function whose error is known a priori.
#[test]
fn a_correctly_rounded_kernel_fits_inside_its_bound() {
    for x in [0.5_f64, 1.0, 2.0, 7.3, 1e10, 1e-10] {
        let bound = forward_error_bound(f64::sqrt, x, 8.0).expect("defined");
        // sqrt(x)^2 should return x to within the bound, doubled for the squaring.
        let round_trip = x.sqrt() * x.sqrt();
        let relative = ((round_trip - x) / x).abs();
        assert!(
            relative <= 2.0 * bound,
            "sqrt round trip at {x} had relative error {relative:e}, bound {bound:e}"
        );
    }
}

/// `UNIT_ROUNDOFF` is `2^-53`, and saying so is not a tautology.
///
/// It is written `f64::EPSILON / 2.0` — "`EPSILON` is the gap at 1, and round-to-nearest halves
/// it" — and mutation testing turned that division into a multiplication and a remainder without
/// anything failing. Every tolerance in this crate is a multiple of this number, so an error here
/// scales every bound in the fleet by a factor of two or four.
///
/// Asserted against `2^-53` rather than against `EPSILON / 2`, so the test states the value
/// independently of the arithmetic that produces it.
#[test]
fn unit_roundoff_is_two_to_the_minus_fifty_three() {
    assert_eq!(UNIT_ROUNDOFF, 2.0_f64.powi(-53));
    assert_eq!(UNIT_ROUNDOFF, 1.110_223_024_625_156_5e-16);
    // And it really is half of EPSILON, which is the gap at one rather than the round-off.
    assert_eq!(UNIT_ROUNDOFF * 2.0, f64::EPSILON);
}

/// But a genuinely subnormal argument is declined — including when its *successor* is normal.
///
/// The guard checks `x` and `next` separately, and the pair straddling the boundary is the case
/// that distinguishes `||` from `&&`: the largest subnormal steps up to the smallest normal, so
/// one end is subnormal and the other is not. Mutation testing found that weakening, and it
/// matters because that step is precisely where the measure's premise fails — a one-ulp step in
/// the subnormals is a percent-scale relative change, not a sixteen-digit one.
#[test]
fn seam_ratio_declines_either_side_of_the_subnormal_boundary() {
    let largest_subnormal = f64::from_bits(f64::MIN_POSITIVE.to_bits() - 1);
    assert!(
        largest_subnormal < f64::MIN_POSITIVE && largest_subnormal != 0.0,
        "precondition: this value must be the largest subnormal"
    );
    assert_eq!(
        next_up(largest_subnormal),
        f64::MIN_POSITIVE,
        "precondition: its successor must be the smallest normal"
    );

    assert_eq!(
        seam_ratio(|v: f64| v, largest_subnormal),
        None,
        "a step out of the subnormals must be declined, not measured"
    );
}

/// An infinite value has no relative condition number.
///
/// `κ = |x·f'(x)/f(x)|`, so an infinite `f(x)` puts an infinity in the *denominator* and the
/// quotient comes out as a perfectly finite **zero** — which reads as "this point is maximally
/// well conditioned" and is the opposite of the truth. Mutation testing weakened the guard that
/// prevents it, and the resulting `Some(0.0)` would flow straight into `forward_error_bound` as a
/// tolerance of `safety · u`, the tightest bound the crate can produce, at the one point where
/// no bound is meaningful.
#[test]
fn condition_at_declines_an_infinite_value() {
    let spikes_at_two = |v: f64| if v == 2.0 { f64::INFINITY } else { 1.0 };
    assert_eq!(
        condition_at(spikes_at_two, 2.0),
        None,
        "an infinite value must be declined, not reported as perfectly conditioned"
    );

    // A zero value is declined for the mirror-image reason: the quotient is infinite.
    assert_eq!(condition_at(|_| 0.0, 1.0), None);
}

/// The two-stage bound keeps **both** terms: `(κ_outer·κ_inner + κ_outer)·safety·u`.
///
/// Nothing was pinning this function. Six mutations of it survived — including replacing the
/// whole body with `Some(1.0)` — because the two property suites built on it both *abstain* when
/// the bound is undefined and both compare against it with headroom, so a corrupted bound makes
/// them quieter rather than louder. That is the worst shape a gap can have: the protection
/// disappears and every test still passes.
///
/// `exp` and `ln` are used because their condition numbers are exactly known, which turns the
/// documented formula into an arithmetic assertion. With `κ_exp(x) = |x|` and
/// `κ_ln(exp x) = 1/|x|`, the amplification is `(1/|x|)·|x| + 1/|x| = 1 + 1/|x|` — and the
/// first term being exactly 1 is the whole point of the module's argument: the two stages are
/// inverses, their condition numbers multiply to one, and the amplification lives entirely in
/// the *second* term, which a naive composite bound discards.
///
/// The tolerance is `1e-6` relative because `condition_at` is a central-difference estimate
/// good to a few digits, not an exact derivative.
#[test]
fn the_two_stage_bound_keeps_the_inner_stages_own_rounding() {
    for x in [0.5_f64, 1.0, 2.0, 4.0] {
        let bound = two_stage_error_bound(f64::exp, f64::ln, x, 1.0)
            .unwrap_or_else(|| panic!("the bound must be defined at x = {x}"));

        // kappa_outer * kappa_inner + kappa_outer, with both known in closed form.
        let predicted = (1.0 + 1.0 / x.abs()) * UNIT_ROUNDOFF;
        let relative = ((bound - predicted) / predicted).abs();
        assert!(
            relative < 1e-6,
            "at x = {x}: bound {bound:e} is not (1 + 1/|x|)u = {predicted:e} (off by {relative:e})"
        );
    }

    // The safety factor scales it linearly, which is what makes it a *factor*.
    let one = two_stage_error_bound(f64::exp, f64::ln, 1.0, 1.0).expect("defined");
    let sixty_four = two_stage_error_bound(f64::exp, f64::ln, 1.0, 64.0).expect("defined");
    let ratio = sixty_four / one;
    assert!(
        (ratio - 64.0).abs() < 1e-6,
        "the safety factor must scale the bound linearly, got a ratio of {ratio}"
    );

    // And it is floored at `safety * u`, so a benign composition still gets a real tolerance.
    let benign = two_stage_error_bound(|v: f64| v, |v: f64| v, 1.0, 1.0).expect("defined");
    assert!(
        benign >= UNIT_ROUNDOFF,
        "the bound must never fall below one unit of round-off, got {benign:e}"
    );
}
