//! The instruments, checked against functions whose behaviour is known by construction.
//!
//! Each finds a planted defect and leaves a clean function alone. A detector that fires on
//! everything is as useless as one that never fires, so both directions are required.

use super::{
    LIKELY_THRESHOLDS, UNIT_ROUNDOFF, condition_at, forward_error_bound, hunt_seams, next_down,
    next_up, scan_for_seam, scan_monotone, scan_window, seam_ratio, two_stage_error_bound,
};

#[test]
fn next_up_steps_exactly_one_representable_value() {
    assert_eq!(next_up(0.0).to_bits(), 1);
    assert_eq!(next_up(1.0), f64::from_bits(1.0_f64.to_bits() + 1));
    // Downward through the negatives is still upward in value.
    assert!(next_up(-1.0) > -1.0);
    assert_eq!(next_up(f64::INFINITY), f64::INFINITY);
    assert!(next_up(f64::NAN).is_nan());
}

// ===== Seams ==============================================================

/// A smooth function must not jump: stepping the input one ulp moves the output about one ulp.
///
/// `ln` is in this list on purpose. It has a root at `x = 1`, which is exactly where the first
/// version of the detector reported a spurious seam, so it is the regression for that bug.
#[test]
fn a_continuous_function_has_no_seam() {
    assert!(
        scan_for_seam(f64::ln, 1.0, 2_000, 64.0).is_none(),
        "root at 1 is not a seam"
    );
    assert!(
        scan_for_seam(f64::sqrt, 0.0, 2_000, 64.0).is_none(),
        "root at 0 is not a seam"
    );
    assert!(scan_for_seam(f64::sin, 0.5, 2_000, 64.0).is_none());
    assert!(scan_for_seam(f64::exp, 1.0, 2_000, 64.0).is_none());
    assert!(scan_for_seam(f64::ln, 3.0, 2_000, 64.0).is_none());
}

/// A planted branch disagreement must be found, and found *at the crossover* — which is the
/// whole point, since nothing told the scanner where the branch was.
#[test]
fn a_planted_seam_is_found_at_its_crossover() {
    let threshold = 1.5;
    // Two branches that disagree by a relative 1e-9 at the boundary: far too small for a
    // casual eye, far too large for a continuous function.
    let discontinuous = move |x: f64| {
        if x < threshold {
            x.sin()
        } else {
            x.sin() * (1.0 + 1e-9)
        }
    };

    let seam = scan_for_seam(discontinuous, 1.5 - 1e-13, 10_000, 1_000.0)
        .expect("the planted discontinuity must be found");
    assert!(
        (seam.at - threshold).abs() < 1e-12,
        "found the seam at {} rather than at the threshold {threshold}",
        seam.at
    );
    assert!(
        seam.ratio > 1e5,
        "a 1e-9 relative step should be millions of times the expected step, got {}",
        seam.ratio
    );
}

/// A directed hunt finds a seam at a round threshold, which is where thresholds actually are.
#[test]
fn the_hunt_finds_a_seam_at_a_round_threshold() {
    let discontinuous = |x: f64| if x < 8.0 { x.ln() } else { x.ln() + 1e-9 };
    let seams = hunt_seams(discontinuous, LIKELY_THRESHOLDS, 64, 1_000.0);
    assert_eq!(seams.len(), 1, "expected exactly one seam, got {seams:?}");
    assert_eq!(seams[0].next, 8.0, "the seam should sit just below 8.0");
}

/// And leaves a smooth kernel alone at every one of those thresholds — including the ones that
/// are roots, which is where the first version of the detector misfired.
#[test]
fn the_hunt_leaves_smooth_functions_alone() {
    for (name, f) in [
        ("ln", f64::ln as fn(f64) -> f64),
        ("sqrt", f64::sqrt),
        ("exp", f64::exp),
        ("sin", f64::sin),
        ("tanh", f64::tanh),
    ] {
        let seams = hunt_seams(f, LIKELY_THRESHOLDS, 64, 64.0);
        assert!(
            seams.is_empty(),
            "{name} reported spurious seams: {seams:?}"
        );
    }
}

#[test]
fn scan_window_looks_below_the_centre_as_well_as_above() {
    // A seam reachable only by starting below the threshold.
    let discontinuous = |x: f64| if x < 2.0 { x } else { x + 1e-9 };
    assert!(
        scan_window(discontinuous, 2.0, 32, 1_000.0).is_some(),
        "a window centred on the threshold must straddle it"
    );
    assert!(
        scan_for_seam(discontinuous, 2.0, 32, 1_000.0).is_none(),
        "scanning upward from the threshold cannot see it, which is why the window exists"
    );
}

#[test]
fn next_down_is_the_inverse_of_next_up() {
    for x in [1.0_f64, -1.0, 1e-300, 1e300, 0.5] {
        assert_eq!(next_down(next_up(x)), x, "round trip failed at {x}");
    }
    assert_eq!(next_up(next_down(0.0)), 0.0);
}

#[test]
fn the_seam_ratio_declines_to_answer_where_it_cannot() {
    assert_eq!(seam_ratio(|_| f64::NAN, 1.0), None);
    assert_eq!(seam_ratio(|x: f64| 1.0 / x, 0.0), None);
}

/// A smooth function scores near 1, which is what makes a threshold meaningful at all.
#[test]
fn a_smooth_function_scores_near_one() {
    for x in [0.5_f64, 1.0, 3.0, 100.0] {
        for (name, f) in [
            ("sin", f64::sin as fn(f64) -> f64),
            ("exp", f64::exp),
            ("sqrt", f64::sqrt),
        ] {
            let ratio = seam_ratio(f, x).expect("defined");
            assert!(
                ratio < 64.0,
                "{name} scored {ratio} at {x}; a smooth function should be near 1"
            );
        }
    }
}

// ===== Wiggle =============================================================

#[test]
fn a_monotone_function_has_no_inversion() {
    assert!(scan_monotone(f64::exp, 1.0, 5_000).is_none());
    assert!(scan_monotone(f64::sqrt, 2.0, 5_000).is_none());
    // A plateau is still monotone: equality is not an inversion.
    assert!(scan_monotone(|_| 1.0, 1.0, 1_000).is_none());
}

/// The defect this detector exists for: a function accurate at every point whose error
/// oscillates by more than the output spacing, so it is not monotone even though no individual
/// value is wrong by much. A pointwise accuracy test cannot see this.
#[test]
fn ulp_level_wiggle_is_found() {
    // `exp` perturbed by an alternating one-ulp error. Every value is within one ulp of the
    // truth; the sequence is not monotone.
    let wiggly = |x: f64| {
        let base = x.exp();
        if x.to_bits().is_multiple_of(2) {
            base
        } else {
            f64::from_bits(base.to_bits().saturating_sub(2))
        }
    };

    let inversion =
        scan_monotone(wiggly, 1.0, 5_000).expect("alternating error must break monotonicity");
    assert!(
        inversion.values.1 < inversion.values.0,
        "an inversion must report a decrease: {:?}",
        inversion.values
    );
}

#[test]
fn a_decreasing_function_inverts_immediately() {
    let inversion = scan_monotone(|x: f64| -x, 1.0, 10).expect("negation decreases");
    assert!(inversion.values.1 < inversion.values.0);
}

// ===== Conditioning =======================================================

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

// ===== The constant everything else is measured in =========================

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

/// `next_down` saturates at negative infinity rather than stepping off the end into NaN.
///
/// The guard is `x.is_nan() || x == f64::NEG_INFINITY`, and weakening that `||` to `&&` makes
/// `next_down(-inf)` decrement the bit pattern of negative infinity — which is a NaN. A scan
/// walking downward would then stop reporting values and start reporting NaN, silently.
#[test]
fn next_down_saturates_at_negative_infinity() {
    assert_eq!(next_down(f64::NEG_INFINITY), f64::NEG_INFINITY);
    assert!(next_down(f64::NAN).is_nan());
    // The upward counterpart, for symmetry.
    assert_eq!(next_up(f64::INFINITY), f64::INFINITY);
    assert!(next_up(f64::NAN).is_nan());
}

// ===== seam_ratio declines where the measure has no meaning ================

/// The window guard declines an infinite argument.
///
/// `window = sqrt(EPSILON) * |x|`, so at an infinite argument the window is infinite too and
/// every comparison downstream degenerates. The guard catches it with `x + window == x` and
/// `x - window == x`, and those two clauses are not decoration: mutation testing found three
/// separate ways to disable them, each of which makes `seam_ratio(-inf)` report a perfectly
/// finite ratio of **zero** — "no seam here, maximally smooth" — for an argument where nothing
/// was measured at all.
///
/// Negative infinity is the case that reaches this guard, and the asymmetry is worth knowing.
/// At `+inf` the earlier `next == x` clause answers first, because `next_up(inf)` is `inf`. At
/// `-inf` it does not: `next_up(-inf)` is `-f64::MAX`, a perfectly ordinary finite value, so the
/// scan walks on and this guard is the only thing left. A test written at `+inf` passes against
/// every mutant.
///
/// The function must also be *constant* here, so that `f(-inf)` and `f(-f64::MAX)` are both
/// finite and the finiteness guard above does not answer instead.
#[test]
fn seam_ratio_declines_an_infinite_argument() {
    assert_eq!(
        next_up(f64::NEG_INFINITY),
        -f64::MAX,
        "precondition: -inf has a finite successor, which is why it reaches the window guard"
    );

    assert_eq!(
        seam_ratio(|_| 1.0, f64::NEG_INFINITY),
        None,
        "an infinite argument gives an infinite window and nothing measurable"
    );
    assert_eq!(seam_ratio(|_| 1.0, f64::INFINITY), None);
}

/// And it declines the argument where the window happens to be exactly one.
///
/// `window` is `2^-26 * |x|`, so at `x = 2^26` it is exactly `1.0` — the one argument at which
/// `x * window == x`. That makes it the witness for a mutation of the guard's arithmetic that no
/// other input can see, and it has to be declined for the ordinary reason: a window of 1.0
/// around 67 million is not a derivative estimate.
#[test]
fn the_window_guard_is_not_fooled_by_a_unit_window() {
    let x = 2.0_f64.powi(26);
    let window = f64::EPSILON.sqrt() * x;
    assert_eq!(
        window, 1.0,
        "precondition: the window must be exactly one here"
    );

    // A real measurement, because the guard's clauses are about `x + window`, not `x * window`.
    assert!(
        seam_ratio(|v: f64| v, x).is_some(),
        "a smooth function at x = 2^26 must still be measurable"
    );
}

/// The derivative window has to **scale with** the argument, not shrink as it grows.
///
/// `window = sqrt(EPSILON) * |x|`, and mutation testing turned that multiplication into a
/// division. The effect is invisible near one and total at large magnitudes: at `x = 1e8` the
/// window becomes `1.5e-16`, far below `ulp(1e8)`, so `x + window == x` and the function declines
/// to measure anything at all. A seam hunt over large arguments would quietly return nothing.
#[test]
fn the_derivative_window_scales_with_the_argument() {
    for x in [1.0_f64, 1e4, 1e8, 1e16, 1e100] {
        assert!(
            seam_ratio(|v: f64| v, x).is_some(),
            "declined to measure a smooth function at x = {x:e}; the derivative window is not \
             scaling with the argument"
        );
    }
}

/// `f64::MIN_POSITIVE` is the smallest **normal**, so it is not subnormal.
///
/// An off-by-one in that comparison — `<` relaxed to `<=` — makes the smallest normal count as
/// subnormal and the whole normal range start one value higher. Trivial in effect and wrong in a
/// way that compounds: the subnormal test is the reason this function declines near zero, and
/// the boundary is where a kernel's own special-casing lives.
#[test]
fn the_smallest_normal_is_not_treated_as_subnormal() {
    assert!(
        seam_ratio(|v: f64| v, f64::MIN_POSITIVE).is_some(),
        "the smallest normal must be measurable"
    );
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

/// A derivative estimate with one infinite end is no estimate.
///
/// The same `||`-to-`&&` weakening as above, one guard further down: the window straddles a point
/// where the kernel overflows, so `f(x + window)` is infinite while `f(x - window)` is not. The
/// derivative then comes out infinite and the ratio with it — another false seam on correct code.
#[test]
fn seam_ratio_declines_a_derivative_with_one_infinite_end() {
    // The window at x = 1 is sqrt(EPSILON) ~ 1.5e-8, so a threshold a nanometre above 1 falls
    // inside it on the upper side only.
    let overflows_just_above_one = |v: f64| if v > 1.0 + 1e-9 { f64::INFINITY } else { 1.0 };
    assert_eq!(
        seam_ratio(overflows_just_above_one, 1.0),
        None,
        "a window reaching into an overflow has no usable derivative"
    );
}

// ===== condition_at declines where the condition number has no meaning =====

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

// ===== The two-stage bound =================================================

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

// ===== The scans ===========================================================

/// A constant function has no seam, even asked at zero tolerance.
///
/// The comparison is `ratio > tolerated`, and relaxing it to `>=` reports a seam wherever the two
/// are equal. A constant function is the case that makes that reachable: its observed step is
/// exactly zero, so its ratio is exactly zero, and at `tolerated = 0.0` the boundary is hit
/// exactly rather than approximately. Mutation testing found the strictness unpinned.
#[test]
fn a_constant_function_has_no_seam_even_at_zero_tolerance() {
    assert_eq!(
        seam_ratio(|_| 1.0, 1.0),
        Some(0.0),
        "precondition: a constant function must score exactly zero"
    );
    assert_eq!(
        scan_for_seam(|_| 1.0, 1.0, 16, 0.0),
        None,
        "a ratio equal to the tolerance is not a seam; the comparison must be strict"
    );
}

/// `scan_window` must reach **above** its centre, not merely up to it.
///
/// It walks `steps` values down and then scans `steps * 2` upward, which is what makes the centre
/// the middle of the window rather than its top — "a threshold sits *between* two branches, so
/// the scan has to start below it". Mutation testing turned that `* 2` into `+ 2`, leaving a
/// window that extends fifty values below the centre and two above, and the existing tests did
/// not notice because they plant their seams at or below the centre.
#[test]
fn scan_window_reaches_above_its_centre() {
    // A jump ten representable values above 1.0 — inside the window, above the centre.
    let mut threshold = 1.0_f64;
    for _ in 0..10 {
        threshold = next_up(threshold);
    }
    let jumps_above_centre = move |v: f64| if v >= threshold { 2.0 } else { 1.0 };

    assert!(
        scan_window(jumps_above_centre, 1.0, 50, 1e3).is_some(),
        "a seam ten ulps above the centre must be inside a 50-step window"
    );
}
