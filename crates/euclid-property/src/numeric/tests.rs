//! The instruments, checked against functions whose behaviour is known by construction.
//!
//! Each finds a planted defect and leaves a clean function alone. A detector that fires on
//! everything is as useless as one that never fires, so both directions are required.

use super::{
    LIKELY_THRESHOLDS, UNIT_ROUNDOFF, condition_at, forward_error_bound, hunt_seams, next_down,
    next_up, scan_for_seam, scan_monotone, scan_window, seam_ratio,
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
