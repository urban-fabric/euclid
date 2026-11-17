//! Seams checks for the numeric module.

use super::*;

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
fn the_seam_ratio_declines_to_answer_where_it_cannot() {
    assert_eq!(seam_ratio(|_| f64::NAN, 1.0), None);
    assert_eq!(seam_ratio(|x: f64| 1.0 / x, 0.0), None);
    for x in [f64::MAX, f64::MAX.next_down(), -f64::MAX] {
        assert_eq!(seam_ratio(|_| 1.0, x), None);
    }
}

#[test]
fn a_linear_function_crossing_zero_scores_one_across_exponents() {
    for exponent in -120..=120 {
        for sign in [-1.0, 1.0] {
            let x = sign * 2.0_f64.powi(exponent);
            assert_eq!(seam_ratio(|point| point - x, x), Some(1.0));
        }
    }
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
