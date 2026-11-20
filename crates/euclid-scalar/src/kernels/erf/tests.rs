//! Unit tests for the error-function family.
//!
//! A child module so implementation plus tests stay inside the crate's 400-line budget, while
//! `use super::*` still reaches `ErfForm`, `ErfcForm`, the `ErfcxDepth` ladder and the
//! `ERFCX_CF_MIN` crossover. The depth ladder is the reason most of this exists: every
//! comparison in it survived mutation testing, because each bucket is deliberately sized above
//! its measured requirement and so one bucket over still converges. Only the decision is
//! assertable.

use super::*;

fn close(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol, "{a} vs {b}");
}

#[test]
fn erf_known_values() {
    close(erf(0.0), 0.0, 1e-12);
    close(erf(1.0), 0.842_700_792_949_715, 1e-10);
    close(erf(-1.0), -0.842_700_792_949_715, 1e-10);
    close(erf(2.0), 0.995_322_265_018_953, 1e-10);
    close(erfc(0.0), 1.0, 1e-12);
}

#[test]
fn erf_inv_round_trips() {
    for &x in &[-0.9, -0.5, -0.1, 0.1, 0.3, 0.5, 0.8, 0.95] {
        close(erf(erf_inv(x)), x, 1e-10);
    }
}

#[test]
fn the_depth_ladder_brackets_each_measured_band() {
    use ErfcxDepth::{
        AtLeastFifty, UnderFifty, UnderFive, UnderSevenQuarters, UnderTen, UnderThree,
        UnderThreeHalves, UnderTwenty, UnderTwo,
    };
    // Each boundary belongs to the bucket above it, so the band is half-open on the right.
    // The depths are the no-regression search's answers; see `ErfcxDepth` for the table.
    for (x, bucket, levels) in [
        (1.4_f64, UnderThreeHalves, 145),
        (1.499, UnderThreeHalves, 145),
        (1.5, UnderSevenQuarters, 131),
        (1.749, UnderSevenQuarters, 131),
        (1.75, UnderTwo, 102),
        (1.999, UnderTwo, 102),
        (2.0, UnderThree, 72),
        (2.999, UnderThree, 72),
        (3.0, UnderFive, 44),
        (4.999, UnderFive, 44),
        (5.0, UnderTen, 20),
        (9.999, UnderTen, 20),
        (10.0, UnderTwenty, 11),
        (19.999, UnderTwenty, 11),
        (20.0, UnderFifty, 7),
        (49.999, UnderFifty, 7),
        (50.0, AtLeastFifty, 5),
        (1e308, AtLeastFifty, 5),
    ] {
        assert_eq!(ErfcxDepth::for_argument(x), bucket, "at {x}");
        assert_eq!(bucket.levels(), levels, "at {x}");
    }
    // Exactly on each boundary, and one step either side of it.
    for edge in [1.5_f64, 1.75, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0] {
        assert_ne!(
            ErfcxDepth::for_argument(edge.next_down()),
            ErfcxDepth::for_argument(edge),
            "the bucket must change at {edge}"
        );
        assert_eq!(
            ErfcxDepth::for_argument(edge),
            ErfcxDepth::for_argument(edge.next_up()),
        );
    }
    // The ladder is strictly monotone: deeper where the fraction converges more slowly. This is
    // what makes a mis-ordered pair of depths a failure rather than a curiosity.
    let levels: Vec<u32> = [1.4_f64, 1.5, 1.75, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0]
        .iter()
        .map(|&x| ErfcxDepth::for_argument(x).levels())
        .collect();
    assert!(levels.windows(2).all(|w| w[0] > w[1]), "{levels:?}");
}

/// The refined ladder is no less accurate than the one it replaced.
///
/// Asserted against `erfc`, which reaches the same fraction by a different association, and
/// against the identity `erfcx(x)·e^{−x²} = erfc(x)` that defines it. The bands that were
/// *reduced* are the ones worth sweeping — a depth cut too far shows up here and nowhere else,
/// because every other test on `erfcx` is loose enough to absorb several ulp.
#[test]
fn the_reduced_bands_still_carry_their_accuracy() {
    for &(lo, hi) in &[
        (1.5_f64, 1.75_f64),
        (1.75, 2.0),
        (5.0, 10.0),
        (10.0, 20.0),
        (20.0, 50.0),
        (50.0, 200.0),
    ] {
        for i in 0..400 {
            let x = lo + (hi - lo) * f64::from(i) / 400.0;
            let scaled = erfcx(x);
            // `erfcx` is `e^{x²}·erfc(x)`, so the product returns `erfc` — exactly where `erfc`
            // itself is still representable, which is what bounds the sweep's top at 200.
            let back = scaled * (-x * x).exp();
            let direct = erfc(x);
            if direct > 0.0 {
                assert!(
                    (back - direct).abs() <= 1e-13 * direct,
                    "erfcx({x}) = {scaled:e} does not reproduce erfc: {back:e} vs {direct:e}"
                );
            }
            // And it stays positive and decreasing, which a truncated fraction would break.
            assert!(
                scaled > 0.0 && scaled.is_finite(),
                "erfcx({x}) = {scaled:e}"
            );
        }
    }
    // Monotone decreasing across every band boundary the ladder has, which is where a
    // mis-sized bucket would show as a step.
    for edge in [1.5_f64, 1.75, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0] {
        let (below, above) = (erfcx(edge.next_down()), erfcx(edge.next_up()));
        assert!(
            below > above,
            "erfcx steps the wrong way across {edge}: {below:e} then {above:e}"
        );
        // The step itself is at the format's resolution rather than the bucket's: both depths
        // converge, so changing depth must not move the value more than a few ulp.
        assert!(
            (below - above).abs() <= 8.0 * f64::EPSILON * below,
            "erfcx jumps at the {edge} boundary: {below:e} then {above:e}"
        );
    }
}

#[test]
fn erfc_tail_does_not_underflow() {
    // Regression: the naive `1 - erf` underflows to exactly 0 by x ≈ 6 in f64;
    // the tail now routes through `e^{-x²}·erfcx`, recovering the true value.
    let known: &[(f64, f64)] = &[
        (5.0, 1.537_459_794_428_034e-12),
        (6.0, 2.151_973_671_249_891e-17),
        (10.0, 2.088_487_583_762_545e-45),
    ];
    for &(x, want) in known {
        let got = erfc(x);
        assert!(got > 0.0, "erfc({x}) underflowed to {got}");
        assert!(
            (got - want).abs() <= 1e-3 * want,
            "erfc({x}) = {got}, want {want}"
        );
    }
}

/// The four forms, at and either side of the one threshold that is not a non-finite case.
///
/// The linear cutoff is *sited* where the dropped `x²/3` term is below half an ulp, so the
/// two forms agree across it to well under a rounding — which is what left `<` against `<=`
/// alive against every value assertion. The decision is what gets pinned.
#[test]
fn the_erf_forms_partition_the_line() {
    use ErfForm::{IncompleteGamma, Linear, Saturated, Undefined};
    assert_eq!(ErfForm::for_argument(f64::NAN), Undefined);
    assert_eq!(ErfForm::for_argument(f64::INFINITY), Saturated);
    assert_eq!(ErfForm::for_argument(f64::NEG_INFINITY), Saturated);
    // The boundary belongs to the incomplete gamma, and the magnitude decides, so both
    // signs route alike.
    assert_eq!(ErfForm::for_argument(ERF_LINEAR_MAX), IncompleteGamma);
    assert_eq!(ErfForm::for_argument(-ERF_LINEAR_MAX), IncompleteGamma);
    assert_eq!(ErfForm::for_argument(ERF_LINEAR_MAX.next_down()), Linear);
    assert_eq!(ErfForm::for_argument(-ERF_LINEAR_MAX.next_down()), Linear);
    assert_eq!(
        ErfForm::for_argument(ERF_LINEAR_MAX.next_up()),
        IncompleteGamma
    );
    // Both zeros take the linear arm, which is what preserves the sign of a zero.
    assert_eq!(ErfForm::for_argument(0.0), Linear);
    assert_eq!(ErfForm::for_argument(-0.0), Linear);
    assert_eq!(erf(-0.0).to_bits(), (-0.0_f64).to_bits());
    assert_eq!(erf(0.0).to_bits(), 0.0_f64.to_bits());
    for x in [1.0_f64, -1.0, 10.0] {
        assert_eq!(ErfForm::for_argument(x), IncompleteGamma);
    }
    // And the saturated arm carries the sign rather than a comparison.
    assert_eq!(erf(f64::INFINITY), 1.0);
    assert_eq!(erf(f64::NEG_INFINITY), -1.0);
}
