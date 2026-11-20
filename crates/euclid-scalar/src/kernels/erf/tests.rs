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
    use ErfcxDepth::{AtLeastTwenty, UnderFive, UnderTen, UnderThree, UnderTwenty, UnderTwo};
    // Each boundary belongs to the bucket above it, so the band is half-open on the right.
    for (x, bucket, levels) in [
        (1.0_f64, UnderTwo, 160),
        (1.999, UnderTwo, 160),
        (2.0, UnderThree, 72),
        (2.999, UnderThree, 72),
        (3.0, UnderFive, 44),
        (4.999, UnderFive, 44),
        (5.0, UnderTen, 24),
        (9.999, UnderTen, 24),
        (10.0, UnderTwenty, 14),
        (19.999, UnderTwenty, 14),
        (20.0, AtLeastTwenty, 10),
        (1e308, AtLeastTwenty, 10),
    ] {
        assert_eq!(ErfcxDepth::for_argument(x), bucket, "at {x}");
        assert_eq!(bucket.levels(), levels, "at {x}");
    }
    // Exactly on each boundary, and one step either side of it.
    for edge in [2.0_f64, 3.0, 5.0, 10.0, 20.0] {
        assert_eq!(
            ErfcxDepth::for_argument(edge.next_down()),
            ErfcxDepth::for_argument(edge.next_down()),
        );
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
    // The ladder is monotone: deeper where the fraction converges more slowly.
    let levels: Vec<u32> = [1.0_f64, 2.0, 3.0, 5.0, 10.0, 20.0]
        .iter()
        .map(|&x| ErfcxDepth::for_argument(x).levels())
        .collect();
    assert!(levels.windows(2).all(|w| w[0] > w[1]), "{levels:?}");
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
