//! Unit tests for the scaled complementary error function.
//!
//! A child module so the parent stays inside the crate's 400-line budget. What these reach is
//! private to [`super`]: the form enum, the handover and the coefficient table.

use super::*;

/// The route table, at and either side of its one threshold.
#[test]
fn the_scaled_forms_partition_the_positive_axis() {
    use ErfcxForm::{Chebyshev, Fraction};
    assert_eq!(ErfcxForm::for_argument(0.0), Chebyshev);
    assert_eq!(ErfcxForm::for_argument(1.0), Chebyshev);
    assert_eq!(
        ErfcxForm::for_argument(ERFCX_CHEBYSHEV_MAX.next_down()),
        Chebyshev
    );
    assert_eq!(ErfcxForm::for_argument(ERFCX_CHEBYSHEV_MAX), Fraction);
    assert_eq!(ErfcxForm::for_argument(50.0), Fraction);
}

/// The series and the fraction agree across the handover.
///
/// Reference-free, and the only comparison that means anything: these are two independent routes
/// to the same value, so where both are sound their disagreement bounds the error of each. At
/// the join the fraction is deep into its accurate range — its own ladder gives it 102 levels
/// there — and the series is at the edge the fit was sized for.
///
/// Measured worst over the arguments below: `2` ulp.
#[test]
fn the_series_and_the_fraction_agree_at_the_handover() {
    let mut worst = 0.0_f64;
    for step in 0..=20 {
        // Approach from below, where the series runs, and compare against the fraction at the
        // same argument — not across the threshold, which would compare two different `x`.
        let x = ERFCX_CHEBYSHEV_MAX - f64::from(step) * 0.01;
        let series = erfcx_chebyshev(x);
        let fraction = erfcx_cf(x);
        let ulps = (series - fraction).abs() / crate::Real::ulp(series);
        worst = worst.max(ulps);
        assert!(
            ulps <= 8.0,
            "at x = {x} the series gives {series} and the fraction {fraction}, {ulps} ulp apart"
        );
    }
    assert!(
        worst > 0.0,
        "the two routes were bit-identical, so this compared nothing"
    );
}

/// `erfcx(x)·e^{−x²} = erfc(x)` below `ERFC_SCALED_MIN`, where `erfc` is independent of this.
///
/// Above that threshold `erfc` *is* `e^{−x²}·erfcx`, so the identity is a tautology there and
/// says nothing. Below it `erfc` takes `1 − erf` and reaches this file by no path at all, which
/// makes the identity a genuine cross-check of the fit against two other kernels.
#[test]
fn the_scaled_form_reproduces_erfc_where_erfc_does_not_use_it() {
    for step in 0..=30 {
        let x = f64::from(step) * 0.0125;
        let direct = super::super::erfc(x);
        let scaled = erfcx_chebyshev(x) * (-x * x).exp();
        let ulps = (scaled - direct).abs() / crate::Real::ulp(direct);
        assert!(
            ulps <= 6.0,
            "at x = {x} the scaled form gives {scaled} and erfc {direct}, {ulps} ulp apart"
        );
    }
}

/// The two values `erfcx` is known at outright, and the shape between them.
///
/// `erfcx(0) = 1` is exact and is what pins the constant term of the fit; a table shifted by a
/// whole ulp still looks smooth and still round-trips through the identity above, but it misses
/// this. `erfcx` is strictly decreasing on the positive axis, which pins the sign of the rest.
#[test]
fn the_fit_matches_the_values_erfcx_is_known_at() {
    assert_eq!(erfcx_chebyshev(0.0), 1.0);
    let at_one = erfcx_chebyshev(1.0);
    let known = 0.427_583_576_155_807_f64;
    assert!(
        (at_one - known).abs() <= 2.0 * crate::Real::ulp(known),
        "erfcx(1) is {at_one}, and e^1·erfc(1) is {known}"
    );
    let mut previous = f64::INFINITY;
    for step in 0..=40 {
        let value = erfcx_chebyshev(f64::from(step) * 0.05);
        assert!(value < previous, "erfcx rose at {}", f64::from(step) * 0.05);
        previous = value;
    }
}
