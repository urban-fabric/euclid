//! The Edgeworth branch against an independent reference, and the exact symmetric median.
//!
//! Part of the [`beta_large_shape`](super) suite, whose preamble carries the `Decimal` reference
//! table and its provenance. The median is the sharper of the two: `I_½(a, a) = ½` is exact for
//! every shape, so it needs no tolerance at all and holds at shapes where nothing else can be
//! computed.

use super::*;

#[test]
fn beta_reg_matches_an_independent_reference_inside_the_edgeworth_branch() {
    for &(a, b, x, want) in EDGEWORTH_REFERENCE {
        assert!(
            a.min(b) >= BETA_EDGEWORTH_MIN,
            "({a:e}, {b:e}) does not reach the branch this table exists to test"
        );
        assert_close(
            Real::beta_reg(x, a, b),
            want,
            EDGEWORTH_TOL,
            &format!("I_x({a:e}, {b:e}) at x = {x:.17e}"),
        );
    }
}

#[test]
fn the_symmetric_median_is_exactly_one_half_at_every_representable_shape() {
    // `Beta(a, a)` is symmetric about ½, so this is an identity rather than an approximation,
    // and `exactly` is the honest bar: the skewness correction vanishes at `a = b` and `Φ(0)`
    // is ½ to the last bit, which is what makes the Edgeworth branch *exact* here.
    //
    // This is the assertion the overflow broke. It returned NaN from `1e103` up, so the
    // exponents past that point are the regression test; the ones below it guard the join.
    for exponent in [
        7.0f64, 7.5, 8.0, 9.0, 10.0, 12.0, 15.0, 20.0, 50.0, 102.0, 103.0, 150.0, 200.0, 300.0,
        307.0,
    ] {
        let a = 10f64.powf(exponent);
        assert_exact(
            Real::beta_reg(0.5, a, a),
            0.5,
            &format!("I_1/2(1e{exponent}, same) must be exactly 1/2"),
        );
    }
    // And where the sum itself overflows, which needs a shape within a half-decade of MAX.
    for &a in &[9e307f64, 1.5e308, f64::MAX] {
        assert!(
            !(a + a).is_finite(),
            "{a:e} was meant to overflow when doubled"
        );
        assert_exact(
            Real::beta_reg(0.5, a, a),
            0.5,
            &format!("I_1/2({a:e}, same) must be exactly 1/2 despite a+b overflowing"),
        );
    }
}
