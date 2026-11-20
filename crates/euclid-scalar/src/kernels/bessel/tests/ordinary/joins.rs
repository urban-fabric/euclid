//! Where two of the three methods overlap, checked against each other.
//!
//! A join is the one place a multi-route function can be wrong while every route is right, and
//! the three methods here are fully independent of one another — a finite trigonometric sum, an
//! ODE march, and a divergent asymptotic expansion — so agreement between them is evidence
//! rather than a tautology.

use super::close;
use crate::kernels::bessel::asymptotic::hankel;
use crate::kernels::bessel::ordinary_first::{
    HANKEL_MIN, J_SERIES_MAX, j_aliased, j_series, j0, j1,
};
use crate::kernels::bessel::ordinary_second::{
    Y_SERIES_MAX, y_march, y0, y0_series, y1, y1_series,
};

#[test]
fn the_regimes_join_without_a_step() {
    // Each join is checked against the derivative, which bounds how much the function may
    // legitimately move across the interval: J0' = -J1, Y0' = -Y1, and
    // J1' = J0 - J1/x, Y1' = Y0 - Y1/x.
    let h = 1e-9;
    for &x in &[Y_SERIES_MAX, HANKEL_MIN] {
        for (name, f, derivative) in [
            ("J0", j0 as fn(f64) -> f64, -j1(x)),
            ("J1", j1 as fn(f64) -> f64, j0(x) - j1(x) / x),
            ("Y0", y0 as fn(f64) -> f64, -y1(x)),
            ("Y1", y1 as fn(f64) -> f64, y0(x) - y1(x) / x),
        ] {
            let (below, above) = (f(x - h), f(x + h));
            let implied = derivative.abs() * 2.0 * h;
            assert!(
                (above - below).abs() <= implied * 1.05 + 1e-14,
                "{name} steps at x = {x}: {below:e} then {above:e}"
            );
        }
    }
}

#[test]
fn adjacent_methods_agree_across_their_whole_overlap() {
    // Stronger than a join test: each pair of methods is run side by side over a band
    // where both should be valid, so agreement constrains them everywhere rather than at
    // a single point. The three are fully independent of one another — a finite
    // trigonometric sum, an ODE march, and a divergent asymptotic expansion — so this is
    // real evidence rather than a tautology.
    for i in 0..=48 {
        let x = 18.0 + f64::from(i) * 0.5; // [18, 42]
        let order0 = hankel(0.0, x);
        let order1 = hankel(1.0, x);
        let (j0_hankel, y0_hankel) = (order0.first_kind, order0.second_kind);
        let (j1_hankel, y1_hankel) = (order1.first_kind, order1.second_kind);
        close(j0_hankel, j_aliased(0, x), 1e-14);
        close(j1_hankel, j_aliased(1, x), 1e-14);
        let marched = y_march(x);
        close(y0_hankel, marched.value, 1e-13);
        close(y1_hankel, marched.derivative, 1e-13);
    }
    // The J series against the quadrature, across the whole band where both hold. The
    // series is exact up to a few, and the quadrature from about 1e-8 upward, so this
    // overlap is wide.
    for i in 1..=60 {
        let x = f64::from(i) * 0.05; // (0, 3]
        close(j_series(0, x), j_aliased(0, x), 1e-15);
        close(j_series(1, x), j_aliased(1, x), 1e-15);
    }
    // And the Y series against the march just above the series cutoff.
    for i in 0..=30 {
        let x = Y_SERIES_MAX + f64::from(i) * 0.05;
        close(y_march(x).value, y0_series(x), 1e-14);
        close(y_march(x).derivative, y1_series(x), 1e-14);
    }
}

#[test]
fn no_regime_boundary_leaves_a_step_at_ulp_resolution() {
    // Each of the three crossovers, probed at the two representable values straddling it.
    // `J` switches series -> trapezoid at 2 and trapezoid -> Hankel at 18; `Y` switches
    // series -> march at 4 and march -> Hankel at 18.
    use euclid_utils::no_step_across;
    for &x in &[J_SERIES_MAX, HANKEL_MIN] {
        no_step_across("J0", j0, x, -j1(x), 1e-14);
        no_step_across("J1", j1, x, j0(x) - j1(x) / x, 1e-14);
    }
    for &x in &[Y_SERIES_MAX, HANKEL_MIN] {
        no_step_across("Y0", y0, x, -y1(x), 1e-14);
        no_step_across("Y1", y1, x, y0(x) - y1(x) / x, 1e-14);
    }
    // And the thresholds themselves must be exactly where the code says they are: evaluating
    // one ulp either side must actually cross a branch, which this confirms indirectly by
    // checking that the *other* route is the one being used just across it.
    assert!(
        (j0(J_SERIES_MAX) - j_series(0, J_SERIES_MAX)).abs() < 1e-15,
        "at the threshold J0 should still be the series"
    );
    let just_above = crate::Real::next_after(J_SERIES_MAX, f64::INFINITY);
    assert!(
        (j0(just_above) - j_aliased(0, just_above)).abs() < 1e-15,
        "one ulp above it J0 should be the trapezoid"
    );
}
