//! The trigonometric helpers whose naive forms cancel.
//!
//! Part of the [`compensated`](super) suite.

use crate::Real;
use euclid_utils::{assert_close, assert_exact, rel, sweep};

// --- cos_m1 --------------------------------------------------------------

#[test]
fn cos_m1_keeps_digits_where_the_naive_form_loses_them() {
    // cos(1e-8) rounds to exactly 1.0, so `cos(x) - 1.0` is exactly 0 — total loss.
    let x = 1e-8f64;
    assert_exact(x.cos() - 1.0, 0.0, "the naive form collapses");
    // cos x - 1 ~ -x^2/2
    assert_close(Real::cos_m1(x), -0.5 * x * x, 1e-12, "cos_m1 at 1e-8");

    // -x^2/2 is the leading term only; the x^4/24 correction is 4e-6 at x = 0.1, so compare
    // against it only where it is genuinely negligible, and against the two-term series above
    // that.
    for i in 4..40 {
        let x = 10f64.powi(-i);
        assert_close(
            Real::cos_m1(x),
            -0.5 * x * x,
            1e-10,
            &format!("cos_m1({x})"),
        );
    }
    // Four terms over [0.01, 0.2]: the first dropped term is x^10/10!, which is 3e-14 at the
    // top of that range. (Three terms over [0.01, 0.3] is *not* enough — x^8/8! reaches
    // 1.1e-12 by x = 0.12, which is the series truncating, not `cos_m1` drifting.)
    for x in sweep(0.01, 0.2, 30) {
        let series = -0.5 * x * x + x.powi(4) / 24.0 - x.powi(6) / 720.0 + x.powi(8) / 40320.0;
        assert_close(
            Real::cos_m1(x),
            series,
            1e-12,
            &format!("cos_m1({x}) vs series"),
        );
    }
}

#[test]
fn cos_m1_agrees_with_the_naive_form_away_from_zero() {
    // Where cancellation is not a problem, the two must match.
    for x in sweep(0.5, 6.0, 40) {
        assert_close(
            Real::cos_m1(x),
            x.cos() - 1.0,
            1e-14,
            &format!("cos_m1({x})"),
        );
    }
}

#[test]
fn cos_m1_is_even_and_bounded() {
    for x in sweep(0.0, 10.0, 50) {
        assert_close(Real::cos_m1(-x), Real::cos_m1(x), 1e-15, "cos_m1 is even");
        let v = Real::cos_m1(x);
        assert!(
            (-2.0..=0.0).contains(&v),
            "cos_m1({x}) = {v} escaped [-2, 0]"
        );
    }
    assert_exact(Real::cos_m1(0.0f64), 0.0, "cos_m1(0) = 0");
}

#[test]
fn haversine_stays_accurate_at_short_distances() {
    // The motivating use. hav(θ) = (1 − cos θ)/2 = −cos_m1(θ)/2. Built the naive way, the
    // central term collapses to zero for nearby points and the distance comes out as 0.
    let earth_radius_m = 6_371_000.0f64;
    let haversine = |lat1: f64, lon1: f64, lat2: f64, lon2: f64| {
        let (dlat, dlon) = (lat2 - lat1, lon2 - lon1);
        let h = -Real::cos_m1(dlat) / 2.0 + lat1.cos() * lat2.cos() * (-Real::cos_m1(dlon) / 2.0);
        2.0 * earth_radius_m * h.sqrt().asin()
    };
    let naive = |lat1: f64, lon1: f64, lat2: f64, lon2: f64| {
        let (dlat, dlon) = (lat2 - lat1, lon2 - lon1);
        let h = (1.0 - dlat.cos()) / 2.0 + lat1.cos() * lat2.cos() * ((1.0 - dlon.cos()) / 2.0);
        2.0 * earth_radius_m * h.sqrt().asin()
    };

    // Two points about 1 metre apart: dlat ~ 1.57e-7 rad.
    let lat = 0.9f64;
    let dlat = 1.0 / earth_radius_m;
    let stable = haversine(lat, 0.0, lat + dlat, 0.0);
    assert_close(stable, 1.0, 1e-9, "1 m separation recovered");

    // At 1 mm the naive form has nothing left.
    let dlat_mm = 1e-3 / earth_radius_m;
    let stable_mm = haversine(lat, 0.0, lat + dlat_mm, 0.0);
    assert_close(stable_mm, 1e-3, 1e-6, "1 mm separation recovered");
    let naive_mm = naive(lat, 0.0, lat + dlat_mm, 0.0);
    assert!(
        rel(naive_mm, 1e-3) > 1e-4,
        "the naive haversine is expected to be inaccurate at 1 mm, got {naive_mm}"
    );
}

// --- sinc ----------------------------------------------------------------

#[test]
fn sinc_fills_the_removable_singularity() {
    assert_exact(Real::sinc(0.0f64), 1.0, "sinc(0) = 1");
    assert_exact(Real::sinc(-0.0f64), 1.0, "sinc(-0) = 1");
    // Continuous across it: sinc -> 1 as x -> 0. Only meaningful once x^2/6 is below the
    // tolerance, which it is from about 1e-8 down.
    for i in 8..40 {
        let x = 10f64.powi(-i);
        assert_close(Real::sinc(x), 1.0, 1e-15, &format!("sinc({x}) -> 1"));
    }
}

#[test]
fn sinc_agrees_with_sin_over_x_away_from_zero() {
    for x in sweep(0.1, 20.0, 80) {
        assert_close(Real::sinc(x), x.sin() / x, 1e-14, &format!("sinc({x})"));
    }
}

#[test]
fn sinc_is_even_and_vanishes_at_multiples_of_pi() {
    for x in sweep(0.5, 15.0, 40) {
        assert_close(Real::sinc(-x), Real::sinc(x), 1e-15, "sinc is even");
    }
    for k in 1..8 {
        let x = f64::from(k) * <f64 as Real>::pi();
        assert!(
            Real::sinc(x).abs() < 1e-15,
            "sinc({k}pi) = {} should vanish",
            Real::sinc(x)
        );
    }
}

#[test]
fn sinc_matches_its_series_near_the_crossover() {
    // The implementation switches between series and sin(x)/x; the join must be seamless.
    for x in sweep(0.009, 0.011, 41) {
        let series = 1.0 - x * x / 6.0 + x.powi(4) / 120.0 - x.powi(6) / 5040.0;
        assert_close(
            Real::sinc(x),
            series,
            1e-14,
            &format!("sinc({x}) at crossover"),
        );
    }
}
