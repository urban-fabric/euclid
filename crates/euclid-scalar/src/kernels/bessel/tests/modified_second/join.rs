//! Where `K`'s power series hands over to its quadrature, and why the handover sits where it does.
//!
//! Split from the parent so each file stays inside the crate's 400-line budget, and the sub-topic
//! is a real one: nothing here is about `K₀` or `K₁` as functions. These three tests are all
//! about the *boundary* — that it introduces no step, that it sits inside the band both routes
//! are accurate on, and that the band has an upper edge which is measured rather than assumed.

use super::*;

#[test]
fn the_second_kind_joins_its_two_regimes_smoothly() {
    // K0' = -K1 and K1' = -(K0 + K1/x), which bound a legitimate change across the join
    // between the power series and the quadrature.
    let x = K_SERIES_MAX;
    let h = 1e-9;
    for (name, f, derivative) in [
        ("K0", k0 as fn(f64) -> f64, -k1(x)),
        ("K1", k1 as fn(f64) -> f64, -(k0(x) + k1(x) / x)),
    ] {
        let (below, above) = (f(x - h), f(x + h));
        let implied = derivative.abs() * 2.0 * h;
        assert!(
            (above - below).abs() <= implied * 1.05 + 1e-16,
            "{name} steps at the join: {below:e} then {above:e}"
        );
    }
}

/// `K_SERIES_MAX` sits inside the band both routes are accurate on, and that band really ends.
///
/// Reference-free, and it is the test the threshold needed: where two independent routes to the
/// same value are both correct they must agree, so their disagreement is an error bound that
/// costs nothing to evaluate. The series and the quadrature share no code — a power series in
/// `x²/4` against a trapezoidal rule on `∫e^{−x(cosh t−1)}dt` — so agreement is evidence and not
/// a tautology.
///
/// Two clauses, worth something only together:
///
/// 1. **The overlap is real.** Below `1.3` the two agree to `9.8e-16` relative, flat in `x`, for
///    both orders. Against a 320-bit reference each is `0`–`4` ulp there.
/// 2. **The threshold is inside it, and the upper edge is not a literal.** Past `1.3` the
///    disagreement climbs monotonically — `1.6e-15`, `2.1e-15`, `3.7e-15`, `5.1e-15`, `6.6e-15`,
///    reaching `1.1e-14` across `[1.9, 2.0)` and `2.2e-14` by `2.3`. All of that growth is the
///    series': the quadrature measures `0`–`3` ulp throughout. So the second clause asserts the
///    degradation *is there*, which is what makes `1.3` a measurement.
///
/// The old `K_SERIES_MAX = 2.0` sat at the far end of that growth, which is the defect this
/// pins: `K₀(1.95)` carried 40 ulp from a route chosen over one carrying 1.
#[test]
fn the_series_quadrature_threshold_sits_inside_the_band_both_routes_are_accurate_on() {
    // The measured edges. Below `OVERLAP_MIN` nothing changes — the quadrature stays accurate
    // all the way to `1e-3` and the series is exact — so the lower edge is a cost boundary,
    // not an accuracy one, and is set generously.
    const OVERLAP_MIN: f64 = 0.5;
    const OVERLAP_MAX: f64 = 1.3;

    // A `const` block, so moving the threshold out of the overlap fails to *compile* rather
    // than failing a test someone might not run. The old `2.0` violated this.
    const {
        assert!(
            K_SERIES_MAX >= OVERLAP_MIN && K_SERIES_MAX <= OVERLAP_MAX,
            "K_SERIES_MAX must sit inside [0.5, 1.3], the band both the power series and the \
             quadrature are measured accurate on"
        );
    }

    // Clause 1: flat agreement across the overlap, both orders, scaled forms so the comparison
    // is not dominated by `e^{−x}`.
    let mut worst = 0.0_f64;
    for i in 0..=400 {
        let x = OVERLAP_MIN + (OVERLAP_MAX - OVERLAP_MIN) * f64::from(i) / 400.0;
        for (series, quadrature) in [
            (k0_series(x) * x.exp(), k_nu_scaled(0.0, x)),
            (k1_series(x) * x.exp(), k_nu_scaled(1.0, x)),
        ] {
            worst = worst.max(((series - quadrature) / quadrature).abs());
        }
    }
    assert!(
        worst <= 1.2e-15,
        "the two routes disagree by {worst:e} inside the overlap, where both measured \
         0-4 ulp against a 320-bit reference; the overlap this threshold rests on is gone"
    );

    // Clause 2: and the overlap genuinely ends, so `OVERLAP_MAX` is not an arbitrary literal.
    // Measured `1.1e-14` across `[1.9, 2.0)`, an order above the overlap's floor.
    let mut beyond = 0.0_f64;
    for i in 0..=200 {
        let x = 1.9 + 0.1 * f64::from(i) / 200.0;
        beyond = beyond
            .max(((k0_series(x) * x.exp() - k_nu_scaled(0.0, x)) / k_nu_scaled(0.0, x)).abs());
    }
    assert!(
        beyond >= 5e-15,
        "the series only disagrees by {beyond:e} near the old threshold of 2.0, so the \
         degradation that moved K_SERIES_MAX down to 1.25 is no longer reproducible"
    );
}

#[test]
fn the_series_quadrature_threshold_leaves_no_step_at_ulp_resolution() {
    // The series -> quadrature switch, probed at the two representable values that straddle it.
    // K decays exponentially, so the comparison is scaled by the value itself. At the threshold
    // the two routes agree to under 1e-15 relative, which is the point of placing it inside
    // their overlap -- see
    // `the_series_quadrature_threshold_sits_inside_the_band_both_routes_are_accurate_on`.
    use euclid_utils::no_step_across;
    let x = K_SERIES_MAX;
    no_step_across("K0", k0, x, -k1(x), 1e-13);
    no_step_across("K1", k1, x, -(k0(x) + k1(x) / x), 1e-13);
    no_step_across("K0e", k0e, x, k0e(x) - k1e(x), 1e-13);
    no_step_across("K1e", k1e, x, k1e(x) - k0e(x) - k1e(x) / x, 1e-13);
}
