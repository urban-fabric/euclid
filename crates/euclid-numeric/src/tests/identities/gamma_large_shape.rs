//! The incomplete gamma at large shape, where its series cannot reach.
//!
//! [`crate::tests::stability::extremes`] sweeps the shape to `1e5` and finds the function
//! robust. The defect began just past that, and it was the quiet kind: `P(a, a)` must tend to
//! `½`, and instead it drifted away and then stopped being a probability. Measured against the
//! known asymptote `½ + 1/(3√(2πa))`:
//!
//! ```text
//! a = 1e7   2.9e-8        a = 1e10   3.2e-1
//! a = 1e8   3.4e-7        a = 1e11   7.5e-1
//! a = 1e9   1.6e-3        a = 1e15   P(a, a+√a) = −9.06
//!                         a = 1e16   NaN
//! ```
//!
//! and in `f32`, four decades earlier: `37%` wrong by `a = 1e6`, `NaN` from `1e8`.
//!
//! Two mechanisms compounded. The series needs `O(√a)` terms and silently stops at its
//! iteration cap, and the front factor `exp(a ln x − x − lnΓ(a))` subtracts three
//! `O(a ln a)` quantities — which past `a ≈ 1e14` leaves nothing but noise.
//!
//! Above the crossover both carriers now use an Edgeworth-corrected normal approximation,
//! whose error is `O(1/a)` and therefore *shrinks* exactly where the series fails. The tests
//! below pin three things: that `P` and `Q` remain probabilities, that they agree with the
//! asymptote, and that the join is where the measurements say it should be.

use crate::Real;

/// `½ + 1/(3√(2πa))`, the limit of `P(a, a)`, with its own error `O(1/a)`.
fn median_asymptote(a: f64) -> f64 {
    0.5 + 1.0 / (3.0 * (core::f64::consts::TAU * a).sqrt())
}

#[test]
fn the_shape_median_approaches_one_half_from_above() {
    // P(a, a) -> 1/2, and the approach is the asymptote above. This is the statement that
    // failed outright: 0.34 at a = 1e10, 0.018 at 1e15, NaN beyond.
    for exponent in [
        5.0f64, 7.0, 7.5, 8.0, 9.0, 10.0, 12.0, 15.0, 20.0, 50.0, 150.0, 300.0,
    ] {
        let a = 10f64.powf(exponent);
        let got = Real::gamma_lr(a, a);
        let want = median_asymptote(a);
        // The bound is the *reference's*: the asymptote drops a term of relative size O(1/a),
        // so below a = 1e7 it is the less accurate side. Both are far inside this.
        let slack = (4.0 / a).max(8.0 * f64::EPSILON);
        assert!(
            (got - want).abs() <= slack,
            "P(1e{exponent}, same) = {got}, expected {want} (slack {slack:e})"
        );
    }
}

#[test]
fn both_tails_stay_probabilities_at_every_shape() {
    // The invariant the old code broke in three different ways: a value outside [0, 1], a
    // negative probability, and a NaN. Swept over the whole representable shape range and
    // several standard deviations either side of the mean.
    for exponent in [3.0f64, 6.0, 9.0, 12.0, 20.0, 60.0, 150.0, 300.0] {
        let a = 10f64.powf(exponent);
        for &deviations in &[-6.0f64, -3.0, -1.0, 0.0, 1.0, 3.0, 6.0] {
            let x = a + deviations * a.sqrt();
            if !(x.is_finite() && x > 0.0) {
                continue;
            }
            let (lower, upper) = (Real::gamma_lr(x, a), Real::gamma_ur(x, a));
            assert!(
                (0.0..=1.0).contains(&lower),
                "P({x:e}, 1e{exponent}) = {lower} is not a probability"
            );
            assert!(
                (0.0..=1.0).contains(&upper),
                "Q({x:e}, 1e{exponent}) = {upper} is not a probability"
            );
            assert!(
                (lower + upper - 1.0).abs() <= 8.0 * f64::EPSILON,
                "P + Q != 1 at ({x:e}, 1e{exponent}): {}",
                lower + upper - 1.0
            );
            // And monotone in x, which a truncated series was not.
            if deviations > -6.0 {
                let previous = Real::gamma_lr(a + (deviations - 1.0) * a.sqrt(), a);
                assert!(
                    lower >= previous - 8.0 * f64::EPSILON,
                    "P not monotone at ({x:e}, 1e{exponent})"
                );
            }
        }
    }
}

#[test]
fn the_normal_limit_is_reached_in_the_shape_limit() {
    // As `a → ∞` the standardised gamma becomes standard normal, so `P(a, a + z√a) → Φ(z)`.
    // The Edgeworth correction is what makes the approach `O(1/a)` rather than `O(1/√a)`;
    // this checks the limit itself, which no series could reach at these shapes.
    for &z in &[-3.0f64, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0] {
        let mut previous_gap = f64::INFINITY;
        for exponent in [12.0f64, 20.0, 40.0, 100.0] {
            let a = 10f64.powf(exponent);
            let x = a + z * a.sqrt();
            // `Φ` is evaluated at the `z` the argument *actually* carries, not the nominal
            // one. Past a = 1e16 the ulp of `a` exceeds 1, so `a + z√a` rounds, and at
            // a = 1e20 that shifts `z` by up to 8e-7 — which `φ(3)` turns into 3.6e-9 of
            // probability, far more than anything the function contributes.
            let normal = Real::ndtr((x - a) / a.sqrt());
            let got = Real::gamma_lr(x, a);
            // The gap closes like `1/√a`, not `1/a`: the gamma's skewness is `2/√a`, and the
            // residual *is* that skewness term. At a = 1e12, z = −3 it is 1.18e-8, which is
            // the function being right rather than wrong — so the bound has to carry it.
            let gap = (got - normal).abs();
            assert!(
                gap <= 10.0 / a.sqrt(),
                "P(1e{exponent}, z = {z}) = {got}, expected Phi(z) = {normal} (gap {gap:e})"
            );
            // And the convergence is monotone in the shape, which is the substance of the
            // limit and cannot be satisfied by any fixed approximation.
            assert!(
                gap <= previous_gap,
                "the normal limit is not approached monotonically at z = {z}: \
                 {previous_gap:e} then {gap:e}"
            );
            previous_gap = gap;
        }
    }
}

#[test]
fn the_crossover_leaves_no_step_in_either_carrier() {
    // The series and the approximation are wholly unrelated — a term-by-term sum against a
    // normal CDF with a skewness correction — so agreement across the threshold is the real
    // evidence that it was sited correctly. Checked at several `z`, since the two routes have
    // quite different `z` dependence.
    let threshold = crate::kernels::gamma::GAMMA_EDGEWORTH_MIN;
    for &z in &[-2.0f64, -1.0, 0.0, 1.0, 2.0] {
        let below = Real::next_after(threshold, f64::NEG_INFINITY);
        let above = Real::next_after(threshold, f64::INFINITY);
        let (lo, hi) = (
            Real::gamma_lr(below + z * below.sqrt(), below),
            Real::gamma_lr(above + z * above.sqrt(), above),
        );
        assert!(
            (hi - lo).abs() < 1e-7,
            "the shape crossover steps at z = {z}: {lo} then {hi}"
        );
    }
    // And the f32 carrier, whose threshold is four decades lower and whose agreement there is
    // correspondingly looser — 4.1e-5 measured, which is still 1e4 times better than the
    // series it replaces.
    for &z in &[-1.0f32, 0.0, 1.0] {
        let (below, above) = (999.0f32, 1001.0f32);
        let (lo, hi) = (
            Real::gamma_lr(below + z * below.sqrt(), below),
            Real::gamma_lr(above + z * above.sqrt(), above),
        );
        assert!(
            (hi - lo).abs() < 1e-3,
            "the f32 shape crossover steps at z = {z}: {lo} then {hi}"
        );
    }
}

#[test]
fn the_f32_carrier_is_a_probability_at_every_shape_too() {
    for exponent in [1.0f32, 3.0, 4.0, 6.0, 10.0, 20.0, 35.0] {
        let a = 10f32.powf(exponent);
        for &deviations in &[-3.0f32, 0.0, 3.0] {
            let x = a + deviations * a.sqrt();
            if !(x.is_finite() && x > 0.0) {
                continue;
            }
            let (lower, upper) = (Real::gamma_lr(x, a), Real::gamma_ur(x, a));
            assert!(
                (0.0..=1.0).contains(&lower) && (0.0..=1.0).contains(&upper),
                "f32 ({x:e}, 1e{exponent}) gave ({lower}, {upper})"
            );
            assert!(
                (lower + upper - 1.0).abs() <= 8.0 * f32::EPSILON,
                "f32 P + Q != 1 at ({x:e}, 1e{exponent})"
            );
        }
    }
    // Against the f64 reference, which is now exact at these shapes.
    for exponent in [3.0f32, 4.0, 5.0, 6.0] {
        let a = 10f32.powf(exponent);
        for &z in &[-1.0f32, 0.0, 1.0] {
            let x = a + z * a.sqrt();
            let got = f64::from(Real::gamma_lr(x, a));
            let truth = Real::gamma_lr(f64::from(x), f64::from(a));
            assert!(
                (got - truth).abs() < 1e-4,
                "f32 P(1e{exponent}, z={z}) = {got} vs f64 {truth}"
            );
        }
    }
}
