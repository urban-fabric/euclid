//! Unit tests for the incomplete beta's routing, fraction and prefactor.
//!
//! A child module so the implementation stays inside the crate's 400-line budget. These reach
//! private items -- `BetaRegMethod`, `CENTRED_MIN`, `beta_cf` -- which is why they live here
//! rather than in the `tests/` tree.

#![allow(clippy::unwrap_used, clippy::panic)]

use super::*;

fn close(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
}

#[test]
fn beta_known_values() {
    close(beta(1.0, 1.0), 1.0, 1e-12);
    close(beta(2.0, 3.0), 1.0 / 12.0, 1e-10);
}

#[test]
fn beta_fraction_agrees_with_the_integrated_polynomial() {
    // For (2, 3), the CDF is x²(6 − 8x + 3x²) and its prefactor
    // divided by a is 6x²(1 − x)³.
    for x in [0.007_812_5_f64, 0.0625, 0.125, 0.25, 0.4] {
        let expected = (6.0 - 8.0 * x + 3.0 * x * x) / (6.0 * (1.0 - x).powi(3));
        let Some(actual) = beta_cf(2.0, 3.0, x) else {
            panic!("fraction failed to converge at {x}");
        };
        close(actual, expected, 2e-15);
    }
}

#[test]
fn both_fraction_orientations_match_the_integrated_cdf() {
    for x in [0.0625, 0.25, 0.375, 0.5, 0.875] {
        // Integral of 20x(1-x)^3, since B(2,4) = 1/20.
        let expected = x * x * (10.0 + x * (-20.0 + x * (15.0 - 4.0 * x)));
        let Some(actual) = beta_fraction_cdf(2.0, 4.0, x) else {
            panic!("integer-shape fraction failed to converge at x={x}");
        };
        close(actual, expected, 4e-15);
    }
}

#[test]
fn beta_reg_endpoints_and_symmetry() {
    assert_eq!(beta_reg(2.0, 3.0, 0.0), 0.0);
    assert_eq!(beta_reg(2.0, 3.0, 1.0), 1.0);
    for &(a, b, x) in &[(2.0, 3.0, 0.4), (0.5, 0.5, 0.3), (5.0, 2.0, 0.7)] {
        close(beta_reg(a, b, x), 1.0 - beta_reg(b, a, 1.0 - x), 1e-10);
    }
    close(beta_reg(1.0, 1.0, 0.37), 0.37, 1e-12);
}

#[test]
fn gamma_approximation_uses_the_shifted_rate_after_the_crossover() {
    assert_eq!(beta_gamma_rate(2.0, 1e8), None);
    assert_eq!(beta_gamma_rate(2.0, 2e8), None);
    assert_eq!(beta_gamma_rate(2.0, 1e9), Some(1e9 + 0.5));
    assert_eq!(beta_gamma_rate(0.5, 1e9), Some(1e9 - 0.25));
    assert!(beta_gamma_rate(GAMMA_EDGEWORTH_MIN.next_down(), 1e30).is_some());
    assert_eq!(beta_gamma_rate(GAMMA_EDGEWORTH_MIN, 1e30), None);
}

#[test]
fn edgeworth_at_the_mean_uses_the_beta_skewness() {
    let (a, b) = (1e7_f64, 3e7_f64);
    let n = a + b;
    // At the mean, z = 0. The unscaled formula fits at these moderate shapes.
    let skewness = 2.0 * (b - a) * (n + 1.0).sqrt() / ((n + 2.0) * (a * b).sqrt());
    let expected = 0.5 + skewness / (6.0 * (2.0 * core::f64::consts::PI).sqrt());
    close(beta_edgeworth(a, b, 0.25), expected, f64::EPSILON);
}

#[test]
fn beta_mean_residual_retains_the_low_bits_of_the_offset() {
    let a = f64::from_bits(0x3ff0_0000_0000_0006);
    let b = f64::from_bits(0x3fdb_bc37_cd1d_a36d);
    let x = f64::from_bits(0x3fe6_533b_1ba2_862f);
    let (n, n_error) = a.two_sum(b);
    // Rounded once from the exact binary rationals x(a+b) − a.
    let expected = f64::from_bits(0xbcaa_a02f_14b3_5725);
    assert_eq!(beta_mean_residual(a, n, n_error, x), expected);
}

/// `CENTRED_MIN` sits where the centred prefactor is actually reachable, and the band is covered.
///
/// The threshold is a *cost* bound, not an accuracy one: `μ` is reachable at any `z ≥ 1` through
/// [`stirling_remainder_from_one`](super::super::stirling_remainder_from_one)'s climb, which runs
/// one step per unit, so `1` is where the climb stays short rather than where the method starts
/// working. A `const` block pins that, and the sweep below is what stops it being an assertion
/// about a literal: it checks the arm is genuinely taken across the band and not somewhere else.
#[test]
fn the_centred_prefactor_is_reached_across_the_band_its_threshold_claims() {
    const {
        assert!(
            CENTRED_MIN >= 1.0,
            "the μ climb in stirling_remainder_from_one costs one step per unit, so a shape \
             below 1 would take thousands of them; its debug_assert says the same"
        );
    }
    // `1 ≤ min(a, b) < 20` against a large partner is the band this threshold opened up, and
    // `a + b ≥ STIRLING_MIN` is the other condition, so at the mode all three must hold.
    let mut reached = 0;
    for &small in &[1.0_f64, 2.5, 7.68, 15.6, 19.5] {
        for &large in &[25.0_f64, 100.0, 571.72, 1000.0] {
            for (a, b) in [(small, large), (large, small)] {
                let mode = a / (a + b);
                assert!(
                    use_centred_prefactor(a, b, mode),
                    "at a={a}, b={b} the mode is not reaching the centred prefactor"
                );
                reached += 1;
            }
        }
    }
    assert_eq!(reached, 40, "the band sweep changed shape");
    // And it is still refused below the threshold and where `a + b` is too small for the series.
    assert!(!use_centred_prefactor(0.5, 1000.0, 0.0005));
    assert!(!use_centred_prefactor(2.0, 3.0, 0.4));
}

/// `I_x(a,b) − I_x(a+1,b) = x^a(1−x)^b/(a·B(a,b))`, which isolates the prefactor.
///
/// The metamorphic relation worth having for this change: its right-hand side **is** the front
/// factor, so a prefactor that has drifted shows up here directly, where the reflection identity
/// would partly cancel it away. Both sides are evaluated through the public surface, and the
/// right-hand side through `ln_beta` rather than the centred form, so the two arrangements are
/// compared against each other rather than one against itself.
///
/// Swept over `1 ≤ min(a, b) < 20` against a large partner — the band `CENTRED_MIN` opened, and
/// the band where `beta_reg`'s worst points sat before it.
///
/// # The relation is only usable where its left side is not a cancelling difference
///
/// At `x` well above the mode both `I_x(a,b)` and `I_x(a+1,b)` round to near one while their
/// difference is the prefactor, which is vanishing — at `a = 50, b = 7.68, x = 0.9999` the right
/// side is `1e-30` and the left is two doubles near `1` subtracting, so the relation reads as
/// `100%` wrong while nothing is.
///
/// So the bound is **normalised by the cancellation** rather than flat. Forming
/// `I_x(a,b) − I_x(a+1,b)` loses `I/difference` of relative accuracy — one ulp of `I` is
/// `ε·I/difference` of the difference — so the resolution this relation can offer degrades
/// exactly as the two terms approach each other. Asserting a flat `5e-14` instead reported
/// `1.4e-11` at `a = 881.7, b = 1.5` as a failure, where the ratio is `3e3` and `1.4e-11` is the
/// floor. Dividing by the ratio is what turns this into a statement about the prefactor.
///
/// Measured worst after that normalisation is `6.1e-14`, at `a = 400, b = 7.68`; the bound is
/// `6e-13`, an order of margin. That is loose as ulp go, and it is what this relation can
/// honestly offer: both sides pass through `beta_reg`, and its own error over this band is tens
/// of ulp. What it catches is a prefactor wrong by a *factor* or a sign, which is the failure a
/// rearrangement actually produces.
#[test]
fn the_prefactor_recurrence_holds_across_the_band_the_centring_governs() {
    use crate::Real;
    let mut worst = (0.0_f64, 0.0, 0.0, 0.0);
    for &small in &[1.5_f64, 3.0, 7.68, 15.6] {
        for &large in &[50.0_f64, 400.0, 881.7] {
            for (a, b) in [(small, large), (large, small)] {
                for step in 0..=3 {
                    // At and below the mode, where the difference is the size of `I` itself.
                    let spread = (a * b / ((a + b) * (a + b) * (a + b + 1.0))).sqrt();
                    let x = a / (a + b) - f64::from(step) * spread;
                    if !(0.0..1.0).contains(&x) {
                        continue;
                    }
                    let lower = Real::beta_reg(x, a, b);
                    let difference = lower - Real::beta_reg(x, a + 1.0, b);
                    let front = (a * x.ln() + b * (-x).ln_1p() - Real::ln_beta(a, b)).exp() / a;
                    // Skip where the subtraction has thrown the answer away entirely; past
                    // this the relation carries no information to assert on.
                    if difference <= 0.0 || difference < 1e-10 * lower {
                        continue;
                    }
                    // Scaled by how much the subtraction cost, so what is left is the
                    // prefactor's own disagreement rather than the cancellation's.
                    let amplification = lower / difference;
                    let relative = ((difference - front) / front).abs() / amplification;
                    if relative > worst.0 {
                        worst = (relative, a, b, x);
                    }
                }
            }
        }
    }
    assert!(
        worst.0 <= 6e-13,
        "the prefactor recurrence is off by {:e} per unit of cancellation at a={}, b={}, \
         x={} — measured 6.1e-14 over this grid, so the front factor or ln_beta has moved",
        worst.0,
        worst.1,
        worst.2,
        worst.3
    );
    assert!(
        worst.0 > 0.0,
        "the two sides were bit-identical, so this compared nothing"
    );
}
