//! Unit tests for the gamma family.
//!
//! A child module in its own file so implementation plus tests stay inside the crate's
//! 400-line budget, while `use super::*` still reaches the private Lanczos coefficients,
//! the Stirling remainder and the Edgeworth crossover.

use super::*;

fn close(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
}

#[test]
fn gamma_matches_factorials() {
    close(gamma(1.0), 1.0, 1e-12);
    close(gamma(5.0), 24.0, 1e-10);
    close(gamma(6.0), 120.0, 1e-10);
    close(gamma(0.5), core::f64::consts::PI.sqrt(), 1e-12);
}

#[test]
fn reflection_preserves_the_negative_tail_and_pole_distance() {
    use crate::Real;

    // 256-bit MPFR reference values, rounded to each destination format.
    for (x, expected) in [
        (-172.5_f64, 0x8000_0034_c532_409a),
        (-174.5, 0x8000_0000_0072_3a81),
        (-176.5, 0x8000_0000_0000_00f2),
        (-177.5, 1),
        (-178.5, 0x8000_0000_0000_0000),
        (-1500.5, 0x8000_0000_0000_0000),
        (-1501.5, 0),
    ] {
        assert_eq!(gamma(x).to_bits(), expected, "x={x}");
    }
    for (x, expected) in [
        (-34.25_f32, 0x8043_8d55),
        (-37.5, 0x1b),
        (-39.5, 0),
        (-40.5, 0x8000_0000),
    ] {
        assert_eq!(Real::gamma(x).to_bits(), expected, "x={x}");
    }
    for (x, expected) in [
        ((-2.0_f64).next_down(), -1_125_899_906_842_623.5),
        ((-2.0_f64).next_up(), 2_251_799_813_685_248.5),
        ((-171.0_f64).next_down(), 2.835_121_658_031_596e-296),
    ] {
        let value = gamma(x);
        assert!(
            (value / expected - 1.0).abs() < 3e-13,
            "x={x}, value={value}"
        );
    }
    close(
        ln_gamma(-1.5),
        (4.0 * core::f64::consts::PI.sqrt() / 3.0).ln(),
        3e-15,
    );
    assert!(ln_gamma(-0.5).is_nan());
}

#[test]
fn ln_gamma_matches_gamma() {
    for &x in &[0.3, 0.7, 1.5, 3.2, 10.0, 50.0] {
        close(ln_gamma(x).exp(), gamma(x), 1e-9);
    }
}

#[test]
fn small_shape_log_normalizer_matches_mpfr_at_the_cutoff() {
    // Regenerate with crates/euclid-scalar/tests/references/extreme_shapes.py.
    let expected = -0.576_393_598_283_369_6;
    assert!((ln_gamma1p_over_a(SMALL_SHAPE) - expected).abs() <= f64::EPSILON / 2.0);
}

#[test]
fn regularized_incomplete_gamma_bounds_and_complement() {
    for &(a, x) in &[(1.0, 0.5), (2.5, 3.0), (5.0, 2.0), (10.0, 12.0)] {
        let p = gamma_lr(a, x);
        let q = gamma_ur(a, x);
        assert!((0.0..=1.0).contains(&p), "P out of range: {p}");
        close(p + q, 1.0, 1e-10);
    }
    close(gamma_lr(1.0, 2.0), 1.0 - (-2.0f64).exp(), 1e-10);
}

#[test]
fn digamma_known_values() {
    // Tightened to 1e-11 after adding the 1/(240·x⁸) asymptotic term: the prior
    // truncation reached only ~1e-10 (vs the ~15-digit intent of the coefficients).
    close(digamma(1.0), -crate::constants::f64::EULER_GAMMA, 1e-11);
    close(
        digamma(2.0),
        1.0 - crate::constants::f64::EULER_GAMMA,
        1e-11,
    );
    // ψ(3) = 3/2 − γ.
    close(
        digamma(3.0),
        1.5 - crate::constants::f64::EULER_GAMMA,
        1e-11,
    );
}

#[test]
fn trigamma_known_values() {
    let pi2 = core::f64::consts::PI * core::f64::consts::PI;
    close(trigamma(1.0), pi2 / 6.0, 1e-10);
    close(trigamma(2.0), pi2 / 6.0 - 1.0, 1e-10);
    close(trigamma(0.5), pi2 / 2.0, 1e-10);
    close(trigamma(3.3) - trigamma(4.3), 1.0 / (3.3 * 3.3), 1e-10);
}

#[test]
fn positive_trigamma_matches_the_series_on_both_sides_of_the_shift() {
    // At integer and half-integer arguments the series is a finite sum
    // subtracted from π²/6 or π²/2. References use 70-digit Decimal sums.
    for (x, expected) in [
        (0.5, 4.934_802_200_544_679),
        (1.0, 1.644_934_066_848_226_4),
        (19.5, 0.052_619_441_213_655_93),
        (20.0, 0.051_270_822_935_203_12),
        (20.5, 0.049_989_592_429_960_99),
        (64.0, 0.015_747_706_064_338_93),
    ] {
        let actual = trigamma_positive(x);
        assert!(
            (actual - expected).abs() < 2e-14 * expected,
            "{x}: {actual} vs {expected}"
        );
    }
}
