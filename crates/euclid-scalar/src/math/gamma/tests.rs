//! Unit tests for the precision-generic gamma machinery.
//!
//! A child module so implementation plus tests stay inside the crate 400-line budget, while
//! `use super::*` still reaches the Lanczos coefficients, the Stirling remainder, the
//! `LanczosLogForm` split and `reflection_sine`. Everything here is generic over the carrier,
//! so each law is asserted once and holds for both precisions — which is the whole reason this
//! code lives in `math` rather than being duplicated per format.

use super::*;
use crate::math::continued_fraction::{LENTZ_TINY_F32, LENTZ_TINY_F64};

#[test]
fn lanczos_terms_match_rational_coefficients() {
    let coefficients = [2.0_f64, 4.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    assert_eq!(lanczos_parts(1.0, 7.0, &coefficients), (0.5, 7.5, 9.0));
    let coefficients = coefficients.map(|c| c as f32);
    assert_eq!(lanczos_parts(2.0, 7.0, &coefficients), (1.5, 8.5, 6.0));
}

#[test]
fn the_shared_log_prefactor_is_half_the_log_of_tau() {
    // `½ln(2π)`. Asserted here because as a term of either form it is thirty-one orders below
    // the result's last bit; see `log_sqrt_tau`. Every mutation of this expression lands at
    // least `0.07` away — `0.2723` for a division, `0.8187` for a sum inside the parentheses
    // — so an `EPSILON` bound is far more than the arithmetic needs to separate them.
    const HALF_LOG_TAU: f64 = 0.918_938_533_204_672_7;
    assert!((log_sqrt_tau::<f64>() - HALF_LOG_TAU).abs() <= f64::EPSILON);
    assert!((log_sqrt_tau::<f32>() - HALF_LOG_TAU as f32).abs() <= f32::EPSILON);
}

#[test]
fn the_lanczos_log_form_follows_the_products_range() {
    use LanczosLogForm::{Direct, Folded};
    for scaled in [0.0_f64, -1.0, 1e308, f64::MIN, f64::MAX] {
        assert_eq!(LanczosLogForm::for_scaled_power(scaled), Direct);
    }
    for scaled in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
        assert_eq!(LanczosLogForm::for_scaled_power(scaled), Folded);
    }
    assert_eq!(LanczosLogForm::for_scaled_power(f32::MAX), Direct);
    assert_eq!(LanczosLogForm::for_scaled_power(f32::INFINITY), Folded);
}

#[test]
fn the_folded_form_is_the_same_sum_rearranged() {
    // Ordinary arguments, where every term is the order of the answer — which is the point of
    // naming the form at all. `power·ln(base)` is nowhere near overflowing here, so these
    // inputs never arise on the folded path and only a direct call can reach this arithmetic.
    // `(2, 3, 4, 2)` gives `½ln(2π) + 2·(3 − 2) + ln 2`; the nearest mutant is a whole unit
    // away, the furthest lands at `−8.39`.
    let folded = lanczos_log_folded(2.0_f64, 3.0, 4.0, 2.0);
    assert!((folded - 3.612_085_713_764_618).abs() <= 4.0 * f64::EPSILON);

    // The two forms are the same quantity: folding distributes `power` over `− base/power`.
    // Checked where both are evaluable, which is the only place the claim can be checked.
    let (power, base, sum) = (10.0_f64, 12.0, 3.0);
    let direct = lanczos_log_direct(power * base.ln(), base, sum);
    let folded = lanczos_log_folded(power, base.ln(), base, sum);
    assert!((direct - folded).abs() <= 2.0 * f64::EPSILON * direct.abs());

    let (power, base, sum) = (10.0_f32, 12.0, 3.0);
    let direct = lanczos_log_direct(power * base.ln(), base, sum);
    let folded = lanczos_log_folded(power, base.ln(), base, sum);
    assert!((direct - folded).abs() <= 2.0 * f32::EPSILON * direct.abs());
}

#[test]
fn reflection_sine_tracks_parity_and_the_nearest_pole() {
    for (x, expected) in [(0.5_f64, 1.0), (-0.5, -1.0), (-1.5, 1.0), (-2.5, -1.0)] {
        assert_eq!(reflection_sine(x), expected);
        assert_eq!(reflection_sine(x as f32), expected as f32);
    }
    let pole = -2.0_f64;
    for x in [pole.next_down(), pole.next_up()] {
        let expected = core::f64::consts::PI * (x - pole);
        assert_eq!(reflection_sine(x), expected);
    }
    let pole = -3.0_f32;
    for x in [pole.next_down(), pole.next_up()] {
        let expected = -core::f32::consts::PI * (x - pole);
        assert!((reflection_sine(x) / expected - 1.0).abs() <= f32::EPSILON);
    }
}

#[test]
fn reflection_switch_excludes_its_own_fixed_point() {
    let half = 0.5_f32;
    assert!(use_gamma_reflection(half.next_down()));
    assert!(!use_gamma_reflection(half));
    assert!(!use_gamma_reflection(half.next_up()));
    let half = 0.5_f64;
    assert!(use_gamma_reflection(half.next_down()));
    assert!(!use_gamma_reflection(half));
}

#[test]
fn gamma_reflection_selects_the_overflow_evaluator_by_range() {
    let ordinary = gamma_reflection(-0.5_f64, |_, _| 0.0);
    assert!((ordinary / (-2.0 * core::f64::consts::PI.sqrt()) - 1.0).abs() < 2e-15);
    let overflow = gamma_reflection(-172.5_f64, |sin, reflected| {
        assert_eq!(sin, -1.0);
        assert_eq!(reflected, 173.5);
        -1.0
    });
    assert_eq!(overflow, -1.0);
}

#[test]
fn gamma_input_and_series_switch_boundaries() {
    for (shape, point, valid) in [
        (0.0_f64, 1.0, false),
        (-1.0, 1.0, false),
        (f64::NAN, 1.0, false),
        (f64::INFINITY, 1.0, false),
        (1.0, -0.125, false),
        (1.0, f64::NAN, false),
        (1.0, -0.0, true),
        (1.0, f64::INFINITY, true),
    ] {
        assert_eq!(valid_incomplete_gamma_inputs(shape, point), valid);
        assert_eq!(
            valid_incomplete_gamma_inputs(shape as f32, point as f32),
            valid
        );
    }
    let x = 4.0_f64;
    assert!(use_lower_gamma_series(3.0, x.next_down()));
    assert!(!use_lower_gamma_series(3.0, x));
    assert!(!use_lower_gamma_series(3.0, x.next_up()));
    assert!(use_lower_gamma_series(4.0, x));
}

#[test]
fn lower_series_step_matches_its_first_terms() {
    let (mut denominator, mut term) = (1.0_f64, 1.0);
    advance_lower_series(1.0, &mut denominator, &mut term);
    assert_eq!((denominator, term), (2.0, 0.5));
    advance_lower_series(1.0, &mut denominator, &mut term);
    assert_eq!((denominator, term), (3.0, 1.0 / 6.0));
    assert_eq!(gamma_lower_series(1.0_f64, 1.0, 0.0, 0), 1.0);
    assert_eq!(gamma_lower_series(1.0_f64, 1.0, 0.0, 1), 1.5);
    assert_eq!(gamma_lower_series(1.0_f32, 1.0, 0.5, 8), 1.5);
}

#[test]
fn lower_series_agrees_with_the_exponential_closed_forms() {
    for x in [0.25_f64, 0.5, 1.5] {
        let e = x.exp_m1();
        for (shape, expected) in [(1.0, e / x), (2.0, (e - x) / (x * x))] {
            let actual = gamma_lower_series(shape, x, 1e-15, 100);
            assert!((actual / expected - 1.0).abs() < 8.0 * f64::EPSILON);
            let actual = gamma_lower_series(shape as f32, x as f32, 1e-6, 100);
            assert!((f64::from(actual) / expected - 1.0).abs() < 5e-7);
        }
    }
}

#[test]
fn upper_fraction_uses_the_exact_coefficient_recurrence_and_budget() {
    let mut denominator = 2.0_f64;
    for (k, numerator, next_denominator) in [
        (1.0, 2.0, 4.0),
        (2.0, 2.0, 6.0),
        (3.0, 0.0, 8.0),
        (4.0, -4.0, 10.0),
    ] {
        assert_eq!(upper_gamma_coefficient(3.0, k, &mut denominator), numerator);
        assert_eq!(denominator, next_denominator);
    }
    assert_eq!(gamma_upper_fraction(3.0, 4.0, LENTZ_TINY_F64, 0.0, 1), 0.5);
    assert_eq!(
        gamma_upper_fraction(3.0, 4.0, LENTZ_TINY_F64, 0.25, 10),
        0.4
    );
}

#[test]
fn upper_fraction_agrees_with_integer_shape_closed_forms() {
    for x in [4.0_f64, 8.0, 18.75, 100.0] {
        for (a, expected) in [
            (1.0, 1.0 / x),
            (2.0, (x + 1.0) / x.powi(2)),
            (3.0, (x * x + 2.0 * x + 2.0) / x.powi(3)),
        ] {
            let actual = gamma_upper_fraction(a, x, LENTZ_TINY_F64, 1e-15, 100);
            assert!((actual / expected - 1.0).abs() <= 2e-15);
            let actual = gamma_upper_fraction(a as f32, x as f32, LENTZ_TINY_F32, 1e-6, 100);
            assert!((f64::from(actual) / expected - 1.0).abs() <= 3e-7);
        }
    }
}

#[test]
fn edgeworth_correction_matches_standardized_points() {
    for z in [-2.0_f64, -1.0, 0.0, 1.0, 2.0] {
        let x = 16.0 + 4.0 * z;
        let expected =
            (z * z - 1.0) * (-z * z / 2.0).exp() / ((2.0 * core::f64::consts::PI).sqrt() * 12.0);
        let (actual_z, correction) = gamma_edgeworth_terms(16.0, x);
        assert_eq!(actual_z, z);
        assert!((correction - expected).abs() <= 7e-18);
    }
    let (z, correction) = gamma_edgeworth_terms(16.0_f32, f32::MAX);
    assert!(z.is_finite());
    assert_eq!(correction, 0.0);
    assert_eq!(gamma_edgeworth(16.0_f64, f64::MAX), (1.0, 0.0));
}

#[test]
fn log_scale_matches_the_prefactor_at_integer_shape() {
    // Γ(2) = 1 and 1.5 * 4² = 24.
    let actual = log_scaled_gamma(2.0_f64, 4.0, 1.5_f64.ln());
    assert!((actual - (24.0_f64.ln() - 4.0)).abs() <= 2e-15);
}
