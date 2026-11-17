//! Closed-form edges: a CDF at special parameters collapses to an unrelated function.

use crate::Real;
use euclid_utils::assert_close;

// --- closed-form edges ---------------------------------------------------

#[test]
fn incomplete_gamma_at_half_shape_equals_the_error_function() {
    // P(1/2, x) = erf(sqrt x). This is the chi-squared-with-1-degree-of-freedom identity, and
    // it is the single strongest edge in the web: the left side runs through the incomplete
    // gamma series / continued fraction, the right through an entirely separate erf
    // approximation. They share no code, so agreement is real evidence.
    for i in 1..200 {
        let x = f64::from(i) * 0.1;
        assert_close(
            Real::gamma_lr(x, 0.5),
            Real::erf(x.sqrt()),
            1e-12,
            &format!("P(1/2, {x}) = erf(sqrt {x})"),
        );
    }
    // The complement likewise: Q(1/2, x) = erfc(sqrt x).
    for i in 1..100 {
        let x = f64::from(i) * 0.25;
        assert_close(
            Real::gamma_ur(x, 0.5),
            Real::erfc(x.sqrt()),
            1e-10,
            &format!("Q(1/2, {x}) = erfc(sqrt {x})"),
        );
    }
}

#[test]
fn incomplete_beta_at_half_parameters_is_the_arcsine_law() {
    // I_x(1/2, 1/2) = (2/pi) asin(sqrt x). Ties the incomplete beta's continued fraction to
    // `asin` and `pi` — again, no shared code.
    for i in 1..100 {
        let x = f64::from(i) * 0.01;
        let want = 2.0 / <f64 as Real>::pi() * Real::asin(x.sqrt());
        assert_close(
            Real::beta_reg(x, 0.5, 0.5),
            want,
            1e-12,
            &format!("I_{x}(1/2,1/2) = arcsine"),
        );
    }
}

#[test]
fn incomplete_beta_at_unit_parameters_collapses_to_powers() {
    // I_x(a, 1) = x^a and I_x(1, b) = 1 - (1-x)^b. Ties `beta_reg` to `powf`.
    for i in 1..100 {
        let x = f64::from(i) * 0.01;
        for &a in &[0.5f64, 1.0, 3.0, 7.5] {
            assert_close(
                Real::beta_reg(x, a, 1.0),
                Real::powf(&x, a),
                1e-12,
                &format!("I_{x}({a},1) = x^{a}"),
            );
            assert_close(
                Real::beta_reg(x, 1.0, a),
                1.0 - Real::powf(&(1.0 - x), a),
                1e-12,
                &format!("I_{x}(1,{a})"),
            );
        }
    }
}

#[test]
fn the_chi_squared_cdf_agrees_through_two_routes() {
    // A chi-squared with k degrees of freedom is Gamma(k/2, 2), so its CDF is P(k/2, x/2).
    // For k = 1 that must also equal erf(sqrt(x/2)), closing a triangle between `gamma_lr`,
    // `erf`, and the k = 2 exponential special case.
    for i in 1..100 {
        let x = f64::from(i) * 0.25;
        // k = 1
        assert_close(
            Real::gamma_lr(x / 2.0, 0.5),
            Real::erf((x / 2.0).sqrt()),
            1e-12,
            "chi2(1) via gamma_lr vs erf",
        );
        // k = 2 is Exponential(1/2): CDF = 1 - e^{-x/2}
        assert_close(
            Real::gamma_lr(x / 2.0, 1.0),
            -Real::exp_m1(&(-x / 2.0)),
            1e-14,
            "chi2(2) is exponential",
        );
    }
}
