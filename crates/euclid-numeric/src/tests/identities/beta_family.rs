//! Identities of the incomplete beta.

use crate::Real;
use euclid_utils::assert_close;

// --- incomplete beta ----------------------------------------------------

#[test]
fn beta_reg_is_symmetric_under_swapping() {
    // I_x(a,b) = 1 − I_{1−x}(b,a)
    for &(a, b) in &[(0.5f64, 0.5f64), (1.0, 1.0), (2.0, 5.0), (7.5, 3.25)] {
        for i in 0..=100 {
            let x = f64::from(i) * 0.01;
            assert_close(
                Real::beta_reg(x, a, b),
                1.0 - Real::beta_reg(1.0 - x, b, a),
                1e-13,
                &format!("beta_reg symmetry at x={x}, a={a}, b={b}"),
            );
        }
    }
}

#[test]
fn beta_reg_endpoints_and_uniform_case() {
    for &(a, b) in &[(0.5f64, 2.0f64), (1.0, 1.0), (4.0, 9.0)] {
        assert_eq!(Real::beta_reg(0.0, a, b), 0.0);
        assert_close(Real::beta_reg(1.0, a, b), 1.0, 1e-14, "I_1(a,b) = 1");
    }
    // I_x(1,1) = x, exactly the uniform CDF.
    for i in 0..=100 {
        let x = f64::from(i) * 0.01;
        assert_close(Real::beta_reg(x, 1.0, 1.0), x, 1e-14, "I_x(1,1) = x");
    }
    // I_x(a,1) = x^a
    for i in 1..=100 {
        let x = f64::from(i) * 0.01;
        assert_close(
            Real::beta_reg(x, 3.0, 1.0),
            x * x * x,
            1e-13,
            "I_x(3,1) = x^3",
        );
    }
}

#[test]
fn ln_beta_matches_the_gamma_quotient() {
    // ln B(a,b) = lnΓ(a) + lnΓ(b) − lnΓ(a+b)
    for &a in &[0.25f64, 1.0, 3.5, 20.0] {
        for &b in &[0.75f64, 2.0, 11.0] {
            assert_close(
                Real::ln_beta(a, b),
                Real::ln_gamma(a) + Real::ln_gamma(b) - Real::ln_gamma(a + b),
                1e-13,
                &format!("ln_beta({a},{b})"),
            );
            // And `beta` is its exponential.
            assert_close(
                Real::beta(a, b),
                Real::ln_beta(a, b).exp(),
                1e-12,
                &format!("beta({a},{b})"),
            );
        }
    }
}
