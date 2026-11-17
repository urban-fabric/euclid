//! Identities of the gamma family, including the incomplete gamma.

use crate::Real;
use euclid_utils::{EULER_GAMMA as EULER, assert_close};

// --- gamma family -------------------------------------------------------

#[test]
fn ln_gamma_satisfies_its_recurrence() {
    // ln Γ(x+1) = ln Γ(x) + ln x
    for i in 1..400 {
        let x = f64::from(i) * 0.1;
        assert_close(
            Real::ln_gamma(x + 1.0),
            Real::ln_gamma(x) + x.ln(),
            1e-13,
            &format!("ln_gamma recurrence at {x}"),
        );
    }
}

#[test]
fn ln_gamma_vanishes_at_one_and_two() {
    // Γ(1) = Γ(2) = 1
    assert!(Real::ln_gamma(1.0).abs() < 1e-14);
    assert!(Real::ln_gamma(2.0).abs() < 1e-14);
}

#[test]
fn ln_gamma_satisfies_the_reflection_formula() {
    // Γ(x)Γ(1−x) = π / sin(πx)  ⇒  lnΓ(x) + lnΓ(1−x) = ln π − ln|sin πx|
    for i in 1..20 {
        let x = f64::from(i) * 0.05; // (0, 1)
        let lhs = Real::ln_gamma(x) + Real::ln_gamma(1.0 - x);
        let rhs = core::f64::consts::PI.ln() - (core::f64::consts::PI * x).sin().abs().ln();
        assert_close(lhs, rhs, 1e-13, &format!("reflection at {x}"));
    }
}

#[test]
fn gamma_matches_factorials() {
    // Γ(n+1) = n!
    let mut factorial = 1.0f64;
    for n in 1..=20u32 {
        factorial *= f64::from(n);
        assert_close(
            Real::gamma(f64::from(n) + 1.0),
            factorial,
            1e-13,
            &format!("gamma({}) = {n}!", n + 1),
        );
    }
    // Γ(1/2) = √π
    assert_close(
        Real::gamma(0.5),
        core::f64::consts::PI.sqrt(),
        1e-14,
        "gamma(1/2) = sqrt(pi)",
    );
}

#[test]
fn gamma_agrees_with_exp_of_ln_gamma() {
    for i in 1..150 {
        let x = f64::from(i) * 0.1;
        assert_close(
            Real::gamma(x),
            Real::ln_gamma(x).exp(),
            1e-12,
            &format!("gamma({x}) = exp(ln_gamma({x}))"),
        );
    }
}

#[test]
fn digamma_satisfies_its_recurrence() {
    // ψ(x+1) = ψ(x) + 1/x
    for i in 1..300 {
        let x = f64::from(i) * 0.05;
        assert_close(
            Real::digamma(x + 1.0),
            Real::digamma(x) + 1.0 / x,
            1e-10,
            &format!("digamma recurrence at {x}"),
        );
    }
}

#[test]
fn digamma_at_one_is_negative_euler_gamma() {
    // digamma reaches ~1.3e-12 here, not full precision; see the accuracy suite.
    assert_close(Real::digamma(1.0), -EULER, 1e-11, "psi(1) = -gamma");
}

#[test]
fn trigamma_satisfies_its_recurrence() {
    // ψ'(x+1) = ψ'(x) − 1/x²
    for i in 1..300 {
        let x = f64::from(i) * 0.05;
        assert_close(
            Real::trigamma(x + 1.0),
            Real::trigamma(x) - 1.0 / (x * x),
            1e-10,
            &format!("trigamma recurrence at {x}"),
        );
    }
}

#[test]
fn trigamma_is_positive_and_decreasing() {
    // ψ' is the variance of a log-gamma; it must stay positive and fall monotonically.
    let mut prev = f64::INFINITY;
    for i in 1..500 {
        let x = f64::from(i) * 0.1;
        let v = Real::trigamma(x);
        assert!(v > 0.0, "trigamma({x}) = {v} must be positive");
        assert!(v < prev, "trigamma must decrease: {prev} -> {v}");
        prev = v;
    }
}

// --- incomplete gamma ---------------------------------------------------

#[test]
fn incomplete_gamma_halves_sum_to_one() {
    // P(a,x) + Q(a,x) = 1
    for &a in &[0.5f64, 1.0, 2.5, 7.0, 30.0] {
        for i in 0..300 {
            let x = f64::from(i) * 0.2;
            assert_close(
                Real::gamma_lr(x, a) + Real::gamma_ur(x, a),
                1.0,
                1e-14,
                &format!("P+Q=1 at x={x}, a={a}"),
            );
        }
    }
}

#[test]
fn gamma_lr_has_the_exponential_closed_form_at_shape_one() {
    // P(1, x) = 1 − e^{−x}
    for i in 0..200 {
        let x = f64::from(i) * 0.1;
        assert_close(
            Real::gamma_lr(x, 1.0),
            -Real::exp_m1(&-x),
            1e-14,
            &format!("P(1,{x}) = 1 - e^-x"),
        );
    }
}

#[test]
fn gamma_lr_is_monotone_in_x_and_hits_its_endpoints() {
    for &a in &[0.5f64, 1.0, 3.0, 20.0] {
        assert_eq!(Real::gamma_lr(0.0, a), 0.0, "P(a,0) = 0");
        let mut prev = -1.0f64;
        for i in 0..400 {
            let v = Real::gamma_lr(f64::from(i) * 0.25, a);
            assert!(v >= prev, "P must be non-decreasing in x (a={a})");
            prev = v;
        }
        // Far out, P → 1.
        assert!(Real::gamma_lr(1e3, a) > 1.0 - 1e-12);
    }
}
