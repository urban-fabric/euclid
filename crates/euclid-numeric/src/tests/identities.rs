//! **Does it satisfy the relations it must?**
//!
//! Identity tests are the strongest tool available here, because they need no external
//! reference: a recurrence, complement, reflection, or round-trip validates itself. They
//! cannot be transcribed wrongly, they sweep whole domains cheaply, and they catch the class
//! of bug that sampled reference values miss — a formula that is right at the points someone
//! happened to tabulate and wrong between them.
//!
//! Each test states the mathematical relation it checks, so a failure says which law broke.

use crate::Real;

const EULER: f64 = 0.577_215_664_901_532_9;

/// Mixed absolute/relative comparison: `|got − want| ≤ tol · max(1, |want|)`.
///
/// A purely relative criterion is the wrong tool twice over here. Several of these quantities
/// pass through zero (`ln Γ` at 1 and 2), where relative error is undefined; and several
/// *references* are built by complementation (`1 − σ(x)`, `1 − I_{1−x}(b,a)`), which cancels
/// to noise once the result is tiny — the reference degrades, not the function under test.
/// Scaling the tolerance by `max(1, |want|)` keeps the check meaningful in both regimes.
#[track_caller]
fn assert_close(got: f64, want: f64, tol: f64, law: &str) {
    let scale = want.abs().max(1.0);
    let err = (got - want).abs();
    assert!(
        err <= tol * scale,
        "{law}: got {got:.17e}, want {want:.17e}, abs err {err:.3e} > {:.3e}",
        tol * scale
    );
}

// --- error function -----------------------------------------------------

#[test]
fn erf_plus_erfc_is_one() {
    for i in -60..60 {
        let x = f64::from(i) * 0.1;
        assert_close(
            Real::erf(x) + Real::erfc(x),
            1.0,
            1e-15,
            &format!("erf({x}) + erfc({x}) = 1"),
        );
    }
}

#[test]
fn erf_is_odd_and_erfc_reflects() {
    for i in 1..60 {
        let x = f64::from(i) * 0.1;
        // erf(−x) = −erf(x)
        assert_close(Real::erf(-x), -Real::erf(x), 1e-15, "erf is odd");
        // erfc(−x) = 2 − erfc(x)
        assert_close(
            Real::erfc(-x),
            2.0 - Real::erfc(x),
            1e-14,
            "erfc(-x) = 2 - erfc(x)",
        );
    }
}

#[test]
fn erfcx_is_the_scaled_erfc() {
    // erfcx(x) = e^{x²} erfc(x), checked where both sides are representable.
    for i in 0..27 {
        let x = f64::from(i) * 0.1;
        assert_close(
            Real::erfcx(x) * (-x * x).exp(),
            Real::erfc(x),
            1e-14,
            &format!("erfcx({x}) e^-x² = erfc({x})"),
        );
    }
}

#[test]
fn erf_inv_round_trips_through_erf() {
    for i in 1..50 {
        let x = f64::from(i) * 0.05;
        assert_close(
            Real::erf_inv(Real::erf(x)),
            x,
            1e-12,
            &format!("erf_inv(erf({x})) = {x}"),
        );
    }
    // And the other direction, on the probability scale.
    for i in 1..20 {
        let p = f64::from(i) * 0.05;
        assert_close(Real::erf(Real::erf_inv(p)), p, 1e-12, "erf(erf_inv(p)) = p");
    }
}

#[test]
fn erf_is_monotone_increasing() {
    let mut prev = f64::NEG_INFINITY;
    for i in -400..400 {
        let v = Real::erf(f64::from(i) * 0.05);
        assert!(v >= prev, "erf must be non-decreasing");
        prev = v;
    }
}

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

// --- logistic family ----------------------------------------------------

#[test]
fn logit_and_sigmoid_are_inverses() {
    for i in 1..100 {
        let p = f64::from(i) * 0.01;
        assert_close(
            Real::sigmoid(&Real::logit(&p)),
            p,
            1e-12,
            &format!("sigmoid(logit({p})) = {p}"),
        );
    }
    // Beyond |x| ≈ 15, σ(x) is within one f64 ulp of 1 and `logit` cannot invert it. The
    // round trip is checked where it carries information.
    for i in -20..20 {
        let x = f64::from(i) * 0.5;
        assert_close(
            Real::logit(&Real::sigmoid(&x)),
            x,
            1e-10,
            &format!("logit(sigmoid({x})) = {x}"),
        );
    }
}

#[test]
fn sigmoid_reflects_about_one_half() {
    // σ(−x) = 1 − σ(x)
    for i in -100..100 {
        let x = f64::from(i) * 0.25;
        assert_close(
            Real::sigmoid(&-x),
            1.0 - Real::sigmoid(&x),
            1e-13,
            "sigmoid(-x) = 1 - sigmoid(x)",
        );
    }
}

#[test]
fn log_sigmoid_forms_agree_with_their_logs() {
    for i in -60..60 {
        let x = f64::from(i) * 0.5;
        assert_close(
            Real::ln_sigmoid(&x),
            Real::sigmoid(&x).ln(),
            1e-12,
            "ln_sigmoid = ln(sigmoid)",
        );
        // Only where `1 - sigmoid(x)` has not cancelled away; `ln_1m_sigmoid` exists exactly
        // because that expression is unusable for larger x. The stable relation
        // `ln(1-sigma(x)) = -softplus(x)` is checked for all x below.
        if x < 15.0 {
            assert_close(
                Real::ln_1m_sigmoid(&x),
                (1.0 - Real::sigmoid(&x)).ln(),
                1e-9,
                "ln_1m_sigmoid = ln(1 - sigmoid)",
            );
        }
        assert_close(
            Real::ln_1m_sigmoid(&x),
            -Real::ln_1p_exp(&x),
            1e-13,
            "ln(1 - sigmoid(x)) = -softplus(x)",
        );
    }
    // softplus relates them: ln σ(x) = −softplus(−x)
    for i in -100..100 {
        let x = f64::from(i) * 0.5;
        assert_close(
            Real::ln_sigmoid(&x),
            -Real::ln_1p_exp(&-x),
            1e-13,
            "ln sigmoid(x) = -softplus(-x)",
        );
    }
}

// --- exp / log pairs ----------------------------------------------------

#[test]
fn ln_1p_and_exp_m1_are_inverses() {
    // `ln_1p` is defined for x > -1 only.
    for i in -9..30 {
        let x = f64::from(i) * 0.1;
        assert_close(
            Real::exp_m1(&Real::ln_1p(&x)),
            x,
            1e-13,
            &format!("exp_m1(ln_1p({x})) = {x}"),
        );
    }
}

#[test]
fn ln_1m_exp_and_ln_1m_relate_through_exp() {
    // ln(1 − e^x) with x = ln p is ln(1 − p)
    for i in 1..99 {
        let p = f64::from(i) * 0.01;
        assert_close(
            Real::ln_1m_exp(&p.ln()),
            Real::ln_1m(&p),
            1e-11,
            &format!("ln_1m_exp(ln {p}) = ln_1m({p})"),
        );
    }
}

#[test]
fn ln_exp_m1_inverts_ln_1p_exp() {
    // softplus and its inverse
    for i in 1..100 {
        let x = f64::from(i) * 0.25;
        let soft = Real::ln_1p_exp(&x);
        assert_close(
            Real::ln_exp_m1(&soft),
            x,
            1e-10,
            &format!("ln_exp_m1(softplus({x})) = {x}"),
        );
    }
}

// --- trigonometry -------------------------------------------------------

#[test]
fn pythagorean_identity_holds() {
    for i in -200..200 {
        let x = f64::from(i) * 0.1;
        let (s, c) = Real::sin_cos(x);
        assert_close(s * s + c * c, 1.0, 1e-14, "sin² + cos² = 1");
        // sin_cos must agree with the separate functions.
        assert_close(s, Real::sin(x), 1e-15, "sin_cos matches sin");
        assert_close(c, Real::cos(x), 1e-15, "sin_cos matches cos");
    }
}

#[test]
fn inverse_hyperbolics_invert() {
    for i in -50..50 {
        let x = f64::from(i) * 0.2;
        assert_close(Real::asinh(Real::sinh(x)), x, 1e-13, "asinh(sinh(x)) = x");
    }
    // `tanh` saturates toward ±1, so the round trip is information-destroying outside a
    // modest range: by |x| = 10, tanh(x) is within 4e-9 of 1 and `atanh` cannot recover the
    // argument. Checked where the round trip is actually invertible.
    for i in -15..15 {
        let x = f64::from(i) * 0.2;
        assert_close(Real::atanh(Real::tanh(x)), x, 1e-12, "atanh(tanh(x)) = x");
    }
    for i in 1..50 {
        let x = 1.0 + f64::from(i) * 0.2;
        assert_close(Real::acosh(Real::cosh(x)), x, 1e-12, "acosh(cosh(x)) = x");
    }
}

#[test]
fn degree_radian_conversions_round_trip() {
    for i in -36..36 {
        let deg = f64::from(i) * 10.0;
        assert_close(
            Real::to_degrees(Real::to_radians(deg)),
            deg,
            1e-14,
            "degrees -> radians -> degrees",
        );
    }
}

// --- cross-precision agreement ------------------------------------------

#[test]
fn f32_and_f64_agree_to_single_precision() {
    // The same formula on both carriers must land within f32's own resolution. This is the
    // check that catches a divergent f32 code path rather than a mere precision difference.
    for i in 1..60 {
        let x64 = f64::from(i) * 0.1;
        let x32 = x64 as f32;
        let pairs: [(f64, f32); 8] = [
            (Real::erf(x64), Real::erf(x32)),
            (Real::erfc(x64), Real::erfc(x32)),
            (Real::ln_gamma(x64), Real::ln_gamma(x32)),
            (Real::digamma(x64), Real::digamma(x32)),
            (Real::trigamma(x64), Real::trigamma(x32)),
            (Real::sigmoid(&x64), Real::sigmoid(&x32)),
            (Real::ln_1p_exp(&x64), Real::ln_1p_exp(&x32)),
            (Real::bessel_i0e(x64), Real::bessel_i0e(x32)),
        ];
        for (idx, (a, b)) in pairs.iter().enumerate() {
            // Mixed criterion: `ln_gamma(1) = 0`, where a relative comparison between two
            // different roundings of zero is meaningless. f32 carries ~1.2e-7 resolution, so
            // the absolute floor has to accommodate it.
            let err = (f64::from(*b) - *a).abs();
            let bound = 1e-5 * a.abs().max(1.0);
            assert!(
                err <= bound,
                "f32/f64 diverged at x={x64}, fn index {idx}: {a} vs {b} \
                 (abs err {err:.3e} > {bound:.3e})"
            );
        }
    }
}
