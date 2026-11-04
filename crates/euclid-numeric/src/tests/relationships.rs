//! **Do the functions agree with each other?** A consistency web that ties the special
//! functions together through the formulas that relate them.
//!
//! The other suites check each function against something external: a reference table, a
//! stability expectation, its own domain edges. This one checks functions against *each other*.
//! That buys something the others cannot: a relation like `ln_gamma' = digamma` or
//! `P(½, x) = erf(√x)` is satisfied by the true functions and essentially never by a wrong
//! one, so each formula constrains every function it mentions. With enough overlapping
//! relations, an error anywhere breaks at least one of them — and because the implementations
//! involved are *independent* (a continued fraction on one side, a polynomial on the other),
//! agreement is strong evidence rather than a tautology.
//!
//! The web has four kinds of edge:
//!
//! - **Derivative edges** — numerical differentiation ties a function to another function that
//!   is supposed to be its derivative (`softplus' = sigmoid`, `ln_gamma' = digamma`).
//! - **Integral edges** — quadrature ties a density to its CDF (`∫ gamma pdf = gamma_lr`),
//!   which also pins the normalizing constant, hence `ln_gamma`/`ln_beta`.
//! - **Closed-form edges** — a CDF at special parameters collapses to an unrelated function
//!   (`gamma_lr(x, ½) = erf(√x)`, `beta_reg(x, ½, ½) = (2/π) asin √x`).
//! - **Discrete↔continuous edges** — a discrete sum equals a continuous CDF (Poisson tail =
//!   `gamma_ur`, binomial CDF = `beta_reg`), which ties the combinatorics on
//!   [`Natural`](crate::Natural) to the incomplete functions.

use crate::{Natural, Real};

fn rel(got: f64, want: f64) -> f64 {
    if want == 0.0 {
        got.abs()
    } else {
        ((got - want) / want).abs()
    }
}

/// Mixed absolute/relative comparison, needed because several of these relations pass through
/// zero (`ln_gamma` at 1 and 2, `digamma` near 1.46).
#[track_caller]
fn assert_rel(got: f64, want: f64, tol: f64, law: &str) {
    let err = (got - want).abs();
    let bound = tol * want.abs().max(1.0);
    assert!(
        err <= bound,
        "{law}: got {got:.12e}, want {want:.12e}, abs err {err:.3e} > {bound:.3e}"
    );
}

// --- numerical machinery -------------------------------------------------

/// Five-point central derivative: `(8Δ₁ − Δ₂) / 12h`, truncation `O(h⁴)`.
///
/// The step scales with `x` so the stencil stays in a sensible relative range. At `h = 1e-4`
/// the truncation term is ~1e-16 and the roundoff term ~`eps/h` ≈ 2e-12, which is why the
/// derivative edges below are checked at ~1e-8 rather than at machine precision.
fn derivative(f: impl Fn(f64) -> f64, x: f64) -> f64 {
    let h = 1e-4 * x.abs().max(1.0);
    let d1 = f(x + h) - f(x - h);
    let d2 = f(x + 2.0 * h) - f(x - 2.0 * h);
    (8.0 * d1 - d2) / (12.0 * h)
}

/// Composite Simpson's rule over `[a, b]` with `panels` (even) subintervals, error `O(h⁴)`.
fn integrate(f: impl Fn(f64) -> f64, a: f64, b: f64, panels: usize) -> f64 {
    let n = panels.max(2) & !1; // force even
    let h = (b - a) / n as f64;
    let mut total = f(a) + f(b);
    for i in 1..n {
        let x = a + h * i as f64;
        total += f(x) * if i % 2 == 1 { 4.0 } else { 2.0 };
    }
    total * h / 3.0
}

// --- densities, written only through the trait ---------------------------

/// `N(0,1)` density, normalized through [`Real::half_ln_tau`].
fn normal_pdf(x: f64) -> f64 {
    (-0.5 * x * x - <f64 as Real>::half_ln_tau()).exp()
}

/// `Gamma(a, 1)` density, normalized through [`Real::ln_gamma`].
fn gamma_pdf(x: f64, a: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    ((a - 1.0) * x.ln() - x - Real::ln_gamma(a)).exp()
}

/// `Beta(a, b)` density, normalized through [`Real::ln_beta`].
fn beta_pdf(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 || x >= 1.0 {
        return 0.0;
    }
    ((a - 1.0) * x.ln() + (b - 1.0) * (1.0 - x).ln() - Real::ln_beta(a, b)).exp()
}

// --- derivative edges ----------------------------------------------------

#[test]
fn softplus_differentiates_to_sigmoid() {
    // d/dx ln(1 + e^x) = sigma(x). Ties `ln_1p_exp` to `sigmoid`, two separately-stabilized
    // implementations that must nevertheless be exact derivatives of each other.
    for i in -60..60 {
        let x = f64::from(i) * 0.25;
        let d = derivative(|t| Real::ln_1p_exp(&t), x);
        assert_rel(
            d,
            Real::sigmoid(&x),
            1e-8,
            &format!("softplus'({x}) = sigmoid"),
        );
    }
}

#[test]
fn sigmoid_differentiates_to_its_own_product_form() {
    // sigma' = sigma(1 - sigma). Ties `sigmoid` to `ln_1m_sigmoid` via the product.
    for i in -40..40 {
        let x = f64::from(i) * 0.25;
        let s = Real::sigmoid(&x);
        let d = derivative(|t| Real::sigmoid(&t), x);
        assert_rel(d, s * (1.0 - s), 1e-8, &format!("sigmoid'({x})"));
        // And the log form agrees: d/dx ln sigma(x) = 1 - sigma(x).
        let dlog = derivative(|t| Real::ln_sigmoid(&t), x);
        assert_rel(dlog, 1.0 - s, 1e-8, &format!("ln_sigmoid'({x})"));
    }
}

#[test]
fn ln_gamma_differentiates_to_digamma_which_differentiates_to_trigamma() {
    // The polygamma ladder. Three functions, two edges — an error in any one of them shows up.
    for i in 2..60 {
        let x = f64::from(i) * 0.25;
        let d = derivative(Real::ln_gamma, x);
        assert_rel(
            d,
            Real::digamma(x),
            1e-7,
            &format!("ln_gamma'({x}) = digamma"),
        );
        let dd = derivative(Real::digamma, x);
        assert_rel(
            dd,
            Real::trigamma(x),
            1e-6,
            &format!("digamma'({x}) = trigamma"),
        );
    }
}

#[test]
fn erf_differentiates_to_the_gaussian_kernel() {
    // d/dx erf(x) = 2/sqrt(pi) e^{-x^2}. Ties `erf` to `exp` and `pi`.
    for i in -30..30 {
        let x = f64::from(i) * 0.1;
        let d = derivative(Real::erf, x);
        let want = 2.0 / <f64 as Real>::pi().sqrt() * (-x * x).exp();
        assert_rel(d, want, 1e-8, &format!("erf'({x})"));
        // erfc is its complement, so its derivative is the negation.
        let dc = derivative(Real::erfc, x);
        assert_rel(dc, -want, 1e-7, &format!("erfc'({x})"));
    }
}

#[test]
fn incomplete_gamma_differentiates_to_the_gamma_density() {
    // d/dx P(a, x) = x^{a-1} e^{-x} / Gamma(a). Ties `gamma_lr` to `ln_gamma` through the
    // normalizer: if either were wrong by a constant factor, this would fail.
    for &a in &[0.75f64, 1.0, 2.5, 6.0] {
        for i in 1..40 {
            let x = f64::from(i) * 0.25;
            let d = derivative(|t| Real::gamma_lr(t, a), x);
            assert_rel(d, gamma_pdf(x, a), 1e-7, &format!("P'({a}, {x}) = pdf"));
        }
    }
}

#[test]
fn incomplete_beta_differentiates_to_the_beta_density() {
    // d/dx I_x(a,b) = x^{a-1}(1-x)^{b-1} / B(a,b). Ties `beta_reg` to `ln_beta`, and through
    // it to `ln_gamma`.
    for &(a, b) in &[(2.0f64, 3.0f64), (1.5, 1.5), (5.0, 2.0)] {
        for i in 1..20 {
            let x = f64::from(i) * 0.05;
            let d = derivative(|t| Real::beta_reg(t, a, b), x);
            assert_rel(
                d,
                beta_pdf(x, a, b),
                1e-6,
                &format!("I'({x}; {a},{b}) = pdf"),
            );
        }
    }
}

#[test]
fn ln_bessel_i0_differentiates_to_the_bessel_ratio() {
    // d/dx ln I0(x) = I1(x)/I0(x) — the von Mises mean resultant length. Ties `ln_bessel_i0`
    // to `bessel_i1_over_i0`, which are computed by different routes.
    //
    // The tolerance is loose because both sides inherit the ~1e-7 A&S polynomial error; see
    // the precision note on `Real::bessel_i0e`.
    // Numerical differentiation *amplifies* the ~1e-7 absolute error in `ln I0` by `1/h`, so
    // the default step (1e-4 relative) would turn it into ~1e-3 of derivative noise. A much
    // wider step cuts the amplification; `I1/I0` is smooth and slowly varying, so the extra
    // truncation error is negligible by comparison.
    let wide_derivative = |f: &dyn Fn(f64) -> f64, x: f64| {
        let h = 0.1f64;
        (8.0 * (f(x + h) - f(x - h)) - (f(x + 2.0 * h) - f(x - 2.0 * h))) / (12.0 * h)
    };
    for i in 2..40 {
        let x = f64::from(i) * 0.5;
        let d = wide_derivative(&Real::ln_bessel_i0, x);
        assert_rel(
            d,
            Real::bessel_i1_over_i0(x),
            1e-3,
            &format!("ln_bessel_i0'({x}) = I1/I0"),
        );
    }
}

#[test]
fn logit_differentiates_to_the_reciprocal_variance() {
    // d/dp logit(p) = 1/(p(1-p)).
    for i in 1..20 {
        let p = f64::from(i) * 0.05;
        let d = derivative(|t| Real::logit(&t), p);
        assert_rel(d, 1.0 / (p * (1.0 - p)), 1e-7, &format!("logit'({p})"));
    }
}

// --- integral edges ------------------------------------------------------

#[test]
fn the_normal_density_integrates_to_the_erf_cdf() {
    // integral_{-x}^{x} phi = erf(x/sqrt 2). Ties `erf` to `exp` and to `half_ln_tau` — the
    // density's normalizing constant is part of what is being checked.
    for i in 1..20 {
        let x = f64::from(i) * 0.2;
        let area = integrate(normal_pdf, -x, x, 2000);
        let want = Real::erf(x * <f64 as Real>::frac_1_sqrt_2());
        assert_rel(area, want, 1e-10, &format!("int phi over [-{x},{x}] = erf"));
    }
    // And the total mass is 1, which pins the normalizer on its own.
    let total = integrate(normal_pdf, -40.0, 40.0, 20000);
    assert_rel(total, 1.0, 1e-10, "normal density integrates to 1");
}

#[test]
fn the_gamma_density_integrates_to_the_incomplete_gamma() {
    // integral_0^x gamma pdf = P(a, x). An independent check on both `gamma_lr` and the
    // `ln_gamma` normalizer, by a route that shares no code with either.
    // Integer `a >= 2` only: the integrand goes like x^(a-1) near 0, so for non-integer `a`
    // some higher derivative is unbounded at the endpoint and Simpson's O(h^4) convergence
    // degrades. Small and fractional `a` are covered by the derivative edge above and the
    // `a = 1/2` closed form below, neither of which needs endpoint smoothness — one of the
    // reasons the web keeps several kinds of edge.
    for &a in &[2.0f64, 3.0, 7.0] {
        for i in 1..12 {
            let x = f64::from(i) * 0.75;
            let area = integrate(|t| gamma_pdf(t, a), 0.0, x, 4000);
            assert_rel(
                area,
                Real::gamma_lr(x, a),
                1e-8,
                &format!("int gamma pdf to {x} = P({a}, {x})"),
            );
        }
    }
}

#[test]
fn the_beta_density_integrates_to_the_incomplete_beta() {
    // integral_0^x beta pdf = I_x(a,b). Ties `beta_reg` to `ln_beta` to `ln_gamma`.
    for &(a, b) in &[(2.0f64, 3.0f64), (4.0, 2.0), (2.5, 2.5)] {
        for i in 1..10 {
            let x = f64::from(i) * 0.1;
            let area = integrate(|t| beta_pdf(t, a, b), 0.0, x, 4000);
            assert_rel(
                area,
                Real::beta_reg(x, a, b),
                1e-7,
                &format!("int beta pdf to {x} = I({a},{b})"),
            );
        }
    }
}

#[test]
fn the_gamma_integral_recovers_the_gamma_function() {
    // Gamma(a) = integral_0^inf x^{a-1} e^{-x} dx, the defining integral. Ties `gamma` and
    // `ln_gamma` to `exp`/`powf` with no incomplete-gamma machinery involved.
    // Integer `a >= 2`, for the endpoint-smoothness reason given above. Note the `a = 1` case
    // is specifically *not* usable here: the integrand is 1 at x = 0, not 0, and Simpson
    // weights that endpoint by h/3 — returning 0 from the guard biases the result by ~3e-4.
    for &a in &[2.0f64, 3.0, 5.0] {
        let f = |x: f64| {
            if x <= 0.0 {
                0.0
            } else {
                Real::powf(&x, a - 1.0) * (-x).exp()
            }
        };
        let area = integrate(f, 0.0, 60.0, 60000);
        assert_rel(area, Real::gamma(a), 1e-7, &format!("int = Gamma({a})"));
        assert_rel(
            area.ln(),
            Real::ln_gamma(a),
            1e-7,
            &format!("ln int = ln_gamma({a})"),
        );
    }
}

// --- closed-form edges ---------------------------------------------------

#[test]
fn incomplete_gamma_at_half_shape_equals_the_error_function() {
    // P(1/2, x) = erf(sqrt x). This is the chi-squared-with-1-degree-of-freedom identity, and
    // it is the single strongest edge in the web: the left side runs through the incomplete
    // gamma series / continued fraction, the right through an entirely separate erf
    // approximation. They share no code, so agreement is real evidence.
    for i in 1..200 {
        let x = f64::from(i) * 0.1;
        assert_rel(
            Real::gamma_lr(x, 0.5),
            Real::erf(x.sqrt()),
            1e-12,
            &format!("P(1/2, {x}) = erf(sqrt {x})"),
        );
    }
    // The complement likewise: Q(1/2, x) = erfc(sqrt x).
    for i in 1..100 {
        let x = f64::from(i) * 0.25;
        assert_rel(
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
        assert_rel(
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
            assert_rel(
                Real::beta_reg(x, a, 1.0),
                Real::powf(&x, a),
                1e-12,
                &format!("I_{x}({a},1) = x^{a}"),
            );
            assert_rel(
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
        assert_rel(
            Real::gamma_lr(x / 2.0, 0.5),
            Real::erf((x / 2.0).sqrt()),
            1e-12,
            "chi2(1) via gamma_lr vs erf",
        );
        // k = 2 is Exponential(1/2): CDF = 1 - e^{-x/2}
        assert_rel(
            Real::gamma_lr(x / 2.0, 1.0),
            -Real::exp_m1(&(-x / 2.0)),
            1e-14,
            "chi2(2) is exponential",
        );
    }
}

// --- discrete <-> continuous edges --------------------------------------

#[test]
fn the_poisson_tail_equals_the_upper_incomplete_gamma() {
    // sum_{i<=k} e^{-lambda} lambda^i / i!  =  Q(k+1, lambda)
    //
    // This ties `Natural::ln_factorial` to `gamma_ur`: the left side is built purely from the
    // combinatorics on the integer tier, the right from the continuous incomplete gamma.
    for &lambda in &[0.5f64, 2.0, 7.0, 20.0] {
        let mut cumulative = 0.0f64;
        for k in 0..40u64 {
            // pmf through the trait's log-factorial, not a local factorial.
            let ln_pmf = -lambda + f64::from(u32::try_from(k).unwrap_or(0)) * lambda.ln()
                - Natural::ln_factorial(&k);
            cumulative += ln_pmf.exp();
            let continuous = Real::gamma_ur(lambda, f64::from(u32::try_from(k).unwrap_or(0)) + 1.0);
            assert_rel(
                cumulative,
                continuous,
                1e-9,
                &format!("Poisson({lambda}) CDF at {k} = Q({}, {lambda})", k + 1),
            );
        }
    }
}

#[test]
fn the_binomial_cdf_equals_the_incomplete_beta() {
    // sum_{i<=k} C(n,i) p^i (1-p)^{n-i}  =  I_{1-p}(n-k, k+1)
    //
    // Ties `Natural::binomial` to `beta_reg`. The discrete side uses exact binomial
    // coefficients; the continuous side a continued fraction.
    for &n in &[5u64, 12, 30] {
        for &p in &[0.2f64, 0.5, 0.75] {
            let mut cumulative = 0.0f64;
            for k in 0..n {
                let c = Natural::binomial(&n, &k);
                cumulative += c
                    * Real::powf(&p, f64::from(u32::try_from(k).unwrap_or(0)))
                    * Real::powf(&(1.0 - p), f64::from(u32::try_from(n - k).unwrap_or(0)));
                let kf = f64::from(u32::try_from(k).unwrap_or(0));
                let nf = f64::from(u32::try_from(n).unwrap_or(0));
                let continuous = Real::beta_reg(1.0 - p, nf - kf, kf + 1.0);
                assert_rel(
                    cumulative,
                    continuous,
                    1e-9,
                    &format!("Binomial({n},{p}) CDF at {k} = I_{{1-p}}"),
                );
            }
        }
    }
}

#[test]
fn log_factorial_and_log_gamma_are_the_same_function() {
    // ln(n!) = ln Gamma(n+1). Ties the integer-tier combinatorics to the continuous gamma.
    for n in 0..100u64 {
        let nf = f64::from(u32::try_from(n).unwrap_or(0));
        assert_rel(
            Natural::ln_factorial(&n),
            Real::ln_gamma(nf + 1.0),
            1e-12,
            &format!("ln({n}!) = ln_gamma({})", n + 1),
        );
    }
}

#[test]
fn log_binomial_matches_the_log_gamma_expression() {
    // ln C(n,k) = lnGamma(n+1) - lnGamma(k+1) - lnGamma(n-k+1). Ties `ln_binomial` to
    // `ln_gamma`, and `binomial` to its own log form.
    for n in 1..40u64 {
        for k in 0..=n {
            let (nf, kf) = (
                f64::from(u32::try_from(n).unwrap_or(0)),
                f64::from(u32::try_from(k).unwrap_or(0)),
            );
            let want =
                Real::ln_gamma(nf + 1.0) - Real::ln_gamma(kf + 1.0) - Real::ln_gamma(nf - kf + 1.0);
            assert_rel(
                Natural::ln_binomial(&n, &k),
                want,
                1e-10,
                &format!("ln C({n},{k})"),
            );
            // And the direct form is its exponential.
            assert_rel(
                Natural::binomial(&n, &k),
                Natural::ln_binomial(&n, &k).exp(),
                1e-9,
                &format!("C({n},{k}) = exp(ln C)"),
            );
        }
    }
}

// --- functional equations -----------------------------------------------

#[test]
fn log_gamma_satisfies_the_duplication_formula() {
    // Gamma(2x) = 2^{2x-1} / sqrt(pi) * Gamma(x) Gamma(x + 1/2), i.e.
    // lnG(2x) = (2x-1) ln 2 - (1/2) ln pi + lnG(x) + lnG(x + 1/2).
    //
    // One relation constraining `ln_gamma` at three different arguments at once, plus `ln_2`
    // and `ln_pi`. A systematic error in the Lanczos coefficients would not survive this.
    for i in 1..80 {
        let x = f64::from(i) * 0.25;
        let lhs = Real::ln_gamma(2.0 * x);
        let rhs = (2.0 * x - 1.0) * <f64 as Real>::ln_2() - 0.5 * <f64 as Real>::ln_pi()
            + Real::ln_gamma(x)
            + Real::ln_gamma(x + 0.5);
        assert_rel(lhs, rhs, 1e-12, &format!("duplication at {x}"));
    }
}

#[test]
fn digamma_satisfies_the_reflection_formula() {
    // psi(1-x) - psi(x) = pi cot(pi x). Ties `digamma` to `tan` and `pi`.
    for i in 1..20 {
        let x = f64::from(i) * 0.05; // (0, 1)
        let lhs = Real::digamma(1.0 - x) - Real::digamma(x);
        let rhs = <f64 as Real>::pi() / Real::tan(<f64 as Real>::pi() * x);
        assert_rel(lhs, rhs, 1e-9, &format!("digamma reflection at {x}"));
    }
}

#[test]
fn digamma_satisfies_the_duplication_formula() {
    // psi(2x) = (1/2) psi(x) + (1/2) psi(x + 1/2) + ln 2.
    for i in 1..60 {
        let x = f64::from(i) * 0.25;
        let lhs = Real::digamma(2.0 * x);
        let rhs = 0.5 * Real::digamma(x) + 0.5 * Real::digamma(x + 0.5) + <f64 as Real>::ln_2();
        assert_rel(lhs, rhs, 1e-9, &format!("digamma duplication at {x}"));
    }
}

#[test]
fn trigamma_satisfies_the_reflection_formula() {
    // psi'(x) + psi'(1-x) = pi^2 / sin^2(pi x).
    for i in 1..20 {
        let x = f64::from(i) * 0.05;
        let lhs = Real::trigamma(x) + Real::trigamma(1.0 - x);
        let s = Real::sin(<f64 as Real>::pi() * x);
        let rhs = <f64 as Real>::pi() * <f64 as Real>::pi() / (s * s);
        assert_rel(lhs, rhs, 1e-9, &format!("trigamma reflection at {x}"));
    }
}

// --- the web closes ------------------------------------------------------

#[test]
fn the_normal_cdf_agrees_through_four_independent_routes() {
    // The summary test. Phi(x) computed four ways that share almost no code:
    //   1. via erf
    //   2. via erfc (the stable-tail route)
    //   3. via erfcx (the scaled route, used for extreme tails)
    //   4. via gamma_lr at half shape (the incomplete-gamma route)
    // All four must agree wherever all four are representable. Any single wrong
    // implementation shows up as a disagreement rather than as a plausible number.
    for i in 1..60 {
        let x = f64::from(i) * 0.1;
        let z = x * <f64 as Real>::frac_1_sqrt_2();

        let via_erf = 0.5 * (1.0 + Real::erf(z));
        let via_erfc = 0.5 * Real::erfc(-z);
        let via_erfcx = 0.5 * Real::erfcx(-z) * (-z * z).exp();
        let via_gamma = 0.5 * (1.0 + Real::gamma_lr(x * x / 2.0, 0.5));

        assert_rel(via_erfc, via_erf, 1e-12, &format!("Phi({x}): erfc vs erf"));
        assert_rel(
            via_erfcx,
            via_erf,
            1e-10,
            &format!("Phi({x}): erfcx vs erf"),
        );
        assert_rel(
            via_gamma,
            via_erf,
            1e-11,
            &format!("Phi({x}): gamma vs erf"),
        );
    }
}

#[test]
fn a_log_density_round_trips_through_its_cdf_and_quantile() {
    // End-to-end: build a Gamma log-density from `ln_gamma`, integrate it with quadrature to
    // get a probability, invert the CDF by bisection using `gamma_lr`, and check the quantile
    // reproduces the probability. Exercises ln_gamma, exp, ln, gamma_lr and the comparison
    // surface in one chain.
    let a = 3.0f64;
    for &p in &[0.1f64, 0.25, 0.5, 0.75, 0.9] {
        // Bisection on P(a, x) = p.
        let (mut lo, mut hi) = (0.0f64, 50.0f64);
        for _ in 0..200 {
            let mid = Real::min(Real::max((lo + hi) * 0.5, lo), hi);
            if Real::gamma_lr(mid, a) < p {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let quantile = (lo + hi) * 0.5;
        // The CDF of the quantile is the probability we asked for.
        assert_rel(
            Real::gamma_lr(quantile, a),
            p,
            1e-9,
            &format!("quantile({p})"),
        );
        // And quadrature of the density up to that point agrees.
        let area = integrate(|t| gamma_pdf(t, a), 0.0, quantile, 4000);
        assert_rel(area, p, 1e-8, &format!("int pdf to quantile({p}) = {p}"));
    }
}

#[test]
fn the_relationship_web_is_sensitive_to_a_perturbation() {
    // A consistency web is only worth having if it would actually catch an error, so this
    // checks the web has teeth: perturbing one function by a *single part in 10^6* — far
    // smaller than a real bug, and invisible to any eyeball check — must break the relations
    // that mention it.
    let perturbed_ln_gamma = |x: f64| Real::ln_gamma(x) * (1.0 + 1e-6);

    // The duplication formula notices.
    let x = 2.25f64;
    let lhs = perturbed_ln_gamma(2.0 * x);
    let rhs = (2.0 * x - 1.0) * <f64 as Real>::ln_2() - 0.5 * <f64 as Real>::ln_pi()
        + perturbed_ln_gamma(x)
        + perturbed_ln_gamma(x + 0.5);
    assert!(
        rel(lhs, rhs) > 1e-9,
        "duplication formula failed to notice a 1e-6 perturbation"
    );

    // So does the derivative edge against digamma.
    let d = derivative(perturbed_ln_gamma, 4.0);
    assert!(
        rel(d, Real::digamma(4.0)) > 1e-8,
        "ln_gamma' = digamma failed to notice a 1e-6 perturbation"
    );

    // And so does the gamma-density integral, through the normalizer.
    let perturbed_pdf = |t: f64, a: f64| {
        if t <= 0.0 {
            0.0
        } else {
            ((a - 1.0) * t.ln() - t - perturbed_ln_gamma(a)).exp()
        }
    };
    let area = integrate(|t| perturbed_pdf(t, 3.0), 0.0, 4.0, 4000);
    assert!(
        rel(area, Real::gamma_lr(4.0, 3.0)) > 1e-8,
        "pdf integral failed to notice a 1e-6 perturbation"
    );
}
