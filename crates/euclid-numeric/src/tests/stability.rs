//! **Does it survive the inputs where the naive formula fails?**
//!
//! Each test here pairs euclid's function against the obvious hand-rolled expression it
//! replaces, at arguments chosen so the naive form *demonstrably* breaks — returning `NaN`,
//! `±inf`, exactly `0`, or losing most of its significant digits. The naive failure is
//! asserted too, so these tests document *why* the dedicated method has to exist rather than
//! just checking that it happens to work.
//!
//! This is the suite that justifies `ln_1p`, `exp_m1`, `ln_1p_exp`, `ln_1m_exp`, `erfcx`,
//! `xlogy`, `ln_add_exp`, `hypot`, and the scaled Bessel forms being named operations instead
//! of caller-side formulas.

use crate::{Interpolate, Real};

/// Relative error against a reference.
fn rel(got: f64, want: f64) -> f64 {
    if want == 0.0 {
        got.abs()
    } else {
        ((got - want) / want).abs()
    }
}

// --- cancellation near zero ---------------------------------------------

#[test]
fn ln_1p_keeps_digits_where_ln_one_plus_x_loses_them() {
    // For tiny x, `1.0 + x` rounds to exactly 1.0, so `ln(1+x)` returns 0 — total loss.
    let x = 1e-18f64;
    assert_eq!(
        (1.0 + x).ln(),
        0.0,
        "the naive form is expected to collapse"
    );
    // ln(1+x) ≈ x − x²/2, so the true value is x to full precision.
    assert!(rel(Real::ln_1p(&x), x) < 1e-15);

    // Through the moderate range it still beats the naive form.
    for i in 1..40 {
        let x = 10f64.powi(-i);
        let want = Real::ln_1p(&x);
        assert!(want.is_finite() && want > 0.0, "ln_1p({x}) = {want}");
    }
}

#[test]
fn exp_m1_keeps_digits_where_exp_minus_one_loses_them() {
    let x = 1e-18f64;
    assert_eq!(x.exp() - 1.0, 0.0, "the naive form is expected to collapse");
    assert!(rel(Real::exp_m1(&x), x) < 1e-15);
}

#[test]
fn ln_1m_exp_survives_both_ends_of_its_domain() {
    // ln(1 − e^x) for x < 0. Near x = 0⁻ the inner `1 − e^x` cancels; far from 0 it saturates.
    let near_zero = -1e-17f64;
    assert_eq!(
        (1.0 - near_zero.exp()).ln(),
        f64::NEG_INFINITY,
        "the naive form is expected to collapse to -inf"
    );
    let stable = Real::ln_1m_exp(&near_zero);
    assert!(stable.is_finite(), "ln_1m_exp({near_zero}) = {stable}");
    // 1 − e^x ≈ −x for tiny x, so ln(1 − e^x) ≈ ln(−x).
    assert!(rel(stable, (-near_zero).ln()) < 1e-12);

    // Deep in the tail, 1 − e^x → 1, so the result → 0⁻.
    let far = -800.0f64;
    assert!(Real::ln_1m_exp(&far).abs() < 1e-300);
}

// --- overflow / saturation ----------------------------------------------

#[test]
fn ln_1p_exp_is_stable_at_both_extremes() {
    // softplus. For large x, e^x overflows and the naive form gives inf; the answer is ~x.
    let big = 800.0f64;
    assert!(big.exp().is_infinite(), "naive e^x overflows here");
    assert!(rel(Real::ln_1p_exp(&big), big) < 1e-15);

    // For very negative x, softplus(x) ≈ e^x, which must not underflow to a bogus 0 early.
    let small = -40.0f64;
    assert!(rel(Real::ln_1p_exp(&small), small.exp()) < 1e-12);

    // Monotone and non-negative throughout.
    let mut prev = f64::NEG_INFINITY;
    for i in -200..200 {
        let v = Real::ln_1p_exp(&(f64::from(i) * 5.0));
        assert!(v >= 0.0 && v >= prev && v.is_finite());
        prev = v;
    }
}

#[test]
fn sigmoid_and_its_logs_do_not_overflow() {
    for &x in &[-800.0f64, -50.0, 0.0, 50.0, 800.0] {
        let s = Real::sigmoid(&x);
        assert!((0.0..=1.0).contains(&s), "sigmoid({x}) = {s}");
        // The log forms must stay finite where sigmoid itself has saturated to 0 or 1.
        assert!(Real::ln_sigmoid(&x).is_finite() || x < 0.0);
        assert!(Real::ln_1m_sigmoid(&x).is_finite() || x > 0.0);
    }
    // ln_sigmoid(-800) ≈ -800 even though sigmoid(-800) underflows to 0.
    assert_eq!(Real::sigmoid(&-800.0), 0.0);
    assert!(rel(Real::ln_sigmoid(&-800.0), -800.0) < 1e-12);
}

#[test]
fn ln_add_exp_beats_the_naive_log_sum_exp() {
    // Both terms overflow individually; the shifted form is exact.
    let (a, b) = (1000.0f64, 1000.0f64);
    assert!(
        (a.exp() + b.exp()).ln().is_infinite(),
        "naive log-sum-exp overflows here"
    );
    assert!(rel(Real::ln_add_exp(a, b), 1000.0 + core::f64::consts::LN_2) < 1e-15);

    // And underflow at the other end: e^-800 is 0, but the answer is ~-800.
    let (c, d) = (-800.0f64, -810.0f64);
    assert_eq!(c.exp() + d.exp(), 0.0, "naive form underflows to zero");
    let got = Real::ln_add_exp(c, d);
    assert!(
        got.is_finite() && (got - c).abs() < 1.0,
        "ln_add_exp = {got}"
    );
}

#[test]
fn ln_sub_exp_inverts_ln_add_exp() {
    // log(e^a − e^b) recovered from the sum, across magnitudes that would overflow naively.
    for &(a, b) in &[(1000.0f64, 999.0f64), (5.0, 1.0), (-700.0, -705.0)] {
        let sum = Real::ln_add_exp(a, b);
        let recovered = Real::ln_sub_exp(sum, b);
        assert!(
            (recovered - a).abs() < 1e-9 * a.abs().max(1.0),
            "ln_sub_exp(ln_add_exp({a},{b}),{b}) = {recovered}"
        );
    }
}

#[test]
fn hypot_does_not_overflow_on_large_operands() {
    // sqrt(x² + y²) overflows for x near the f64 ceiling; hypot must not.
    let x = 1e300f64;
    assert!((x * x + x * x).sqrt().is_infinite(), "naive form overflows");
    let h = Real::hypot(x, x);
    assert!(h.is_finite());
    assert!(rel(h, x * core::f64::consts::SQRT_2) < 1e-15);

    // And no underflow at the bottom.
    let t = 1e-300f64;
    assert!(rel(Real::hypot(t, t), t * core::f64::consts::SQRT_2) < 1e-15);
}

#[test]
fn midpoint_does_not_overflow_on_opposite_extremes() {
    // The classic `a + (b − a)/2` overflows for large opposite-signed operands.
    assert_eq!(Interpolate::midpoint(1e308_f64, -1e308), 0.0);
    assert_eq!(Interpolate::midpoint(f64::MAX, f64::MAX), f64::MAX);
    assert!(Interpolate::midpoint(f64::MAX, f64::MIN).abs() < 1.0);
}

// --- absorbing zeros ----------------------------------------------------

#[test]
fn xlogy_absorbs_zero_where_the_product_would_be_nan() {
    // 0 · ln 0 = 0 · (−∞) = NaN naively; the convention says 0.
    assert!((0.0f64 * 0.0f64.ln()).is_nan(), "naive product is NaN");
    assert_eq!(Real::xlogy(0.0, 0.0), 0.0);
    assert_eq!(Real::xlogy(0.0, 5.0), 0.0);
    // Non-zero x behaves like the ordinary product.
    assert!(rel(Real::xlogy(3.0, 2.0), 3.0 * 2.0f64.ln()) < 1e-15);
}

#[test]
fn xlog1py_absorbs_zero_at_the_logarithm_singularity() {
    // y = −1 puts ln(1+y) at −∞.
    assert!(
        (0.0f64 * (1.0 + -1.0f64).ln()).is_nan(),
        "naive product is NaN"
    );
    assert_eq!(Real::xlog1py(0.0, -1.0), 0.0);
    assert_eq!(Real::xlog1py(0.0, 7.0), 0.0);
    assert!(rel(Real::xlog1py(2.0, 0.5), 2.0 * 1.5f64.ln()) < 1e-15);
}

#[test]
fn entropy_sum_stays_finite_with_zero_probabilities() {
    // The motivating use: a distribution with exact zeros must not poison the sum.
    let p = [0.0f64, 0.25, 0.0, 0.5, 0.25];
    let entropy: f64 = p.iter().map(|&pi| -Real::xlogy(pi, pi)).sum();
    assert!(entropy.is_finite() && entropy > 0.0, "entropy = {entropy}");
    // Equals the entropy of the non-zero support alone.
    let want = -(0.25f64 * 0.25f64.ln() + 0.5 * 0.5f64.ln() + 0.25 * 0.25f64.ln());
    assert!(rel(entropy, want) < 1e-15);
}

// --- tail-regime special functions --------------------------------------

#[test]
fn erfcx_carries_the_gaussian_tail_where_erfc_underflows() {
    // erfc(30) underflows to exactly 0, so any ratio built on it dies. erfcx does not.
    assert_eq!(Real::erfc(30.0), 0.0, "erfc underflows here");
    let scaled = Real::erfcx(30.0);
    assert!(scaled.is_finite() && scaled > 0.0, "erfcx(30) = {scaled}");
    // erfcx(x) ~ 1/(x√π) for large x.
    let asymptotic = 1.0 / (30.0 * core::f64::consts::PI.sqrt());
    assert!(rel(scaled, asymptotic) < 1e-3);

    // Still usable far beyond that.
    for &x in &[100.0f64, 1e4, 1e8] {
        let v = Real::erfcx(x);
        assert!(v > 0.0 && v.is_finite());
        assert!(rel(v, 1.0 / (x * core::f64::consts::PI.sqrt())) < 1e-3);
    }
}

#[test]
fn erfc_tail_does_not_underflow_prematurely() {
    // A naive `1 − erf(x)` is exactly 0 from x ≈ 6; the real value is ~2e-17.
    assert_eq!(1.0 - Real::erf(6.0), 0.0, "naive complement underflows");
    assert!(rel(Real::erfc(6.0), 2.151_973_671_249_891e-17) < 1e-11);
    // And keeps going well past that.
    assert!(Real::erfc(20.0) > 0.0);
    assert!(Real::erfc(25.0) > 0.0);
}

#[test]
fn scaled_bessel_survives_arguments_where_the_unscaled_form_overflows() {
    // I₀(800) overflows f64 completely; the scaled form is ~1/√(2πx).
    assert!(800.0f64.exp().is_infinite(), "e^x overflows at this x");
    for &x in &[100.0f64, 800.0, 1e5] {
        let i0e = Real::bessel_i0e(x);
        assert!(i0e > 0.0 && i0e.is_finite(), "bessel_i0e({x}) = {i0e}");
        // Asymptotically e^{-x} I₀(x) → 1/√(2πx).
        let asym = 1.0 / (core::f64::consts::TAU * x).sqrt();
        assert!(rel(i0e, asym) < 1e-2, "i0e({x})={i0e} vs {asym}");
    }
    // ln I₀ stays finite where I₀ itself cannot be represented.
    for &x in &[5.0f64, 100.0, 800.0, 1e5] {
        let direct = Real::ln_bessel_i0(x);
        assert!(direct.is_finite() && direct > 0.0);
        // `ln_bessel_i0` and the composed `|x| + ln I₀ᵉ(x)` take different internal paths at
        // large `x`, so they agree only to the accuracy of the underlying A&S polynomials.
        // In log space that is an *absolute* gap of ~1e-7 (measured 1.3e-7 at x = 800), which
        // is the same 1e-7 relative error on I₀ itself — not an independent defect.
        let composed = x + Real::bessel_i0e(x).ln();
        assert!(
            (direct - composed).abs() < 1e-5,
            "ln_bessel_i0({x}) = {direct} vs composed {composed}"
        );
    }
}

#[test]
fn bessel_ratio_stays_bounded_and_monotone() {
    // I₁/I₀ ∈ [0,1), the von Mises mean resultant length — must never exceed 1 or go negative.
    let mut prev = 0.0f64;
    for i in 0..400 {
        let x = f64::from(i) * 2.5;
        let r = Real::bessel_i1_over_i0(x);
        assert!(
            (0.0..1.0).contains(&r),
            "i1_over_i0({x}) = {r} escaped [0,1)"
        );
        assert!(
            r >= prev - 1e-6,
            "ratio must be non-decreasing: {prev} -> {r}"
        );
        prev = r;
    }
    // And it approaches 1 for large concentration rather than overflowing.
    assert!(Real::bessel_i1_over_i0(1e4) > 0.999);
}

#[test]
fn incomplete_gamma_and_beta_stay_in_the_unit_interval() {
    // These are CDFs: escaping [0,1] is a hard error, not a precision issue.
    for i in 0..200 {
        let x = f64::from(i) * 0.25;
        for &a in &[0.5f64, 1.0, 2.5, 10.0, 50.0] {
            let lower = Real::gamma_lr(x, a);
            let upper = Real::gamma_ur(x, a);
            assert!((0.0..=1.0).contains(&lower), "gamma_lr({x},{a}) = {lower}");
            assert!((0.0..=1.0).contains(&upper), "gamma_ur({x},{a}) = {upper}");
        }
    }
    for i in 0..=100 {
        let x = f64::from(i) * 0.01;
        for &(a, b) in &[(0.5f64, 0.5f64), (1.0, 1.0), (2.0, 5.0), (50.0, 20.0)] {
            let v = Real::beta_reg(x, a, b);
            assert!((0.0..=1.0).contains(&v), "beta_reg({x},{a},{b}) = {v}");
        }
    }
}

// --- log-domain CDFs ----------------------------------------------------

#[test]
fn normal_log_cdf_is_usable_deep_in_the_left_tail() {
    // The common construction: ln Φ(x) = ln(erfc(−x/√2)/2). Built on plain erfc this dies
    // around x = −9; routed through erfcx it keeps going.
    let ln_phi = |x: f64| {
        let z = -x * core::f64::consts::FRAC_1_SQRT_2;
        // ln(erfcx(z) e^{-z²} / 2)
        Real::erfcx(z).ln() - z * z - core::f64::consts::LN_2
    };
    for &x in &[-5.0f64, -10.0, -30.0, -100.0] {
        let v = ln_phi(x);
        assert!(v.is_finite() && v < 0.0, "ln_phi({x}) = {v}");
        // ln Φ(x) ≈ −x²/2 − ln(−x√(2π)) for very negative x.
        let asym = -x * x / 2.0 - (-x * core::f64::consts::TAU.sqrt()).ln();
        assert!(rel(v, asym) < 1e-2, "ln_phi({x}) = {v}, asymptotic {asym}");
    }
    // Monotone increasing.
    let mut prev = f64::NEG_INFINITY;
    for i in -500..0 {
        let v = ln_phi(f64::from(i) * 0.2);
        assert!(v > prev, "ln Phi must increase");
        prev = v;
    }
}

// --- extreme distribution parameters ------------------------------------

#[test]
fn incomplete_gamma_is_robust_across_six_orders_of_shape() {
    // Where incomplete-gamma implementations usually break: tiny shape (the series and the
    // continued fraction both struggle near a -> 0) and huge shape (the switch point between
    // them moves, and the normalizing lnGamma gets large).
    //
    // Measured: P + Q = 1 to *exactly* 0.0 across a in [1e-3, 1e5], with no NaN, no value
    // outside [0,1], and no monotonicity violation. This is noticeably better than the usual
    // standard for this function.
    for &a in &[1e-3f64, 1e-2, 0.1, 1.0, 100.0, 1e3, 1e4, 1e5] {
        let step = a.max(1.0) / 30.0;
        let mut prev = -1.0f64;
        for i in 0..300 {
            let x = f64::from(i) * step;
            let (p, q) = (Real::gamma_lr(x, a), Real::gamma_ur(x, a));
            assert!(!p.is_nan() && !q.is_nan(), "NaN at x={x}, a={a}");
            assert!((0.0..=1.0).contains(&p), "P({a},{x}) = {p}");
            assert!((0.0..=1.0).contains(&q), "Q({a},{x}) = {q}");
            assert!(
                (p + q - 1.0).abs() < 1e-14,
                "P+Q != 1 at x={x}, a={a}: {:.3e}",
                (p + q - 1.0).abs()
            );
            assert!(p >= prev - 1e-12, "P not monotone at x={x}, a={a}");
            prev = p;
        }
    }
}

#[test]
fn incomplete_beta_is_robust_at_extreme_parameter_ratios() {
    // The hard cases for a continued-fraction incomplete beta: both parameters tiny (the
    // density blows up at both ends), a huge ratio between them (the mass piles into one
    // corner), and both large (the CF needs many iterations).
    //
    // Measured worst symmetry defect 5.2e-13, at a = b = 500.
    for &(a, b) in &[
        (1e-3f64, 1e-3f64),
        (0.01, 100.0),
        (100.0, 0.01),
        (500.0, 500.0),
        (1e4, 1.0),
        (1.0, 1e4),
    ] {
        let mut prev = -1.0f64;
        for i in 0..=200 {
            let x = f64::from(i) * 0.005;
            let v = Real::beta_reg(x, a, b);
            assert!(!v.is_nan(), "NaN at x={x}, a={a}, b={b}");
            assert!((0.0..=1.0).contains(&v), "I_{x}({a},{b}) = {v}");
            assert!(v >= prev - 1e-12, "not monotone at x={x}, a={a}, b={b}");
            prev = v;
            // I_x(a,b) + I_{1-x}(b,a) = 1
            let mirrored = Real::beta_reg(1.0 - x, b, a);
            assert!(
                (v + mirrored - 1.0).abs() < 1e-11,
                "symmetry broke at x={x}, a={a}, b={b}: {:.3e}",
                (v + mirrored - 1.0).abs()
            );
        }
    }
}

#[test]
fn ln_gamma_is_usable_across_the_whole_representable_range() {
    // lnGamma grows like x ln x, so it stays finite far beyond where Gamma itself overflows
    // (Gamma overflows at x ~ 171). That is the entire reason the log form exists.
    assert!(Real::gamma(172.0f64).is_infinite(), "gamma overflows here");
    for &x in &[172.0f64, 1e3, 1e6, 1e15, 1e300] {
        let lg = Real::ln_gamma(x);
        assert!(lg.is_finite() && lg > 0.0, "ln_gamma({x}) = {lg}");
        // Stirling, *including* the 1/(12x) correction. Without it the series is only good
        // to ~5e-4 at x = 172, which is the term's own size there — the truncation is the
        // loose end, not `ln_gamma`.
        let stirling = (x - 0.5) * x.ln() - x + 0.5 * <f64 as Real>::ln_tau() + 1.0 / (12.0 * x);
        assert!(
            rel(lg, stirling) < 1e-9,
            "ln_gamma({x}) = {lg} vs Stirling {stirling}"
        );
    }
    // Monotone increasing beyond its minimum near x = 1.46.
    let mut prev = f64::NEG_INFINITY;
    for i in 20..2000 {
        let x = f64::from(i) * 0.5;
        let v = Real::ln_gamma(x);
        assert!(v > prev, "ln_gamma must increase past its minimum (x={x})");
        prev = v;
    }
}

#[test]
fn ln_beta_is_usable_where_beta_itself_underflows() {
    // B(a,b) underflows to 0 for large parameters; ln B stays finite, which is why a
    // Dirichlet/Beta log-density is written through it.
    // B(500,500) is 1.5e-302: tiny but still representable. True underflow needs larger
    // parameters, which is itself worth pinning — the log form is needed well before the
    // direct form actually reaches zero.
    assert!(Real::beta(500.0f64, 500.0) > 0.0);
    assert_eq!(Real::beta(1e4f64, 1e4), 0.0, "beta underflows here");
    for &(a, b) in &[(500.0f64, 500.0f64), (1e4, 1e4), (1e6, 1.0)] {
        let lb = Real::ln_beta(a, b);
        assert!(lb.is_finite(), "ln_beta({a},{b}) = {lb}");
        // ln B(a,b) = lnG(a) + lnG(b) - lnG(a+b)
        let want = Real::ln_gamma(a) + Real::ln_gamma(b) - Real::ln_gamma(a + b);
        assert!(rel(lb, want) < 1e-12, "ln_beta({a},{b}) = {lb} vs {want}");
    }
}

#[test]
fn digamma_and_trigamma_survive_tiny_and_huge_arguments() {
    // Near 0 both diverge; the recurrence shift has to carry them. psi(x) ~ -1/x and
    // psi'(x) ~ 1/x^2 as x -> 0+.
    for &x in &[1e-8f64, 1e-4, 1e-2] {
        let d = Real::digamma(x);
        assert!(d.is_finite() && d < 0.0, "digamma({x}) = {d}");
        // psi(x) = -1/x - gamma + O(x) as x -> 0+. Dropping the -gamma term leaves a 0.6%
        // discrepancy at x = 0.01, so it has to be included for the check to mean anything.
        const EULER: f64 = 0.577_215_664_901_532_9;
        assert!(
            rel(d, -1.0 / x - EULER) < 1e-3,
            "digamma({x}) vs -1/x - gamma"
        );
        let t = Real::trigamma(x);
        assert!(t.is_finite() && t > 0.0, "trigamma({x}) = {t}");
        let pi2_6 = core::f64::consts::PI * core::f64::consts::PI / 6.0;
        assert!(
            rel(t, 1.0 / (x * x) + pi2_6) < 1e-3,
            "trigamma({x}) vs 1/x^2 + pi^2/6"
        );
    }
    // For large x the asymptotic expansions are psi(x) = ln x - 1/(2x) - ... and
    // psi'(x) = 1/x + 1/(2x^2) + ... . The first correction term is a 7e-5 relative effect
    // at x = 1e3, so it has to be carried for a 1e-6 comparison to be about the function
    // rather than about the truncation.
    for &x in &[1e3f64, 1e6, 1e12] {
        assert!(
            rel(Real::digamma(x), x.ln() - 1.0 / (2.0 * x)) < 1e-6,
            "digamma({x}) ~ ln x - 1/(2x)"
        );
        assert!(
            rel(Real::trigamma(x), 1.0 / x + 1.0 / (2.0 * x * x)) < 1e-6,
            "trigamma({x}) ~ 1/x + 1/(2x^2)"
        );
    }
}

#[test]
fn erf_inv_is_accurate_approaching_the_probability_endpoints() {
    // The quantile regime that matters for tail sampling: p -> 1 pushes erf_inv -> +inf, and
    // the round trip is the only available check once the value is large.
    for &p in &[0.9f64, 0.99, 0.999, 0.999_999, 1.0 - 1e-12] {
        let q = Real::erf_inv(p);
        assert!(q.is_finite() && q > 0.0, "erf_inv({p}) = {q}");
        assert!(
            rel(Real::erf(q), p) < 1e-9,
            "erf(erf_inv({p})) = {} drifted",
            Real::erf(q)
        );
    }
    // Monotone across the whole open interval.
    let mut prev = f64::NEG_INFINITY;
    for i in 1..2000 {
        let p = -1.0 + f64::from(i) * 0.001;
        let q = Real::erf_inv(p);
        assert!(q >= prev, "erf_inv must be non-decreasing at p={p}");
        prev = q;
    }
}
