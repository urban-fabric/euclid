//! Robustness across extreme distribution parameters.

use crate::Real;
use euclid_utils::{EULER_GAMMA as EULER, rel};

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
