//! The gamma and beta quantiles, and value-level accuracy checks at large shape.
//!
//! The large-shape tests exist because of a defect this suite did not originally catch. The
//! earlier robustness test asserted `P(a,x) + Q(a,x) == 1` and monotonicity across shapes from
//! `1e-3` to `1e5`, and passed — while `P(1e5, median)` was returning `0.237` instead of
//! `0.5`. Both invariants were blind to it:
//!
//! - `P + Q == 1` held because `gamma_lr` computes the large-`x` branch *as* `1 - gamma_ur`,
//!   so the two shared the same truncation error and it cancelled out of the sum.
//! - Monotonicity held because the truncation error grew smoothly with `x`, and the jump at
//!   the branch switch happened to be upward.
//!
//! Structural invariants are cheap and catch a lot, but they cannot substitute for checking
//! the value against something independent. These tests do that.

use crate::Real;
use euclid_utils::{assert_close, assert_nondecreasing, assert_probability, rel, sweep};

// --- the defect the invariants missed -----------------------------------

#[test]
fn incomplete_gamma_is_accurate_at_large_shape() {
    // For large `a` the gamma distribution is asymptotically normal with mean `a` and variance
    // `a`, so `P(a, a + z sqrt a) -> Phi(z)`. The approximation's own error is `O(1/sqrt a)`
    // from the skewness, which is why the tolerance loosens as `a` shrinks rather than
    // tightens — the loose end here is the normal approximation, not `gamma_lr`.
    for &(a, tol) in &[(1e3f64, 5e-3f64), (1e4, 2e-3), (1e5, 6e-4), (1e6, 2e-4)] {
        for z in [-2.0f64, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0] {
            let x = a + z * a.sqrt();
            let got = Real::gamma_lr(x, a);
            let want = Real::ndtr(z);
            assert!(
                (got - want).abs() < tol,
                "P({a}, a + {z} sqrt a) = {got}, normal approximation {want}"
            );
        }
    }
}

#[test]
fn the_gamma_median_is_near_the_shape_minus_a_third() {
    // A sharper check that needs no approximation slack: the median of Gamma(a, 1) is
    // `a - 1/3 + O(1/a)`, so `P(a, a - 1/3)` must be very close to 1/2 and converge to it.
    for &a in &[1e2f64, 1e3, 1e4, 1e5] {
        let p = Real::gamma_lr(a - 1.0 / 3.0, a);
        assert!(
            (p - 0.5).abs() < 1e-4,
            "P({a}, a - 1/3) = {p}, expected ~0.5"
        );
    }
    // And it tightens with a, which a truncating series would not do.
    let errs: Vec<f64> = [1e3f64, 1e4, 1e5, 1e6]
        .iter()
        .map(|&a| (Real::gamma_lr(a - 1.0 / 3.0, a) - 0.5).abs())
        .collect();
    for w in errs.windows(2) {
        assert!(
            w[1] <= w[0] * 1.5,
            "the median error should not grow with shape: {:?}",
            errs
        );
    }
}

#[test]
fn incomplete_gamma_is_continuous_across_its_branch_switch() {
    // The implementation switches from a series to a continued fraction at `x = a + 1`. When
    // the series was truncating, the two branches disagreed by 0.027 at `a = 1e4` — a visible
    // step in a function that must be smooth. The density bounds how much the CDF can change
    // over a small interval, which is what makes this detectable without a reference value.
    for &a in &[1.0f64, 10.0, 1e3, 1e4, 1e5] {
        let switch = a + 1.0;
        let h = 1e-6 * switch;
        let below = Real::gamma_lr(switch - h, a);
        let above = Real::gamma_lr(switch + h, a);
        // The density at the switch bounds the admissible change.
        let ln_pdf = (a - 1.0) * switch.ln() - switch - Real::ln_gamma(a);
        let bound = ln_pdf.exp() * 2.0 * h * 10.0 + 1e-14;
        assert!(
            (above - below).abs() <= bound,
            "discontinuity at the a = {a} branch switch: {below} then {above} \
             (change {:.3e} exceeds the density-implied bound {bound:.3e})",
            (above - below).abs()
        );
    }
}

#[test]
fn incomplete_gamma_matches_the_exact_integer_shape_tail_at_large_shape() {
    // Cancellation-free exact reference: P(a, x) = e^-x sum_{k>=a} x^k/k! for integer a.
    fn tail(a: u32, x: f64) -> f64 {
        // The leading term `x^a / a!` overflows long before `e^-x` scales it back, so factor
        // it out in the log domain and accumulate the rest *relative* to it. All the ratios
        // are positive, so there is still no cancellation anywhere.
        let af = f64::from(a);
        let ln_leading = af * x.ln() - Real::ln_gamma(af + 1.0);
        let (mut relative_sum, mut ratio, mut k) = (1.0f64, 1.0f64, af);
        loop {
            k += 1.0;
            ratio *= x / k;
            relative_sum += ratio;
            if ratio < relative_sum * 1e-19 || k > 300_000.0 {
                break;
            }
        }
        (ln_leading - x).exp() * relative_sum
    }
    for a in [500u32, 1_000, 2_000] {
        for mult in [0.9f64, 1.0, 1.1] {
            let x = f64::from(a) * mult;
            assert_close(
                Real::gamma_lr(x, f64::from(a)),
                tail(a, x),
                1e-10,
                &format!("P({a}, {x}) vs the exact tail"),
            );
        }
    }
}

// --- the gamma quantile -------------------------------------------------

#[test]
fn gamma_quantile_round_trips_across_shapes() {
    for &a in &[0.1f64, 0.5, 1.0, 2.5, 10.0, 100.0, 1e4] {
        for p in sweep(0.01, 0.99, 50) {
            let x = Real::gamma_lr_inv(p, a);
            assert!(x.is_finite() && x >= 0.0, "gamma_lr_inv({p},{a}) = {x}");
            assert_close(
                Real::gamma_lr(x, a),
                p,
                1e-9,
                &format!("round trip at p={p}, a={a}"),
            );
        }
    }
}

#[test]
fn gamma_quantile_matches_the_exponential_closed_form() {
    // P(1, x) = 1 - e^-x, so the quantile is -ln(1 - p) exactly.
    for p in sweep(0.01, 0.99, 100) {
        assert_close(
            Real::gamma_lr_inv(p, 1.0),
            -Real::ln_1p(&-p),
            1e-10,
            &format!("gamma quantile at shape 1, p={p}"),
        );
    }
}

#[test]
fn gamma_quantile_matches_the_chi_squared_and_normal_routes() {
    // Shape 1/2 is chi-squared with one degree of freedom, whose quantile comes from the
    // normal: P(1/2, x) = erf(sqrt x), so x = erf_inv(p)^2.
    for p in sweep(0.01, 0.99, 60) {
        let want = Real::erf_inv(p) * Real::erf_inv(p);
        assert_close(
            Real::gamma_lr_inv(p, 0.5),
            want,
            1e-9,
            &format!("shape 1/2 quantile at p={p}"),
        );
    }
}

#[test]
fn gamma_quantile_is_monotone_and_handles_its_endpoints() {
    let values: Vec<f64> = sweep(0.001, 0.999, 300)
        .map(|p| Real::gamma_lr_inv(p, 3.0))
        .collect();
    assert_nondecreasing(values.iter().copied(), 0.0, "gamma quantile is monotone");
    assert_eq!(Real::gamma_lr_inv(0.0f64, 2.0), 0.0);
    assert!(Real::gamma_lr_inv(1.0f64, 2.0).is_infinite());
    assert!(Real::gamma_lr_inv(1.5f64, 2.0).is_nan());
    assert!(Real::gamma_lr_inv(0.5f64, -1.0).is_nan());
    assert!(Real::gamma_lr_inv(f64::NAN, 2.0).is_nan());
}

// --- the beta quantile --------------------------------------------------

#[test]
fn beta_quantile_round_trips_across_parameters() {
    for &(a, b) in &[
        (1.0f64, 1.0f64),
        (2.0, 5.0),
        (5.0, 2.0),
        (0.5, 0.5),
        (100.0, 50.0),
        (1e3, 1e3),
    ] {
        // See the conditioning note in `special::quantile`: a U-shaped beta cannot resolve
        // the probability more finely than one ulp of `x` allows near the endpoints.
        let tol = if a < 1.0 || b < 1.0 { 1e-5 } else { 1e-9 };
        for p in sweep(0.01, 0.99, 40) {
            let x = Real::beta_reg_inv(p, a, b);
            assert!((0.0..=1.0).contains(&x), "beta_reg_inv({p},{a},{b}) = {x}");
            assert_close(
                Real::beta_reg(x, a, b),
                p,
                tol,
                &format!("round trip at p={p}, a={a}, b={b}"),
            );
        }
    }
}

#[test]
fn beta_quantile_matches_its_closed_forms() {
    for p in sweep(0.01, 0.99, 80) {
        // I_x(1,1) = x
        assert_close(
            Real::beta_reg_inv(p, 1.0, 1.0),
            p,
            1e-12,
            "uniform quantile",
        );
        // I_x(a,1) = x^a  =>  quantile p^(1/a)
        assert_close(
            Real::beta_reg_inv(p, 3.0, 1.0),
            Real::powf(&p, 1.0 / 3.0),
            1e-10,
            "I(a,1) quantile",
        );
        // I_x(1,b) = 1 - (1-x)^b  =>  quantile 1 - (1-p)^(1/b)
        assert_close(
            Real::beta_reg_inv(p, 1.0, 4.0),
            1.0 - Real::powf(&(1.0 - p), 0.25),
            1e-10,
            "I(1,b) quantile",
        );
        // Arcsine law: I_x(1/2,1/2) = (2/pi) asin(sqrt x)  =>  x = sin^2(pi p / 2)
        let s = Real::sin(<f64 as Real>::pi() * p / 2.0);
        assert_close(
            Real::beta_reg_inv(p, 0.5, 0.5),
            s * s,
            1e-9,
            "arcsine quantile",
        );
    }
}

#[test]
fn beta_quantile_is_monotone_symmetric_and_handles_its_endpoints() {
    let values: Vec<f64> = sweep(0.001, 0.999, 300)
        .map(|p| Real::beta_reg_inv(p, 2.0, 5.0))
        .collect();
    assert_nondecreasing(values.iter().copied(), 0.0, "beta quantile is monotone");
    assert_probability(values.iter().copied(), "beta quantile stays in [0,1]");
    // Swapping the parameters mirrors the quantile: Q_{a,b}(p) = 1 - Q_{b,a}(1-p).
    for p in sweep(0.05, 0.95, 30) {
        assert_close(
            Real::beta_reg_inv(p, 2.0, 5.0),
            1.0 - Real::beta_reg_inv(1.0 - p, 5.0, 2.0),
            1e-9,
            "beta quantile mirror symmetry",
        );
    }
    assert_eq!(Real::beta_reg_inv(0.0f64, 2.0, 3.0), 0.0);
    assert_eq!(Real::beta_reg_inv(1.0f64, 2.0, 3.0), 1.0);
    assert!(Real::beta_reg_inv(-0.1f64, 2.0, 3.0).is_nan());
    assert!(Real::beta_reg_inv(0.5f64, 0.0, 3.0).is_nan());
}

// --- quantiles tie into the rest of the web -----------------------------

#[test]
fn the_quantiles_invert_their_cdfs_as_functions() {
    // Composition in the other direction: starting from x, the quantile of its CDF returns x.
    for &a in &[0.5f64, 2.0, 10.0] {
        for x in sweep(0.1, 20.0, 40) {
            let p = Real::gamma_lr(x, a);
            if p <= 0.0 || p >= 1.0 {
                continue;
            }
            assert!(
                rel(Real::gamma_lr_inv(p, a), x) < 1e-7,
                "gamma_lr_inv(gamma_lr(x)) != x at x={x}, a={a}"
            );
        }
    }
    for &(a, b) in &[(2.0f64, 3.0f64), (5.0, 1.5)] {
        for x in sweep(0.02, 0.98, 40) {
            let p = Real::beta_reg(x, a, b);
            assert!(
                rel(Real::beta_reg_inv(p, a, b), x) < 1e-7,
                "beta_reg_inv(beta_reg(x)) != x at x={x}, a={a}, b={b}"
            );
        }
    }
}

#[test]
fn the_chi_squared_quantile_agrees_with_the_normal_one() {
    // chi-squared with 1 degree of freedom is the square of a standard normal, so its
    // quantile is ndtri((1+p)/2)^2. Reaching that through `gamma_lr_inv` ties the gamma
    // quantile to the normal one.
    for p in sweep(0.02, 0.98, 50) {
        let via_gamma = 2.0 * Real::gamma_lr_inv(p, 0.5);
        let z = Real::ndtri(0.5 * (1.0 + p));
        assert_close(
            via_gamma,
            z * z,
            1e-8,
            &format!("chi2(1) quantile at p={p}"),
        );
    }
}

#[test]
fn the_quantiles_are_available_on_f32() {
    for p in [0.1f32, 0.5, 0.9] {
        let x = Real::gamma_lr_inv(p, 2.0f32);
        assert!(x.is_finite() && x > 0.0);
        assert!((Real::gamma_lr(x, 2.0f32) - p).abs() < 1e-4);
        let y = Real::beta_reg_inv(p, 2.0f32, 3.0);
        assert!((0.0..=1.0).contains(&y));
        assert!((Real::beta_reg(y, 2.0f32, 3.0) - p).abs() < 1e-4);
    }
}
