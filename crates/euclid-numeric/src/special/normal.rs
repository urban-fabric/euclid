//! The standard normal distribution: CDF, log-CDF, and quantile.
//!
//! These are the functions every probabilistic consumer reaches for, and the ones most easily
//! got wrong. All three are built on the error function rather than on a separate
//! approximation, so they inherit its accuracy and cannot drift away from it.
//!
//! The log-CDF is the delicate one. Written the obvious way, `ln(ndtr(x))` dies around
//! `x = -9` because `ndtr` has underflowed to zero, taking the log to `-inf` while the true
//! value is a perfectly representable `-41`. Routing through the scaled complementary error
//! function keeps it finite as far as the exponent range allows.

use super::erf::{erf_inv, erfc, erfcx};

/// `ln(2)`, for the `1/2` factor in terms of logs.
const LN_2: f64 = core::f64::consts::LN_2;
/// `1/sqrt(2)`.
const FRAC_1_SQRT_2: f64 = core::f64::consts::FRAC_1_SQRT_2;
/// `0.5 * ln(2 pi)`, the normal density's log normalizer.
const HALF_LN_TAU: f64 = 0.918_938_533_204_672_7;

/// The standard normal CDF `Phi(x) = P(Z <= x)`.
///
/// `Phi(x) = erfc(-x/sqrt 2) / 2`. Using `erfc` rather than `1 - erf` keeps the left tail
/// accurate: it is the tail where the answer is small, so the complement form would cancel.
pub(crate) fn ndtr(x: f64) -> f64 {
    0.5 * erfc(-x * FRAC_1_SQRT_2)
}

/// `ln Phi(x)`, finite as far into the left tail as the exponent range allows.
///
/// Two regimes. For `x` above about `-1`, `Phi(x)` is comfortably representable and the direct
/// `ln(erfc(z)/2)` is accurate. Below that, `erfc(z)` underflows long before `ln Phi` stops
/// being representable, so the identity `erfc(z) = erfcx(z) e^{-z^2}` is used to move the
/// vanishing factor into the log:
///
/// ```text
/// ln Phi(x) = ln erfcx(z) - z^2 - ln 2,   z = -x / sqrt 2
/// ```
///
/// At `x = -100` this returns about `-5005.5`; the naive form returns `-inf` from `x = -38`.
pub(crate) fn log_ndtr(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    let z = -x * FRAC_1_SQRT_2;
    if x > -1.0 {
        // erfc(z) is order 1 here, so take its log directly.
        (0.5 * erfc(z)).ln()
    } else {
        // z >= 1/sqrt(2) > 0, so erfcx is well-conditioned and strictly positive.
        erfcx(z).ln() - z * z - LN_2
    }
}

/// Starting point for the quantile deep in the lower tail, from the asymptotic
/// `Phi(x) ~ phi(x)/|x|`.
///
/// Inverting `ln p ~ -x^2/2 - ln|x| - ln sqrt(2 pi)` by one round of substitution gives
/// `|x| ~ t - (ln t + ln sqrt(2 pi))/t` with `t = sqrt(-2 ln p)`, good to a few digits — ample
/// for Newton to finish from.
fn tail_start(p: f64) -> f64 {
    let t = (-2.0 * p.ln()).sqrt();
    -(t - (t.ln() + HALF_LN_TAU) / t)
}

/// The standard normal quantile (probit) `Phi^{-1}(p)`.
///
/// Solved rather than approximated: an initial guess is refined by Newton's method on
/// `ln Phi(x) = ln p`, whose derivative `phi(x)/Phi(x)` is available in closed form. Working in
/// the log domain is what makes the tail work — in probability space both the residual and the
/// derivative underflow, while their logs stay finite.
///
/// The guess comes from `erf_inv` in the central region and from [`tail_start`] below about
/// `p = 1e-15`, where `2p - 1` has rounded to exactly `-1` and `erf_inv` can only return
/// `-inf`.
///
/// # Accuracy
/// Full double precision for `p` in roughly `[1e-300, 0.5]`, and the same by symmetry above
/// `0.5` *as far as `1 - p` is representable*. For an extreme upper quantile, ask for
/// `-ndtri(q)` with the small `q` directly rather than `ndtri(1 - q)`; the subtraction is what
/// loses the digits, not the solver.
pub(crate) fn ndtri(p: f64) -> f64 {
    if p.is_nan() || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::NEG_INFINITY;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    if p == 0.5 {
        return 0.0;
    }
    if p > 0.5 {
        // Phi is symmetric: Phi^{-1}(p) = -Phi^{-1}(1 - p).
        return -ndtri_lower(1.0 - p);
    }
    ndtri_lower(p)
}

/// [`ndtri`] restricted to `p` in `(0, 0.5)`, so the result is negative.
fn ndtri_lower(p: f64) -> f64 {
    let ln_p = p.ln();
    // An `erf_inv` start is better where `2p - 1` still carries information.
    let mut x = if p >= 1e-15 {
        erf_inv(2.0 * p - 1.0) / FRAC_1_SQRT_2
    } else {
        tail_start(p)
    };
    if !x.is_finite() {
        x = tail_start(p);
    }

    // Newton on g(x) = ln Phi(x) - ln p. g'(x) = phi(x)/Phi(x), so the step is
    // (ln Phi - ln p) * exp(ln Phi - ln phi). Converges quadratically; four steps is ample
    // from a guess good to a few digits, and the iteration is monotone here.
    for _ in 0..6 {
        let l_phi = log_ndtr(x);
        let ln_pdf = -0.5 * x * x - HALF_LN_TAU;
        let step = (l_phi - ln_p) * (l_phi - ln_pdf).exp();
        if !step.is_finite() {
            break;
        }
        let next = x - step;
        if next == x {
            break;
        }
        x = next;
    }
    x
}

/// The inverse complementary error function: the `x` with `erfc(x) = y`.
///
/// Derived from the quantile rather than from `erf_inv(1 - y)`, which cancels for small `y`.
/// Since `erfc(x) = 2 Phi(-x sqrt 2)`, the inverse is `-ndtri(y/2) / sqrt 2`, which stays
/// accurate as far into the tail as [`ndtri`] does.
pub(crate) fn erfc_inv(y: f64) -> f64 {
    if y.is_nan() || !(0.0..=2.0).contains(&y) {
        return f64::NAN;
    }
    if y == 0.0 {
        return f64::INFINITY;
    }
    if y == 2.0 {
        return f64::NEG_INFINITY;
    }
    -ndtri(0.5 * y) * FRAC_1_SQRT_2
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) {
        assert!(
            (a - b).abs() <= tol * b.abs().max(1.0),
            "{a:e} vs {b:e} (tol {tol:e})"
        );
    }

    #[test]
    fn ndtr_hits_its_known_values() {
        close(ndtr(0.0), 0.5, 1e-15);
        close(ndtr(1.0), 0.841_344_746_068_543, 1e-14);
        close(ndtr(-1.0), 0.158_655_253_931_457, 1e-14);
        close(ndtr(1.959_963_984_540_054), 0.975, 1e-12);
        close(ndtr(-3.0), 1.349_898_031_630_095e-3, 1e-12);
    }

    #[test]
    fn log_ndtr_stays_finite_where_the_naive_form_dies() {
        assert_eq!(ndtr(-40.0), 0.0, "Phi underflows here");
        assert!(ndtr(-40.0).ln().is_infinite(), "so its log is -inf");
        let v = log_ndtr(-40.0);
        assert!(v.is_finite() && v < 0.0, "log_ndtr(-40) = {v}");
        // ln Phi(x) ~ -x^2/2 - ln|x| - ln sqrt(2 pi)
        close(v, -800.0 - 40.0f64.ln() - HALF_LN_TAU, 1e-3);
        for &x in &[-100.0f64, -1e3, -1e6] {
            let got = log_ndtr(x);
            assert!(got.is_finite(), "log_ndtr({x}) = {got}");
            close(got, -0.5 * x * x - (-x).ln() - HALF_LN_TAU, 1e-6);
        }
    }

    #[test]
    fn ndtri_round_trips_through_ndtr() {
        for i in 1..100 {
            let p = f64::from(i) * 0.01;
            close(ndtr(ndtri(p)), p, 1e-12);
        }
    }

    #[test]
    fn ndtri_round_trips_deep_in_the_tail_through_the_log_cdf() {
        for &p in &[1e-20f64, 1e-50, 1e-100, 1e-200, 1e-300] {
            let x = ndtri(p);
            assert!(x.is_finite() && x < 0.0, "ndtri({p}) = {x}");
            close(log_ndtr(x), p.ln(), 1e-12);
        }
    }

    #[test]
    fn erfc_inv_inverts_erfc_including_small_arguments() {
        for i in 1..40 {
            let y = f64::from(i) * 0.05;
            close(erfc(erfc_inv(y)), y, 1e-12);
        }
        // Small y is the regime `erf_inv(1 - y)` cannot reach.
        for &y in &[1e-10f64, 1e-50, 1e-200] {
            let x = erfc_inv(y);
            assert!(x.is_finite() && x > 0.0, "erfc_inv({y}) = {x}");
            // Check through the scaled form, since erfc(x) itself has underflowed.
            close(erfcx(x).ln() - x * x, y.ln(), 1e-9);
        }
    }
}
