//! The beta function family: `ln_beta` / `beta`, and the regularized incomplete
//! beta `beta_reg` `Iₓ(a, b)` (the CDF kernel for the Beta, Student's-t, and F
//! distributions). Continued fraction from Numerical Recipes.

use core::cmp::Ordering;

use super::gamma::{GAMMA_EDGEWORTH_MIN, ln_gamma, ln_rising_factorial};
use crate::Real;
use crate::constants::f64::SQRT_TAU;
use crate::math::continued_fraction::{LENTZ_TINY_F64, converged, lentz_step, nonzero_pivot};

/// `ln B(a, b) = ln Γ(a) + ln Γ(b) − ln Γ(a + b)`.
///
/// # Why it is not written that way
///
/// Written literally, that expression subtracts two nearly-equal large logarithms whenever
/// `a + b ≫ min(a, b)` — and the result is `O(ln)` while the terms are `O(a ln a)`. The damage
/// is total, not gradual. Against the exact `ln B(a, 1) = −ln a`:
///
/// ```text
/// a = 1e2   1.7e-14 relative        a = 1e15   4e-2 relative
/// a = 1e8   1.4e-9  relative        a = 1e16   ln B returned 0, and B returned 1
/// ```
///
/// Past `a = 1/ε` the sum `a + b` rounds back to `a`, so `lnΓ(a + b)` and `lnΓ(a)` become
/// *bit-identical* and the whole answer cancels away. `B(1e300, 1)` came back as `1` where the
/// true value is `1e-300`.
///
/// Rewriting it around the rising factorial removes the subtraction entirely:
///
/// ```text
/// ln B(a, b) = ln Γ(b) − ln (a)_b,     (a)_b = Γ(a+b)/Γ(a)
/// ```
///
/// which is the same identity read differently — and
/// [`ln_rising_factorial`] already computes
/// that ratio without ever forming the two logs apart, by an exact product for small integer
/// `b` and an integrated-digamma asymptotic for large `a`. The larger argument is passed as
/// the base so the increment is the short one.
pub(crate) fn ln_beta(a: f64, b: f64) -> f64 {
    let (base, increment) = if a >= b { (a, b) } else { (b, a) };
    if increment >= 1e8 && base.is_finite() {
        // Combine Stirling's formulas before forming the large terms. Both
        // leading contributions have the same sign, and base + increment can
        // exceed the format even when ln B is finite.
        let ratio = increment / base;
        let log_sum_ratio = ratio.ln_1p();
        return increment * (ratio.ln() - log_sum_ratio) - base * log_sum_ratio
            + (0.5 * (log_sum_ratio - increment.ln()) + crate::constants::f64::HALF_LN_TAU);
    }
    ln_gamma(increment) - ln_rising_factorial(base, increment)
}

/// The beta function `B(a, b)`.
pub(crate) fn beta(a: f64, b: f64) -> f64 {
    ln_beta(a, b).exp()
}

/// The regularized incomplete beta `Iₓ(a, b)`, `x ∈ [0, 1]`, `a, b > 0`.
pub(crate) fn beta_reg(a: f64, b: f64, x: f64) -> f64 {
    if !a.is_finite() || !b.is_finite() || a <= 0.0 || b <= 0.0 {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    if a.max(b) <= 1e-22 && !x.is_nan() {
        // The endpoint masses are b/(a+b) at zero and a/(a+b) at one.
        // Throughout the representable interior, the relative correction is
        // O(max(a,b) * max(|ln x|, |ln(1-x)|)), below 1e-19 here.
        return b / (a + b);
    }
    // Integrating the density gives exact forms that also avoid the continued
    // fraction's large intermediates when one shape is much larger than the other.
    if a == 1.0 {
        return -(b * (-x).ln_1p()).exp_m1();
    }
    if b == 1.0 {
        return x.powf(a);
    }
    if let Some(rate) = beta_gamma_rate(a, b) {
        return super::gamma::gamma_lr(a, -rate * (-x).ln_1p());
    }
    if let Some(rate) = beta_gamma_rate(b, a) {
        // Reflection in log coordinates preserves the small tail.
        return super::gamma::gamma_ur(b, -rate * x.ln());
    }
    if a.min(b) >= BETA_EDGEWORTH_MIN {
        return beta_edgeworth(a, b, x);
    }
    beta_fraction_cdf(a, b, x)
        .unwrap_or_else(|| beta_edgeworth(a, b, x))
        .clamp(0.0, 1.0)
}

/// Common factor `x^a(1-x)^b/B(a,b)` for the direct and reflected fractions.
/// `ln1p(-x)` preserves the logarithm's relative accuracy at small `x`.
fn beta_fraction_prefactor(a: f64, b: f64, x: f64) -> f64 {
    (a * x.ln() + b * (-x).ln_1p() - ln_beta(a, b)).exp()
}

/// Select the faster-converging orientation around the beta mean.
fn use_lower_beta_fraction(a: f64, b: f64, x: f64) -> bool {
    x < (a + 1.0) / (a + b + 2.0)
}

/// Evaluate the fraction in the chosen orientation. An unconverged fraction
/// leaves the decision to use an approximation with the caller.
fn beta_fraction_cdf(a: f64, b: f64, x: f64) -> Option<f64> {
    let front = beta_fraction_prefactor(a, b, x);
    if use_lower_beta_fraction(a, b, x) {
        beta_cf(a, b, x).map(|h| front * h / a)
    } else {
        beta_cf(b, a, 1.0 - x).map(|h| 1.0 - front * h / b)
    }
}

/// Rate for the gamma approximation to `T = -ln(1-X)`, with `X ~ Beta(a,b)`.
///
/// `(1-e^-t)^(a-1) = t^(a-1) e^(-(a-1)t/2) (sinh(t/2)/(t/2))^(a-1)`.
/// Moving the first exponential into the rate leaves a log-density correction
/// of order `(a-1)t²/24`. Near the mass, `t` is about `a/b`, so the switch
/// bounds this correction at the floating-point scale.
fn beta_gamma_rate(a: f64, b: f64) -> Option<f64> {
    // Once gamma itself would use Edgeworth, standardize directly in beta
    // coordinates. Rounding the transformed argument can erase its distance
    // from the gamma mean, even when the distance is many standard deviations.
    if a >= GAMMA_EDGEWORTH_MIN {
        return None;
    }
    let shift = a + 1.0;
    (b / shift / shift.sqrt() >= 5e7).then_some(b + 0.5 * (a - 1.0))
}

/// Iteration cap for the continued fraction.
///
/// The requirement grows with the parameters, because the fraction converges slowest exactly at
/// `x ≈ a/(a+b)` — the median, where the branch switch below gives no help. Measured iterations
/// to converge at `a = b`, `x = ½`:
///
/// | `a` | 1e2 | 1e3 | 1e4 | 1e5 | 1e6 | 1e7 | 1e8 |
/// |-----|-----|-----|-----|-----|-----|-----|-----|
/// | iterations | 25 | 54 | 115 | 243 | 526 | 1091 | 2282 |
///
/// This used to be `300`, which silently truncated from about `a = 1e4` on and left
/// `I_{1/2}(1e8, 1e8)` returning `0.687` where the true value is exactly `½`. `4000` carries the
/// fraction past [`BETA_EDGEWORTH_MIN`], where it hands over anyway, with room to spare.
const MAX_ITER: usize = 4000;
const EPS: f64 = 1e-15;

/// At or above this `min(a, b)` the Edgeworth-corrected normal replaces the continued fraction.
///
/// # Why `min(a, b)` and not `a + b`
///
/// `Beta(a,b)` approaches a normal only as **both** shape parameters grow; its skewness is
/// `≈ 2/√min(a,b)`, so the expansion parameter is the smaller one alone. Gating on the sum
/// instead admits `Beta(1, 1e8)` — an exponential-shaped density with skewness 2, nothing like
/// a normal — and the approximation is then wrong in the first digit. That mistake was made
/// here first and caught by the `tests::identities::beta_range` suite, which checks the exact
/// `I_x(1, b) = 1 − (1−x)^b`.
///
/// Measured worst `|Edgeworth − converged fraction|` over `z ∈ [−4, 4]`, which confirms the
/// error depends on `min` and essentially not on the ratio `b/a`:
///
/// | `min(a,b)` | ratio 1 | ratio 10 | ratio 1e2 | ratio 1e4 |
/// |---|---|---|---|---|
/// | 1e5 | `6.1e-7` | `4.3e-7` | `6.1e-7` | `6.3e-7` |
/// | 1e6 | `6.0e-8` | `4.3e-8` | `6.1e-8` | `6.3e-8` |
///
/// # Where the crossover falls
///
/// Two independent error sources meet, and neither is fixable by iterating harder:
///
/// - **The fraction's prefactor cancels.** `front` is `exp(a ln x + b ln(1−x) − ln B(a,b))`,
///   three terms of magnitude `≈ 1.386n` for `n = a + b` summing to an `O(1)` exponent. The
///   exponent's absolute error is `≈ 1.386nε`, which `exp` hands back as *relative* error —
///   `3.1e-9` at `n = 1e7`, `3.1e-8` at `n = 1e8`. That is the subtraction, not the truncation.
/// - **The Edgeworth form improves as `≈ 0.0605/min(a,b)`**, from the table above.
///
/// Setting those equal at the worst case `a = b` (so `n = 2min`) gives `min ≈ 1e7`, where both
/// carry about `6e-9`. Below it the fraction is better; above it the approximation is.
///
/// # What it buys
///
/// The symmetric case — the one that was catastrophically wrong — becomes **exact** rather than
/// merely better: `Beta(a,a)` has zero skewness, so the correction vanishes identically and
/// `I_{1/2}(a,a)` is `Φ(0) = ½` to the last bit, at every `a`. Measured against that closed
/// form, `a = 1e7` went from `9.5e-3` to `0`, and `a = 1e8` from `0.187` to `0`.
pub(crate) const BETA_EDGEWORTH_MIN: f64 = 1e7;

/// The signed distance from the mean in units of `1/(a+b)`.
fn beta_mean_residual(a: f64, n: f64, n_error: f64, x: f64) -> f64 {
    // Near the mean, product − a is exact by Sterbenz's lemma. Rounding
    // a/n first can move x by many standard deviations at large shapes.
    let (product, product_error) = x.two_product(n);
    let (offset, offset_error) = (product - a).two_sum(product_error);
    x.mul_add(n_error, offset) + offset_error
}

/// `Iₓ(a, b)` from the Edgeworth-corrected normal approximation; see [`BETA_EDGEWORTH_MIN`].
///
/// The peer of [`super::gamma::gamma_lr`]'s `gamma_edgeworth`, and the same expansion: a normal
/// with the distribution's own mean and variance, plus the first skewness correction
/// `−φ(z)·γ₁(z²−1)/6`. For `Beta(a,b)`,
///
/// ```text
/// μ = a/n,   σ² = ab/(n²(n+1)),   γ₁ = 2(b−a)√(n+1)/((n+2)√(ab)),   n = a + b
/// ```
///
/// `Φ` is formed as `½erfc(−z/√2)` so a small lower tail keeps its relative accuracy instead of
/// being lost to a complement. The `erfc` call reaches into [`super::erf`], which is built on
/// `gamma_lr` — but that recursion terminates at once, since `erfc` asks for shape `½`, far
/// below any threshold that returns here.
fn beta_edgeworth(a: f64, b: f64, x: f64) -> f64 {
    let (n, n_error) = a.two_sum(b);
    if !n.is_finite() {
        // Even the smaller shape is at least 2^970 here. Near the mean,
        // the lowest bits of the binary64 products in x(a+b) − a are
        // hundreds of bits above the transition width √(ab/(a+b+1)).
        // Halving keeps the sum finite and preserves the residual's sign.
        let half_a = 0.5 * a;
        let (half_n, half_error) = half_a.two_sum(0.5 * b);
        return match beta_mean_residual(half_a, half_n, half_error, x).partial_cmp(&0.0) {
            Some(Ordering::Less) => 0.0,
            Some(Ordering::Equal) => 0.5,
            Some(Ordering::Greater) => 1.0,
            None => f64::NAN,
        };
    }

    // z = (x·n − a)√(n+1)/√(ab). Taking the roots separately keeps the
    // factors in range even when a·b overflows or σ² underflows.
    let inverse_width = (n + 1.0).sqrt() / a.sqrt() / b.sqrt();
    let z = beta_mean_residual(a, n, n_error, x) * inverse_width;
    // Past `|z| = 40` the standard normal tail has left the format in both directions: `erfc`
    // has saturated to `0` or `2`, and `e^{−z²/2}` underflowed at `|z| ≈ 38.6`. The answer is
    // a certainty, so returning it directly changes no value the expression below could
    // produce — while keeping `z·z` out of an overflow that would make the correction `NaN`.
    if z.abs() > 40.0 {
        return f64::from(u8::from(z > 0.0));
    }
    let skew = 2.0 * ((b - a) / (n + 2.0)) * inverse_width;
    let density = (-0.5 * z * z).exp() / SQRT_TAU;
    let correction = density * skew * (z * z - 1.0) / 6.0;
    let lower = 0.5 * super::erf::erfc(-z * core::f64::consts::FRAC_1_SQRT_2) - correction;
    lower.clamp(0.0, 1.0)
}

/// Lentz continued fraction for the incomplete beta (`betacf` in Numerical Recipes).
///
/// Returns `None` when it exhausts [`MAX_ITER`] without converging, rather than handing back a
/// truncated partial sum. Returning the partial sum is what made the large-parameter defect
/// *silent*: the loop exited by either route and the caller could not tell which.
fn beta_cf(a: f64, b: f64, x: f64) -> Option<f64> {
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = nonzero_pivot(1.0 - qab * x / qap, LENTZ_TINY_F64).recip();
    let mut h = d;
    // `1..=MAX_ITER`, not `1..MAX_ITER`: the exclusive form ran one step short of the cap, so
    // the constant did not mean what its name said.
    for m in 1..=MAX_ITER {
        let m = m as f64;
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        h *= lentz_step(aa, 1.0, &mut c, &mut d, LENTZ_TINY_F64);
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        let del = lentz_step(aa, 1.0, &mut c, &mut d, LENTZ_TINY_F64);
        h *= del;
        if converged(del, 0.0, EPS) {
            return Some(h);
        }
    }
    None
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn beta_known_values() {
        close(beta(1.0, 1.0), 1.0, 1e-12);
        close(beta(2.0, 3.0), 1.0 / 12.0, 1e-10);
    }

    #[test]
    fn beta_fraction_agrees_with_the_integrated_polynomial() {
        // For (2, 3), the CDF is x²(6 − 8x + 3x²) and its prefactor
        // divided by a is 6x²(1 − x)³.
        for x in [0.007_812_5_f64, 0.0625, 0.125, 0.25, 0.4] {
            let expected = (6.0 - 8.0 * x + 3.0 * x * x) / (6.0 * (1.0 - x).powi(3));
            let Some(actual) = beta_cf(2.0, 3.0, x) else {
                panic!("fraction failed to converge at {x}");
            };
            close(actual, expected, 2e-15);
        }
    }

    #[test]
    fn fraction_orientation_switches_at_the_exact_dyadic_boundary() {
        let x = 0.375_f64;
        assert!(use_lower_beta_fraction(2.0, 4.0, x.next_down()));
        assert!(!use_lower_beta_fraction(2.0, 4.0, x));
        assert!(!use_lower_beta_fraction(2.0, 4.0, x.next_up()));
        assert!(use_lower_beta_fraction(4.0, 2.0, x));
    }

    #[test]
    fn fraction_prefactor_matches_integer_shape_polynomial() {
        for x in [2.0_f64.powi(-50), 0.125, 0.25, 0.75, 1.0_f64.next_down()] {
            let expected = 12.0 * x.powi(2) * (1.0 - x).powi(3);
            let actual = beta_fraction_prefactor(2.0, 3.0, x);
            assert!(
                (actual / expected - 1.0).abs() <= 3e-14,
                "x={x}, actual={actual}"
            );
        }
    }

    #[test]
    fn both_fraction_orientations_match_the_integrated_cdf() {
        for x in [0.0625, 0.25, 0.375, 0.5, 0.875] {
            // Integral of 20x(1-x)^3, since B(2,4) = 1/20.
            let expected = x * x * (10.0 + x * (-20.0 + x * (15.0 - 4.0 * x)));
            let Some(actual) = beta_fraction_cdf(2.0, 4.0, x) else {
                panic!("integer-shape fraction failed to converge at x={x}");
            };
            close(actual, expected, 4e-15);
        }
    }

    #[test]
    fn beta_reg_endpoints_and_symmetry() {
        assert_eq!(beta_reg(2.0, 3.0, 0.0), 0.0);
        assert_eq!(beta_reg(2.0, 3.0, 1.0), 1.0);
        for &(a, b, x) in &[(2.0, 3.0, 0.4), (0.5, 0.5, 0.3), (5.0, 2.0, 0.7)] {
            close(beta_reg(a, b, x), 1.0 - beta_reg(b, a, 1.0 - x), 1e-10);
        }
        close(beta_reg(1.0, 1.0, 0.37), 0.37, 1e-12);
    }

    #[test]
    fn gamma_approximation_uses_the_shifted_rate_after_the_crossover() {
        assert_eq!(beta_gamma_rate(2.0, 1e8), None);
        assert_eq!(beta_gamma_rate(2.0, 2e8), None);
        assert_eq!(beta_gamma_rate(2.0, 1e9), Some(1e9 + 0.5));
        assert_eq!(beta_gamma_rate(0.5, 1e9), Some(1e9 - 0.25));
        assert!(beta_gamma_rate(GAMMA_EDGEWORTH_MIN.next_down(), 1e30).is_some());
        assert_eq!(beta_gamma_rate(GAMMA_EDGEWORTH_MIN, 1e30), None);
    }

    #[test]
    fn edgeworth_at_the_mean_uses_the_beta_skewness() {
        let (a, b) = (1e7_f64, 3e7_f64);
        let n = a + b;
        // At the mean, z = 0. The unscaled formula fits at these moderate shapes.
        let skewness = 2.0 * (b - a) * (n + 1.0).sqrt() / ((n + 2.0) * (a * b).sqrt());
        let expected = 0.5 + skewness / (6.0 * (2.0 * core::f64::consts::PI).sqrt());
        close(beta_edgeworth(a, b, 0.25), expected, f64::EPSILON);
    }

    #[test]
    fn beta_mean_residual_retains_the_low_bits_of_the_offset() {
        let a = f64::from_bits(0x3ff0_0000_0000_0006);
        let b = f64::from_bits(0x3fdb_bc37_cd1d_a36d);
        let x = f64::from_bits(0x3fe6_533b_1ba2_862f);
        let (n, n_error) = a.two_sum(b);
        // Rounded once from the exact binary rationals x(a+b) − a.
        let expected = f64::from_bits(0xbcaa_a02f_14b3_5725);
        assert_eq!(beta_mean_residual(a, n, n_error, x), expected);
    }
}
