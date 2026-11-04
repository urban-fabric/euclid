//! The beta function family: `ln_beta` / `beta`, and the regularized incomplete
//! beta `beta_reg` `Iₓ(a, b)` (the CDF kernel for the Beta, Student's-t, and F
//! distributions). Continued fraction from Numerical Recipes.

use super::gamma::{ln_gamma, ln_rising_factorial};
use crate::constants::f64::SQRT_TAU;

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
    ln_gamma(increment) - ln_rising_factorial(base, increment)
}

/// The beta function `B(a, b)`.
pub(crate) fn beta(a: f64, b: f64) -> f64 {
    ln_beta(a, b).exp()
}

/// The regularized incomplete beta `Iₓ(a, b)`, `x ∈ [0, 1]`, `a, b > 0`.
pub(crate) fn beta_reg(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    if a.min(b) >= BETA_EDGEWORTH_MIN {
        return beta_edgeworth(a, b, x);
    }
    // `b·ln(1 − x)` as `b·ln1p(−x)`. The literal form is a cancellation in the *logarithm*:
    // `1 − x` rounds with absolute error `ε/2`, which is `ε/2x` *relative* once divided
    // through by `ln(1−x) ≈ −x`, so the log carries `1.1e-4` of relative error at `x = 1e-12`.
    // Multiplied by `b = 1e8` that is `1.1e-8` absolute in the exponent, and the `exp` hands
    // it back as relative error in the result — measured 2.2e-8 against the exact
    // `I_x(1, b) = 1 − (1−x)^b`.
    let front = (a * x.ln() + b * (-x).ln_1p() - ln_beta(a, b)).exp();
    // The fraction is taken on whichever side of the median converges faster, then complemented
    // if needed. `None` means it did not converge at all, which the Edgeworth form backs up
    // rather than letting a truncated partial sum through — see [`beta_cf`].
    let converged = if x < (a + 1.0) / (a + b + 2.0) {
        beta_cf(a, b, x).map(|h| front * h / a)
    } else {
        beta_cf(b, a, 1.0 - x).map(|h| 1.0 - front * h / b)
    };
    converged.unwrap_or_else(|| beta_edgeworth(a, b, x))
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
    let n = a + b;
    let mean = a / n;
    let sd = (a * b / (n * n * (n + 1.0))).sqrt();
    let z = (x - mean) / sd;
    let skew = 2.0 * (b - a) * (n + 1.0).sqrt() / ((n + 2.0) * (a * b).sqrt());
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
    let tiny = 1e-300;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < tiny {
        d = tiny;
    }
    d = 1.0 / d;
    let mut h = d;
    // `1..=MAX_ITER`, not `1..MAX_ITER`: the exclusive form ran one step short of the cap, so
    // the constant did not mean what its name said.
    for m in 1..=MAX_ITER {
        let m = m as f64;
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + aa / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = 1.0 + aa / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
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
    fn beta_reg_endpoints_and_symmetry() {
        assert_eq!(beta_reg(2.0, 3.0, 0.0), 0.0);
        assert_eq!(beta_reg(2.0, 3.0, 1.0), 1.0);
        for &(a, b, x) in &[(2.0, 3.0, 0.4), (0.5, 0.5, 0.3), (5.0, 2.0, 0.7)] {
            close(beta_reg(a, b, x), 1.0 - beta_reg(b, a, 1.0 - x), 1e-10);
        }
        close(beta_reg(1.0, 1.0, 0.37), 0.37, 1e-12);
    }
}
