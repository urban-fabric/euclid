//! Two transcendental functions that belong to no one family: the Lambert `W` and the Riemann
//! zeta.
//!
//! The other classical functions in this tree group naturally — the Bessel family in
//! [`bessel`](super::bessel), the non-elementary antiderivatives in
//! [`integrals`](super::integrals) — and these two do not. `W` is the inverse of `x ↦ x e^x`,
//! the function needed whenever an unknown appears both inside and outside an exponential,
//! which is a shape that recurs across analysis rather than within one field. `ζ` is the
//! central object of analytic number theory and the value of several otherwise-unnameable
//! constants.

use crate::constants::f64::NEG_INV_E;
use crate::kernels::gamma::ln_gamma;
use crate::kernels::polygamma::hurwitz_zeta;

/// Iteration cap for the `W` refinement. Halley's method triples the correct digits each
/// step, so convergence is reached in a handful from any of the starting guesses below.
const W_STEPS: u32 = 60;

/// The principal branch of the Lambert `W` function: the `w ≥ −1` with `w e^w = x`.
///
/// Defined for `x ≥ −1/e`; `NaN` below that, where the real branches end.
///
/// # Method
///
/// Halley's method on `f(w) = w e^w − x`, which converges cubically, from a starting guess
/// chosen per regime:
///
/// - **Near the branch point** (`x` close to `−1/e`) the derivative of `w e^w` vanishes, so
///   Newton-type iteration stalls there. The series in `p = √(2(1 + ex))` is used instead,
///   which is well-behaved precisely where the iteration is not.
/// - **Large `x`**: `W(x) ≈ ln x − ln ln x`, from `w + ln w = ln x`.
/// - **Otherwise**: the Taylor series `x − x² + 3x³/2` about the origin.
pub(crate) fn lambert_w(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x < NEG_INV_E {
        return f64::NAN;
    }
    if x == 0.0 {
        // Returning `x` rather than the literal `0.0`, so a negative zero survives: `W` is
        // `x - x^2 + ...` near the origin, hence odd there.
        return x;
    }
    if x.is_infinite() {
        return f64::INFINITY;
    }

    let mut w = if x < NEG_INV_E + 1e-3 {
        // Branch-point series: p = sqrt(2(1 + e x)).
        let p = (2.0 * (1.0 + core::f64::consts::E * x)).max(0.0).sqrt();
        -1.0 + p - p * p / 3.0 + 11.0 / 72.0 * p * p * p
    } else if x > 3.0 {
        let l = x.ln();
        l - l.ln()
    } else {
        x * (1.0 - x + 1.5 * x * x)
    };

    for _ in 0..W_STEPS {
        let ew = w.exp();
        let f = w * ew - x;
        if f == 0.0 {
            return w;
        }
        // Halley: w -= f / (e^w (1+w) - f (w+2)/(2w+2))
        let denom_newton = ew * (1.0 + w);
        let correction = f * (w + 2.0) / (2.0 * w + 2.0);
        let denom = denom_newton - correction;
        let step = if denom == 0.0 || !denom.is_finite() {
            // Fall back to Newton where Halley's denominator degenerates.
            f / denom_newton
        } else {
            f / denom
        };
        if !step.is_finite() {
            break;
        }
        let next = w - step;
        if next == w {
            break;
        }
        w = next;
    }
    w
}

/// Terms used by [`zeta_critical_strip`]. The Cohen–Rodriguez Villegas–Zagier acceleration
/// converges like `(3 + √8)^{−n} ≈ 5.83^{−n}`, so 24 terms give `≈ 2e-18`.
const CVZ_TERMS: u32 = 24;

/// `ζ(s)` for `0 < s < 1`, through the alternating (Dirichlet eta) series.
///
/// ```text
/// η(s) = Σ_{k≥1} (−1)^{k−1} k^{−s} = (1 − 2^{1−s})·ζ(s)
/// ```
///
/// `η` converges — conditionally — for every `s > 0`, which is what gets inside the critical
/// strip where both of `zeta`'s other routes fail. Conditional convergence is far too slow to
/// use directly (the terms only decay like `k^{−s}`, so `s = 1/2` would need `1e32` of them),
/// so the sum is accelerated by the Cohen–Rodriguez Villegas–Zagier algorithm, which
/// reweights the first `n` terms by coefficients derived from Chebyshev polynomials and turns
/// the conditional convergence into geometric convergence at rate `(3 + √8)^{−n}`.
///
/// # The divisor near the pole
///
/// `1 − 2^{1−s}` vanishes at `s = 1`, which the caller has already handled as the pole — but
/// writing it that way is a cancellation bug just below it. For `s = 1 − δ` the true divisor
/// is `−δ ln 2`, and the literal form computes it as `1 − (1 + δ ln 2)`: by `δ = 1e-15` that
/// has lost 4% of its value, and at `δ = 1e-16` the sum rounds to exactly `1`, so the divisor
/// is `0` and `ζ` comes back `+∞` with the wrong sign where the true value is `−9.0e15`.
///
/// Phrased with `exp_m1` the subtraction never happens — `1 − e^{u} = −expm1(u)` — and the
/// divisor is exact right up to the pole.
fn zeta_critical_strip(s: f64) -> f64 {
    debug_assert!(s > 0.0 && s < 1.0);
    let n = f64::from(CVZ_TERMS);

    // d = ((3 + sqrt(8))^n + (3 - sqrt(8))^n) / 2, formed as (t + 1/t)/2 since the two roots
    // are reciprocal.
    let t = (3.0 + 8.0f64.sqrt()).powf(n);
    let d = (t + 1.0 / t) * 0.5;

    let mut b = -1.0f64;
    let mut c = -d;
    let mut total = 0.0f64;
    for k in 0..CVZ_TERMS {
        let kf = f64::from(k);
        c = b - c;
        // The kth term of eta is (k+1)^{-s}.
        total += c * (kf + 1.0).powf(-s);
        b = (kf + n) * (kf - n) * b / ((kf + 0.5) * (kf + 1.0));
    }
    let eta = total / d;
    // 1 - 2^{1-s} = -expm1((1-s) ln 2); see the note above.
    eta / -(((1.0 - s) * core::f64::consts::LN_2).exp_m1())
}

/// The Riemann zeta function `ζ(s)` for real `s`.
///
/// # Method
///
/// Three regimes, because no single series covers the line:
///
/// For `s > 1` this is the Hurwitz zeta at `a = 1`, so it reuses that Euler–Maclaurin
/// summation directly rather than repeating it.
///
/// For `0 < s < 1` — the critical strip — neither of the other two routes applies, and
/// [`zeta_critical_strip`] takes over. It is worth being explicit about why the functional
/// equation does *not* suffice here, since it is the obvious thing to reach for: the equation
/// relates `s` to `1 − s`, and for `s` in `(0, 1)` the reflected point `1 − s` is still in
/// `(0, 1)`. The strip is mapped onto itself, so no amount of reflection escapes it.
///
/// For `s < 0` the functional equation does work, since `1 − s > 1`:
///
/// ```text
/// ζ(s) = 2^s π^{s−1} sin(πs/2) Γ(1−s) ζ(1−s)
/// ```
///
/// which maps the whole left half-plane onto `s > 1` where the series converges. The trivial
/// zeros at the negative even integers come out of the `sin(πs/2)` factor, and are returned
/// exactly rather than left to cancellation.
///
/// `s = 1` is the pole; `+∞` is returned, matching the limit from above.
pub(crate) fn zeta(s: f64) -> f64 {
    if s.is_nan() {
        return f64::NAN;
    }
    if s == 1.0 {
        return f64::INFINITY; // the simple pole
    }
    if s.is_infinite() {
        return if s > 0.0 { 1.0 } else { f64::NAN };
    }
    if s > 1.0 {
        return hurwitz_zeta(s, 1.0);
    }
    // zeta(0) = -1/2, and the trivial zeros at s = -2, -4, -6, ...
    if s == 0.0 {
        return -0.5;
    }
    if s > 0.0 {
        // The critical strip, which neither of the other two routes reaches: the
        // Euler-Maclaurin sum needs s > 1, and the functional equation is no help because it
        // maps (0, 1) onto itself -- 1 - s is still in the strip. The alternating series is
        // the way in.
        return zeta_critical_strip(s);
    }
    if s < 0.0 && s.fract() == 0.0 && (s / 2.0).fract() == 0.0 {
        return 0.0; // negative even integers
    }
    // Functional equation. Everything is formed in the log domain where it could overflow:
    // Gamma(1-s) grows without bound as s -> -inf, and zeta does too, so the product must be
    // allowed to reach infinity rather than producing NaN on the way.
    let one_minus_s = 1.0 - s;
    let sin_term = (core::f64::consts::FRAC_PI_2 * s).sin();
    if sin_term == 0.0 {
        return 0.0;
    }
    let ln_magnitude = s * core::f64::consts::LN_2
        + (s - 1.0) * core::f64::consts::PI.ln()
        + sin_term.abs().ln()
        + ln_gamma(one_minus_s);
    let tail = hurwitz_zeta(one_minus_s, 1.0);
    let magnitude = ln_magnitude.exp() * tail.abs();
    let sign = if (sin_term < 0.0) != (tail < 0.0) {
        -1.0
    } else {
        1.0
    };
    sign * magnitude
}

#[cfg(test)]
mod tests;
