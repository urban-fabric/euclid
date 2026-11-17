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

/// The starting guess for [`lambert_w`]'s Halley iteration, chosen per regime.
///
/// A named function so a test can reach it. The three regimes are described in `lambert_w`'s
/// own documentation; what matters here is that the guess is only ever a *guess*, and the
/// quality that matters is not its accuracy but **how many steps it saves**.
///
/// # Why its accuracy is the wrong thing to assert
///
/// Sixteen mutants lived in these three lines and survived the whole suite, and the reason is
/// not that the seed is good. It is that the seed does not have to be: the Taylor branch returns
/// `34.5` at `x = 3`, where `W(3)` is `1.05`, and Halley recovers from that without trouble
/// because it is allowed sixty steps. So a test that demanded the seed be *close* would either
/// fail against the real kernel or be too loose to notice a mutation.
///
/// # Nor is its cost, as the code stands
///
/// The next idea was to pin the *iterations*: a worse seed should cost steps even when it costs
/// no accuracy. Measured, it does not discriminate. Counting steps until `w` lands within four
/// ulps of the converged answer, this seed needs **22 at `x = 3`** — the top of the Taylor
/// regime, where the guess is `34.5` — while four mutated variants need eleven, twenty-two,
/// twenty-two and twenty-two over the same range. The real seed is not better than the mutants
/// by the measure that would have to catch them.
///
/// So these sixteen survivors are genuinely unreachable from outside, and the measurement says
/// why: the Taylor branch is carried to `x = 3`, far past where `x − x² + 3x³/2` is a useful
/// approximation, and `W_STEPS = 60` is generous enough to absorb the difference. Narrowing that
/// regime — the large-`x` form is already better by `x ≈ 1` — would both save the iterations and
/// make the seed observable. That is a change to the kernel on cost grounds rather than
/// correctness, so it is recorded in MUTANTS.md rather than made here.
fn lambert_seed(x: f64) -> f64 {
    if x < NEG_INV_E + 1e-3 {
        // Branch-point series: p = sqrt(2(1 + e x)).
        let p = (2.0 * (1.0 + core::f64::consts::E * x)).max(0.0).sqrt();
        -1.0 + p - p * p / 3.0 + 11.0 / 72.0 * p * p * p
    } else if x > 3.0 {
        let l = x.ln();
        l - l.ln()
    } else {
        x * (1.0 - x + 1.5 * x * x)
    }
}

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

    let mut w = lambert_seed(x);

    for _ in 0..W_STEPS {
        let ew = w.exp();
        // For positive x, divide the residual and both denominators by exp(w).
        // W(MAX) is finite, but w*exp(w) can overflow after a rounding-sized
        // overshoot, and f*(w+2) can overflow even before that. The scaled
        // equation has the same Halley step without either large product.
        // Keep the unscaled residual near the negative branch point, where
        // subtracting x/exp(w) would introduce a different cancellation.
        let (f, denom_newton) = if x > 0.0 {
            (w - x / ew, 1.0 + w)
        } else {
            (w * ew - x, ew * (1.0 + w))
        };
        if f == 0.0 {
            return w;
        }
        // Halley: w -= f / (e^w (1+w) - f (w+2)/(2w+2))
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

/// Terms used by [`zeta_eta`]. The Cohen–Rodriguez Villegas–Zagier acceleration
/// converges like `(3 + √8)^{−n} ≈ 5.83^{−n}`, so 24 terms give `≈ 2e-18`.
const CVZ_TERMS: u32 = 24;

/// `T_n(3)`, the integer Chebyshev normalization for the eta accelerator.
/// Fits in `u64` through `n = 25`.
const fn cvz_normalizer(terms: u32) -> u64 {
    if terms == 0 {
        return 1;
    }
    let (mut previous, mut current) = (1, 3);
    let mut index = 1;
    while index < terms {
        (previous, current) = (current, 6 * current - previous);
        index += 1;
    }
    current
}

fn next_cvz_coefficient(k: f64, n: f64, coefficient: f64) -> f64 {
    (k + n) * (k - n) * coefficient / ((k + 0.5) * (k + 1.0))
}

/// `1 - 2^(1-s)`, retaining the difference close to the pole.
fn eta_divisor(s: f64) -> f64 {
    -((1.0 - s) * core::f64::consts::LN_2).exp_m1()
}

/// `ζ(s)` for `-1 < s < 1`, through the accelerated Dirichlet eta series.
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
/// The acceleration also continues eta across zero. Using it for `-1 < s < 0` avoids
/// reflection's loss of `s` in `1-s`: for tiny negative `s` that rounds to exactly `1`,
/// spuriously evaluating the Hurwitz-zeta pole even though `ζ(0) = -1/2` is regular.
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
fn zeta_eta(s: f64) -> f64 {
    debug_assert!(s > -1.0 && s < 1.0);
    let n = f64::from(CVZ_TERMS);

    const NORMALIZER: u64 = cvz_normalizer(CVZ_TERMS);
    let d = NORMALIZER as f64;

    let mut b = -1.0f64;
    let mut c = -d;
    let mut total = 0.0f64;
    for k in 0..CVZ_TERMS {
        let kf = f64::from(k);
        c = b - c;
        // The kth term of eta is (k+1)^{-s}.
        total += c * (kf + 1.0).powf(-s);
        b = next_cvz_coefficient(kf, n, b);
    }
    let eta = total / d;
    eta / eta_divisor(s)
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum ZetaRoute {
    Value(f64),
    Hurwitz,
    Eta,
    Reflection,
}

fn zeta_route(s: f64) -> ZetaRoute {
    if s.is_nan() || s == f64::NEG_INFINITY {
        ZetaRoute::Value(f64::NAN)
    } else if s == f64::INFINITY {
        ZetaRoute::Value(1.0)
    } else if s > 1.0 {
        ZetaRoute::Hurwitz
    } else if s == 1.0 {
        ZetaRoute::Value(f64::INFINITY)
    } else if s == 0.0 {
        ZetaRoute::Value(-0.5)
    } else if s > -1.0 {
        ZetaRoute::Eta
    } else if s % 2.0 == 0.0 {
        ZetaRoute::Value(0.0) // Negative even integers are trivial zeros.
    } else {
        ZetaRoute::Reflection
    }
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
/// For `-1 < s < 1`, including the critical strip, [`zeta_eta`] takes over. Reflection near
/// zero would lose the argument in `1-s`, so eta is used on both sides of zero. It is worth
/// being explicit about why the functional
/// equation does *not* suffice here, since it is the obvious thing to reach for: the equation
/// relates `s` to `1 − s`, and for `s` in `(0, 1)` the reflected point `1 − s` is still in
/// `(0, 1)`. The strip is mapped onto itself, so no amount of reflection escapes it.
///
/// For `s <= -1` the functional equation is used, since `1 − s >= 2`:
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
    match zeta_route(s) {
        ZetaRoute::Value(value) => value,
        ZetaRoute::Hurwitz => hurwitz_zeta(s, 1.0),
        ZetaRoute::Eta => zeta_eta(s),
        ZetaRoute::Reflection => zeta_reflection(s),
    }
}

fn zeta_reflection(s: f64) -> f64 {
    // Functional equation. Everything is formed in the log domain where it could overflow:
    // Gamma(1-s) grows without bound as s -> -inf, and zeta does too, so the product must be
    // allowed to reach infinity rather than producing NaN on the way.
    let one_minus_s = 1.0 - s;
    let sin_term = crate::math::gamma::reflection_sine(0.5 * s);
    let ln_magnitude = s * core::f64::consts::LN_2
        + (s - 1.0) * core::f64::consts::PI.ln()
        + sin_term.abs().ln()
        + ln_gamma(one_minus_s);
    // ζ(1-s) is positive throughout the reflection domain.
    (ln_magnitude.exp() * hurwitz_zeta(one_minus_s, 1.0)).copysign(sin_term)
}

#[cfg(test)]
mod tests;
