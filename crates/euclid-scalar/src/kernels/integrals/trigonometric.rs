//! The trigonometric integrals `Si` and `Ci`.
//!
//! A child module so the parent stays inside the crate's 400-line budget. These are the
//! imaginary ray of the same continued fraction the exponential integrals use — `E₁(ix) = −Ci(x)
//! + i(Si(x) − π/2)` — so one pass returns both, which is what [`TrigIntegrals`] carries.

use super::*;

/// The two trigonometric integrals at one argument, which [`trig_integrals`] produces together.
///
/// Named fields rather than a `(f64, f64)`: the pair is genuinely two different functions, not a
/// coordinate, and every call site previously picked one out by index. `Ci` and `Si` differ in
/// parity, in domain and in what they do at the origin — a reader asking which of `.0` and `.1`
/// was which had to go and look, and a caller that swapped them would still typecheck.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct TrigIntegrals {
    /// `Ci(x) = γ + ln x + ∫₀ˣ (cos t − 1)/t dt`, the cosine integral.
    cosine: f64,
    /// `Si(x) = ∫₀ˣ sin(t)/t dt`, the sine integral.
    sine: f64,
}

/// Which route evaluates the pair `(Ci, Si)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum TrigIntegralRoute {
    /// The alternating power series, below [`TRIG_SERIES_MAX`].
    Series,
    /// One pass of the continued fraction along the imaginary axis, which returns both.
    Fraction,
}

impl TrigIntegralRoute {
    pub(super) fn for_argument(x: f64) -> Self {
        if x < TRIG_SERIES_MAX {
            Self::Series
        } else {
            Self::Fraction
        }
    }
}

/// Both trigonometric integrals at `x > 0`, from a single evaluation.
///
/// The two are the real and imaginary parts of one exponential integral, so taking them
/// together is both cheaper than two passes and guarantees they remain mutually consistent.
pub(super) fn trig_integrals(x: f64) -> TrigIntegrals {
    debug_assert!(x > 0.0 && x.is_finite());
    match TrigIntegralRoute::for_argument(x) {
        TrigIntegralRoute::Series => {
            // Ci(x) = gamma + ln x + sum_{k>=1} (-1)^k x^{2k} / ((2k)(2k)!)
            let mut power_over_factorial = 1.0f64; // x^{2k}/(2k)!
            let cosine = EULER_GAMMA
                + x.ln()
                + compensated_series(|k| {
                    let m = 2.0 * (f64::from(k) + 1.0);
                    power_over_factorial *= x * x / ((m - 1.0) * m);
                    let term = power_over_factorial / m;
                    Some(if k.is_multiple_of(2) { -term } else { term })
                });
            // Si(x) = sum_{k>=0} (-1)^k x^{2k+1} / ((2k+1)(2k+1)!)
            let mut power_over_factorial = x; // x^{2k+1}/(2k+1)!
            let sine = compensated_series(|k| {
                if k > 0 {
                    let m = 2.0 * f64::from(k);
                    power_over_factorial *= x * x / (m * (m + 1.0));
                }
                let term = power_over_factorial / (2.0 * f64::from(k) + 1.0);
                Some(if k.is_multiple_of(2) { term } else { -term })
            });
            TrigIntegrals { cosine, sine }
        }
        TrigIntegralRoute::Fraction => {
            // E1(ix) = -Ci(x) + i (Si(x) - pi/2), with e^{-ix} = cos x - i sin x.
            let (re, im) = e1_continued_fraction((0.0, x), (x.cos(), -x.sin()));
            TrigIntegrals {
                cosine: -re,
                sine: core::f64::consts::FRAC_PI_2 + im,
            }
        }
    }
}

/// `Si(x) = ∫₀ˣ sin(t)/t dt`, the sine integral. Odd in `x`.
pub(crate) fn sin_int(x: f64) -> f64 {
    match SinIntRoute::for_argument(x) {
        SinIntRoute::Undefined => f64::NAN,
        // Returning the argument rather than a literal zero, so a negative zero survives: `Si`
        // is odd, hence `Si(−0) = −0`.
        SinIntRoute::AtOrigin => x,
        SinIntRoute::AtInfinity => core::f64::consts::FRAC_PI_2.copysign(x),
        SinIntRoute::Quadrature => {
            let si = trig_integrals(x.abs()).sine;
            // A sign-bit test rather than `* x.signum()`: multiplying and dividing by `±1` give
            // the same answer, so that spelling buys mutants with no observable difference. The
            // argument is never zero here — [`SinIntRoute::AtOrigin`] takes that case.
            if x.is_sign_negative() { -si } else { si }
        }
    }
}

/// Which route evaluates `Si` at this argument.
///
/// `Si` is odd and entire, so no arm here is a singularity: the quadrature takes `|x|` and the
/// sign is applied afterwards. What the arms divide is the format — the two ends of it, and the
/// signed zero in the middle that the general route would round away.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SinIntRoute {
    /// Not a number.
    Undefined,
    /// `x = ±0`, where `Si` is zero with the argument's own sign.
    AtOrigin,
    /// `±∞`, where `Si` converges to `±π/2` — the Dirichlet integral.
    AtInfinity,
    /// Everywhere else, by quadrature on `|x|` with the sign applied afterwards.
    Quadrature,
}

impl SinIntRoute {
    pub(super) fn for_argument(x: f64) -> Self {
        if x.is_nan() {
            Self::Undefined
        } else if x == 0.0 {
            Self::AtOrigin
        } else if x.is_infinite() {
            Self::AtInfinity
        } else {
            Self::Quadrature
        }
    }
}

/// Which route evaluates `Ci` at this argument.
///
/// Unlike [`SinIntRoute`] this one *is* about the function: `Ci` carries a `ln x` term, so it has
/// a genuine singularity at the origin and no real value at all to the left of it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CosIntRoute {
    /// `NaN`, and the whole negative half-line including `−∞`, where the `ln x` term puts `Ci` on
    /// the far side of a branch cut.
    Undefined,
    /// `x = 0`, where `ln x` diverges and `Ci` goes to `−∞`.
    AtOrigin,
    /// `+∞`, where `Ci` decays to zero.
    AtInfinity,
    /// Everywhere else, by quadrature.
    Quadrature,
}

impl CosIntRoute {
    pub(super) fn for_argument(x: f64) -> Self {
        if x.is_nan() || x < 0.0 {
            Self::Undefined
        } else if x == 0.0 {
            Self::AtOrigin
        } else if x.is_infinite() {
            Self::AtInfinity
        } else {
            Self::Quadrature
        }
    }
}

/// `Ci(x) = γ + ln x + ∫₀ˣ (cos t − 1)/t dt`, the cosine integral, for `x > 0`.
pub(crate) fn cos_int(x: f64) -> f64 {
    match CosIntRoute::for_argument(x) {
        CosIntRoute::Undefined => f64::NAN,
        CosIntRoute::AtOrigin => f64::NEG_INFINITY,
        CosIntRoute::AtInfinity => 0.0,
        CosIntRoute::Quadrature => trig_integrals(x).cosine,
    }
}

#[cfg(test)]
mod tests;
