//! The gamma function family: `ln_gamma`/`gamma` (Lanczos), the regularized lower/upper
//! incomplete gamma `gamma_lr`/`gamma_ur` (the Poisson/Gamma/chi-squared CDF kernels), and
//! the polygamma `digamma`/`trigamma`.

use crate::math::continued_fraction::LENTZ_TINY_F64;
use crate::math::gamma::{
    gamma_edgeworth, gamma_lower_series, gamma_reflection, gamma_upper_fraction, lanczos_gamma,
    lanczos_log_gamma, lanczos_reflected_gamma, log_gamma_reflection, log_scaled_gamma,
    reflection_sine, use_gamma_reflection, use_lower_gamma_series, valid_incomplete_gamma_inputs,
};
use core::f64::consts::PI;

/// Lanczos `g` parameter and coefficients (g = 7, n = 9) — ~15-digit accuracy.
const G: f64 = 7.0;
const LANCZOS: [f64; 9] = [
    0.999_999_999_999_809_9,
    676.520_368_121_885_1,
    -1_259.139_216_722_402_8,
    771.323_428_777_653_1,
    -176.615_029_162_140_6,
    12.507_343_278_686_905,
    -0.138_571_095_265_720_12,
    9.984_369_578_019_572e-6,
    1.505_632_735_149_311_6e-7,
];

/// Above this argument `Γ(x)` exceeds the format's range, and ``+∞`` is returned directly.
///
/// `Γ` overflows where `lnΓ(x)` passes `ln(f64::MAX) = 709.78`, which is `x ≈ 171.62`. The
/// threshold sits just above that, so nothing representable is saturated: `Γ(171)` is
/// `7.26e306` and is returned in full.
///
/// Needed as an explicit test rather than left to the arithmetic. The Lanczos form below keeps
/// its intermediates in range only while the *result* is in range, which is exactly what makes
/// it accurate — but once `x` is far past the boundary the half-power overflows while `e^{−t/2}`
/// underflows, and `inf · 0` is `NaN`. Saturating deliberately is both correct and cheaper.
const GAMMA_OVERFLOW_MIN: f64 = 171.7;

/// Natural log of the gamma function on intervals where `Γ(x)` is positive.
/// Which form evaluates `lnΓ` at this argument.
///
/// Seven arms, and more than the single-precision peer's three: this kernel carries the two
/// zero-neighbourhood series and the pole test that one does not. Matched rather than tested as a
/// chain of early returns, so the ordering is readable in one place and a new regime cannot be
/// added without the compiler asking what it returns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LnGammaMethod {
    /// `NaN`, and `−∞` where no limit exists.
    Undefined,
    /// `+∞`, and the non-positive integers.
    ///
    /// The poles have to be tested for rather than left to the reflection below, because
    /// `sin(πx)` at an integer `x` does not evaluate to zero — it evaluates to the rounding
    /// residue of `πx`, whose *sign* is then arbitrary. That made `ln_gamma` return `NaN` at `−1`
    /// and `−5` (residue negative, so `ln` of a negative) but a plausible finite number at `−2`,
    /// `−20` and `−1e5`: `36.4`, `−7.55`, `−1.05e6`. All three were noise. `ln|Γ|` does tend to
    /// `+∞` at a pole, from both sides, so the limit is well defined and is returned.
    Divergent,
    /// `|x − 1| ≤ `[`SMALL_SHAPE`]: the shifted series about the first zero.
    ///
    /// `lnΓ` has zeros at `1` and `2` and nowhere else, and beside a zero the general route
    /// delivers an *absolute* error where a caller needs a *relative* one. Lanczos returns
    /// `−8.88e-16` at `x = 1`, where the answer is `0`, and `0` at `x = 1 + ulp`, where the answer
    /// is `−γ·ulp = −1.28e-16`. Neither is far wrong absolutely; both are wrong by everything
    /// relatively, and it is the relative form a log-density normalisation at shape one propagates
    /// — `ln_beta(1, 1)` carried the same `−8.9e-16` for the same reason.
    NearUnitZero,
    /// `|x − 2| ≤ SMALL_SHAPE`: the same, through `Γ(2+b) = (1+b)·Γ(1+b)`.
    ///
    /// Both shifts are **exact** — `x − 1` and `x − 2` lose nothing this close to them, by
    /// Sterbenz — and both forms are exactly zero at the zero, so the two exact zeros fall out of
    /// the general branch rather than needing an equality test of their own.
    NearTwoZero,
    /// `0 < x ≤ SMALL_SHAPE`: the pole at the origin, through `lnΓ(1+x)/x`.
    ///
    /// Thresholded at [`SMALL_SHAPE`], which is the range that polynomial is fitted for; past it
    /// its own truncation takes over, reaching `9.9e-11` by `a = 0.1`.
    SmallShape,
    /// Negative arguments: reflect to the positive side through `lnΓ(1−x)`.
    Reflected,
    /// Everything else: the Lanczos log form, whose cancellation `lnΓ` is by then big enough to
    /// absorb.
    Lanczos,
}

impl LnGammaMethod {
    /// The non-finite cases and the poles first, then the three neighbourhoods, then the sign.
    fn for_argument(x: f64) -> Self {
        if x.is_nan() || x == f64::NEG_INFINITY {
            Self::Undefined
        } else if x == f64::INFINITY || (x <= 0.0 && x == x.trunc()) {
            Self::Divergent
        } else if (x - 1.0).abs() <= SMALL_SHAPE {
            Self::NearUnitZero
        } else if (x - 2.0).abs() <= SMALL_SHAPE {
            Self::NearTwoZero
        } else if x.is_sign_positive() && x <= SMALL_SHAPE {
            Self::SmallShape
        } else if use_gamma_reflection(x) {
            Self::Reflected
        } else {
            Self::Lanczos
        }
    }
}

pub(crate) fn ln_gamma(x: f64) -> f64 {
    match LnGammaMethod::for_argument(x) {
        LnGammaMethod::Undefined => f64::NAN,
        // `lnΓ` grows without bound, so the limit at `+∞` is `+∞` rather than `NaN`.
        LnGammaMethod::Divergent => f64::INFINITY,
        LnGammaMethod::NearUnitZero => {
            let a = x - 1.0;
            a * ln_gamma1p_over_a(a)
        }
        LnGammaMethod::NearTwoZero => {
            let b = x - 2.0;
            b.ln_1p() + b * ln_gamma1p_over_a(b)
        }
        LnGammaMethod::SmallShape => x * ln_gamma1p_over_a(x) - x.ln(),
        LnGammaMethod::Reflected => log_gamma_reflection(reflection_sine(x), ln_gamma(1.0 - x)),
        LnGammaMethod::Lanczos => lanczos_log_gamma(x, G, &LANCZOS),
    }
}

/// The gamma function `Γ(x)`.
/// Which form evaluates `Γ` at this argument.
///
/// Four arms, and the two pole cases are separate because `Γ` answers them differently: at the
/// origin the sign of the zero says which side is meant, so `±∞` is the limit, while at a
/// negative integer `Γ` has *opposite* signs on the two sides and no limit exists.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GammaMethod {
    /// `x = ±0`: the sign of the zero selects which infinity.
    SignedPole,
    /// A negative integer: no limit, so `NaN`.
    ///
    /// Tested for rather than left to the reflection, for the reason [`LnGammaMethod::Divergent`]
    /// gives: `sin(πx)` returns the rounding residue of `πx` at an integer argument, not zero.
    /// That made `Γ(−1) = −2.57e16`, `Γ(−5) = −4.28e13` and — worst, because it looks entirely
    /// plausible — `Γ(−20) = 5.27e-4`.
    UnsignedPole,
    /// Below `½`: reflect through `Γ(1−x)`.
    Reflected,
    /// Above [`GAMMA_OVERFLOW_MIN`], where `Γ` has left the format.
    Saturated,
    /// `x ≥ `[`STIRLING_MIN`](crate::math::gamma::STIRLING_MIN): the Stirling series.
    ///
    /// Not a speed choice. The Lanczos approximation's *truncation* reaches `456` ulp by
    /// `x = 170` — measured by evaluating the same formula and coefficients at 60 digits, which
    /// reproduces the kernel's error exactly — and Stirling is `0.1` to `1.8` ulp there.
    Stirling,
    /// Everything else: the Lanczos product form.
    Lanczos,
}

impl GammaMethod {
    /// The poles first — they are what the reflection cannot be trusted with — then the sign,
    /// then the range.
    fn for_argument(x: f64) -> Self {
        if x <= 0.0 && x == x.trunc() {
            if x == 0.0 {
                Self::SignedPole
            } else {
                Self::UnsignedPole
            }
        } else if use_gamma_reflection(x) {
            Self::Reflected
        } else if x > GAMMA_OVERFLOW_MIN {
            Self::Saturated
        } else if x >= crate::math::gamma::STIRLING_MIN {
            Self::Stirling
        } else {
            Self::Lanczos
        }
    }
}

pub(crate) fn gamma(x: f64) -> f64 {
    match GammaMethod::for_argument(x) {
        GammaMethod::SignedPole => f64::INFINITY.copysign(x),
        GammaMethod::UnsignedPole => f64::NAN,
        GammaMethod::Reflected => gamma_reflection(x, |sin, reflected| {
            lanczos_reflected_gamma(sin, reflected, G, REFLECTED_ZERO_MIN, &LANCZOS)
        }),
        GammaMethod::Saturated => f64::INFINITY,
        GammaMethod::Stirling => crate::math::gamma::stirling_gamma(x),
        GammaMethod::Lanczos => lanczos_gamma(x, G, &LANCZOS),
    }
}

/// Reflected argument past which `Γ` on the negative side has certainly rounded to signed zero.
///
/// At a reflected argument of 200, the integral over `[199, 200]` alone gives
/// `Γ(a) ≥ exp(−200)·199^199`. The smallest nonzero distance of these `binary64` inputs to an
/// integer is at least `2^-46`, so even with that sine factor the reflected result rounds to
/// signed zero from here on — and clamping keeps the Lanczos intermediates in range rather than
/// letting them overflow to `inf` and produce `NaN`.
const REFLECTED_ZERO_MIN: f64 = 200.0;

/// Iteration cap for the incomplete-gamma series and continued fraction.
///
/// Both need roughly `O(sqrt(a))` terms near `x ~ a`, so a small cap silently truncates at
/// large shape: at 200 the series stopped short from about `a = 1e3`, leaving `P(1e5, median)`
/// at `0.237` instead of `0.5` and putting a visible discontinuity at the `x = a + 1` branch
/// switch. All the series terms are positive, so this is pure truncation with no cancellation —
/// raising the cap fixes it outright, and costs nothing at small shape because both loops
/// break as soon as the relative term falls below `EPS`.
const MAX_ITER: usize = 100_000;
const EPS: f64 = 1e-15;

/// Below this shape the complement needs a series that retains terms of order `a`.
const SMALL_SHAPE: f64 = 1e-3;

/// `ln Γ(1+a) / a` from the Taylor series at zero. The omitted term is
/// smaller than 2e-19 throughout the small-shape branch.
fn ln_gamma1p_over_a(a: f64) -> f64 {
    -crate::constants::f64::EULER_GAMMA
        + a * (0.822_467_033_424_113_2
            + a * (-0.400_685_634_386_531_4
                + a * (0.270_580_808_427_784_55
                    + a * (-0.207_385_551_028_674 + a * 0.169_557_176_997_408_2))))
}

/// DLMF 8.7.1 with the constant term separated, for `x < 1+a`.
mod incomplete;
mod psi;

// Re-exported so the family keeps one path per function: callers reached
// `kernels::gamma::gamma_lr` before the split and still do. `SMALL_SHAPE`,
// `ln_gamma1p_over_a` and the iteration limits stay in this file because both
// halves use them — `lnΓ` for its zero neighbourhoods, the incomplete pair for
// its small-shape route.
pub(crate) use incomplete::{GAMMA_EDGEWORTH_MIN, gamma_lr, gamma_ur};
pub(crate) use psi::{digamma, trigamma};

mod rising_factorial;
pub(crate) use rising_factorial::ln_rising_factorial;

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests;
