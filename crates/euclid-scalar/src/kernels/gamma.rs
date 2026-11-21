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
    /// `½ ≤ x ≤ 5/2`: the two zeros factored out, by [`ln_gamma_unit_interval`].
    ///
    /// Ordered *after* the two zero neighbourhoods rather than before, so those keep the slivers
    /// they already covered on both sides of `1` and `2` and nothing that worked changes. This
    /// arm takes everything else in the window, which is where Lanczos was losing digits to
    /// cancellation — inside `[1,2]` because `lnΓ` is small there, and either side of it because
    /// the two zeros are still close enough to dominate the answer's size.
    UnitInterval,
    /// `5/2 < x ≤ 4`: one or two steps of `lnΓ(x) = lnΓ(x−1) + ln(x−1)` into the window above.
    ///
    /// Lanczos still carried `53` ulp here against a condition number of `5.8`, for a weaker
    /// version of the same reason it carried `2014` inside `[1,2]`: the zeros at `1` and `2` are
    /// close enough that `lnΓ` is still small compared with the terms Lanczos builds it from.
    /// Stepping down is exact — `x − 1` loses nothing on `(5/2, 4]` — and the `ln(x−1)` it adds
    /// is the same size as the answer, so nothing cancels.
    ///
    /// It stops at `4` because that is where Lanczos reaches `9` ulp on its own and the stepping
    /// would start costing more logarithms than it saves digits.
    Recurrence,
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
        } else if (LN_GAMMA_UNIT_MIN..=LN_GAMMA_UNIT_MAX).contains(&x) {
            Self::UnitInterval
        } else if (LN_GAMMA_UNIT_MAX..=LN_GAMMA_RECURRENCE_MAX).contains(&x) {
            Self::Recurrence
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
        LnGammaMethod::UnitInterval => ln_gamma_unit_interval(x),
        LnGammaMethod::Recurrence => {
            let down = x - 1.0;
            ln_gamma(down) + down.ln()
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
    /// `1 ≤ x < STIRLING_MIN`: reduce to `[1, 2)` and exponentiate the logarithm there.
    ///
    /// # Lanczos's truncation is the error, and `lnΓ` no longer shares it
    ///
    /// The coefficients' own truncation — not the evaluation — is what cost this range, and the
    /// earlier attribution concluded it needed a new fitted table, which was data this project
    /// could not generate. It turns out not to need one. `lnΓ` is now accurate on `[1, 2]`
    /// through its own factored window, and **there `lnΓ` is small**: at most `0.1214` in
    /// magnitude, so `exp` of it carries `|lnΓ|·ε` and not the argument's own size. That is a
    /// quarter of an ulp, and the recurrence `Γ(x) = (x−1)·Γ(x−1)` carries it upward on exact
    /// subtractions.
    ///
    /// Measured against a 400-bit reference, worst ulp per band:
    ///
    /// ```text
    /// x          [1,2]  [2,4]  [4,8]  [8,15]  [15,20]
    /// Lanczos       21     18     44      43       34   ulp
    /// reduced        1      1      3       4        4
    /// ```
    ///
    /// The same trick does not extend above [`STIRLING_MIN`](crate::math::gamma::STIRLING_MIN),
    /// and not because of the recurrence: `exp(lnΓ)` is useless once `lnΓ` is large, which is
    /// the defect this crate has fixed three times elsewhere. Stirling already measures `0.1` to
    /// `1.8` ulp there, so there is nothing to take from it.
    Reduced,
    /// Everything else — `[½, 1)` — the Lanczos product form.
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
        } else if (1.0..crate::math::gamma::STIRLING_MIN).contains(&x) {
            Self::Reduced
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
        GammaMethod::Reduced => reduced_gamma(x),
        GammaMethod::Lanczos => lanczos_gamma(x, G, &LANCZOS),
    }
}

/// `Γ(x)` for `1 ≤ x < STIRLING_MIN`, by reduction to `[1, 2)`.
///
/// `Γ(x) = (x−1)(x−2)⋯r·Γ(r)` with `r ∈ [1, 2)`. Every subtraction is exact — the result lands
/// in a binade no coarser than `x`'s — so the only error is the product's roundings and
/// `Γ(r)`'s, and `Γ(r)` is `exp` of a logarithm bounded by `0.1214`.
///
/// The loop runs at most eighteen times, since it is entered only below
/// [`STIRLING_MIN`](crate::math::gamma::STIRLING_MIN), and it cannot be entered at all by a
/// `NaN` — the range test in [`GammaMethod::for_argument`] is false for one.
fn reduced_gamma(x: f64) -> f64 {
    let mut remaining = x;
    let mut product = 1.0;
    while remaining >= 2.0 {
        remaining -= 1.0;
        product *= remaining;
    }
    product * ln_gamma(remaining).exp()
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

/// `lnΓ` on `[1, 2]`, with both of its zeros factored out exactly.
///
/// # Lanczos computes a small number as a difference of large ones here
///
/// `lnΓ` has simple zeros at `1` and `2` and a minimum of `−0.1214` between them, so across this
/// whole interval it is a *small* quantity. Lanczos forms it as `ln(series) + (x−½)ln t − t`
/// with `t ≈ 8`, so terms of size eight produce an answer of size an eighth: about two digits
/// go, and they go everywhere in the interval rather than at a point. Measured against a
/// 400-bit reference, worst ulp per band with the condition number beside it:
///
/// ```text
/// band        [1.01,1.2]  [1.2,1.46]  [1.46,1.8]  [1.8,1.99]
/// was                799         121         172        1902   ulp
/// kappa             89.1        1.12        5.86         118
/// ```
///
/// The outer bands are near the zeros, where a large `κ` is the honest explanation — `lnΓ`
/// passes through zero, so relative error there is unbounded for any implementation. `[1.2,1.46]`
/// is not: `κ` is `1.12` and the error is `121` ulp, an amplification of about a hundred.
///
/// # Factoring is structural, not a refinement
///
/// `lnΓ(x) = (x−1)(x−2)·R(x)`, and on `[1,2]` **both subtractions are exact** by Sterbenz. So
/// the product carries only `R`'s own error and two roundings, and the zeros come out as exactly
/// zero rather than as whatever the general route rounds to.
///
/// `R` is analytic here with removable singularities at both ends — `R(1) = γ` and
/// `R(2) = 1 − γ`, which is the check that the factoring is the right one. See
/// [`LN_GAMMA_UNIT`].
fn ln_gamma_unit_interval(x: f64) -> f64 {
    // Clenshaw on the Chebyshev series, in `t = x − 3/2` which maps the window to `[−1,1]`.
    // Clenshaw rather than a monomial Horner because converting a degree-21 Chebyshev series to
    // monomials loses most of what the fit was for: the monomial coefficients alternate and grow,
    // so their sum cancels where the Chebyshev form does not.
    let t = x - 1.5;
    let twice = 2.0 * t;
    let (mut nearer, mut further) = (0.0_f64, 0.0_f64);
    for &coefficient in LN_GAMMA_UNIT.iter().skip(1).rev() {
        let next = twice * nearer - further + coefficient;
        further = nearer;
        nearer = next;
    }
    let r = t * nearer - further + LN_GAMMA_UNIT[0];
    (x - 1.0) * (x - 2.0) * r
}

/// Chebyshev coefficients of `R(x) = lnΓ(x)/((x−1)(x−2))` on `[½, 5/2]`, in `t = x − 3/2`.
///
/// Derived rather than transcribed, by `lngamma_unit.py`: `R` evaluated at Chebyshev nodes
/// through MPFR and the transform run in 60-digit decimal. Run in `f64` the coefficients floor
/// at `1e-16` around the eighteenth and look converged; they are not, and they keep falling by
/// about a factor of six per order to `c₂₁ = −3.6e-18`. Two separate things had to leave `f64`
/// before that showed — the node values *and* the `π` in the node angles.
///
/// Thirty-nine terms, which is where the tail drops below `2e-18` — a quarter ulp of `R`. The
/// narrower `[1,2]` needs only twenty-two; the extra seventeen buy the bands either side, where
/// the measurement said the error was worst of all (`1845` ulp on `[2.01, 2.5]`).
///
/// The window is bounded by where the series stops paying rather than by where it stops working:
/// `R`'s nearest singularity is `lnΓ`'s pole at the origin, so widening costs terms geometrically
/// — `[0.5, 2.5]` takes 39 where `[0.5, 4]` would take 56.
const LN_GAMMA_UNIT: [f64; 39] = [
    0.5233830675936226,
    -0.17797233729529863,
    0.043707765509376,
    -0.012436054693716324,
    0.0038166273622453444,
    -0.0012248310110168689,
    0.0004046307309337695,
    -0.0001363999286478383,
    4.6669854564403646e-05,
    -1.6152571568530024e-05,
    5.641861294289138e-06,
    -1.985468608994096e-06,
    7.031292350156658e-07,
    -2.503431672683324e-07,
    8.954655195462511e-08,
    -3.216038816813161e-08,
    1.1591674196198932e-08,
    -4.191331343092517e-09,
    1.5198287942838018e-09,
    -5.525248244354367e-10,
    2.0133461407895695e-10,
    -7.351973399346738e-11,
    2.6898575436827364e-11,
    -9.85882886617758e-12,
    3.619354400654772e-12,
    -1.3307382122555227e-12,
    4.899613435234918e-13,
    -1.8063242404502505e-13,
    6.667388160934133e-14,
    -2.4638096705345538e-14,
    9.114199669747885e-15,
    -3.3749037208856614e-15,
    1.2508630658961872e-15,
    -4.640235454740319e-16,
    1.7227826138316332e-16,
    -6.401198458457187e-17,
    2.3802040365336956e-17,
    -8.85670810564711e-18,
    3.2977685786612905e-18,
];

/// The window the factored route covers.
///
/// `[½, 5/2]`, and the bound that matters is `x − 1` and `x − 2` staying exact. On `[½, 2]`
/// Sterbenz gives `x − 1` outright, and above `2` the difference lands in a binade whose ulp is
/// no coarser than `x`'s, so it is exact there too. The one place it is not is `x − 2` below
/// `x = 1`, where the result sits in `[−3/2, −1]` with a coarser ulp than `x` has — that costs
/// at most half an ulp of `x` against a `|x − 2|` of about `3/2`, which is `0.7` ulp in the
/// answer and is why the window is not pushed below a half.
const LN_GAMMA_UNIT_MIN: f64 = 0.5;
const LN_GAMMA_UNIT_MAX: f64 = 2.5;

/// Above this the Lanczos form is accurate enough to keep.
///
/// `4`, which takes at most two steps to reach the window — `(5/2, 7/2]` lands in it directly
/// and `(7/2, 4]` needs one more. Lanczos measures `9` ulp on `[4, 10]` and `4` beyond, so there
/// is little left to win and each further step is another logarithm.
const LN_GAMMA_RECURRENCE_MAX: f64 = 4.0;

/// DLMF 8.7.1 with the constant term separated, for `x < 1+a`.
mod incomplete;
mod psi;
mod temme;

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
