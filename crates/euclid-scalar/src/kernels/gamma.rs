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
fn small_shape_log_lower(a: f64, x: f64) -> f64 {
    let mut power = -x;
    let mut sum = power / (a + 1.0);
    for k in 2..=24 {
        let k = f64::from(k);
        power *= -x / k;
        sum += power / (a + k);
    }
    let v = a * sum;
    // ln1p(v)/v rounds near one at this cutoff. Avoid dividing subnormal values.
    let log_ratio = if v.abs() <= f64::EPSILON {
        1.0
    } else {
        v.ln_1p() / v
    };
    a * (x.ln() - ln_gamma1p_over_a(a) + sum * log_ratio)
}

/// At or above this shape the incomplete gamma switches to an Edgeworth-corrected normal
/// approximation.
///
/// Both of the series routes fail above it, and they fail *quietly*. Measured `P(a, a)`
/// against its known asymptote `½ + 1/(3√(2πa))`:
///
/// ```text
/// a = 1e7   2.9e-8        a = 1e9.5  7.5e-2
/// a = 1e8   3.4e-7        a = 1e10   3.2e-1
/// a = 1e9   1.6e-3        a = 1e11   7.5e-1
/// ```
///
/// and further out it stops being a probability at all: `P(1e15, a + √a)` came back as
/// `−9.06`, and from `a = 1e16` as `NaN`. Two mechanisms compound — the series needs `O(√a)`
/// terms and runs into [`MAX_ITER`], and the front factor `exp(a ln x − x − lnΓ(a))` subtracts
/// three `O(a ln a)` quantities, which past `a ≈ 1e14` leaves nothing at all.
///
/// A `Gamma(a, 1)` standardised by its own mean and variance has skewness `2/√a`, so the
/// Edgeworth expansion
///
/// ```text
/// P(a, x) ≈ Φ(z) − φ(z)·(z² − 1)/(3√a),      z = (x − a)/√a
/// ```
///
/// has absolute error `O(1/a)` — *improving* exactly where the series collapses, and exact in
/// the limit. Verified against the series over `a ∈ [1e6, 1e8]` and `z ∈ [−3, 6]`, where the
/// measured discrepancy is `1.8e-5`, `1.8e-6`, `3.4e-7` — one order per decade of `a`, as
/// `O(1/a)` predicts. At the threshold both routes carry about `3e-8`, so the join costs
/// nothing; above it the approximation is the more accurate of the two.
///
/// No tail accuracy is given up. `Q` decays like `e^{−a·D(x/a)}`, which underflows for any
/// `a` past a few thousand once `x` is away from `a` — `Q(1e6, 2e6)` is already exactly zero.
pub(crate) const GAMMA_EDGEWORTH_MIN: f64 = 3e7;

/// Regularized **lower** incomplete gamma `P(a, x) = γ(a, x) / Γ(a)`, `a > 0`, `x ≥ 0`.
/// Which series or approximation evaluates the regularized incomplete gamma.
///
/// Both tails route identically and differ only in what each arm returns, so the decision is
/// named once here and [`gamma_lr`] and [`gamma_ur`] are each nothing but a `match` over it. The
/// single-precision peer carries the same enum with the same arms; the ordering below is this
/// kernel's own, and the two small-shape bounds are mutually exclusive with the Edgeworth one, so
/// placing Edgeworth first changes nothing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum IncompleteGammaMethod {
    /// Not a positive finite shape with a non-negative argument.
    Invalid,
    /// `x = 0`, where the lower tail is `0` and the upper is `1`.
    AtZero,
    /// `x = +∞`, the complementary limit.
    ///
    /// Tested for rather than left to the series: without it the routes below produce `NaN` at
    /// an infinite argument, which then propagates into everything built on `gamma_lr` — `erf`,
    /// and through it the whole normal family.
    AtInfinity,
    /// `a ≥ `[`GAMMA_EDGEWORTH_MIN`]: the Edgeworth-corrected normal, where both series fail.
    Edgeworth,
    /// `a ≤ `[`SMALL_SHAPE`] and `x < a + 1`: the DLMF 8.7.1 series, which keeps the `O(a)` terms
    /// the ordinary routes drop.
    SmallShapeSeries,
    /// `a ≤ SMALL_SHAPE` and `x ≥ a + 1`: the upper fraction with the tiny shape applied
    /// **after** it, so the probability takes a single subnormal rounding.
    ///
    /// Separate from [`Self::SmallShapeSeries`] because that series is an expansion in `x` and is
    /// only valid for `x < 1 + a`. Collapsing the two into one small-shape arm reads as harmless
    /// and is not: in the `f32` peer it put the series to work at `x = 4`, where `P + Q` came
    /// back as `1.0000009537`.
    SmallShapeFraction,
    /// `x < a + 1`: the lower-tail series converges fastest here.
    LowerSeries,
    /// `x ≥ a + 1`: the upper-tail continued fraction does.
    UpperFraction,
}

impl IncompleteGammaMethod {
    /// The invalid and limiting arguments first, then the shape, then which tail converges.
    fn for_arguments(a: f64, x: f64) -> Self {
        if !valid_incomplete_gamma_inputs(a, x) {
            Self::Invalid
        } else if x == 0.0 {
            Self::AtZero
        } else if x.is_infinite() {
            Self::AtInfinity
        } else if a >= GAMMA_EDGEWORTH_MIN {
            Self::Edgeworth
        } else if use_lower_gamma_series(a, x) {
            // The small shape still has to pick a tail: the series is an expansion in `x`.
            if a <= SMALL_SHAPE {
                Self::SmallShapeSeries
            } else {
                Self::LowerSeries
            }
        } else if a <= SMALL_SHAPE {
            Self::SmallShapeFraction
        } else {
            Self::UpperFraction
        }
    }
}

pub(crate) fn gamma_lr(a: f64, x: f64) -> f64 {
    match IncompleteGammaMethod::for_arguments(a, x) {
        IncompleteGammaMethod::Invalid => f64::NAN,
        IncompleteGammaMethod::AtZero => 0.0,
        IncompleteGammaMethod::AtInfinity => 1.0,
        IncompleteGammaMethod::Edgeworth => {
            let (lower, _) = gamma_edgeworth(a, x);
            lower
        }
        IncompleteGammaMethod::SmallShapeSeries => small_shape_log_lower(a, x).exp(),
        IncompleteGammaMethod::LowerSeries => {
            let series = gamma_lower_series(a, x, EPS, MAX_ITER);
            log_scaled_gamma(a, x, series.ln()).exp()
        }
        // Either upper arm is reached through the complement, which is what keeps the lower tail
        // from being formed by a cancelling subtraction where it is the small one.
        IncompleteGammaMethod::SmallShapeFraction | IncompleteGammaMethod::UpperFraction => {
            1.0 - gamma_ur(a, x)
        }
    }
}

pub(crate) fn gamma_ur(a: f64, x: f64) -> f64 {
    match IncompleteGammaMethod::for_arguments(a, x) {
        IncompleteGammaMethod::Invalid => f64::NAN,
        IncompleteGammaMethod::AtZero => 1.0,
        IncompleteGammaMethod::AtInfinity => 0.0,
        IncompleteGammaMethod::Edgeworth => {
            let (_, upper) = gamma_edgeworth(a, x);
            upper
        }
        IncompleteGammaMethod::SmallShapeSeries => -small_shape_log_lower(a, x).exp_m1(),
        IncompleteGammaMethod::LowerSeries => 1.0 - gamma_lr(a, x),
        IncompleteGammaMethod::SmallShapeFraction => {
            let h = gamma_upper_fraction(a, x, LENTZ_TINY_F64, EPS, MAX_ITER);
            // Γ(1+a) = a Γ(a). Multiply the tiny shape only after evaluating the fraction so the
            // probability undergoes a single subnormal rounding.
            a * (-x + a * (x.ln() - ln_gamma1p_over_a(a)) + h.ln()).exp()
        }
        IncompleteGammaMethod::UpperFraction => {
            let h = gamma_upper_fraction(a, x, LENTZ_TINY_F64, EPS, MAX_ITER);
            log_scaled_gamma(a, x, 0.0).exp() * h
        }
    }
}

/// Argument above which the Stirling-type asymptotic series for `digamma`/`trigamma` is
/// evaluated; below it the recurrence walks the argument up to here first.
///
/// This was `10`, which left both functions a thousand times weaker than the rest of the crate
/// for no good reason. Their series are truncated after the `x^{-8}` and `x^{-9}` terms, so the
/// first *omitted* term sets the floor: for `digamma` that is `1/(132x^10)`, which is `7.6e-13`
/// at `x = 10` — and the measured error was `7.4e-13`, i.e. entirely this.
///
/// Walking to `20` instead puts the omitted term at `1/(132·20^10) = 7.4e-16`, below `ε`.
/// Measured against the exact closed forms `ψ(n) = −γ + H_{n−1}` and
/// `ψ₁(n) = ζ(2) − Σ_{k<n} k^{−2}`:
///
/// | shift | 10 | 15 | **20** | 25 | 30 |
/// |---|---|---|---|---|---|
/// | `digamma` | `7.4e-13` | `1.3e-14` | **`8.8e-16`** | `4.4e-16` | `8.9e-16` |
/// | `trigamma` | `7.0e-12` | `1.2e-13` | **`8.5e-15`** | `8.5e-15` | `8.5e-15` |
///
/// It flattens at 20 — past that the series is at its own floor and further shifting only costs
/// recurrence steps. A nonnegative argument takes at most twenty steps.
const PSI_ASYMPTOTIC_MIN: u8 = 20;

/// The digamma function `ψ(x) = d/dx ln Γ(x)`.
/// The `f64` nearest `ψ`'s only zero on the positive axis.
///
/// The zero is at `1.4616321449683623412626595423257213284681967140551221919873…`; this constant
/// is `9.55e-17` below it, which is why [`psi_near_zero`]'s constant term is not itself zero.
const PSI_ZERO: f64 = 1.461_632_144_968_362_2;

/// Half-width of the window on which [`psi_near_zero`] is used.
///
/// Set by the series' truncation at the edge, measured with the twenty-one coefficients below:
/// `2.4e-30` relative at `5e-2`, `2.6e-24` at `1e-1`, `2.8e-18` at `2e-1`, `9.6e-15` at `3e-1`.
/// `2e-1` holds `0.013` ulp at the edge, and the general route there is `16` ulp.
///
/// # Why this is wide rather than tight
///
/// A first version used ten coefficients and `2e-2`, which fixed the zero itself and left the
/// band `[0.02, 0.4]` carrying `10`–`160` ulp: the general route's error is **absolute** and flat
/// at about `3ε`, so its *relative* error is `3ε/|ψ(x)|` and does not fall under a few ulp until
/// `|d| ≈ 0.4`. Covering the band needs the window to reach it, and the window is limited by the
/// coefficient count — the `aₖ` decay by only about `0.68` per order, so `2e-1` needs twenty of
/// them where `2e-2` needed ten.
const PSI_ZERO_WINDOW: f64 = 2e-1;

/// `ψ(x₀ + d)` by its Taylor series about the zero, for `|d| ≤ PSI_ZERO_WINDOW`.
///
/// # Why a branch is needed at all
///
/// The general route recurses `x` up to `PSI_ASYMPTOTIC_MIN` accumulating `−1/x` and then adds
/// `ln x`. At `x ≈ 1.46` that is a sum of about `2.3` against a logarithm of about `2.86`,
/// differencing to a value near zero — so the result carries an **absolute** error of roughly
/// `3ε ≈ 7e-16` however close to the zero the argument is. Measured, `ψ` came back `1.2e-15` at
/// the zero and the relative error reached **479%** one ulp away from it.
///
/// That is the same shape as the `lnΓ(1)` defect: beside a zero an absolute error is what the
/// route delivers and a relative one is what a caller needs. `super::cross_precision` carries an
/// `ulp(1)` *envelope* for `ψ` to accommodate it, and notes `ψ` was "a latent flake, passing only
/// because this seed's climb had not yet reached their zero" — an envelope is the symptom, and
/// this is the cause.
///
/// # Provenance of the coefficients
///
/// Twenty-one of them, `aₖ = ψ⁽ᵏ⁾(x₀)/k!` at the `f64` [`PSI_ZERO`], computed at 80 decimal digits by a local
/// generator using `ψ⁽ᵏ⁾(y) ~ (−1)^{k−1}[(k−1)!/yᵏ + k!/(2y^{k+1}) + Σ B₂ⱼ(2j+k−1)!/((2j)! y^{2j+k})]`
/// with the argument first recursed up by `80`. The generator validates against seven closed forms
/// it shares no code with before emitting — `ψ(1) = −γ`, `ψ′(1) = π²/6`, `ψ″(1) = −2ζ(3)`,
/// `ψ‴(1) = π⁴/15`, `ψ⁗(1) = −24ζ(5)`, `ψ⁵(1) = 8π⁶/63`, `ψ⁶(1) = −720ζ(7)` — and agrees with all
/// seven to between 43 and 49 digits. A first run reported "failure" at `1e-40`, which was the
/// threshold being tighter than the method's own truncation rather than an error.
///
/// `a₀` is a consistency check in itself: it must equal `a₁ ×` the offset from the true zero, and
/// `−9.2413e-17` against `0.96767 × −9.55e-17` agrees to every digit printed.
fn psi_near_zero(d: f64) -> f64 {
    // Horner, from `a₉` inward.
    const A: [f64; 21] = [
        -9.241_265_521_729_427e-17,
        0.967_672_245_447_621_2,
        -0.442_763_168_983_592_2,
        0.258_499_760_955_651_1,
        -0.163_942_705_442_406_58,
        0.107_824_050_691_262_41,
        -0.072_199_561_256_454_74,
        0.048_804_288_164_143_13,
        -0.033_161_126_474_847_376,
        0.022_597_648_232_218_118,
        -0.015_424_765_904_948_97,
        0.010_538_791_616_612_184,
        -0.007_204_534_386_356_875,
        0.004_926_781_395_729_858,
        -0.003_369_801_655_439_331_2,
        0.002_305_126_326_734_93,
        -0.001_576_936_771_430_199,
        0.001_078_825_201_916_297_8,
        -0.000_738_070_938_996_006,
        0.000_504_953_265_834_602_7,
        -0.000_345_468_025_106_308_2,
    ];
    let mut acc = A[20];
    for &a in A[..20].iter().rev() {
        acc = acc * d + a;
    }
    acc
}

/// How `ψ` reaches an argument its asymptotic series can be evaluated at.
///
/// The series is only valid for a large positive argument, so every other arm is a reduction
/// onto that case — and the reductions are different functions of the input rather than
/// variations of one formula, which is why they are named. The `f32` peer in
/// [`super::f32::gamma`] carries the same vocabulary as `PsiReduction`, with one arm fewer:
/// it reduces onto `[1, 2)` with a factored polynomial where this one has a Taylor series about
/// `ψ`'s real zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PsiReduction {
    /// No answer: `NaN`, or `−∞` where the function oscillates without limit.
    Undefined,
    /// A negative integer — a pole. `ψ` has opposite signs on the two sides of one, so no limit
    /// exists there and the answer is a `NaN` like an unusable argument's.
    Pole,
    /// `x < 0` and not an integer: reflect across the poles with the cotangent identity.
    Reflected,
    /// Within [`PSI_ZERO_WINDOW`] of `ψ`'s real zero, where the series and the recurrence both
    /// lose the answer to cancellation; see [`psi_near_zero`].
    NearZero,
    /// Everywhere else on the positive axis. Walks up to [`PSI_ASYMPTOTIC_MIN`] by the
    /// recurrence first where the argument is below it, accumulating the shed terms, then
    /// evaluates the series — including at `+∞`, where it gives `ψ(+∞) = +∞`.
    Asymptotic,
}

impl PsiReduction {
    fn for_digamma(x: f64) -> Self {
        if !x.is_finite() {
            // `ψ(+∞) = +∞` falls out of the series; `NaN` and `−∞` have no answer.
            return if x == f64::INFINITY {
                Self::Asymptotic
            } else {
                Self::Undefined
            };
        }
        if x < 0.0 {
            // A negative integer is a pole, and `x - x.round()` is the distance to the nearest
            // one — formed before multiplying by `π` so rounding cannot erase it.
            return if x == x.round() {
                Self::Pole
            } else {
                Self::Reflected
            };
        }
        if (x - PSI_ZERO).abs() <= PSI_ZERO_WINDOW {
            Self::NearZero
        } else {
            Self::Asymptotic
        }
    }
}

pub(crate) fn digamma(x: f64) -> f64 {
    match PsiReduction::for_digamma(x) {
        PsiReduction::Undefined | PsiReduction::Pole => f64::NAN,
        PsiReduction::Reflected => {
            // DLMF 5.5.4. Reduce the cotangent's argument before multiplying by π
            // so rounding cannot erase the distance to the nearest pole.
            let offset = x - x.round();
            digamma(1.0 - x) - PI / (PI * offset).tan()
        }
        // `x - PSI_ZERO` is exact for every `x` in this window, by Sterbenz, so the shift
        // costs nothing.
        PsiReduction::NearZero => psi_near_zero(x - PSI_ZERO),
        PsiReduction::Asymptotic => digamma_asymptotic(x),
    }
}

/// `ψ(x)` for positive `x` away from the zero, by recurrence up to [`PSI_ASYMPTOTIC_MIN`]
/// followed by the Stirling-type series.
///
/// The walk-up is a no-op for an argument already at or above the threshold.
fn digamma_asymptotic(mut x: f64) -> f64 {
    let mut result = 0.0;
    for _ in 0..PSI_ASYMPTOTIC_MIN {
        if x < f64::from(PSI_ASYMPTOTIC_MIN) {
            result -= 1.0 / x;
            x += 1.0;
        } else {
            break;
        }
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    result + x.ln()
        - 0.5 * inv
        - inv2 * (1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 * (1.0 / 252.0 - inv2 / 240.0)))
}

/// The trigamma function `ψ₁(x) = d²/dx² ln Γ(x)` — the derivative of [`digamma`].
pub(crate) fn trigamma(x: f64) -> f64 {
    if x <= 0.0 && x == x.floor() {
        return f64::INFINITY;
    }
    if x < 0.5 {
        // DLMF 5.15.6. Squaring the sine makes its integer shift immaterial.
        let offset = x - x.round();
        let cosecant = PI / (PI * offset).sin();
        return cosecant * cosecant - trigamma_positive(1.0 - x);
    }
    trigamma_positive(x)
}

/// Trigamma after reflection, where the recurrence moves towards positive infinity.
fn trigamma_positive(mut x: f64) -> f64 {
    let mut result = 0.0;
    for _ in 0..PSI_ASYMPTOTIC_MIN {
        if x < f64::from(PSI_ASYMPTOTIC_MIN) {
            result += 1.0 / (x * x);
            x += 1.0;
        } else {
            break;
        }
    }
    let r = 1.0 / x;
    let r2 = r * r;
    result + r + 0.5 * r2 + r2 * r * (1.0 / 6.0 - r2 * (1.0 / 30.0 - r2 * (1.0 / 42.0 - r2 / 30.0)))
}
mod rising_factorial;
pub(crate) use rising_factorial::ln_rising_factorial;

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests;
