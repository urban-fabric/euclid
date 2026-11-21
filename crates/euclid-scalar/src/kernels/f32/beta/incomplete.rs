//! The regularized incomplete beta `I_x(a, b)`, at single precision.
//!
//! A child module so the parent stays inside the crate's 400-line budget. Its own topic rather
//! than a continuation of `B(a, b)`: this is a CDF in three arguments with an Edgeworth regime,
//! a continued fraction and a gamma-limit route, and which one runs turns on the relationship
//! between the shapes and `x` rather than on any single magnitude.

use super::*;

/// At or above this `min(a, b)` the Edgeworth-corrected normal replaces the continued fraction.
///
/// **`512`, where the `f64` peer uses `1e7`**, and this threshold is the whole reason a native
/// `f32` `beta_reg` is possible at all.
///
/// The fraction's cost is its `exp` prefactor, `exp(a ln x + b ln(1−x) − ln B(a,b))`: three terms
/// each of magnitude `≈1.386(a+b)` summing to an `O(1)` exponent, so the exponent carries
/// `≈1.386(a+b)·ε` of absolute error and `exp` hands it back as *relative* error. At `f32` that
/// is `1.65e-7(a+b)`, already `3e-4` at `a = b = 1e3`.
///
/// **Compensated summation cannot rescue it**, and it is worth saying why, because the primitives
/// are right there in this crate. The error is not in adding the three terms — it is inside
/// `a·ln x`, because `ln x` was rounded to `f32` before the multiplication. Measured, that term
/// alone carries `a·ulp(ln ½)/2`: `1.9e-8` at `a = 10`, `1.9e-3` at `a = 1e6`, `1.9e-1` at
/// `a = 1e8`. No summation rule recovers information that was discarded by the logarithm, and an
/// extended-precision `ln` is a far larger undertaking than this function.
///
/// What *does* work is handing over sooner. The Edgeworth expansion's error is `≈0.0605/min(a,b)`
/// — it *improves* exactly where the fraction degrades — so the two cross far below `f64`'s
/// handover. Measured against the `f64` kernel over shapes `10^1..10^7` at ratios `1`, `2` and
/// `10`, sampling `x` at `±½σ` and `±2σ` about the mean:
///
/// | `min(a,b)` | 1e1 | 1e2 | 1e3 | 1e4 | 1e5 | 1e6 | 1e7 |
/// |---|---|---|---|---|---|---|---|
/// | fraction | `6.1e-7` | `3.7e-5` | `2.5e-4` | `3.0e-4` | `6.4e-3` | `1.7e-2` | `2.7e-1` |
/// | Edgeworth | `5.9e-3` | `6.0e-4` | `6.1e-5` | `6.1e-6` | `6.9e-7` | `7.5e-8` | `1.0e-8` |
///
/// They cross between `1e2` and `1e3`. The worst error of the finished function as the handover
/// moves: `6.0e-4` at `1e2`, **`6.1e-5`** anywhere in `[2e2, 1e3]`, and `5.4e-4` from `2e3` up.
/// `512` sits in the flat middle of that and is a power of two, so the switch is exact.
///
/// # What this costs against narrowing
///
/// `6.1e-5` absolute on a probability, where narrowing gives `≈1e-9`. That is a real loss and
/// the largest of any port in this tree — but it is a loss, not a failure. The claim this
/// function used to carry was that a native port is "not merely less accurate but unusable",
/// with a table showing `6.9e23` at `a = 1e8` and `inf` at `1e10`. Those numbers are real and
/// they are the *fraction* evaluated where it must not be: a port keeping `f64`'s `1e7`
/// handover. Retuning the handover is the port.
pub(super) const BETA_EDGEWORTH_MIN: f32 = 512.0;

/// Iteration cap for the continued fraction. The `f64` peer allows 4,000.
///
/// The requirement grows with the parameters, because the fraction converges slowest at
/// `x ≈ a/(a+b)`. It is only used below [`BETA_EDGEWORTH_MIN`] here, where the `f64` peer's own
/// measurements put the requirement at well under a hundred iterations.
const MAX_ITER: usize = 256;

/// Relative size at which the fraction's running factor is taken to have converged.
const EPS: f32 = 1e-7;

/// The signed distance from the mean in units of `1/(a+b)`.
///
/// Compensated throughout, and it has to be: near the mean `x·n − a` is a difference of two
/// nearly-equal `O(a)` quantities, so a naive form loses the residual entirely and `z` comes
/// back as noise. Sterbenz makes `product − a` exact; the rest carries the discarded low bits.
pub(super) fn beta_mean_residual(a: f32, n: f32, n_error: f32, x: f32) -> f32 {
    let (product, product_error) = x.two_product(n);
    let (offset, offset_error) = (product - a).two_sum(product_error);
    x.mul_add(n_error, offset) + offset_error
}

/// `Iₓ(a, b)` from the Edgeworth-corrected normal approximation; see [`BETA_EDGEWORTH_MIN`].
///
/// A normal with the distribution's own mean and variance, plus the first skewness correction:
///
/// ```text
/// μ = a/n,   σ² = ab/(n²(n+1)),   γ₁ = 2(b−a)√(n+1)/((n+2)√(ab)),   n = a + b
/// ```
///
/// `Φ` is formed as `½erfc(−z/√2)` so a small lower tail keeps its relative accuracy instead of
/// being lost to a complement.
pub(super) fn beta_edgeworth(a: f32, b: f32, x: f32) -> f32 {
    let (n, n_error) = a.two_sum(b);
    if !n.is_finite() {
        // `a + b` has left the format, which here needs only `a ≈ 1.7e38` — `f64` does not reach
        // this until `2^970`, so the branch is far more than a formality in this format and
        // `f32::MAX` as a shape hits it directly. Both shapes are enormous, so the transition
        // width `√(ab/(a+b+1))` is hundreds of bits below the lowest bits of `x(a+b) − a`: the
        // answer is a certainty either side of the mean. Halving keeps the sum finite and
        // preserves the residual's sign.
        let half_a = 0.5 * a;
        let (half_n, half_error) = half_a.two_sum(0.5 * b);
        return match beta_mean_residual(half_a, half_n, half_error, x).partial_cmp(&0.0) {
            Some(core::cmp::Ordering::Less) => 0.0,
            Some(core::cmp::Ordering::Equal) => 0.5,
            Some(core::cmp::Ordering::Greater) => 1.0,
            None => f32::NAN,
        };
    }
    let inverse_width = edgeworth_inverse_width(a, b, n);
    let z = beta_mean_residual(a, n, n_error, x) * inverse_width;
    match EdgeworthTail::for_standardised(z) {
        EdgeworthTail::CertainLower => 0.0,
        EdgeworthTail::CertainUpper => 1.0,
        EdgeworthTail::Corrected => {
            let skew = edgeworth_skew(a, b, n, inverse_width);
            let lower = 0.5 * crate::kernels::f32::erf::erfc(-z * core::f32::consts::FRAC_1_SQRT_2)
                - edgeworth_correction(z, skew);
            lower.clamp(0.0, 1.0)
        }
    }
}

/// Past this `|z|` the standard normal tail has left this format in both directions.
///
/// `erfc` has saturated and `e^{−z²/2}` underflowed, so the answer is a certainty — and returning
/// it directly keeps `z·z` out of an overflow that would make the correction `NaN`. That is not
/// hypothetical: with `a = 512` and `b = 1e38` the standardised argument is `2e36`, whose square
/// leaves the format. (`f64` can carry this to 40.)
pub(super) const EDGEWORTH_Z_MAX: f32 = 10.0;

/// Which side of the correction's reachable range `z` falls on.
///
/// # Mutation testing is what asked for this
///
/// Three mutants here survived. Narrowing `|z| > 10` to `==` leaves every ordinary argument on the
/// corrected path, where the formula happens to agree — `erfc` saturates and the density
/// underflows, so it returns the same certainty by a longer route — and it only *differs* where
/// `z·z` overflows. Widening to `>=` differs at exactly `|z| = 10`, which cannot be reached
/// through this function's own arithmetic at all: `z` is a product of two rounded quantities.
/// And the sign test inside the old shortcut, `z > 0.0` against `z >= 0.0`, was reachable only at
/// `z = 0`, which `|z| > 10` excludes — an equivalent mutant by construction.
///
/// Splitting the certainty into its two sides removes the sign test entirely and makes the cut a
/// decision that takes `z` directly, so both boundaries are three assertions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum EdgeworthTail {
    /// `z < −10`: the lower tail is below this format's resolution and the answer is `0`.
    CertainLower,
    /// `z > 10`: the upper tail likewise, and the answer is `1`.
    CertainUpper,
    /// `|z| ≤ 10`, and `NaN`: the normal plus its first skewness correction.
    Corrected,
}

impl EdgeworthTail {
    /// The cut is strict on both sides, so `|z| = 10` is still corrected. A `NaN` compares false
    /// either way and reaches the correction, where it propagates.
    pub(super) fn for_standardised(z: f32) -> Self {
        if z > EDGEWORTH_Z_MAX {
            Self::CertainUpper
        } else if z < -EDGEWORTH_Z_MAX {
            Self::CertainLower
        } else {
            Self::Corrected
        }
    }
}

/// `√(n+1)/√(ab)`, the reciprocal of `σ` in units of `1/n`.
///
/// Taking the roots separately keeps the factors in range even when `a·b` overflows or `σ²`
/// underflows — a much nearer hazard in `f32`, where `a·b` leaves the format from `a = b ≈
/// 1.8e19`. The `n + 1` is the beta's own variance denominator and is not `n`: dropping the one
/// shifts the width by `1/2n`, which is under a whole-function tolerance at every shape this arm
/// is selected for, and so needs asserting here.
pub(super) fn edgeworth_inverse_width(a: f32, b: f32, n: f32) -> f32 {
    (n + 1.0).sqrt() / a.sqrt() / b.sqrt()
}

/// The beta skewness `γ₁ = 2(b−a)√(n+1)/((n+2)√(ab))`, given the inverse width.
pub(super) fn edgeworth_skew(a: f32, b: f32, n: f32, inverse_width: f32) -> f32 {
    2.0 * ((b - a) / (n + 2.0)) * inverse_width
}

/// The first skewness correction to the normal, `φ(z)·γ₁·(z²−1)/6`.
///
/// Signed, and subtracted from the lower tail by the caller. It vanishes at `z = ±1` rather than
/// at the mean, where it is largest in magnitude — which is why a test at the mean alone pins
/// almost none of this arithmetic.
pub(super) fn edgeworth_correction(z: f32, skew: f32) -> f32 {
    let density = (-0.5 * z * z).exp() / crate::constants::f32::SQRT_TAU;
    density * skew * (z * z - 1.0) / 6.0
}

/// Evaluate the fraction in the chosen orientation. An unconverged fraction leaves the decision
/// to use an approximation with the caller.
pub(super) fn beta_fraction_cdf(a: f32, b: f32, x: f32) -> Option<f32> {
    let front = crate::math::beta::beta_fraction_prefactor(a, b, x, ln_beta(a, b));
    if crate::math::beta::use_lower_beta_fraction(a, b, x) {
        beta_cf(a, b, x).map(|h| front * h / a)
    } else {
        beta_cf(b, a, 1.0 - x).map(|h| 1.0 - front * h / b)
    }
}

/// Lentz continued fraction for the incomplete beta.
///
/// Returns `None` when it exhausts [`MAX_ITER`] without converging, rather than handing back a
/// truncated partial sum — which is what made the `f64` peer's large-parameter defect *silent*.
pub(super) fn beta_cf(a: f32, b: f32, x: f32) -> Option<f32> {
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = nonzero_pivot(1.0 - qab * x / qap, LENTZ_TINY_F32).recip();
    let mut h = d;
    for m in 1..=MAX_ITER {
        let m = m as f32;
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        h *= lentz_step(aa, 1.0, &mut c, &mut d, LENTZ_TINY_F32);
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        let del = lentz_step(aa, 1.0, &mut c, &mut d, LENTZ_TINY_F32);
        h *= del;
        if converged(del, 0.0, EPS) {
            return Some(h);
        }
    }
    None
}

/// Which closed form or approximation evaluates `Iₓ(a, b)`.
///
/// Seven arms, which is what this function is: most of them are exact forms that exist to keep
/// the fraction away from arguments where its intermediates misbehave, and naming them is the
/// only way the ordering stays legible.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum BetaRegMethod {
    /// Not two positive finite shapes.
    Invalid,
    /// `x` outside `(0, 1)`, or both shapes so small the distribution is two endpoint masses.
    Value(f32),
    /// `a = 1`: `I = 1 − (1−x)^b`, exact.
    UnitLower,
    /// `b = 1`: `I = x^a`, exact.
    UnitUpper,
    /// One shape vastly larger than the other, where the beta tends to a gamma in `−ln(1−x)`.
    GammaLimit,
    /// The same limit reflected, which preserves the small tail.
    GammaLimitReflected,
    /// `min(a, b) ≥ `[`BETA_EDGEWORTH_MIN`]: the Edgeworth-corrected normal.
    Edgeworth,
    /// Otherwise the continued fraction, with Edgeworth as its fallback if it fails to converge.
    Fraction,
}

impl BetaRegMethod {
    pub(super) fn for_arguments(a: f32, b: f32, x: f32) -> Self {
        // `x` is checked for `NaN` alongside the shapes: the comparisons below are all false
        // for it, so without this it would fall through to the fraction and return a number.
        if !a.is_finite() || !b.is_finite() || a <= 0.0 || b <= 0.0 || x.is_nan() {
            Self::Invalid
        } else if x <= 0.0 {
            Self::Value(0.0)
        } else if x >= 1.0 {
            Self::Value(1.0)
        } else if a.max(b) <= 1e-18 {
            // The endpoint masses are b/(a+b) at zero and a/(a+b) at one. Throughout the
            // representable interior the relative correction is O(max(a,b)·max(|ln x|,
            // |ln(1−x)|)), which this bound keeps under an ulp. `f64` can place it at `1e-22`.
            Self::Value(b / (a + b))
        } else if a == 1.0 {
            Self::UnitLower
        } else if b == 1.0 {
            Self::UnitUpper
        } else if beta_gamma_rate(a, b).is_some() {
            Self::GammaLimit
        } else if beta_gamma_rate(b, a).is_some() {
            Self::GammaLimitReflected
        } else if a.min(b) >= BETA_EDGEWORTH_MIN {
            Self::Edgeworth
        } else {
            Self::Fraction
        }
    }
}

/// Rate for the gamma approximation to `T = −ln(1−X)`, with `X ~ Beta(a,b)`.
///
/// Moving the leading exponential into the rate leaves a log-density correction of order
/// `(a−1)t²/24`; near the mass `t ≈ a/b`, so the switch bounds that at the format's scale.
pub(super) fn beta_gamma_rate(a: f32, b: f32) -> Option<f32> {
    // Once gamma itself would use Edgeworth, standardise directly in beta coordinates.
    if a >= crate::kernels::f32::gamma::GAMMA_EDGEWORTH_MIN {
        return None;
    }
    let shift = a + 1.0;
    // `5e7` in `f64`; scaled to this format's resolution.
    (b / shift / shift.sqrt() >= 1e4).then_some(b + 0.5 * (a - 1.0))
}

/// The regularized incomplete beta `Iₓ(a, b)`, `x ∈ [0, 1]`, `a, b > 0`.
///
/// Native single precision, where this used to narrow the `f64` kernel. See
/// [`BETA_EDGEWORTH_MIN`] for what made that possible and what it costs.
pub(crate) fn beta_reg(a: f32, b: f32, x: f32) -> f32 {
    match BetaRegMethod::for_arguments(a, b, x) {
        BetaRegMethod::Invalid => f32::NAN,
        BetaRegMethod::Value(value) => value,
        BetaRegMethod::UnitLower => -(b * (-x).ln_1p()).exp_m1(),
        BetaRegMethod::UnitUpper => x.powf(a),
        BetaRegMethod::GammaLimit => {
            let rate = beta_gamma_rate(a, b).unwrap_or(b);
            crate::kernels::f32::gamma::gamma_lr(a, -rate * (-x).ln_1p())
        }
        BetaRegMethod::GammaLimitReflected => {
            let rate = beta_gamma_rate(b, a).unwrap_or(a);
            crate::kernels::f32::gamma::gamma_ur(b, -rate * x.ln())
        }
        BetaRegMethod::Edgeworth => beta_edgeworth(a, b, x),
        BetaRegMethod::Fraction => beta_fraction_cdf(a, b, x)
            .unwrap_or_else(|| beta_edgeworth(a, b, x))
            .clamp(0.0, 1.0),
    }
}

#[cfg(test)]
mod tests;
