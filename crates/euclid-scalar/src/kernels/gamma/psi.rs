//! The polygamma pair `ψ` and `ψ₁`, the first two logarithmic derivatives of `Γ`.
//!
//! A child module so the parent stays inside the crate's 400-line budget. Separated from `Γ`
//! itself because the machinery does not overlap: these reach their asymptotic series by
//! *recurrence* rather than by Lanczos, reflect with a cotangent rather than a sine, and `ψ`
//! carries a Taylor series about its real zero that has no analogue for `Γ`. See
//! [`PsiReduction`], which both share.

use super::*;

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
///
/// # The flattening was `trigamma`'s truncation, not a floor, and the table could not see it
///
/// That row is flat at `8.5e-15` for shifts of 20, 25 and 30, and the conclusion drawn from it
/// was that the series had reached its limit. It had not. The experiment varied the **shift**
/// and held the **term count** fixed, so it could only ever measure one of the two things that
/// bound the answer — and for `trigamma` the binding one was the series, whose first omitted
/// term `5/(66x¹¹)` is `6.7e-15` relative at `x = 20`. Flat across shifts is exactly what a
/// term-count limit looks like from an experiment that cannot vary the term count.
///
/// Swept against a reference rather than at integers, the shape gives it away: the error *falls*
/// with `x` — `53` ulp at `x = 20`, `7` at `24`, `1` beyond `40` — which no floor does. Two more
/// Bernoulli terms in [`trigamma_positive`] cost two fused multiply-adds and remove it:
///
/// ```text
/// x        [1,4]  [4,8]  [8,12]  [12,16]  [16,20]  [20,24]  [24,40]
/// was          7     15      28       27       53       53        7   ulp
/// now          4      3       2        2        2        1        1
/// ```
///
/// `digamma`'s own omitted term is `1/(132x¹⁰)`, which is `7.6e-16` at `x = 20` — already under
/// `ε`, which is why only one of the two needed extending and why the original threshold work
/// was right about `digamma`.
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
/// [`f32::gamma`](crate::kernels::f32::gamma) carries the same vocabulary as `PsiReduction`, with
/// one arm fewer: it reduces onto `[1, 2)` with a factored polynomial where this one has a Taylor
/// series about `ψ`'s real zero.
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
    result
        + r
        + 0.5 * r2
        + r2 * r
            * (1.0 / 6.0
                - r2 * (1.0 / 30.0
                    - r2 * (1.0 / 42.0
                        - r2 * (1.0 / 30.0 - r2 * (5.0 / 66.0 - r2 * (691.0 / 2730.0))))))
}

#[cfg(test)]
mod tests;
