//! The error function family — `erf`, `erfc`, `erf_inv` — behind the Normal CDF
//! (`Φ(x) = ½ erfc(-x/√2)`) and quantile. `erf` is an all-positive series below
//! [`ERFCX_CF_MIN`] and `1 − erfc` above it; `erfc` is `1 − erf` below and a continued fraction
//! above; `erf_inv` uses Giles's rational approximation plus Newton steps.
//!
//! **`erf` reduced to `P(½, x²)` until it was measured.** That identity is exact and the route
//! is not: `P`'s prefactor exponentiates a cancelling sum of size `|ln x|`, which cost `erf`
//! about `|ln x|` ulp everywhere — `32` at `x = 1e-8` against a condition number of `1`. See
//! [`erf_series`]. Nothing in this file reaches the incomplete gamma now.
//!
//! The single-precision peer is [`crate::kernels::f32::erf`], in the mirrored tree with every
//! other family. Its thresholds are this file's derivations re-evaluated at `f32`'s epsilon
//! rather than inherited — they are a *different measurement of the same quantity*, which is why
//! each lives beside the code it governs. What the two genuinely share is precision-generic and
//! lives in [`crate::math`].

mod depth;
mod inverse;
mod reflection;

use depth::ErfcxDepth;

pub(crate) use inverse::erf_inv;

/// Below this argument `erf` is its own leading term, `2x/√π`.
///
/// Not an optimisation but a correctness requirement, because `erf` is reached through
/// `gamma_lr(½, x²)` and `x²` leaves the format long before `x` does. At `x = 1e-154` the
/// square is already subnormal and carries only a few digits — `erf(1e-160)` was wrong by
/// `5.6e-6` — and by `x = 1.6e-162` the square is exactly zero, so `erf` returned **zero**
/// where the true value `1.13e-162` is a perfectly ordinary number.
///
/// The series is `erf(x) = (2/√π)(x − x³/3 + ⋯)`, so dropping everything past the linear term
/// costs `x²/3` relative. At the threshold that is ``3.3e-17``, below half an ulp, making the
/// linear form correctly rounded rather than merely adequate.
///
/// Taking `x = 0` through this path as well is what makes `erf(−0.0)` return `−0.0`: the old
/// `if x == 0.0 { 0.0 }` dropped the sign of the zero, which an odd function should not.
const ERF_LINEAR_MAX: f64 = 1e-8;

use core::f64::consts::PI;

/// Which form evaluates `erf` at this magnitude.
///
/// Matched rather than tested as a chain of early returns: the four arms are exhaustive over the
/// line, so a new one cannot be added without the compiler asking what it returns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ErfForm {
    /// Not a number.
    Undefined,
    /// `±∞`, where `erf` saturates at exactly `±1` rather than going undefined.
    Saturated,
    /// `|x| < `[`ERF_LINEAR_MAX`]: `erf` is its own leading term `2x/√π`. Taking `x = 0` through
    /// here is what makes `erf(−0.0)` return `−0.0`, which an odd function should.
    Linear,
    /// `ERF_LINEAR_MAX ≤ |x| < `[`ERFCX_CF_MIN`]: the all-positive series of [`erf_series`].
    Series,
    /// `|x| ≥ `[`ERFCX_CF_MIN`]: `1 − erfc(x)`, with `erfc` on its continued fraction.
    ///
    /// No cancellation worth the name: `erfc` is at most `0.052` here, so subtracting it from
    /// one is an `O(1)` operation on an `O(1)` answer. It is the mirror of [`ErfcForm`]'s own
    /// `Complement` arm, which is usable only in the opposite direction — and the two do not
    /// recurse, because each calls the other strictly across the threshold.
    Complement,
}

impl ErfForm {
    /// The non-finite cases first, then the magnitude.
    fn for_argument(x: f64) -> Self {
        if x.is_nan() {
            Self::Undefined
        } else if x.is_infinite() {
            Self::Saturated
        } else if x.abs() < ERF_LINEAR_MAX {
            Self::Linear
        } else if x.abs() < ERFCX_CF_MIN {
            Self::Series
        } else {
            Self::Complement
        }
    }
}

/// The error function `erf(x) = (2/√π) ∫₀ˣ e^{-t²} dt`.
pub(crate) fn erf(x: f64) -> f64 {
    match ErfForm::for_argument(x) {
        ErfForm::Undefined => f64::NAN,
        ErfForm::Saturated => 1.0_f64.copysign(x),
        ErfForm::Linear => 2.0 / PI.sqrt() * x,
        // `erf` is odd, so the magnitude comes from the series and the sign from the argument.
        ErfForm::Series => erf_series(x.abs()).copysign(x),
        ErfForm::Complement => (1.0 - erfc(x.abs())).copysign(x),
    }
}

/// `erf` by its all-positive series, for `x ∈ [ERF_LINEAR_MAX, ERFCX_CF_MIN)`.
///
/// # Why `erf` no longer goes through the incomplete gamma
///
/// `erf(x) = P(½, x²)` is exact and was the whole implementation. It is also an **exponential of
/// a cancelling sum**: `P`'s prefactor is formed as `a·ln x − x − lnΓ(a)`, and at `a = ½` that
/// is `ln|x| − x² − lnΓ(½)`, whose magnitude is `|ln x|` — `18.4` at `x = 1e-8`. `exp` turns the
/// exponent's absolute error into the answer's relative error, so `erf` carried about `|ln x|`
/// ulp across its whole domain. Measured against a 400-bit reference, log-uniformly within each
/// decade, with the condition number beside it:
///
/// ```text
/// |x|       1e-8…1e-7  1e-6…1e-5  1e-4…1e-3  1e-2…1e-1  1e-1…1
/// was              32         19         18         13       7   ulp
/// kappa             1          1          1          1    0.65
/// ```
///
/// A condition number of `1` means there is nothing to amplify: a correctly rounded `erf` is
/// half an ulp, and the route was thirty times that. The same defect set the floor for `erf_inv`
/// (whose Newton step divides by an `O(1)` derivative, so it inherited the error one for one)
/// and for the uniform expansion in `super::gamma::temme`, which calls `erfc` here.
///
/// # The series
///
/// DLMF 7.6.2, with the Gaussian factored out so every term is positive:
///
/// ```text
/// erf(x) = (2x/√π)·e^{−x²}·Σ_{n≥0} (2x²)^n/(2n+1)!!
/// ```
///
/// Nothing cancels, at any `x` in range, and the only exponential is `e^{−x²}` whose argument is
/// at most `1.9` — so its absolute error is `1.9ε`, not `|ln x|·ε`. The alternating form
/// `Σ(−1)^n x^{2n+1}/(n!(2n+1))` is the one to avoid: its terms reach `e^{x²}` before cancelling
/// back down, which is the same defect in a different coat.
///
/// It is bounded above by [`ERFCX_CF_MIN`] rather than carried further because the term count
/// grows like `x²` and the terms themselves like `e^{x²}`; above the threshold `erfc`'s
/// continued fraction is both accurate and cheaper, and `1 − erfc` is well conditioned there.
fn erf_series(x: f64) -> f64 {
    let square = 2.0 * x * x;
    let mut term = 1.0;
    let mut sum = 1.0;
    for n in 1..ERF_SERIES_MAX_TERMS {
        term *= square / (2.0 * f64::from(n) + 1.0);
        sum += term;
        if term <= f64::EPSILON * sum {
            break;
        }
    }
    2.0 / PI.sqrt() * x * (-x * x).exp() * sum
}

/// Terms [`erf_series`] is allowed before it gives up.
///
/// The ratio `2x²/(2n+1)` falls below a half once `n > 2x²`, so at the top of the range
/// (`2x² = 3.78`) the tail is geometric from about the fourth term and the sum is converged by
/// the twenty-fifth. Forty is that with headroom; the loop exits on its own well before.
const ERF_SERIES_MAX_TERMS: u32 = 40;

/// Lower edge of the continued-fraction regime, for both `erfc` and `erfcx`.
///
/// # Why it is not 4, and no longer 2
///
/// It was `4.0`, on the reasoning that below that point `1 − erf` had not yet "lost too many
/// bits, `erf` having rounded to `1.0`". That describes the wrong failure. `erf` does not round
/// to exactly `1.0` until about `x = 5.9` — but `erfc = 1 − erf` is a subtraction whose
/// *relative* error is `≈ ε/erfc(x)`, and `erfc` shrinks fast, so the loss arrives far earlier
/// and grows smoothly rather than switching on. Measured against an independent 45-digit
/// reference:
///
/// | `x` | 1.0 | 1.5 | 2.0 | 2.5 | 3.0 | 3.5 | 3.9 |
/// |---|---|---|---|---|---|---|---|
/// | `1 − erf` | `3.7e-15` | `1.0e-15` | `4.6e-15` | `1.2e-13` | `2.4e-12` | `6.7e-11` | `1.2e-9` |
/// | continued fraction | `2.9e-9` | `1.9e-13` | `2.2e-16` | `2.7e-16` | `3.1e-16` | `1.4e-16` | `2.7e-15` |
///
/// So `4.0` left a band `x ∈ [2, 4]` where the worse of the two routes was in use, peaking at
/// `1.2e-9` just below the handover. Moving the join to `2.0` fixed that.
///
/// # Why `2.0` was still too high, and what the table above got wrong
///
/// The `1 − erf` row is a row of **point samples**, and this route's error is not a smooth
/// function that can be sampled. `erf(x)` is an `O(1)` quantity, so it moves in steps of
/// `ulp(erf(x)) ≈ 1.1e-16`; dividing by a small `erfc(x)` turns each step into a *relative* jump
/// of `1.1e-16/erfc(x)`. The route is therefore a **staircase**, constant over a run of
/// neighbouring `x` and then jumping, and a sampled point says only where in a step it happened
/// to land — `4.6e-15` at `x = 2.0` against a step height of `2.4e-14`. The envelope is the step
/// height, and that is what has to be compared:
///
/// | `x` | 1.65 | 1.68 | **1.6875** | 1.69 | 1.75 | 1.90 | 2.00 |
/// |---|---|---|---|---|---|---|---|
/// | `1 − erf` step height | `5.7e-15` | `6.3e-15` | **`6.4e-15`** | `6.6e-15` | `8.3e-15` | `1.5e-14` | `2.4e-14` |
/// | fraction, 60 terms | `1.1e-14` | `6.4e-15` | **`6.4e-15`** | `5.5e-15` | `1.8e-15` | `2.1e-16` | `<1e-16` |
///
/// The fraction's column is its truncation error, measured in-crate by running the same
/// recurrence to 2,000 terms and differencing — no external reference needed, since the only
/// question is how far 60 terms is from the fraction's own limit.
///
/// The two cross at `27/16`, where both carry about `6.4e-15`, so the join costs nothing in
/// either direction — the same criterion the gamma and beta Edgeworth crossovers use. A dyadic
/// value keeps the switch exactly representable. This takes the worst case over the whole axis
/// from `2.4e-14`, just under the old join, to `6.4e-15`.
///
/// # `1.375`, lowered from `1.6875` once the fraction was given enough levels
///
/// The old figure was placed as "where the two carry equal error" — true, and measured against a
/// fraction evaluated to only 60 levels. Deepening it to 120 (see the loop in `erfcx_fraction`)
/// moves the fraction's own floor down with it, so the balance point moves too.
///
/// Measured against a 100-digit reference — the `erf` series and the `erfcx` fraction evaluated
/// independently in `Decimal`, agreeing to between 15 and 67 digits:
///
/// ```text
/// x        complement   fraction @60   fraction @120
/// 1.00       16.96          --            1617 ulp
/// 1.10        4.36          --              94
/// 1.25        9.52          --            1.36
/// 1.40       15.50          --            0.74
/// 1.60       21.25          --            0.82
/// 1.6875       --          25.56          0.85
/// 1.75         --           8.77          0.00
/// 1.90         --           0.94          0.94
/// ```
///
/// So at 120 levels the fraction is the better route from about `x = 1.25`, where the complement
/// form's `1 − erf` staircase — `≈ ε/(2·erfc(x))`, which is `21` ulp by `x = 1.6` — has already
/// overtaken it. `1.375` keeps the fraction's own cliff at a distance: it is `1.36` ulp at `1.25`
/// and `94` at `1.1`, the same super-exponential falloff `I_ASYMPTOTIC_MIN` has, so the threshold
/// sits above the knee rather than on it.
///
/// Together the two changes take `[1.25, 1.9]` from `4`–`25.6` ulp to under `1.4`.
///
/// # The second step, to `1.0`, needed `erfc`'s negatives restructured first
///
/// Measured with a trustworthy reference — the `erf` series against a 6000-level fraction,
/// agreeing to between 24 and 73 digits — the complement form below here costs:
///
/// ```text
/// x       0.1   0.3   0.5   0.7   0.8   0.9   1.0   1.1   1.2    1.3   1.374
/// ulp    1.12  0.68  4.87  4.75  9.20  14.8  17.0  4.36  26.4   1.40    2.18
/// ```
///
/// `26` ulp at `1.2` is the loosest point left in the family, and the fraction reaches `x = 1.0`
/// at `220` levels — affordable. Dropping the threshold to `1.0` duly takes `[1.0, 1.375]` to
/// under `0.7` ulp.
///
/// It also **breaks `Φ(x) + Φ(−x) = 1`**, which `identities::normal` asserts at one ulp. That law
/// held because both sides took the *same* route: `erfc(t)` was `1 − erf(t)` and `erfc(−t)` is
/// `1 − erf(−t) = 1 + erf(t)`, so the sum is `2` with `erf`'s error cancelling **identically**.
/// Making the positive side independently more accurate removes the cancellation and the sum
/// drifts to one or two ulp.
///
/// # Why it stops at `1.375`, and what the real obstacle is
///
/// Lowering to `1.0` works numerically — `[1.0, 1.375]` measures **under `0.7` ulp** against the
/// `4`–`26.4` the complement form costs there, with a `280`-level bucket. It was tried, and
/// reverted, because it breaks two *identities* rather than any value:
///
/// - `Φ(x) + Φ(−x) = 1`. Fixed, and worth having independently: `erfc` now reflects on the
///   negatives, so that law is exact by construction rather than by the two sides sharing an
///   error. See `erfc`.
/// - `erf(x) + erfc(x) = 1`. **Not** fixable the same way. It holds exactly only while `erfc` *is*
///   `1 − erf`, so every argument moved onto the fraction is an argument where the identity
///   becomes inexact by `erfc`'s own error. The old threshold put that boundary at `1.6875`;
///   lowering it to `1.0` moved `erf_and_erfc_are_complementary`'s failure in with it.
///
/// That is a genuine trade, not an oversight: an independently-computed `erfc` is more accurate
/// *and* less consistent with `erf`, and which matters depends on whether a caller composes the
/// two. Resolving it means deciding that question explicitly — and relaxing the identity's
/// tolerance in the band where the fraction runs — rather than moving a constant.
///
/// Three laws in this crate have now turned out to hold because two errors agreed rather than
/// because either was accurate: this one, the `Φ` reflection, and `B(½,½) = π`. It is worth
/// treating a one-ulp identity here as evidence of shared routing until checked.
///
/// Also measured and deliberately out of reach: `[0.8, 1.0)` stays at `9`–`17` ulp, because the
/// fraction needs `380` levels at `0.75` and `840` at `0.5`. The depth requirement grows faster
/// than the error it would remove.
pub(crate) const ERFCX_CF_MIN: f64 = 1.375;

/// Which of the two algebraically equal forms of `erfc` is evaluated at a given argument.
///
/// Named rather than left as an `if`, because the choice *is* the numerics: the two forms agree
/// in exact arithmetic and fail in opposite directions in floating point, so which one runs is
/// the whole content of [`ERFCX_CF_MIN`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ErfcForm {
    /// `1 − erf(x)`. The subtraction itself is exact — `erf ∈ [½, 1)` here, so Sterbenz applies
    /// — but `erf` is an `O(1)` quantity quantised at `ulp(erf)`, which divided by a small
    /// `erfc` becomes a relative staircase of height `≈1.1e-16/erfc(x)`. Usable only while
    /// `erfc` has not yet shrunk.
    Complement,
    /// `e^{−x²}·erfcx(x)` through the continued fraction. No cancellation anywhere, so it holds
    /// to the underflow of `e^{−x²}` — but it has to have converged, which costs the small-`x`
    /// end.
    Fraction,
}

impl ErfcForm {
    /// The regime `x` falls in. The boundary is where the two carry equal error; see
    /// [`ERFCX_CF_MIN`] for the measurements that place it.
    pub(crate) fn for_argument(x: f64) -> Self {
        if x >= ERFCX_CF_MIN {
            Self::Fraction
        } else {
            Self::Complement
        }
    }
}

/// The complementary error function `erfc(x) = 1 - erf(x)`.
///
/// Evaluated by whichever of [`ErfcForm`]'s two routes is accurate at the argument. The naive
/// `1 - erf(x)` cancels catastrophically in the right tail — by `x ≈ 6` `erf` rounds to exactly
/// `1.0` and `erfc` would underflow to `0` — while the continued fraction needs `x` away from
/// zero to converge.
pub(crate) fn erfc(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return if x.is_sign_positive() { 0.0 } else { 2.0 };
    }
    // Negative arguments **reflect** rather than evaluate: `erfc(−t) = 2 − erfc(t)`, saturating
    // where the subtraction has nothing left to say. Both the reason and the boundary are in
    // [`reflection`], which exists because that boundary is unreachable from here.
    if x < 0.0 {
        return reflection::reflected_erfc(erfc(-x));
    }
    match ErfcForm::for_argument(x) {
        ErfcForm::Complement => 1.0 - erf(x),
        ErfcForm::Fraction => (-x * x).exp() * erfcx_cf(x),
    }
}

/// Continued-fraction core of `erfcx` for `x ≥ ERFCX_CF_MIN`, depending only on `x`
/// (not on `erfc`), so `erfc` and `erfcx` can both call it without mutual recursion.
///
/// The depth is chosen from the argument and the closing association from its own range; both
/// live in [`ErfcxDepth`] and [`crate::math::continued_fraction::reciprocal_of_product`].
fn erfcx_cf(x: f64) -> f64 {
    let mut t = x;
    for k in (1..=ErfcxDepth::for_argument(x).levels()).rev() {
        t = x + f64::from(k) * 0.5 / t;
    }
    crate::math::continued_fraction::reciprocal_of_product(PI.sqrt(), t)
}

/// The scaled complementary error function `erfcx(x) = e^{x²}·erfc(x)`, finite in the right
/// tail where `erfc` underflows — the stable Gaussian/Wald log-CDF kernel. A continued
/// fraction carries the large-`x` regime.
pub(crate) fn erfcx(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        // erfcx(x) ~ 1/(x sqrt(pi)) -> 0 as x -> +inf; the mirrored branch diverges.
        return if x.is_sign_positive() {
            0.0
        } else {
            f64::INFINITY
        };
    }
    if x < 0.0 {
        return 2.0 * (x * x).exp() - erfcx(-x);
    }
    match ErfcForm::for_argument(x) {
        // Below the crossover `erfc` takes its `Complement` branch, which is the stable one
        // there, so scaling it back up is safe.
        ErfcForm::Complement => (x * x).exp() * erfc(x),
        ErfcForm::Fraction => erfcx_cf(x),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests;
