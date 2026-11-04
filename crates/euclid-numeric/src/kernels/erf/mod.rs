//! The error function family — `erf`, `erfc`, `erf_inv` — behind the Normal CDF
//! (`Φ(x) = ½ erfc(-x/√2)`) and quantile. `erf`/`erfc` reduce to the regularized incomplete
//! gamma `P(½, x²)`; `erf_inv` uses Giles's rational approximation plus one Newton step.
//!
//! The single-precision peer is the child [`f32`](mod@crate::kernels::erf::f32) module rather
//! than a file in a parallel `kernels/f32/` tree. Its thresholds are this file's derivations
//! re-evaluated at `f32`'s epsilon, so the two belong within one directory of each other; see
//! [`crate::kernels`] for why the rest of the tree has not been folded this way yet.

pub(crate) mod f32;

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

use super::gamma::gamma_lr;
use core::f64::consts::PI;

/// The error function `erf(x) = (2/√π) ∫₀ˣ e^{-t²} dt`.
pub(crate) fn erf(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    // erf saturates: the limits are exactly +/-1, not NaN.
    if x.is_infinite() {
        return if x > 0.0 { 1.0 } else { -1.0 };
    }
    if x.abs() < ERF_LINEAR_MAX {
        return 2.0 / PI.sqrt() * x;
    }
    if x > 0.0 {
        gamma_lr(0.5, x * x)
    } else {
        -gamma_lr(0.5, x * x)
    }
}

/// Lower edge of the continued-fraction regime, for both `erfc` and `erfcx`.
///
/// # Why it is 2 and not 4
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
/// So the old threshold left a band `x ∈ [2, 4]` where the worse of the two routes was in use,
/// peaking at `1.2e-9` just below the handover — six orders off machine precision, in a range
/// the Gaussian tail functions use constantly.
///
/// The fraction, at its existing 60 terms, is already at machine precision from `x = 1.8`
/// (`8.0e-16`), so no extra depth is needed; it is only unusable below about `1.5`, where it has
/// not converged. Moving the join to `2.0` takes the worst case over the whole axis from `1.2e-9`
/// to `≈4.6e-15`, which is then the `1 − erf` floor just under the join and irreducible without a
/// separate small-argument series for `erfc` itself.
///
/// `2.0` is also what the `f32` peer already used — the single-precision port sited this
/// correctly and the `f64` original was the outlier.
const ERFCX_CF_MIN: f64 = 2.0;

/// The complementary error function `erfc(x) = 1 - erf(x)`.
///
/// The naive `1 - erf(x)` cancels catastrophically in the right tail — by `x ≈ 6`
/// `erf` rounds to exactly `1.0` and `erfc` would underflow to `0`. For `x ≥ ERFCX_CF_MIN`
/// we instead evaluate the algebraically equal `erfc(x) = e^{-x²}·erfcx(x)` through the
/// continued fraction, which keeps full precision down to the underflow of `e^{-x²}`.
pub(crate) fn erfc(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return if x > 0.0 { 0.0 } else { 2.0 };
    }
    if x >= ERFCX_CF_MIN {
        return (-x * x).exp() * erfcx_cf(x);
    }
    1.0 - erf(x)
}

/// Continued-fraction core of `erfcx` for `x ≥ ERFCX_CF_MIN`, depending only on `x`
/// (not on `erfc`), so `erfc` and `erfcx` can both call it without mutual recursion.
fn erfcx_cf(x: f64) -> f64 {
    let mut t = x;
    for k in (1..=60).rev() {
        t = x + (k as f64) * 0.5 / t;
    }
    1.0 / (PI.sqrt() * t)
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
        return if x > 0.0 { 0.0 } else { f64::INFINITY };
    }
    if x < 0.0 {
        return 2.0 * (x * x).exp() - erfcx(-x);
    }
    if x < ERFCX_CF_MIN {
        // `x < ERFCX_CF_MIN`, so `erfc(x)` here takes its stable `1 - erf` branch.
        return (x * x).exp() * erfc(x);
    }
    erfcx_cf(x)
}

/// The inverse error function: `erf_inv(erf(x)) = x` for `x ∈ (-1, 1)`.
pub(crate) fn erf_inv(x: f64) -> f64 {
    if x <= -1.0 {
        return f64::NEG_INFINITY;
    }
    if x >= 1.0 {
        return f64::INFINITY;
    }
    if x == 0.0 {
        return 0.0;
    }
    let w = -((1.0 - x) * (1.0 + x)).ln();
    let mut p;
    if w < 5.0 {
        let w = w - 2.5;
        p = 2.810_226_36e-08;
        p = 3.432_739_39e-07 + p * w;
        p = -3.523_387_7e-06 + p * w;
        p = -4.391_506_54e-06 + p * w;
        p = 0.000_218_580_87 + p * w;
        p = -0.001_253_725_03 + p * w;
        p = -0.004_177_681_64 + p * w;
        p = 0.246_640_727 + p * w;
        p = 1.501_409_41 + p * w;
    } else {
        let w = w.sqrt() - 3.0;
        p = -0.000_200_214_257;
        p = 0.000_100_950_24 + p * w;
        p = 0.001_349_343_22 + p * w;
        p = -0.003_673_428_4 + p * w;
        p = 0.005_739_507_73 + p * w;
        p = -0.007_622_461_3 + p * w;
        p = 0.009_438_870_47 + p * w;
        p = 1.001_674_06 + p * w;
        p = 2.832_976_82 + p * w;
    }
    let mut result = p * x;
    let err = erf(result) - x;
    result -= err / (2.0 / PI.sqrt() * (-result * result).exp());
    result
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol, "{a} vs {b}");
    }

    #[test]
    fn erf_known_values() {
        close(erf(0.0), 0.0, 1e-12);
        close(erf(1.0), 0.842_700_792_949_715, 1e-10);
        close(erf(-1.0), -0.842_700_792_949_715, 1e-10);
        close(erf(2.0), 0.995_322_265_018_953, 1e-10);
        close(erfc(0.0), 1.0, 1e-12);
    }

    #[test]
    fn erf_inv_round_trips() {
        for &x in &[-0.9, -0.5, -0.1, 0.1, 0.3, 0.5, 0.8, 0.95] {
            close(erf(erf_inv(x)), x, 1e-10);
        }
    }

    #[test]
    fn erfc_tail_does_not_underflow() {
        // Regression: the naive `1 - erf` underflows to exactly 0 by x ≈ 6 in f64;
        // the tail now routes through `e^{-x²}·erfcx`, recovering the true value.
        let known: &[(f64, f64)] = &[
            (5.0, 1.537_459_794_428_034e-12),
            (6.0, 2.151_973_671_249_891e-17),
            (10.0, 2.088_487_583_762_545e-45),
        ];
        for &(x, want) in known {
            let got = erfc(x);
            assert!(got > 0.0, "erfc({x}) underflowed to {got}");
            assert!(
                (got - want).abs() <= 1e-3 * want,
                "erfc({x}) = {got}, want {want}"
            );
        }
    }
}
