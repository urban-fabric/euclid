//! Native `f32` error-function family — the single-precision port of its parent [`super`],
//! which it sits beside rather than in a mirrored tree. `erf`/`erfc` reduce to the regularized
//! incomplete gamma `P(½, x²)`; `erf_inv` uses Giles's rational approximation plus one Newton
//! step. Evaluated in `f32` throughout.
//!
//! Keeping the two precisions in one directory is the point of the layout: the thresholds below
//! are the *same derivation* as the `f64` ones with a different `ε`, so a change to one is
//! almost always a change to both, and a reader comparing them should not have to cross the
//! tree to do it. [`crate::kernels::f32`] re-exports this module so the call paths stay uniform
//! with the families that have not been folded yet.

/// Below this argument `erf` is its own leading term, `2x/√π`.
///
/// Not an optimisation but a correctness requirement, because `erf` is reached through
/// `gamma_lr(½, x²)` and `x²` leaves the format long before `x` does. At `x = 1e-154` the
/// square is already subnormal and carries only a few digits — `erf(1e-160)` was wrong by
/// `5.6e-6` — and by `x = 1.6e-162` the square is exactly zero, so `erf` returned **zero**
/// where the true value `1.13e-162` is a perfectly ordinary number.
///
/// The series is `erf(x) = (2/√π)(x − x³/3 + ⋯)`, so dropping everything past the linear term
/// costs `x²/3` relative. At the threshold that is ``3.3e-9``, below half an ulp, making the
/// linear form correctly rounded rather than merely adequate.
///
/// Taking `x = 0` through this path as well is what makes `erf(−0.0)` return `−0.0`: the old
/// `if x == 0.0 { 0.0 }` dropped the sign of the zero, which an odd function should not.
const ERF_LINEAR_MAX: f32 = 1e-4;

use crate::kernels::f32::gamma::gamma_lr;
use core::f32::consts::PI;

/// The error function `erf(x) = (2/√π) ∫₀ˣ e^{-t²} dt`.
pub(crate) fn erf(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
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

/// Lower edge of the continued-fraction regime. Lower than the `f64` cutoff (`4.0`)
/// because `f32`'s `1 - erf` cancellation sets in much earlier — by `x ≈ 3.5` it has
/// lost most of its bits, and by `x ≈ 4` `erf` rounds to `1.0` so `erfc` underflows to
/// `0`. The 60-term backward continued fraction is already accurate from `2.0` upward.
const ERFCX_CF_MIN: f32 = 2.0;

/// The complementary error function `erfc(x) = 1 - erf(x)`.
///
/// For `x ≥ ERFCX_CF_MIN` the `1 - erf` form cancels (in `f32` it underflows to `0`
/// by `x ≈ 4`), so there we evaluate the equal, stable `erfc(x) = e^{-x²}·erfcx(x)`.
pub(crate) fn erfc(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
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
fn erfcx_cf(x: f32) -> f32 {
    let mut t = x;
    for k in (1..=60).rev() {
        t = x + (k as f32) * 0.5 / t;
    }
    1.0 / (PI.sqrt() * t)
}

/// The scaled complementary error function `erfcx(x) = e^{x²}·erfc(x)`, finite in the right
/// tail where `erfc` underflows. A continued fraction carries the large-`x` regime.
pub(crate) fn erfcx(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    if x.is_infinite() {
        return if x > 0.0 { 0.0 } else { f32::INFINITY };
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
pub(crate) fn erf_inv(x: f32) -> f32 {
    if x <= -1.0 {
        return f32::NEG_INFINITY;
    }
    if x >= 1.0 {
        return f32::INFINITY;
    }
    if x == 0.0 {
        return 0.0;
    }
    let w = -((1.0 - x) * (1.0 + x)).ln();
    let mut p;
    if w < 5.0 {
        let w = w - 2.5;
        p = 2.810_226_4e-8;
        p = 3.432_739_4e-7 + p * w;
        p = -3.523_387_7e-06 + p * w;
        p = -4.391_506_5e-6 + p * w;
        p = 0.000_218_580_87 + p * w;
        p = -0.001_253_725 + p * w;
        p = -0.004_177_681_6 + p * w;
        p = 0.246_640_73 + p * w;
        p = 1.501_409_4 + p * w;
    } else {
        let w = w.sqrt() - 3.0;
        p = -0.000_200_214_26;
        p = 0.000_100_950_24 + p * w;
        p = 0.001_349_343_2 + p * w;
        p = -0.003_673_428_4 + p * w;
        p = 0.005_739_507_7 + p * w;
        p = -0.007_622_461_3 + p * w;
        p = 0.009_438_870_5 + p * w;
        p = 1.001_674 + p * w;
        p = 2.832_976_8 + p * w;
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

    fn agrees(a: f32, b: f64, tol: f32) {
        let b = b as f32;
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    }

    #[test]
    fn erf_matches_f64_reference() {
        for &x in &[-2.0, -1.0, -0.3, 0.0, 0.3, 1.0, 2.0, 3.0] {
            agrees(erf(x), crate::kernels::erf::erf(x as f64), 1e-5);
            agrees(erfc(x), crate::kernels::erf::erfc(x as f64), 1e-5);
        }
        for &x in &[-3.0, -0.5, 0.5, 2.0, 6.0] {
            agrees(erfcx(x), crate::kernels::erf::erfcx(x as f64), 1e-4);
        }
    }

    #[test]
    fn erf_inv_round_trips() {
        for &x in &[-0.9, -0.5, -0.1, 0.1, 0.3, 0.5, 0.8, 0.95] {
            agrees(erf(erf_inv(x)), x as f64, 1e-5);
        }
    }

    #[test]
    fn erfcx_tail_is_accurate() {
        // Regression: the old `x < 4.0` threshold (copied from f64) routed [≈3.5, 4)
        // through the catastrophic `1 - erf` cancellation, where the relative error
        // reached 71% at 3.9 and 100% (returned 0.0) at 3.99. The lowered threshold
        // sends that zone through the continued fraction instead.
        for &x in &[1.5f32, 2.0, 2.5, 3.0, 3.5, 3.9, 3.99, 4.0, 5.0, 8.0] {
            let got = erfcx(x);
            let reference = crate::kernels::erf::erfcx(x as f64);
            let rel = ((got as f64) - reference).abs() / reference.abs();
            assert!(
                rel < 1e-5,
                "erfcx({x}) = {got}, ref {reference}, rel {rel:.2e}"
            );
        }
    }

    #[test]
    fn erfc_tail_does_not_underflow() {
        // Regression: `1 - erf` underflowed to exactly 0 by x ≈ 4 in f32; the tail now
        // routes through `e^{-x²}·erfcx`, staying positive and accurate to the
        // underflow of `e^{-x²}` itself.
        for &x in &[3.0f32, 3.5, 4.0, 5.0, 6.0, 8.0] {
            let got = erfc(x);
            let reference = crate::kernels::erf::erfc(x as f64);
            assert!(got > 0.0, "erfc({x}) underflowed to {got}");
            let rel = ((got as f64) - reference).abs() / reference.abs();
            assert!(
                rel < 1e-4,
                "erfc({x}) = {got}, ref {reference}, rel {rel:.2e}"
            );
        }
    }
}
