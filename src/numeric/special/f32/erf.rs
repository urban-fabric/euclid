//! Native `f32` error-function family — the single-precision port of [`mod@super::super::erf`].
//! `erf`/`erfc` reduce to the regularized incomplete gamma `P(½, x²)`; `erf_inv` uses Giles's
//! rational approximation plus one Newton step. Evaluated in `f32` throughout.

use super::gamma::gamma_lr;
use core::f32::consts::PI;

/// The error function `erf(x) = (2/√π) ∫₀ˣ e^{-t²} dt`.
pub(crate) fn erf(x: f32) -> f32 {
    if x == 0.0 {
        0.0
    } else if x > 0.0 {
        gamma_lr(0.5, x * x)
    } else {
        -gamma_lr(0.5, x * x)
    }
}

/// The complementary error function `erfc(x) = 1 - erf(x)`.
pub(crate) fn erfc(x: f32) -> f32 {
    1.0 - erf(x)
}

/// The scaled complementary error function `erfcx(x) = e^{x²}·erfc(x)`, finite in the right
/// tail where `erfc` underflows. A continued fraction carries the large-`x` regime.
pub(crate) fn erfcx(x: f32) -> f32 {
    if x < 0.0 {
        return 2.0 * (x * x).exp() - erfcx(-x);
    }
    if x < 4.0 {
        return (x * x).exp() * erfc(x);
    }
    let mut t = x;
    for k in (1..=60).rev() {
        t = x + (k as f32) * 0.5 / t;
    }
    1.0 / (PI.sqrt() * t)
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
            agrees(erf(x), crate::numeric::special::erf::erf(x as f64), 1e-5);
            agrees(erfc(x), crate::numeric::special::erf::erfc(x as f64), 1e-5);
        }
        for &x in &[-3.0, -0.5, 0.5, 2.0, 6.0] {
            agrees(
                erfcx(x),
                crate::numeric::special::erf::erfcx(x as f64),
                1e-4,
            );
        }
    }

    #[test]
    fn erf_inv_round_trips() {
        for &x in &[-0.9, -0.5, -0.1, 0.1, 0.3, 0.5, 0.8, 0.95] {
            agrees(erf(erf_inv(x)), x as f64, 1e-5);
        }
    }
}
