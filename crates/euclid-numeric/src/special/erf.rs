//! The error function family — `erf`, `erfc`, `erf_inv` — behind the Normal CDF
//! (`Φ(x) = ½ erfc(-x/√2)`) and quantile. `erf`/`erfc` reduce to the regularized incomplete
//! gamma `P(½, x²)`; `erf_inv` uses Giles's rational approximation plus one Newton step.

use super::gamma::gamma_lr;
use core::f64::consts::PI;

/// The error function `erf(x) = (2/√π) ∫₀ˣ e^{-t²} dt`.
pub(crate) fn erf(x: f64) -> f64 {
    if x == 0.0 {
        0.0
    } else if x > 0.0 {
        gamma_lr(0.5, x * x)
    } else {
        -gamma_lr(0.5, x * x)
    }
}

/// Lower edge of the continued-fraction regime. At and above this point `1 - erf`
/// has lost too many bits (`erf` has rounded to `1.0`) while the continued fraction
/// has converged, so both `erfc` and `erfcx` switch to the stable scaled form there.
const ERFCX_CF_MIN: f64 = 4.0;

/// The complementary error function `erfc(x) = 1 - erf(x)`.
///
/// The naive `1 - erf(x)` cancels catastrophically in the right tail — by `x ≈ 6`
/// `erf` rounds to exactly `1.0` and `erfc` would underflow to `0`. For `x ≥ ERFCX_CF_MIN`
/// we instead evaluate the algebraically equal `erfc(x) = e^{-x²}·erfcx(x)` through the
/// continued fraction, which keeps full precision down to the underflow of `e^{-x²}`.
pub(crate) fn erfc(x: f64) -> f64 {
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
