//! Native `f32` modified-Bessel family — the single-precision port of
//! [`mod@super::super::bessel`]. Abramowitz & Stegun polynomial approximations (≈ 1e-7 relative
//! accuracy — already at the `f32` precision floor), the scaled forms `Iₙᵉ`, the stable ratio
//! `I₁/I₀`, and `ln I₀`, evaluated in `f32` throughout.

use core::f32::consts::PI;

/// `I₀` large-argument asymptotic polynomial in `t = 3.75/x` (A&S 9.8.2).
fn i0_asymp(t: f32) -> f32 {
    0.398_942_3
        + t * (0.013_285_92
            + t * (0.002_253_19
                + t * (-0.001_575_65
                    + t * (0.009_162_81
                        + t * (-0.020_577_06
                            + t * (0.026_355_37 + t * (-0.016_476_33 + t * 0.003_923_77)))))))
}

/// `I₁` large-argument asymptotic polynomial in `t = 3.75/x` (A&S 9.8.4).
fn i1_asymp(t: f32) -> f32 {
    0.398_942_3
        + t * (-0.039_880_24
            + t * (-0.003_620_18
                + t * (0.001_638_01
                    + t * (-0.010_315_55
                        + t * (0.022_829_67
                            + t * (-0.028_953_12 + t * (0.017_876_54 + t * (-0.004_200_59))))))))
}

/// The modified Bessel function `I₀(x)` (even in `x`).
pub(crate) fn i0(x: f32) -> f32 {
    let ax = x.abs();
    if ax < 3.75 {
        let t = (x / 3.75) * (x / 3.75);
        1.0 + t
            * (3.515_622_9
                + t * (3.089_942_4
                    + t * (1.206_749_2 + t * (0.265_973_2 + t * (0.036_076_8 + t * 0.004_581_3)))))
    } else {
        (ax.exp() / ax.sqrt()) * i0_asymp(3.75 / ax)
    }
}

/// The modified Bessel function `I₁(x)` (odd in `x`).
pub(crate) fn i1(x: f32) -> f32 {
    let ax = x.abs();
    let result = if ax < 3.75 {
        let t = (x / 3.75) * (x / 3.75);
        ax * (0.5
            + t * (0.878_905_94
                + t * (0.514_988_69
                    + t * (0.150_849_34
                        + t * (0.026_587_33 + t * (0.003_015_32 + t * 0.000_324_11))))))
    } else {
        (ax.exp() / ax.sqrt()) * i1_asymp(3.75 / ax)
    };
    if x < 0.0 { -result } else { result }
}

/// The **exponentially-scaled** `I₀ᵉ(x) = e^{−|x|} I₀(x)`, finite for all `x` where
/// `I₀` itself overflows. `ln I₀(x) = ln I₀ᵉ(x) + |x|`.
pub(crate) fn i0e(x: f32) -> f32 {
    let ax = x.abs();
    if ax < 3.75 {
        i0(x) * (-ax).exp()
    } else {
        i0_asymp(3.75 / ax) / ax.sqrt()
    }
}

/// The **exponentially-scaled** `I₁ᵉ(x) = e^{−|x|} I₁(x)` (odd in `x`).
pub(crate) fn i1e(x: f32) -> f32 {
    let ax = x.abs();
    if ax < 3.75 {
        i1(x) * (-ax).exp()
    } else {
        let r = i1_asymp(3.75 / ax) / ax.sqrt();
        if x < 0.0 { -r } else { r }
    }
}

/// The ratio `I₁(x)/I₀(x) ∈ [0, 1)` for `x ≥ 0` (the von Mises mean resultant length).
/// Stable for large `x`: the shared `eˣ/√x` growth factor cancels, so it never overflows.
pub(crate) fn i1_over_i0(x: f32) -> f32 {
    let ax = x.abs();
    if ax < 3.75 {
        i1(ax) / i0(ax)
    } else {
        let t = 3.75 / ax;
        i1_asymp(t) / i0_asymp(t)
    }
}

/// `ln I₀(x)` for `x ≥ 0`, stable for large `x` where `I₀` itself overflows: uses
/// the asymptotic `ln I₀(x) ≈ x − ½ln(2πx) + ln(1 + 1/8x + 9/128x² + …)`.
pub(crate) fn ln_i0(x: f32) -> f32 {
    let ax = x.abs();
    if ax < 15.0 {
        i0(ax).ln()
    } else {
        let r = 1.0 / ax;
        let series = 1.0 + r * (0.125 + r * (0.070_312_5 + r * 0.073_242_19));
        ax - 0.5 * (2.0 * PI * ax).ln() + series.ln()
    }
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
    fn bessel_matches_f64_reference() {
        for &x in &[0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 20.0] {
            agrees(i0(x), crate::special::bessel::i0(x as f64), 1e-5);
            agrees(i1(x), crate::special::bessel::i1(x as f64), 1e-5);
            agrees(i0e(x), crate::special::bessel::i0e(x as f64), 1e-5);
            agrees(i1e(x), crate::special::bessel::i1e(x as f64), 1e-5);
        }
    }

    #[test]
    fn ratio_and_logs_match_f64_reference() {
        for &x in &[0.5, 2.0, 6.0, 100.0] {
            agrees(
                i1_over_i0(x),
                crate::special::bessel::i1_over_i0(x as f64),
                1e-5,
            );
            agrees(ln_i0(x), crate::special::bessel::ln_i0(x as f64), 1e-5);
        }
    }
}
