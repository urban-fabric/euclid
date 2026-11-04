//! Single-precision peer of [`super::super::bessel`]. Same peak-factored power series; see
//! that module for why it replaced the Abramowitz & Stegun polynomials.
//!
//! The term cutoff is loosened to the `f32` noise floor, which shortens the outward walk.

use super::gamma::ln_gamma;

/// Cap on the outward walk in either direction from the peak. The significant terms span
/// `O(√x)` indices, so this covers arguments past `1e8` before it binds.
const MAX_TERMS: u32 = 100_000;

/// Relative size at which an outward term stops contributing.
const TERM_EPS: f32 = 1e-12;

/// The `I₀` series summed **relative to its peak term** at index `k_peak`, so the result is
/// `O(√x)` and cannot overflow. Shared by `ln_i0_sum` and the ratio.
fn i0_rel_sum(h: f32, k_peak: f32) -> f32 {
    let mut sum = 1.0f32;

    // Upward: t_{k+1}/t_k = h^2/(k+1)^2.
    let mut ratio = 1.0f32;
    let mut k = k_peak;
    for _ in 0..MAX_TERMS {
        let step = h / (k + 1.0);
        ratio *= step * step;
        sum += ratio;
        k += 1.0;
        if ratio < sum * TERM_EPS {
            break;
        }
    }

    // Downward: t_{k-1}/t_k = k^2/h^2.
    ratio = 1.0;
    k = k_peak;
    for _ in 0..MAX_TERMS {
        if k < 1.0 {
            break;
        }
        let step = k / h;
        ratio *= step * step;
        k -= 1.0;
        sum += ratio;
        if ratio < sum * TERM_EPS {
            break;
        }
    }

    sum
}

/// `ln I₀(|x|)`.
fn ln_i0_sum(ax: f32) -> f32 {
    debug_assert!(ax >= 0.0);
    if ax == 0.0 {
        return 0.0; // I0(0) = 1
    }
    let h = ax * 0.5;
    let k_peak = h.round().max(0.0);
    let ln_peak = 2.0 * k_peak * h.ln() - 2.0 * ln_gamma(k_peak + 1.0);
    ln_peak + i0_rel_sum(h, k_peak).ln()
}

/// The `I₁` series summed relative to its peak term at the same index.
fn i1_rel_sum(h: f32, k_peak: f32) -> f32 {
    let mut sum = 1.0f32;

    // Upward: t_{k+1}/t_k = h^2/((k+1)(k+2)).
    let mut ratio = 1.0f32;
    let mut k = k_peak;
    for _ in 0..MAX_TERMS {
        ratio *= (h / (k + 1.0)) * (h / (k + 2.0));
        sum += ratio;
        k += 1.0;
        if ratio < sum * TERM_EPS {
            break;
        }
    }

    // Downward: t_{k-1}/t_k = k(k+1)/h^2.
    ratio = 1.0;
    k = k_peak;
    for _ in 0..MAX_TERMS {
        if k < 1.0 {
            break;
        }
        ratio *= (k / h) * ((k + 1.0) / h);
        k -= 1.0;
        sum += ratio;
        if ratio < sum * TERM_EPS {
            break;
        }
    }

    sum
}

/// `ln I₁(|x|)`.
fn ln_i1_sum(ax: f32) -> f32 {
    debug_assert!(ax >= 0.0);
    if ax == 0.0 {
        return f32::NEG_INFINITY; // I1(0) = 0
    }
    let h = ax * 0.5;
    let k_peak = h.round().max(0.0);
    let ln_peak = (2.0 * k_peak + 1.0) * h.ln() - ln_gamma(k_peak + 1.0) - ln_gamma(k_peak + 2.0);
    ln_peak + i1_rel_sum(h, k_peak).ln()
}

/// The modified Bessel function `I₀(x)` (even in `x`).
///
/// Overflows for `|x| ≳ 710`, as the true value does; use [`i0e`] or [`ln_i0`] there.
pub(crate) fn i0(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    ln_i0_sum(x.abs()).exp()
}

/// The modified Bessel function `I₁(x)` (odd in `x`).
pub(crate) fn i1(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    let magnitude = ln_i1_sum(x.abs()).exp();
    if x < 0.0 { -magnitude } else { magnitude }
}

/// The exponentially-scaled `I₀ᵉ(x) = e^{−|x|} I₀(x)`, finite for every argument.
pub(crate) fn i0e(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    let ax = x.abs();
    if ax.is_infinite() {
        return 0.0;
    }
    (ln_i0_sum(ax) - ax).exp()
}

/// The exponentially-scaled `I₁ᵉ(x) = e^{−|x|} I₁(x)`.
pub(crate) fn i1e(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    let ax = x.abs();
    if ax.is_infinite() {
        return 0.0;
    }
    let magnitude = (ln_i1_sum(ax) - ax).exp();
    if x < 0.0 { -magnitude } else { magnitude }
}

/// `I₁(x) / I₀(x)`, in `(−1, 1)`.
///
/// Both series peak at the same index `k`, and at that index the two peak terms differ by
/// exactly `ln h − ln(k+1)`. So the ratio is
///
/// ```text
/// I₁/I₀ = h/(k+1) · sum₁/sum₀
/// ```
///
/// where both sums are the peak-relative ones. Every quantity here is `O(1)`: the large
/// factors cancel *symbolically* rather than by subtracting two big logarithms. That matters
/// because those logarithms are `O(x)` — at `x = 400` they are around 400 while their
/// difference is around 0.0025, so even a relative error of `1e-7` in each would swamp the
/// answer. This form has no such cancellation at any argument or precision.
pub(crate) fn i1_over_i0(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    let ax = x.abs();
    if ax == 0.0 {
        return 0.0;
    }
    if ax.is_infinite() {
        return if x < 0.0 { -1.0 } else { 1.0 };
    }
    let h = ax * 0.5;
    let k_peak = h.round().max(0.0);
    let ratio = (h / (k_peak + 1.0)) * (i1_rel_sum(h, k_peak) / i0_rel_sum(h, k_peak));
    if x < 0.0 { -ratio } else { ratio }
}

/// `ln I₀(x)`, the overflow-free log of `I₀`.
///
/// Taken straight from the peak-factored sum, so it needs no `e^{|x|}` round trip and stays
/// accurate for every argument.
pub(crate) fn ln_i0(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    let ax = x.abs();
    if ax.is_infinite() {
        return f32::INFINITY;
    }
    ln_i0_sum(ax)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn agrees(a: f32, b: f64, tol: f32) {
        let b = b as f32;
        assert!(
            (a - b).abs() <= tol * b.abs().max(1.0),
            "{a:e} vs {b:e} (tol {tol:e})"
        );
    }

    #[test]
    fn the_scaled_forms_match_the_f64_reference_to_single_precision() {
        // The f32 path is now limited by f32 itself rather than by a 1e-7 approximation, so
        // it tracks the f64 implementation to a few ulp instead of coincidentally sharing its
        // error.
        for &(x, i0_ref, i1_ref) in &[
            (
                0.5f32,
                6.450_352_704_491_501e-1f64,
                1.564_208_031_848_717e-1f64,
            ),
            (1.0, 4.657_596_075_936_404e-1, 2.079_104_153_497_084e-1),
            (3.74, 2.147_726_486_128_186e-1, 1.831_416_223_205_881e-1),
            (3.76, 2.141_428_724_247_597e-1, 1.827_955_634_912_299e-1),
            (10.0, 1.278_333_371_634_286e-1, 1.212_626_813_844_555e-1),
            (50.0, 5.656_162_664_745_419e-2, 5.599_312_389_289_54e-2),
        ] {
            agrees(i0e(x), i0_ref, 1e-5);
            agrees(i1e(x), i1_ref, 1e-5);
        }
    }

    #[test]
    fn large_arguments_stay_finite_in_f32() {
        for &x in &[50.0f32, 100.0, 1e3, 1e4] {
            let v = i0e(x);
            assert!(v > 0.0 && v.is_finite(), "i0e({x}) = {v}");
            let asym = 1.0 / (core::f32::consts::TAU * x).sqrt();
            assert!((v - asym).abs() < 1e-2 * asym);
        }
        // ln I0 stays finite far past where I0 is representable in f32.
        assert!(ln_i0(1e4f32).is_finite());
    }

    #[test]
    fn the_ratio_is_bounded_and_monotone() {
        let mut prev = 0.0f32;
        for i in 1..400 {
            let x = (i as f32) * 0.5;
            let r = i1_over_i0(x);
            assert!((0.0..1.0).contains(&r), "i1_over_i0({x}) = {r}");
            assert!(r >= prev - 1e-6);
            prev = r;
        }
    }
}
