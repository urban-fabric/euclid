//! Modified **Bessel functions of the first kind** `I₀`/`I₁`, their exponentially-scaled forms
//! `Iₙᵉ(x) = e^{−|x|} Iₙ(x)`, the ratio `I₁/I₀`, and `ln I₀`.
//!
//! # Method: a peak-factored power series
//!
//! These previously used the Abramowitz & Stegun polynomial approximations, which are accurate
//! to only about 1e-7 *by construction* — roughly `f32` resolution out of an `f64` function,
//! and eight orders of magnitude weaker than the gamma and beta families beside them.
//!
//! The defining series is used instead, which has two properties that make it the better
//! choice:
//!
//! - **Every term is positive**, so there is no cancellation at any argument.
//! - **The terms peak sharply.** For `I₀`, `tₖ = (x/2)^{2k}/(k!)²` is maximal near `k = x/2`
//!   with a Gaussian-like width of `O(√x)`, so only `O(√x)` terms matter.
//!
//! The one obstacle is scale: the sum is about `e^x/√(2πx)`, which overflows `f64` past
//! `x ≈ 710` even though the *scaled* result `I₀ᵉ` is a small, perfectly representable number.
//! So the peak term is factored out in the log domain and the rest accumulated **relative to
//! it**, walking outward by ratio:
//!
//! ```text
//! tₖ₊₁/tₖ = (x/2)² / (k+1)²        tₖ₋₁/tₖ = k² / (x/2)²
//! ```
//!
//! Every accumulated value is then at most 1, the sum is `O(√x)` well-conditioned additions,
//! and a single `exp(ln t_peak − |x|)` at the end carries the scale.
//!
//! The result is full double precision at any argument, with no approximation coefficients to
//! transcribe and no large-argument expansion to join on — the old branch switch at
//! `|x| = 3.75`, and the join error there, are gone. The cost is `O(√x)` multiply-adds rather
//! than a fixed-length polynomial.

use super::gamma::ln_gamma;

/// Cap on the outward walk in either direction from the peak. The significant terms span
/// `O(√x)` indices, so this covers arguments past `1e8` before it binds.
const MAX_TERMS: u32 = 100_000;

/// Relative size at which an outward term stops contributing.
const TERM_EPS: f64 = 1e-22;

/// The `I₀` series summed **relative to its peak term** at index `k_peak`, so the result is
/// `O(√x)` and cannot overflow. Shared by `ln_i0_sum` and the ratio.
fn i0_rel_sum(h: f64, k_peak: f64) -> f64 {
    let mut sum = 1.0f64;

    // Upward: t_{k+1}/t_k = h^2/(k+1)^2.
    let mut ratio = 1.0f64;
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
fn ln_i0_sum(ax: f64) -> f64 {
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
fn i1_rel_sum(h: f64, k_peak: f64) -> f64 {
    let mut sum = 1.0f64;

    // Upward: t_{k+1}/t_k = h^2/((k+1)(k+2)).
    let mut ratio = 1.0f64;
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
fn ln_i1_sum(ax: f64) -> f64 {
    debug_assert!(ax >= 0.0);
    if ax == 0.0 {
        return f64::NEG_INFINITY; // I1(0) = 0
    }
    let h = ax * 0.5;
    let k_peak = h.round().max(0.0);
    let ln_peak = (2.0 * k_peak + 1.0) * h.ln() - ln_gamma(k_peak + 1.0) - ln_gamma(k_peak + 2.0);
    ln_peak + i1_rel_sum(h, k_peak).ln()
}

/// The modified Bessel function `I₀(x)` (even in `x`).
///
/// Overflows for `|x| ≳ 710`, as the true value does; use [`i0e`] or [`ln_i0`] there.
pub(crate) fn i0(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    ln_i0_sum(x.abs()).exp()
}

/// The modified Bessel function `I₁(x)` (odd in `x`).
pub(crate) fn i1(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    let magnitude = ln_i1_sum(x.abs()).exp();
    if x < 0.0 { -magnitude } else { magnitude }
}

/// The exponentially-scaled `I₀ᵉ(x) = e^{−|x|} I₀(x)`, finite for every argument.
pub(crate) fn i0e(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    let ax = x.abs();
    if ax.is_infinite() {
        return 0.0;
    }
    (ln_i0_sum(ax) - ax).exp()
}

/// The exponentially-scaled `I₁ᵉ(x) = e^{−|x|} I₁(x)`.
pub(crate) fn i1e(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
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
pub(crate) fn i1_over_i0(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
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
pub(crate) fn ln_i0(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    let ax = x.abs();
    if ax.is_infinite() {
        return f64::INFINITY;
    }
    ln_i0_sum(ax)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) {
        assert!(
            (a - b).abs() <= tol * b.abs().max(1.0),
            "{a:e} vs {b:e} (tol {tol:e})"
        );
    }

    /// 60-digit `Decimal` reference values for `e^{-x} I0(x)` and `e^{-x} I1(x)`.
    const SCALED: &[(f64, f64, f64)] = &[
        (0.0, 1.0, 0.0),
        (0.5, 6.450_352_704_491_501e-1, 1.564_208_031_848_717e-1),
        (1.0, 4.657_596_075_936_404e-1, 2.079_104_153_497_084e-1),
        (2.0, 3.085_083_225_536_71e-1, 2.152_692_892_489_376e-1),
        (3.74, 2.147_726_486_128_186e-1, 1.831_416_223_205_881e-1),
        (3.76, 2.141_428_724_247_597e-1, 1.827_955_634_912_299e-1),
        (5.0, 1.835_408_126_093_284e-1, 1.639_722_669_445_424e-1),
        (10.0, 1.278_333_371_634_286e-1, 1.212_626_813_844_555e-1),
        (20.0, 8.978_031_188_482_602e-2, 8.750_622_218_328_867e-2),
        (22.1, 8.535_495_638_436_524e-2, 8.340_093_783_969_195e-2),
        (50.0, 5.656_162_664_745_419e-2, 5.599_312_389_289_54e-2),
        (100.0, 3.994_437_929_909_668e-2, 3.974_415_302_513_025e-2),
    ];

    #[test]
    fn the_scaled_forms_now_reach_machine_precision() {
        // The point of the rewrite: these were accurate to ~1e-7 before.
        for &(x, i0_ref, i1_ref) in SCALED {
            close(i0e(x), i0_ref, 1e-13);
            close(i1e(x), i1_ref, 1e-13);
        }
    }

    #[test]
    fn there_is_no_longer_a_branch_point_at_3_75() {
        // The old implementation switched formula there and the two sides disagreed. There is
        // no branch now, so the function is smooth across it.
        let step = 1e-8;
        assert!(
            (i0e(3.75 + step) - i0e(3.75 - step)).abs() < 1e-8,
            "join discontinuity at 3.75"
        );
    }

    #[test]
    fn large_arguments_stay_finite_and_accurate() {
        for &x in &[700.0f64, 1e3, 1e4, 1e5] {
            let v = i0e(x);
            assert!(v > 0.0 && v.is_finite(), "i0e({x}) = {v}");
            // I0e(x) ~ (1 + 1/(8x) + 9/(128x^2)) / sqrt(2 pi x)
            let asym = (1.0 + 1.0 / (8.0 * x) + 9.0 / (128.0 * x * x))
                / (core::f64::consts::TAU * x).sqrt();
            close(v, asym, 1e-8);
        }
    }

    #[test]
    fn unscaled_forms_match_the_scaled_ones() {
        for &(x, _, _) in SCALED.iter().take(8) {
            close(i0(x), i0e(x) * x.exp(), 1e-13);
            if x > 0.0 {
                close(i1(x), i1e(x) * x.exp(), 1e-13);
            }
        }
    }

    #[test]
    fn ln_i0_is_consistent_and_overflow_free() {
        for &(x, i0_ref, _) in SCALED {
            close(ln_i0(x), i0_ref.ln() + x, 1e-13);
        }
        // Far past where I0 itself is representable.
        for &x in &[800.0f64, 1e4, 1e6] {
            let v = ln_i0(x);
            assert!(v.is_finite() && v > 0.0, "ln_i0({x}) = {v}");
            close(v, x - 0.5 * (core::f64::consts::TAU * x).ln(), 1e-6);
        }
    }

    #[test]
    fn the_ratio_is_bounded_monotone_and_odd() {
        assert_eq!(i1_over_i0(0.0), 0.0);
        let mut prev = 0.0f64;
        for i in 1..2000 {
            let x = f64::from(i) * 0.5;
            let r = i1_over_i0(x);
            assert!((0.0..1.0).contains(&r), "i1_over_i0({x}) = {r}");
            assert!(r >= prev, "the ratio must increase: {prev} then {r}");
            prev = r;
        }
        assert!(i1_over_i0(1e5) > 0.99999);
        close(i1_over_i0(-3.0), -i1_over_i0(3.0), 1e-15);
        for &(x, _, _) in SCALED.iter().skip(1) {
            close(i1_over_i0(x), i1e(x) / i0e(x), 1e-13);
        }
    }

    #[test]
    fn the_derivative_relation_holds() {
        // d/dx ln I0(x) = I1(x)/I0(x). Now that both sides are accurate to ~1e-14 rather
        // than ~1e-7, this can be checked at a far tighter tolerance than before.
        for i in 1..40 {
            let x = f64::from(i) * 0.5;
            let h = 1e-5 * x;
            let d = (ln_i0(x + h) - ln_i0(x - h)) / (2.0 * h);
            close(d, i1_over_i0(x), 1e-9);
        }
    }

    #[test]
    fn parity_and_non_finite_inputs() {
        for &(x, _, _) in SCALED.iter().skip(1) {
            close(i0e(-x), i0e(x), 1e-15);
            close(i1e(-x), -i1e(x), 1e-15);
        }
        assert!(i0(f64::NAN).is_nan());
        assert!(i1(f64::NAN).is_nan());
        assert_eq!(i0e(f64::INFINITY), 0.0);
        assert_eq!(i1e(f64::INFINITY), 0.0);
        assert_eq!(ln_i0(f64::INFINITY), f64::INFINITY);
    }
}
