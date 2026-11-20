//! Unit tests for the single-precision incomplete gamma pair.
//!
//! Beside [`super`] rather than with `Γ`'s tests, because what they reach is this module's: the
//! small-shape logarithm, its term count and the upper fraction. Both small-shape routes are
//! checked against the `f64` kernel, which is the only available truth for them.

use super::*;

/// Cross-check the native `f32` result against the `f64` reference, narrowed — the guard
/// against the two implementations silently drifting apart.
fn agrees(a: f32, b: f64, tol: f32) {
    let b = b as f32;
    assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
}

#[test]
fn incomplete_gamma_matches_f64_reference() {
    for &(a, x) in &[(1.0, 0.5), (2.5, 3.0), (5.0, 2.0), (10.0, 12.0)] {
        let p = gamma_lr(a, x);
        let q = gamma_ur(a, x);
        agrees(p, crate::kernels::gamma::gamma_lr(a as f64, x as f64), 1e-5);
        agrees(p + q, 1.0, 1e-6);
    }
}

/// The small-shape lower-tail series, whose value is a logarithm of a probability.
///
/// `exp` of it is `P(a, x)` exactly — that is how `gamma_ur` uses it — so the `f64` kernel is a
/// reference for every term. Eight mutants lived in here because the terms past the first are
/// each below a whole-function tolerance on `Q`.
#[test]
fn the_small_shape_series_is_the_log_of_the_lower_tail() {
    for (a, x) in [
        (SMALL_SHAPE, 0.5_f32),
        (SMALL_SHAPE, 0.9),
        (1e-3, 0.25),
        (1e-3, 1.0),
        (1e-5, 0.5),
        (1e-7, 0.75),
    ] {
        let got = small_shape_log_lower(a, x);
        let wide = crate::kernels::gamma::gamma_lr(f64::from(a), f64::from(x)).ln() as f32;
        assert!(
            (got - wide).abs() <= 2e-5 * wide.abs().max(1.0),
            "small_shape_log_lower({a:e}, {x}): {got} vs {wide}"
        );
    }
    // The `ln1p(v)/v → 1` guard, reached where `v = a·sum` falls under the format's resolution.
    // Both arms have to agree there, which is the whole reason the cutoff is safe.
    let (a, x) = (0.5_f32, 1e-10_f32);
    let got = small_shape_log_lower(a, x);
    assert!(got.is_finite(), "{got}");
    let wide = crate::kernels::gamma::gamma_lr(f64::from(a), f64::from(x)).ln() as f32;
    assert!((got - wide).abs() <= 1e-4 * wide.abs(), "{got} vs {wide}");
    // And the series is alternating in `x`, so the sum is bounded by its first term.
    assert!(small_shape_log_lower(1e-3, 0.5) < 0.0);
}

/// The small-shape **fraction** arm of the upper tail, which only a tiny shape with a large
/// argument reaches.
///
/// Four mutants lived in its exponent: `a ≤ 0.01` together with `x ≥ a + 1` is a corner no
/// whole-function sweep in this crate visits, and the arm's whole purpose is that the tiny shape
/// multiplies in *after* the fraction so the probability takes a single subnormal rounding.
#[test]
fn the_small_shape_upper_fraction_tracks_the_wide_kernel() {
    for (a, x) in [
        (SMALL_SHAPE, 2.0_f32),
        (SMALL_SHAPE, 5.0),
        (1e-3, 2.0),
        (1e-5, 10.0),
        (1e-7, 3.0),
    ] {
        assert_eq!(
            IncompleteGammaMethod::for_arguments(a, x),
            IncompleteGammaMethod::SmallShapeFraction,
            "at a={a:e}, x={x}"
        );
        let got = gamma_ur(a, x);
        let wide = crate::kernels::gamma::gamma_ur(f64::from(a), f64::from(x)) as f32;
        assert!(
            (got - wide).abs() <= 2e-5 * wide.abs().max(1e-30),
            "gamma_ur({a:e}, {x}): {got:e} vs {wide:e}"
        );
        // A tiny shape puts almost all the mass below `x`, so the upper tail is small and
        // strictly positive rather than rounded away.
        assert!(got > 0.0 && got < 1e-2, "{got:e}");
    }
}
