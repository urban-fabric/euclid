//! Unit tests for the single-precision `K₀`/`K₁`.
//!
//! A child module so implementation plus tests stay inside the crate's 400-line budget, while
//! `use super::*` still reaches the private series and the quadrature's truncation time.

use super::*;
use crate::kernels::bessel::modified_second as reference;
use crate::kernels::f32::bessel::modified_first::{i0e, i1e};
use euclid_utils::assert_close_f32;

/// The native values against the `f64` kernel, at budgets a native path earns.
#[test]
fn the_native_values_track_the_f64_reference() {
    for i in 1..60 {
        let x = (i as f32) * 0.25;
        let wide = f64::from(x);
        assert_close_f32(k0(x), reference::k0(wide), 16.0, "K0");
        assert_close_f32(k1(x), reference::k1(wide), 16.0, "K1");
        assert_close_f32(k0e(x), reference::k0e(wide), 16.0, "K0e");
        assert_close_f32(k1e(x), reference::k1e(wide), 16.0, "K1e");
    }
    // Deep into the tail, where only the scaled forms stay representable.
    for &x in &[20.0f32, 50.0, 88.0, 200.0, 1e4] {
        let wide = f64::from(x);
        assert_close_f32(k0e(x), reference::k0e(wide), 16.0, "K0e tail");
        assert_close_f32(k1e(x), reference::k1e(wide), 16.0, "K1e tail");
    }
}

/// The Wronskian `I₀ᵉ(x)K₁ᵉ(x) + I₁ᵉ(x)K₀ᵉ(x) = 1/x`, tying the two kinds together.
///
/// Worth more than a value comparison: it holds only if both families are right *and* wired
/// to each other correctly, and the scaled forms make it checkable across the whole axis
/// rather than only where `K` itself is representable.
#[test]
fn the_wronskian_holds_against_the_native_first_kind() {
    for i in 1..80 {
        let x = (i as f32) * 0.5;
        let w = i0e(x) * k1e(x) + i1e(x) * k0e(x);
        assert!(
            (w * x - 1.0).abs() <= 1e-5,
            "I/K Wronskian broken at x = {x}: {}",
            w * x
        );
    }
}

#[test]
fn the_f32_surface_keeps_the_f64_conventions() {
    for f in [
        k0 as fn(f32) -> f32,
        k1 as fn(f32) -> f32,
        k0e as fn(f32) -> f32,
        k1e as fn(f32) -> f32,
    ] {
        assert!(f(f32::NAN).is_nan());
        assert!(f(-1.0).is_nan(), "K is defined for positive argument");
        assert_eq!(f(0.0), f32::INFINITY);
    }
    assert_eq!(k0(f32::INFINITY), 0.0);
    assert_eq!(k1(f32::INFINITY), 0.0);
    // The unscaled forms underflow far sooner in this format than in f64.
    assert_eq!(k0(200.0), 0.0);
    assert!(k0e(200.0) > 0.0, "the scaled form stays representable");
}

/// Each route must be reached, and the two must agree across the join.
#[test]
fn the_routes_agree_across_their_join() {
    assert_eq!(KRoute::for_argument(1.0), KRoute::Series);
    assert_eq!(KRoute::for_argument(K_SERIES_MAX), KRoute::Series);
    assert_eq!(KRoute::for_argument(K_SERIES_MAX + 0.1), KRoute::Quadrature);
    for x in [K_SERIES_MAX - 1e-3, K_SERIES_MAX, K_SERIES_MAX + 1e-3] {
        let wide = f64::from(x);
        assert_close_f32(k0(x), reference::k0(wide), 16.0, "K0 at the join");
        assert_close_f32(k1(x), reference::k1(wide), 16.0, "K1 at the join");
    }
}

/// The quadrature's truncation time, as its own value.
///
/// `T = 2·asinh(√(K_DECAY/2x))`, the half-angle form of `cosh T − 1 = K_DECAY/x`. Spelling it as
/// `acosh(1 + K_DECAY/x)` is a cancellation bug rather than a loss of digits — past `x = 1.7e8`
/// the sum rounds to exactly one and the quadrature integrates nothing — so the association here
/// is the fix, and only a direct call pins it.
#[test]
fn the_truncation_time_solves_the_decay_equation() {
    for (root_x, want) in [
        (1.0_f32, 3.737_102_3_f32),
        (2.0, 2.477_888_8),
        (0.5, 5.087_558_3),
    ] {
        let got = k_truncation_time(root_x);
        assert!(
            (got - want).abs() <= 1e-6 * want,
            "at √x={root_x}: {got} vs {want}"
        );
        // It is the solution of its own equation: `cosh T − 1 = K_DECAY/x`.
        let x = root_x * root_x;
        let residual = got.cosh() - 1.0 - K_DECAY / x;
        assert!(residual.abs() <= 1e-4 * (K_DECAY / x), "{residual:e}");
    }
    // Decreasing in the argument, and still positive where `acosh(1 + K_DECAY/x)` would have
    // collapsed to zero — `√x = 1e4` is `x = 1e8`, past the `1.7e8` threshold in the comment.
    assert!(k_truncation_time(1.0) > k_truncation_time(2.0));
    let far = k_truncation_time(1e4);
    assert!(far > 0.0, "{far:e}");
    assert!((far - 6.324_555e-4).abs() <= 1e-9, "{far:e}");
    // And the radicand is formed by dividing *after* the roots, so a subnormal argument does not
    // overflow it.
    assert!(k_truncation_time(f32::MIN_POSITIVE).is_finite());
}

/// Both series against the `f64` kernel, across the range they are actually selected on.
///
/// Seven mutants lived in their terms. The series run only over `x ≤ K_SERIES_MAX = 1.5`, and the
/// harmonic-number factors are what the deep terms carry — each below a whole-function tolerance
/// at small `x`, which is exactly where the series is used.
#[test]
fn both_k_series_track_the_wide_kernel_across_their_range() {
    for x in [1e-3_f32, 0.01, 0.1, 0.5, 1.0, K_SERIES_MAX] {
        assert_eq!(KRoute::for_argument(x), KRoute::Series, "at {x}");
        let got = k0_series(x);
        let want = reference::k0(f64::from(x)) as f32;
        assert!(
            (got - want).abs() <= 2e-5 * want.abs(),
            "k0_series({x}): {got} vs {want}"
        );
        let got = k1_series(x);
        let want = reference::k1(f64::from(x)) as f32;
        assert!(
            (got - want).abs() <= 2e-5 * want.abs(),
            "k1_series({x}): {got} vs {want}"
        );
    }
    // `K₀` diverges logarithmically at the origin and `K₁` like `1/x`, so both are large and
    // positive at the bottom of the range and decreasing across it.
    assert!(k0_series(1e-3) > k0_series(1.0));
    assert!(k1_series(1e-3) > k1_series(1.0));
    assert!(k0_series(1.0) > 0.0 && k1_series(1.0) > 0.0);
}

/// The **cost** of both series, which is all the convergence break changes.
///
/// Four of the seven mutants on those two lines force the loop to [`MAX_TERMS`] — narrowing `<`
/// to `==`, which never fires, and `k > 1` to `==` or `<`, which never admits a break — and all
/// four leave the returned value identical. The remaining three differ only when a term lands
/// *exactly* on the tolerance, or let `K₁` break one index sooner than it ever converges: those
/// are equivalent mutants, recorded rather than chased.
#[test]
fn both_k_series_converge_well_inside_the_term_cap() {
    // Measured, and monotone in the argument: three terms at the bottom of the range, eight at
    // the top, against a cap of thirty-two.
    for (x, k0_want, k1_want) in [
        (1e-3_f32, 3, 3),
        (0.01, 3, 3),
        (0.1, 4, 4),
        (0.5, 5, 6),
        (1.0, 6, 8),
        (K_SERIES_MAX, 7, 8),
    ] {
        let (value, terms) = k0_series_terms(x);
        assert_eq!(terms, k0_want, "k0 at x = {x}");
        assert!(value.is_finite() && value > 0.0);
        assert!(terms < MAX_TERMS, "k0 at x = {x} reached the cap");
        let (value, terms) = k1_series_terms(x);
        assert_eq!(terms, k1_want, "k1 at x = {x}");
        assert!(value.is_finite() && value > 0.0);
        assert!(terms < MAX_TERMS, "k1 at x = {x} reached the cap");
    }
}
