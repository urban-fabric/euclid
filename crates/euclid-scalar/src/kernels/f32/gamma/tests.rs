//! Unit tests for the single-precision gamma family, reaching its private routes and series.
//!
//! A child module so the implementation stays inside the crate's 400-line budget. What is added
//! here beyond the original three tests is the **routing** — `PsiReduction`, `LnGammaMethod`,
//! `IncompleteGammaMethod` — and the individual series. Mutation testing found twenty-four
//! survivors across those, all of the same shape: a reduction that reaches the right answer by a
//! longer path, or a series term whose contribution is below a whole-function tolerance.

use super::*;

/// Cross-check the native `f32` result against the `f64` reference, narrowed — the guard
/// against the two implementations silently drifting apart.
fn agrees(a: f32, b: f64, tol: f32) {
    let b = b as f32;
    assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
}

#[test]
fn gamma_matches_f64_reference() {
    for &x in &[0.3, 0.7, 1.5, 3.2, 5.0, 10.0] {
        agrees(gamma(x), crate::kernels::gamma::gamma(x as f64), 1e-5);
        agrees(ln_gamma(x), crate::kernels::gamma::ln_gamma(x as f64), 1e-5);
    }
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

#[test]
fn polygamma_matches_f64_reference() {
    for &x in &[0.5, 1.0, 2.0, 3.3, 7.5] {
        agrees(digamma(x), crate::kernels::gamma::digamma(x as f64), 1e-5);
        agrees(trigamma(x), crate::kernels::gamma::trigamma(x as f64), 1e-5);
    }
}

/// Both digamma-family reductions, as tables.
///
/// Every one of these arms reaches a correct answer by a different route, so a misrouted argument
/// still returns something plausible — which is why six mutants lived here.
#[test]
fn the_psi_reductions_are_tables_over_the_whole_line() {
    use PsiReduction::{Asymptotic, Pole, Recurrence, Reflected, Undefined};
    // `ψ`: reflection over the whole negative axis, recurrence up to the asymptotic threshold.
    for (x, want) in [
        (f32::NAN, Undefined),
        (f32::NEG_INFINITY, Undefined),
        (f32::INFINITY, Asymptotic),
        (-1.0, Pole),
        (-2.0, Pole),
        (-1e7, Pole),
        (-0.5, Reflected),
        (-1.5, Reflected),
        (0.0, Recurrence),
        (-0.0, Recurrence),
        (0.5, Recurrence),
        (PSI_ASYMPTOTIC_MIN.next_down(), Recurrence),
        (PSI_ASYMPTOTIC_MIN, Asymptotic),
        (1e30, Asymptotic),
    ] {
        assert_eq!(PsiReduction::for_digamma(x), want, "digamma at {x}");
    }
    // `ψ₁`: reflection below one half — its own fixed point — and zero is a pole it *has*.
    for (x, want) in [
        (f32::NAN, Undefined),
        (f32::NEG_INFINITY, Undefined),
        (f32::INFINITY, Asymptotic),
        (0.0, Pole),
        (-0.0, Pole),
        (-1.0, Pole),
        (-3.0, Pole),
        (-0.5, Reflected),
        (-1.5, Reflected),
        (0.25, Reflected),
        (0.5, Recurrence),
        (0.5_f32.next_down(), Reflected),
        (3.0, Recurrence),
        (PSI_ASYMPTOTIC_MIN.next_down(), Recurrence),
        (PSI_ASYMPTOTIC_MIN, Asymptotic),
    ] {
        assert_eq!(PsiReduction::for_trigamma(x), want, "trigamma at {x}");
    }
}

#[test]
fn the_log_gamma_method_prefers_the_reduction_on_the_positive_side() {
    use LnGammaMethod::{Lanczos, Reduced, Reflected};
    // The sign test is on the *sign bit*, so a negative zero reflects rather than reducing.
    assert_eq!(LnGammaMethod::for_argument(0.0), Reduced);
    assert_eq!(LnGammaMethod::for_argument(1.0), Reduced);
    assert_eq!(
        LnGammaMethod::for_argument(LN_GAMMA_RECURRENCE_MAX.next_down()),
        Reduced
    );
    assert_eq!(
        LnGammaMethod::for_argument(LN_GAMMA_RECURRENCE_MAX),
        Lanczos
    );
    assert_eq!(LnGammaMethod::for_argument(1e30), Lanczos);
    assert_eq!(LnGammaMethod::for_argument(-0.0), Reflected);
    assert_eq!(LnGammaMethod::for_argument(-1.5), Reflected);
    assert_eq!(LnGammaMethod::for_argument(f32::NEG_INFINITY), Reflected);
}

/// The `ψ` walk, which both functions share and only the shed term distinguishes.
#[test]
fn the_psi_recurrence_walks_to_the_threshold_and_accumulates() {
    // A unit term makes the accumulation a step count, so the walk is exactly checkable: from
    // `3` it takes three steps to reach `6`.
    assert_eq!(psi_recurrence(3.0, |_| 1.0), (6.0, 3.0));
    assert_eq!(psi_recurrence(1.0, |_| 1.0), (6.0, 5.0));
    assert_eq!(psi_recurrence(5.5, |_| 1.0), (6.5, 1.0));
    // At or above the threshold the loop does not run at all, so the argument comes back
    // untouched and nothing is shed.
    assert_eq!(psi_recurrence(PSI_ASYMPTOTIC_MIN, |_| 1.0), (6.0, 0.0));
    assert_eq!(psi_recurrence(10.0, |_| 1.0), (10.0, 0.0));
    // And the term is evaluated at the *pre-increment* argument, which is what makes the two
    // callers' identities hold: `ψ` sheds `1/x` and `ψ₁` sheds `1/x²`.
    assert_eq!(psi_recurrence(4.0, |x| x), (6.0, 9.0));
    let (shifted, shed) = psi_recurrence(4.0, |x| 1.0 / x);
    assert_eq!(shifted, 6.0);
    assert!((shed - (0.25 + 0.2)).abs() < 1e-6, "{shed}");
}

/// The two asymptotic series and the reduction, against the `f64` kernel.
#[test]
fn the_psi_series_track_the_wide_kernel() {
    for x in [6.0_f32, 7.5, 10.0, 100.0, 1e4] {
        let wide = crate::kernels::gamma::digamma(f64::from(x)) as f32;
        let got = digamma_asymptotic(x);
        assert!(
            (got - wide).abs() <= 1e-5 * wide.abs().max(1.0),
            "digamma_asymptotic({x}): {got} vs {wide}"
        );
        let wide = crate::kernels::gamma::trigamma(f64::from(x)) as f32;
        let got = trigamma_asymptotic(x);
        assert!(
            (got - wide).abs() <= 1e-5 * wide.abs().max(1.0),
            "trigamma_asymptotic({x}): {got} vs {wide}"
        );
    }
    // The reduction onto the unit interval, on both sides of its own `x < 1` split.
    for x in [0.25_f32, 0.75, 1.0, 1.5, 2.0, 3.5, 5.75] {
        let wide = crate::kernels::gamma::digamma(f64::from(x)) as f32;
        let got = digamma_reduced(x);
        assert!(
            (got - wide).abs() <= 1e-4 * wide.abs().max(1.0),
            "digamma_reduced({x}): {got} vs {wide}"
        );
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

/// The two correction polynomials, as their own values.
///
/// Inside the asymptotic series these are rounding corrections — the deepest Horner terms sit at
/// or below one `f32` ulp of `ψ` at `x = 6`, the smallest argument the series sees, which is why
/// their signs survived. Called directly the nest *is* the answer, and the same flips move it by
/// `7.2e-5`, `2.1e-6` and `8.6e-6` relative: three orders clear of the format either way.
#[test]
fn the_psi_correction_polynomials_are_pinned_by_their_own_values() {
    for (inv, want) in [
        (1.0_f32 / 6.0, 0.002_308_467_6_f32),
        (0.1, 0.000_832_504),
        (1.0 / 60.0, 2.314_750_8e-5),
    ] {
        let got = digamma_correction(inv);
        assert!(
            (got - want).abs() <= 1e-7 * want,
            "digamma_correction({inv}): {got:e} vs {want:e}"
        );
    }
    for (r, want) in [(1.0_f32 / 6.0, 0.000_767_400_1_f32), (0.1, 0.000_166_335_7)] {
        let got = trigamma_tail(r);
        assert!(
            (got - want).abs() <= 1e-7 * want,
            "trigamma_tail({r}): {got:e} vs {want:e}"
        );
    }
    // Both are positive and strictly decreasing in the argument, since every leading coefficient
    // is positive and the nest is dominated by its first term.
    assert!(digamma_correction(1.0 / 6.0) > digamma_correction(0.1));
    assert!(trigamma_tail(1.0 / 6.0) > trigamma_tail(0.1));
    assert_eq!(digamma_correction(0.0), 0.0);
    assert_eq!(trigamma_tail(0.0), 0.0);
}

/// The reduction's `x < 1` split, which the recurrence identity makes invisible at the boundary.
#[test]
fn the_digamma_reduction_agrees_with_itself_at_the_unit_boundary() {
    // At `x = 1` the two arms are the *same* identity read two ways: the `x < 1` arm gives
    // `ψ(1+x) − 1/x` and the shift arm gives `ψ(1+0)`, and `ψ(2) − 1 = ψ(1)` exactly. So
    // widening the comparison to `<=` is an equivalent mutant at the only argument it moves —
    // recorded here rather than left to be rediscovered.
    let wide = crate::kernels::gamma::digamma(1.0) as f32;
    let got = digamma_reduced(1.0);
    assert!((got - wide).abs() <= 2.0 * f32::EPSILON, "{got} vs {wide}");
    // Either side of it the arms are genuinely different and both track the wide kernel.
    for x in [1.0_f32.next_down(), 1.0, 1.0_f32.next_up()] {
        let wide = crate::kernels::gamma::digamma(f64::from(x)) as f32;
        let got = digamma_reduced(x);
        assert!(
            (got - wide).abs() <= 1e-5 * wide.abs().max(1.0),
            "digamma_reduced({x}): {got} vs {wide}"
        );
    }
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
