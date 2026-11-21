//! Unit tests for the shared machinery behind all four integrals.
//!
//! A child module so implementation plus tests stay inside the crate's 400-line budget. What
//! stays here is what both halves share: the compensated accumulator, the exact `two_sum` it is
//! built on, the Lentz nudge, and the crossover sweeps that have to span every route at once.
//! The per-function laws sit beside their own modules, in [`super::exponential`] and
//! [`super::trigonometric`].

use super::*;
use euclid_utils::no_step_across;

fn close(a: f64, b: f64, tol: f64) {
    assert!(
        (a - b).abs() <= tol * b.abs().max(1.0),
        "{a:e} vs {b:e} (tol {tol:e})"
    );
}

// --- reference values ------------------------------------------------
// 600-digit `Decimal` evaluations of the defining power series, which at that precision
// are immune to the cancellation that defeats the same series in `f64`.

#[test]
fn the_integrals_differentiate_to_their_integrands() {
    // E1' = -e^-x/x, Ei' = e^x/x, Si' = sin(x)/x, Ci' = cos(x)/x.
    for i in 1..60 {
        let x = f64::from(i) * 0.5;
        let h = 1e-6 * x;
        let d_e1 = (exp_int_e1(x + h) - exp_int_e1(x - h)) / (2.0 * h);
        close(d_e1, -(-x).exp() / x, 1e-6);
        let d_si = (sin_int(x + h) - sin_int(x - h)) / (2.0 * h);
        close(d_si, x.sin() / x, 1e-6);
        let d_ci = (cos_int(x + h) - cos_int(x - h)) / (2.0 * h);
        close(d_ci, x.cos() / x, 1e-6);
    }
    for i in 1..40 {
        let x = f64::from(i) * 0.25;
        let h = 1e-6 * x;
        let d = (exp_int_ei(x + h) - exp_int_ei(x - h)) / (2.0 * h);
        close(d, x.exp() / x, 1e-6);
    }
}

#[test]
fn every_crossover_leaves_no_step_at_ulp_resolution() {
    // All three switches in this module, each probed at the two representable values that
    // straddle it -- the smallest interval the format allows, so the function's own slope
    // contributes almost nothing and what is left is the agreement between the two methods.
    //
    // `Ei` is the exception that proves the derivative term is needed: it grows like e^x, so
    // across one ulp of x = 40 it legitimately moves by Ei'(40)*ulp(40) = 42.
    no_step_across(
        "Si",
        sin_int,
        TRIG_SERIES_MAX,
        TRIG_SERIES_MAX.sin() / TRIG_SERIES_MAX,
        1e-14,
    );
    no_step_across(
        "Ci",
        cos_int,
        TRIG_SERIES_MAX,
        TRIG_SERIES_MAX.cos() / TRIG_SERIES_MAX,
        1e-14,
    );
    // `E₁` has two joins now, both edges of the Chebyshev window, and neither is
    // `E1_SERIES_MAX` — which no longer routes anything and has been retired.
    for threshold in [
        super::exponential::E1_CHEBYSHEV_MIN,
        super::exponential::E1_CHEBYSHEV_MAX,
    ] {
        no_step_across(
            "E1",
            exp_int_e1,
            threshold,
            -(-threshold).exp() / threshold,
            1e-14,
        );
    }
    no_step_across(
        "Ei",
        exp_int_ei,
        EI_ASYMPTOTIC_MIN,
        EI_ASYMPTOTIC_MIN.exp() / EI_ASYMPTOTIC_MIN,
        1e-14,
    );
}

/// Knuth's two-sum, pinned by its defining identity rather than through a caller.
///
/// A broken decomposition can still reconstruct the sum. If the primary sum changes
/// from `a + b` to `a - b`, the residual can absorb the difference. At
/// `(0.1, 0.2)` the mutated pair is `s = -0.1, e = 0.4` against the real `s = 0.30000000000000004,
/// e = -2.78e-17`, and every downstream caller adds the two straight back together, so a test
/// that checks only a sum cannot see it.
///
/// What is destroyed is the *decomposition*, and that is the whole product: `total` and
/// `correction` are accumulated separately across terms, so a residual carrying `0.4` instead of
/// `-2.78e-17` makes the compensation meaningless and the error `O(n·ε·Σ|tₖ|)` again. The
/// identity has to be asserted on the two halves, not on their sum.
///
/// Each case is exact, so the assertions are equalities and none of the values is transcribed:
/// where `a + b` is representable the residual is zero, and where it needs more than 53 bits the
/// residual is exactly the part that did not fit.
#[test]
fn two_sum_splits_a_sum_into_a_rounded_total_and_its_exact_residual() {
    let two_pow_60 = 2f64.powi(60);
    for (a, b, residual) in [
        // An exactly representable sum leaves nothing over.
        (1.0f64, 2.0f64, 0.0f64),
        (0.5, 0.25, 0.0),
        // A sum needing a 61st bit: the trailing one does not fit, and is the residual.
        (two_pow_60, 1.0, 1.0),
        (two_pow_60, -1.0, -1.0),
        // The same below the ulp of 1, in both directions.
        (1.0, 1e-20, 1e-20),
        (1.0, -1e-20, -1e-20),
    ] {
        let (s, e) = two_sum(a, b);
        assert_eq!(
            s,
            a + b,
            "the primary sum is the rounded sum, at ({a:e}, {b:e})"
        );
        assert_eq!(e, residual, "and the residual is exact, at ({a:e}, {b:e})");
        // Which is the identity the compensation rests on.
        assert_eq!(s + e, a + b, "s + e reconstructs at ({a:e}, {b:e})");
    }
    // A residual that is not representable alongside the total is what the accumulator is for:
    // 0.1 + 0.2 is inexact, so the residual is non-zero and carries the lost bits.
    let (s, e) = two_sum(0.1_f64, 0.2);
    assert_eq!(s, 0.1 + 0.2, "the rounded sum");
    assert!(
        e != 0.0,
        "an inexact sum has a non-zero residual, got {e:e}"
    );
    assert!(
        e.abs() < f64::EPSILON * s.abs(),
        "and it is below one ulp of the total: {e:e}"
    );
    // Commutative in the total, as a sum should be.
    assert_eq!(two_sum(0.1, 0.2).0, two_sum(0.2, 0.1).0);
}

/// `compensated_series` must actually compensate, which only a series that defeats naive
/// summation can require.
///
/// `correction += e` was replaceable with `correction *= e` — zeroing the compensation outright,
/// since `correction` starts at zero — and the suite did not notice, because every existing
/// caller is checked against a tolerance that plain summation already meets. That is the
/// `compensated_sum` finding this repository's mutation config was written for, in a second
/// function.
///
/// The series below is the standard witness: one unit term followed by a thousand at `1e-17`,
/// each individually below half an ulp of the running total. Added naively every one of them is
/// lost and the answer is exactly `1.0`; compensated, they sum in the correction accumulator and
/// the answer is `1 + 1e-14`. No reference value is needed — the exact sum is known by
/// construction.
#[test]
fn compensated_series_recovers_terms_naive_summation_drops_entirely() {
    let terms = |k: u32| {
        if k == 0 {
            Some(1.0)
        } else if k <= 1_000 {
            Some(1e-17)
        } else {
            None
        }
    };

    // Naive summation loses all thousand: 1e-17 is below half an ulp of 1.0.
    let mut naive = 0.0f64;
    for k in 0..=1_001 {
        match terms(k) {
            Some(t) => naive += t,
            None => break,
        }
    }
    assert_eq!(
        naive, 1.0,
        "the premise: added naively the small terms vanish"
    );

    let compensated = compensated_series(terms);
    let exact = 1.0 + 1e-14; // 1000 × 1e-17, which is representable alongside 1.0
    assert!(
        compensated > 1.0,
        "the compensation must recover something, got {compensated:e}"
    );
    close(compensated, exact, 1e-16);
    // And the recovered amount is the whole of what naive summation dropped — to the resolution
    // the format has there, which is the only claim available: `1 + 1e-14` sits 45 ulps above
    // `1.0`, so the difference can be read back only to within one ulp of it (9.992e-15 for an
    // exact 1e-14). That is still a hundred million times the signal needed to separate a
    // working compensation from none at all.
    let recovered = compensated - naive;
    assert!(
        (recovered - 1e-14).abs() <= f64::EPSILON,
        "recovered {recovered:e}, expected 1e-14 to within one ulp of 1.0"
    );
}

#[test]
fn compensated_series_checks_convergence_after_four_terms() {
    let terms = [1.0, 0.0, 0.0, 0.0, 4.0];
    let mut evaluated = Vec::new();
    let sum = super::compensated_series(|k| {
        evaluated.push(k);
        terms.get(k as usize).copied()
    });
    assert_eq!(sum, 1.0);
    assert_eq!(evaluated, [0, 1, 2, 3]);
}

/// The two regime crossovers.
///
/// `exp_int_ei`'s `x < EI_ASYMPTOTIC_MIN` and `trig_integrals`' `x < TRIG_SERIES_MAX` decide
/// which route evaluates the threshold *exactly*. Both routes are accurate there — that is what
/// makes a crossover a free choice — and the existing join tests measure the step against the
/// derivative. This measures the thing the mutation actually changes: the two routes' agreement
/// at the single argument that moves between them.
#[test]
fn the_integral_crossovers_agree_at_their_thresholds() {
    // `Ei` at its asymptotic switch: both routes, one argument.
    let x = super::EI_ASYMPTOTIC_MIN;
    let from_below = super::exp_int_ei(x - f64::EPSILON * x);
    let at_threshold = super::exp_int_ei(x);
    let from_above = super::exp_int_ei(x + f64::EPSILON * x);
    let spread = ((from_above - from_below) / at_threshold).abs();
    assert!(
        spread < 1e-13,
        "the two Ei routes must agree across the switch at x = {x}: spread {spread:e}"
    );

    // `Si`/`Ci` at theirs.
    let t = super::TRIG_SERIES_MAX;
    for (name, f) in [
        ("Si", super::sin_int as fn(f64) -> f64),
        ("Ci", super::cos_int as fn(f64) -> f64),
    ] {
        let below = f(t - f64::EPSILON * t);
        let above = f(t + f64::EPSILON * t);
        let scale = f(t).abs().max(1.0);
        assert!(
            ((above - below) / scale).abs() < 1e-13,
            "{name}'s two routes must agree at the join: {below:e} against {above:e}"
        );
    }
}
