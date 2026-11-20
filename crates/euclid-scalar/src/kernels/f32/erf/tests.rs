//! Unit tests for the native single-precision error-function family.
//!
//! A child module so implementation plus tests stay inside the crate 400-line budget, while
//! `use super::*` still reaches `ErfForm`, `ErfcForm`, `ErfInvRoute`, `ErfInvSeed` and the
//! `ERFCX_CF_MIN` crossover. The inverse is the reason most of this exists: dropping the
//! Newton step its `f64` peer takes made the seed coefficients observable, and
//! `the_f32_inverse_is_as_accurate_as_the_format_allows` is what pins them.

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

/// **The inverse is accurate to about two `f32` ulps, and that is now checkable.**
///
/// Compared against the f64 kernel, which is four orders of magnitude more accurate than
/// `f32` can represent and so serves as the truth here. Measured maximum over this sweep is
/// `2.4e-7`; the bound is `3e-7`.
///
/// This is the test the Newton step made impossible. While the step was there the function
/// was accurate only to `1.9e-6`, and every one of its thirty-four internal mutants sat
/// inside that — the step destroyed the seed's precision and then added nothing of its own,
/// so no bound could separate a correct coefficient table from a damaged one. Removing it
/// improved the function eightfold and made its coefficients observable in the same move.
///
/// A relative bound throughout, which is the right measure for an inverse whose range is
/// unbounded: near `|p| = 1` the answer is large and a relative claim is the only meaningful
/// one, and near zero `erf_inv(p) ≈ p·√π/2` stays well away from zero relative to `p`.
#[test]
fn the_f32_inverse_is_as_accurate_as_the_format_allows() {
    let mut worst = 0.0_f64;
    let mut at = 0.0_f64;
    for i in 1..4000 {
        #[allow(clippy::cast_precision_loss)]
        let p = -0.999_f32 + 2.0 * 0.999_f32 * (i as f32) / 4000.0;
        if p == 0.0 {
            continue;
        }
        let truth = crate::kernels::erf::erf_inv(f64::from(p));
        let gap = ((f64::from(erf_inv(p)) - truth) / truth).abs();
        if gap > worst {
            worst = gap;
            at = f64::from(p);
        }
    }
    assert!(
        worst <= 3e-7,
        "f32 erf_inv is accurate to {worst:.4e} (worst at p = {at}), budget 3e-7",
    );

    // The far tail, past the `w >= 5` crossover into the second coefficient table. The
    // seed is the whole computation out here, since a Newton correction's `e^{-r²}` factor
    // underflows to zero — which is why removing it changed nothing in this range.
    for &(lo, hi, bound) in &[
        (0.9966_f32, 0.99999_f32, 3e-7),
        (0.99999, 0.999_999_94, 2e-7),
    ] {
        for i in 0..=1000 {
            #[allow(clippy::cast_precision_loss)]
            let p = lo + (hi - lo) * (i as f32) / 1000.0;
            if p >= 1.0 {
                continue;
            }
            let truth = crate::kernels::erf::erf_inv(f64::from(p));
            let gap = ((f64::from(erf_inv(p)) - truth) / truth).abs();
            assert!(gap <= bound, "f32 erf_inv at p = {p} is off by {gap:.3e}");
        }
    }

    // Both signs, since the seed is applied as `p(w)·x` and the sign rides on `x`.
    for &p in &[0.5_f32, -0.5, 0.9, -0.9, 0.01, -0.01] {
        let truth = crate::kernels::erf::erf_inv(f64::from(p));
        let gap = ((f64::from(erf_inv(p)) - truth) / truth).abs();
        assert!(gap <= 3e-7, "f32 erf_inv at p = {p} is off by {gap:.3e}");
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

/// The four forms, at and either side of the one threshold that is not a non-finite case.
///
/// The linear cutoff is *sited* where the dropped `x²/3` term is below half an ulp, so the
/// two forms agree across it to well under a rounding — which is what left `<` against `<=`
/// alive against every value assertion. The decision is what gets pinned.
#[test]
fn the_erf_forms_partition_the_line() {
    use ErfForm::{IncompleteGamma, Linear, Saturated, Undefined};
    assert_eq!(ErfForm::for_argument(f32::NAN), Undefined);
    assert_eq!(ErfForm::for_argument(f32::INFINITY), Saturated);
    assert_eq!(ErfForm::for_argument(f32::NEG_INFINITY), Saturated);
    // The boundary belongs to the incomplete gamma, and the magnitude decides, so both
    // signs route alike.
    assert_eq!(ErfForm::for_argument(ERF_LINEAR_MAX), IncompleteGamma);
    assert_eq!(ErfForm::for_argument(-ERF_LINEAR_MAX), IncompleteGamma);
    assert_eq!(ErfForm::for_argument(ERF_LINEAR_MAX.next_down()), Linear);
    assert_eq!(ErfForm::for_argument(-ERF_LINEAR_MAX.next_down()), Linear);
    assert_eq!(
        ErfForm::for_argument(ERF_LINEAR_MAX.next_up()),
        IncompleteGamma
    );
    // Both zeros take the linear arm, which is what preserves the sign of a zero.
    assert_eq!(ErfForm::for_argument(0.0), Linear);
    assert_eq!(ErfForm::for_argument(-0.0), Linear);
    assert_eq!(erf(-0.0).to_bits(), (-0.0_f32).to_bits());
    assert_eq!(erf(0.0).to_bits(), 0.0_f32.to_bits());
    for x in [1.0_f32, -1.0, 10.0] {
        assert_eq!(ErfForm::for_argument(x), IncompleteGamma);
    }
    // And the saturated arm carries the sign rather than a comparison.
    assert_eq!(erf(f32::INFINITY), 1.0);
    assert_eq!(erf(f32::NEG_INFINITY), -1.0);
}

/// The inverse's domain routes and its two seed tables.
///
/// `AtNegativeOne` and `AtOne` are the ends of `erf`'s range, where the inverse diverges;
/// `AtOrigin` is the signed zero the polynomial cannot produce. The seed split is a separate
/// decision in a separate variable — the central table is a polynomial in `w − 5/2` and the
/// tail's in `√w − 3` — so neither is a continuation of the other.
#[test]
fn the_inverse_routes_split_the_range_and_the_seed_tables() {
    use ErfInvRoute::{AtNegativeOne, AtOne, AtOrigin, Interior};
    use ErfInvSeed::{Central, Tail};

    for x in [-1.0, (-1.0_f32).next_down(), -2.0, f32::NEG_INFINITY] {
        assert_eq!(ErfInvRoute::for_argument(x), AtNegativeOne, "at {x:e}");
        assert_eq!(erf_inv(x), f32::NEG_INFINITY, "at {x:e}");
    }
    for x in [1.0, 1.0_f32.next_up(), 2.0, f32::INFINITY] {
        assert_eq!(ErfInvRoute::for_argument(x), AtOne, "at {x:e}");
        assert_eq!(erf_inv(x), f32::INFINITY, "at {x:e}");
    }
    // Odd at the origin, which is what `copysign` is for.
    for x in [0.0_f32, -0.0] {
        assert_eq!(ErfInvRoute::for_argument(x), AtOrigin);
        assert_eq!(erf_inv(x).to_bits(), x.to_bits(), "signed zero at {x:e}");
    }
    for x in [
        (-1.0_f32).next_up(),
        -0.5,
        -f32::MIN_POSITIVE,
        f32::MIN_POSITIVE,
        0.5,
        1.0_f32.next_down(),
        f32::NAN,
    ] {
        assert_eq!(ErfInvRoute::for_argument(x), Interior, "at {x:e}");
    }
    // `NaN` reaches the interior and propagates rather than being caught by a guard.
    assert!(erf_inv(f32::NAN).is_nan());

    // The seed tables, at and either side of the boundary.
    assert_eq!(ErfInvSeed::for_log_weight(5.0), Tail);
    assert_eq!(ErfInvSeed::for_log_weight(5.0_f32.next_down()), Central);
    assert_eq!(ErfInvSeed::for_log_weight(5.0_f32.next_up()), Tail);
    assert_eq!(ErfInvSeed::for_log_weight(0.0), Central);
    assert_eq!(ErfInvSeed::for_log_weight(f32::NAN), Tail);
    // Both tables are evaluated in different variables, so they are genuinely two fits —
    // yet they agree at the join, which is what makes the split free.
    let join = 5.0_f32;
    let (central, tail) = (Central.evaluate(join), Tail.evaluate(join));
    assert!(
        (central - tail).abs() <= 2e-2 * tail.abs(),
        "the seed tables disagree at the join: {central} vs {tail}"
    );
}
