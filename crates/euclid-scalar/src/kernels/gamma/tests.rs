//! Unit tests for the gamma family.
//!
//! A child module in its own file so implementation plus tests stay inside the crate's
//! 400-line budget, while `use super::*` still reaches the private Lanczos coefficients,
//! the Stirling remainder and the Edgeworth crossover.

use super::*;

fn close(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
}

#[test]
fn gamma_matches_factorials() {
    close(gamma(1.0), 1.0, 1e-12);
    close(gamma(5.0), 24.0, 1e-10);
    close(gamma(6.0), 120.0, 1e-10);
    close(gamma(0.5), core::f64::consts::PI.sqrt(), 1e-12);
}

#[test]
fn reflection_preserves_the_negative_tail_and_pole_distance() {
    use crate::Real;

    // 256-bit MPFR reference values, rounded to each destination format.
    for (x, expected) in [
        (-172.5_f64, 0x8000_0034_c532_409a),
        (-174.5, 0x8000_0000_0072_3a81),
        (-176.5, 0x8000_0000_0000_00f2),
        (-177.5, 1),
        (-178.5, 0x8000_0000_0000_0000),
        (-1500.5, 0x8000_0000_0000_0000),
        (-1501.5, 0),
    ] {
        assert_eq!(gamma(x).to_bits(), expected, "x={x}");
    }
    // The `f32` reflection is native, so the normal-range value is allowed one ulp against the
    // MPFR reference while the **subnormal tail stays bit-exact**. That split is the result
    // worth pinning: preserving the subnormal structure was the stated reason the path used to
    // widen to `f64`, and it survives the port intact — `0x1b` at `-37.5` is three significant
    // bits of a value near `3.8e-44`, reproduced exactly. Only `-34.25`, the one case in the
    // normal range, moves at all.
    for (x, expected, budget) in [
        (-34.25_f32, 0x8043_8d55u32, 1i64),
        (-37.5, 0x1b, 0),
        (-39.5, 0, 0),
        (-40.5, 0x8000_0000, 0),
    ] {
        let got: f32 = Real::gamma(x);
        let gap = i64::from(got.to_bits()) - i64::from(expected);
        assert!(
            gap.abs() <= budget,
            "x={x}: got 0x{:08x}, expected 0x{expected:08x} ({gap} ulp, budget {budget})",
            got.to_bits()
        );
    }
    // Past `REFLECTED_ZERO_MIN` the clamp must give signed zero rather than the `inf · 0 = NaN`
    // an unclamped `f32` Lanczos intermediate would produce.
    for x in [-50.5_f32, -60.5, -100.5, -1000.5] {
        let got: f32 = Real::gamma(x);
        assert_eq!(got.to_bits(), 0x8000_0000, "x={x} must be negative zero");
    }
    for (x, expected) in [
        ((-2.0_f64).next_down(), -1_125_899_906_842_623.5),
        ((-2.0_f64).next_up(), 2_251_799_813_685_248.5),
        ((-171.0_f64).next_down(), 2.835_121_658_031_596e-296),
    ] {
        let value = gamma(x);
        assert!(
            (value / expected - 1.0).abs() < 3e-13,
            "x={x}, value={value}"
        );
    }
    close(
        ln_gamma(-1.5),
        (4.0 * core::f64::consts::PI.sqrt() / 3.0).ln(),
        3e-15,
    );
    assert!(ln_gamma(-0.5).is_nan());
}

#[test]
fn ln_gamma_matches_gamma() {
    for &x in &[0.3, 0.7, 1.5, 3.2, 10.0, 50.0] {
        close(ln_gamma(x).exp(), gamma(x), 1e-9);
    }
}

#[test]
fn small_shape_log_normalizer_matches_mpfr_at_the_cutoff() {
    // From `extreme_shapes`, a local generator over MPFR at 256 bits. See CONTRIBUTING.md,
    // "Reference values are pasted in": the generator is not committed, so the precision is
    // recorded here instead of a path.
    let expected = -0.576_393_598_283_369_6;
    assert!((ln_gamma1p_over_a(SMALL_SHAPE) - expected).abs() <= f64::EPSILON / 2.0);
}

#[test]
fn regularized_incomplete_gamma_bounds_and_complement() {
    for &(a, x) in &[(1.0, 0.5), (2.5, 3.0), (5.0, 2.0), (10.0, 12.0)] {
        let p = gamma_lr(a, x);
        let q = gamma_ur(a, x);
        assert!((0.0..=1.0).contains(&p), "P out of range: {p}");
        close(p + q, 1.0, 1e-10);
    }
    close(gamma_lr(1.0, 2.0), 1.0 - (-2.0f64).exp(), 1e-10);
}

#[test]
fn digamma_known_values() {
    // Tightened to 1e-11 after adding the 1/(240·x⁸) asymptotic term: the prior
    // truncation reached only ~1e-10 (vs the ~15-digit intent of the coefficients).
    close(digamma(1.0), -crate::constants::f64::EULER_GAMMA, 1e-11);
    close(
        digamma(2.0),
        1.0 - crate::constants::f64::EULER_GAMMA,
        1e-11,
    );
    // ψ(3) = 3/2 − γ.
    close(
        digamma(3.0),
        1.5 - crate::constants::f64::EULER_GAMMA,
        1e-11,
    );
}

#[test]
fn trigamma_known_values() {
    let pi2 = core::f64::consts::PI * core::f64::consts::PI;
    close(trigamma(1.0), pi2 / 6.0, 1e-10);
    close(trigamma(2.0), pi2 / 6.0 - 1.0, 1e-10);
    close(trigamma(0.5), pi2 / 2.0, 1e-10);
    close(trigamma(3.3) - trigamma(4.3), 1.0 / (3.3 * 3.3), 1e-10);
}

#[test]
fn positive_trigamma_matches_the_series_on_both_sides_of_the_shift() {
    // At integer and half-integer arguments the series is a finite sum
    // subtracted from π²/6 or π²/2. References use 70-digit Decimal sums.
    for (x, expected) in [
        (0.5, 4.934_802_200_544_679),
        (1.0, 1.644_934_066_848_226_4),
        (19.5, 0.052_619_441_213_655_93),
        (20.0, 0.051_270_822_935_203_12),
        (20.5, 0.049_989_592_429_960_99),
        (64.0, 0.015_747_706_064_338_93),
    ] {
        let actual = trigamma_positive(x);
        assert!(
            (actual - expected).abs() < 2e-14 * expected,
            "{x}: {actual} vs {expected}"
        );
    }
}

/// The Taylor series about `ψ`'s zero, at the edge of the window it is used on.
///
/// # Why the edge, and why against pasted digits
///
/// Five of the twenty-one coefficients' signs survived the suite. They are the deep ones, and
/// their contribution falls off as `d^k`: at the window's half-width of `2e-1` flipping `a₁₂`
/// moves the result by `2.1e6` ulp and flipping `a₁₈` by `14`, while anywhere nearer the zero
/// they vanish — at `d = 1e-1` the `a₁₄` flip is already down to `4` ulp and `a₁₆` to nothing.
/// So the edge is the only argument that pins the tail of the series.
///
/// The references are `ψ` at 40 digits from Stirling plus a sixty-step upward recurrence,
/// computed outside the crate; the series itself is accurate to **0 to 2 ulp** against them,
/// which is what lets the bound be four. `a₂₀` is *not* reachable: its flip moves the value by
/// one ulp at the edge and zero everywhere else, so it sits at the format's resolution and no
/// assertion on this function can separate it.
#[test]
fn the_series_about_the_psi_zero_is_pinned_at_its_window_edge() {
    for (d, want) in [
        (0.2_f64, 0.177_660_044_996_198_25_f64),
        (-0.2, -0.213_615_129_720_246_74),
        (0.1, 0.092_582_708_956_254_55),
        (-0.05, -0.049_523_892_169_472),
        (0.15, 0.135_985_547_380_247_57),
    ] {
        let got = psi_near_zero(d);
        let ulp = want.abs() * f64::EPSILON;
        assert!(
            (got - want).abs() <= 4.0 * ulp,
            "psi_near_zero({d}) = {got:e}, want {want:e} ({:.1} ulp)",
            (got - want).abs() / ulp
        );
    }
    // The constant term is the offset from the true zero, not zero itself: `ψ` vanishes
    // `9.55e-17` above the representable `PSI_ZERO`, and the high-precision reference agrees
    // with `a₀` to every digit it carries.
    assert!((psi_near_zero(0.0) - -9.241_265_521_729_427e-17).abs() <= 1e-31);
    // Odd about the zero to leading order, since `a₁` dominates across the window.
    assert!(psi_near_zero(0.2) > 0.0 && psi_near_zero(-0.2) < 0.0);
}

/// The seven `lnΓ` regimes, at and either side of every boundary.
///
/// Each arm reaches a correct answer by a different route, so a misrouted argument still returns
/// something plausible — which is exactly how the pole residue defect survived before the poles
/// were tested for. The table is what makes the ordering checkable.
#[test]
fn the_log_gamma_methods_partition_the_line() {
    use LnGammaMethod::{
        Divergent, Lanczos, NearTwoZero, NearUnitZero, Reflected, SmallShape, Undefined,
    };
    for (x, want) in [
        (f64::NAN, Undefined),
        (f64::NEG_INFINITY, Undefined),
        (f64::INFINITY, Divergent),
        // The poles: every non-positive integer, including both zeros.
        (0.0, Divergent),
        (-0.0, Divergent),
        (-1.0, Divergent),
        (-5.0, Divergent),
        (-1e5, Divergent),
        // The two zeros of `lnΓ`, and their neighbourhoods.
        (1.0, NearUnitZero),
        (1.0 - 0.9 * SMALL_SHAPE, NearUnitZero),
        (1.0 + SMALL_SHAPE, NearUnitZero),
        (2.0, NearTwoZero),
        (2.0 - SMALL_SHAPE, NearTwoZero),
        (2.0 + SMALL_SHAPE, NearTwoZero),
        // `1 − SMALL_SHAPE` is `0.999`, whose distance back to one *rounds up* to
        // `1.0000000000000009e-3` — just outside the closed neighbourhood, by one ulp of the
        // subtraction. The other three edges land inside, because `1.001`, `1.999` and `2.001`
        // all sit at a coarser spacing and round the other way. Worth pinning rather than
        // tidying: it is the arithmetic, not the threshold, and a reader checking the bound
        // against the constant would otherwise conclude the routing was wrong.
        (1.0 - SMALL_SHAPE, Lanczos),
        // The pole at the origin, approached from above only.
        (SMALL_SHAPE, SmallShape),
        (1e-30, SmallShape),
        // The negative axis reflects; the rest is Lanczos.
        (-0.5, Reflected),
        (-1.5, Reflected),
        (-1e5 - 0.5, Reflected),
        (3.0, Lanczos),
        (1e30, Lanczos),
    ] {
        assert_eq!(LnGammaMethod::for_argument(x), want, "at {x:e}");
    }
    // The neighbourhoods are closed, so one step outside each falls through to the general
    // route rather than staying on the shifted series.
    assert_eq!(
        LnGammaMethod::for_argument((1.0 + SMALL_SHAPE).next_up()),
        Lanczos
    );
    assert_eq!(
        LnGammaMethod::for_argument((2.0 + SMALL_SHAPE).next_up()),
        Lanczos
    );
    // Just above the small-shape cutoff the route is *Reflected*, not Lanczos: the reflection
    // applies to everything below `½`, which includes the small positives the shifted series
    // stops covering. That ordering is the original's and is preserved here.
    assert_eq!(
        LnGammaMethod::for_argument(SMALL_SHAPE.next_up()),
        Reflected
    );
    assert_eq!(LnGammaMethod::for_argument(0.25), Reflected);
    assert_eq!(LnGammaMethod::for_argument(0.5), Lanczos);
    // No continuity assertion across the zero-neighbourhood joins, and that is the point of
    // them: the shifted series and Lanczos *deliberately* disagree there by about an ulp of one
    // — measured `2.53e-16` at `x = 1.001` — because Lanczos carries an absolute error of
    // `−8.88e-16` beside a zero where the answer is itself near zero. The shifted arm is the
    // correct one, so a step at the join is the fix working rather than a defect. See
    // `LnGammaMethod::NearUnitZero`.
}

/// The eight incomplete-gamma routes, at and either side of every boundary.
///
/// Both tails dispatch on this one decision, so a misroute moves *two* public functions at once
/// — and every arm returns a probability in `[0, 1]`, so a wrong one still looks plausible. The
/// table is the only thing that separates them.
#[test]
fn the_incomplete_gamma_methods_partition_the_quadrant() {
    use IncompleteGammaMethod::{
        AtInfinity, AtZero, Edgeworth, Invalid, LowerSeries, SmallShapeFraction, SmallShapeSeries,
        UpperFraction,
    };
    for (a, x, want) in [
        // Rejected inputs: a non-positive or non-finite shape, a negative argument, any NaN.
        (0.0, 1.0, Invalid),
        (-1.0, 1.0, Invalid),
        (f64::INFINITY, 1.0, Invalid),
        (f64::NAN, 1.0, Invalid),
        (1.0, -0.125, Invalid),
        (1.0, f64::NAN, Invalid),
        // The two limits, which both tails return as exact constants.
        (2.0, 0.0, AtZero),
        (2.0, -0.0, AtZero),
        (2.0, f64::INFINITY, AtInfinity),
        // The Edgeworth handover, closed on the left.
        (GAMMA_EDGEWORTH_MIN, 1.0, Edgeworth),
        (GAMMA_EDGEWORTH_MIN.next_down(), 1.0, LowerSeries),
        (1e30, 1.0, Edgeworth),
        // The small shape, also closed, and it still picks a tail because the series is an
        // expansion in `x` valid only below `a + 1`.
        (SMALL_SHAPE, 0.5, SmallShapeSeries),
        (SMALL_SHAPE, 2.0, SmallShapeFraction),
        (SMALL_SHAPE.next_up(), 0.5, LowerSeries),
        (SMALL_SHAPE.next_up(), 2.0, UpperFraction),
        (1e-30, 0.5, SmallShapeSeries),
        // And the tail split at `x = a + 1`, which belongs to the fraction.
        (2.0, 1.0, LowerSeries),
        (2.0, 3.0_f64.next_down(), LowerSeries),
        (2.0, 3.0, UpperFraction),
        (2.0, 5.0, UpperFraction),
    ] {
        assert_eq!(
            IncompleteGammaMethod::for_arguments(a, x),
            want,
            "at a={a:e}, x={x:e}"
        );
    }

    // The two tails agree about the route, which is what lets each be a bare `match`: every
    // argument gives a complementary pair, whichever arm produced it.
    for (a, x) in [
        (SMALL_SHAPE, 0.5),
        (SMALL_SHAPE, 4.0),
        (2.0, 1.0),
        (2.0, 5.0),
        (1e4, 1e4),
        (GAMMA_EDGEWORTH_MIN, 3e7),
    ] {
        let (lower, upper) = (gamma_lr(a, x), gamma_ur(a, x));
        assert!((lower + upper - 1.0).abs() <= 1e-12, "at a={a:e}, x={x:e}");
        assert!((0.0..=1.0).contains(&lower) && (0.0..=1.0).contains(&upper));
    }
    // The limits are exact in both tails, not merely close.
    assert_eq!((gamma_lr(2.0, 0.0), gamma_ur(2.0, 0.0)), (0.0, 1.0));
    assert_eq!(
        (gamma_lr(2.0, f64::INFINITY), gamma_ur(2.0, f64::INFINITY)),
        (1.0, 0.0)
    );
    assert!(gamma_lr(0.0, 1.0).is_nan() && gamma_ur(0.0, 1.0).is_nan());
}

/// The five `Γ` routes, with the two poles kept apart.
///
/// The pole split is the load-bearing part: `Γ` has opposite signs across a negative integer so
/// no limit exists, while at the origin the sign of the zero says which side is meant. Routing
/// either one into the reflection is how `Γ(−20) = 5.27e-4` happened — a number that looks
/// entirely plausible.
#[test]
fn the_gamma_methods_keep_the_two_pole_kinds_apart() {
    use GammaMethod::{Lanczos, Reflected, Saturated, SignedPole, UnsignedPole};
    for (x, want) in [
        (0.0, SignedPole),
        (-0.0, SignedPole),
        (-1.0, UnsignedPole),
        (-5.0, UnsignedPole),
        (-20.0, UnsignedPole),
        (-1e15, UnsignedPole),
        // Non-integers below a half reflect; the half itself is the fixed point and does not.
        (-0.5, Reflected),
        (-20.5, Reflected),
        (0.25, Reflected),
        (0.5_f64.next_down(), Reflected),
        (0.5, Lanczos),
        (1.0, Lanczos),
        (GAMMA_OVERFLOW_MIN, Lanczos),
        (GAMMA_OVERFLOW_MIN.next_up(), Saturated),
        (1e300, Saturated),
    ] {
        assert_eq!(GammaMethod::for_argument(x), want, "at {x:e}");
    }
    // And the values the two pole arms produce, which is what the split is for.
    assert_eq!(gamma(0.0), f64::INFINITY);
    assert_eq!(gamma(-0.0), f64::NEG_INFINITY);
    assert!(gamma(-1.0).is_nan() && gamma(-20.0).is_nan());
    assert_eq!(gamma(GAMMA_OVERFLOW_MIN.next_up()), f64::INFINITY);
    // A non-integer just off a pole is finite and enormous, not NaN — the distinction the
    // integer test exists to preserve.
    assert!(gamma(-20.0 + 1e-9).is_finite());
}

/// The digamma reductions, which divide the line by how `ψ` is *reached* rather than by value.
///
/// `Undefined` and `Pole` both answer `NaN`, so only the decision separates them — and the
/// distinction is real: `−∞` has no answer because `ψ` oscillates there, while a negative integer
/// has none because `ψ` takes opposite signs on the two sides of it.
#[test]
fn the_digamma_reductions_name_every_region() {
    use super::PsiReduction::{Asymptotic, NearZero, Pole, Reflected, Undefined};

    assert_eq!(PsiReduction::for_digamma(f64::NAN), Undefined);
    assert_eq!(PsiReduction::for_digamma(f64::NEG_INFINITY), Undefined);
    // `+∞` is not undefined: the series gives `ψ(+∞) = +∞`.
    assert_eq!(PsiReduction::for_digamma(f64::INFINITY), Asymptotic);
    assert_eq!(digamma(f64::INFINITY), f64::INFINITY);

    // Negative integers are poles; everything else left of zero reflects.
    for x in [-1.0, -2.0, -39.0, -1e15] {
        assert_eq!(PsiReduction::for_digamma(x), Pole, "at {x:e}");
        assert!(digamma(x).is_nan(), "at {x:e}");
    }
    for x in [
        -0.5,
        -1.5,
        -0.001,
        (-1.0_f64).next_up(),
        (-1.0_f64).next_down(),
    ] {
        assert_eq!(PsiReduction::for_digamma(x), Reflected, "at {x:e}");
        assert!(digamma(x).is_finite(), "at {x:e}");
    }

    // The Taylor window about `ψ`'s real zero, at and either side of both edges.
    let (lo, hi) = (PSI_ZERO - PSI_ZERO_WINDOW, PSI_ZERO + PSI_ZERO_WINDOW);
    assert_eq!(PsiReduction::for_digamma(PSI_ZERO), NearZero);
    assert_eq!(PsiReduction::for_digamma(lo), NearZero);
    assert_eq!(PsiReduction::for_digamma(hi), NearZero);
    assert_eq!(PsiReduction::for_digamma(lo.next_down()), Asymptotic);
    assert_eq!(PsiReduction::for_digamma(hi.next_up()), Asymptotic);
    // `ψ` really does vanish there, which is what the window exists to resolve.
    assert!(digamma(PSI_ZERO).abs() < 1e-15, "{}", digamma(PSI_ZERO));

    // Zero itself, and the rest of the positive axis.
    for x in [0.0, 1.0, 0.5, 19.0, f64::from(PSI_ASYMPTOTIC_MIN), 1e300] {
        assert_eq!(PsiReduction::for_digamma(x), Asymptotic, "at {x:e}");
    }
    // The two arms of the walk-up agree across the threshold: both are the same series after the
    // recurrence, so the only thing the comparison changes is how many terms are shed.
    let threshold = f64::from(PSI_ASYMPTOTIC_MIN);
    close(digamma(threshold.next_down()), digamma(threshold), 1e-14);
}
