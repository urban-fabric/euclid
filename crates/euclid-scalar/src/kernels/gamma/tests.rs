//! Unit tests for `Γ` and `lnΓ` themselves.
//!
//! A child module so implementation plus tests stay inside the crate's 400-line budget, while
//! `use super::*` still reaches the private Lanczos coefficients and both method tables. The
//! incomplete pair and the polygammas keep their tests beside their own modules, in
//! [`super::incomplete`] and [`super::psi`], because what those assert is private to them.

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

/// The seven `lnΓ` regimes, at and either side of every boundary.
///
/// Each arm reaches a correct answer by a different route, so a misrouted argument still returns
/// something plausible — which is exactly how the pole residue defect survived before the poles
/// were tested for. The table is what makes the ordering checkable.
#[test]
fn the_log_gamma_methods_partition_the_line() {
    use LnGammaMethod::{
        Divergent, Lanczos, NearTwoZero, NearUnitZero, Recurrence, Reflected, SmallShape,
        Undefined, UnitInterval,
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
        //
        // It used to fall through to `Lanczos`, which is what made this edge worth a note in
        // the first place; the factored window now catches it, so missing the neighbourhood by
        // one ulp costs nothing at all.
        (1.0 - SMALL_SHAPE, UnitInterval),
        // The factored window and the recurrence above it, at and either side of both edges.
        (LN_GAMMA_UNIT_MIN, UnitInterval),
        // Just below the window the reflection takes over — and lands back inside it, since
        // `1 − x` is then just above a half.
        (LN_GAMMA_UNIT_MIN.next_down(), Reflected),
        (LN_GAMMA_UNIT_MAX, UnitInterval),
        (LN_GAMMA_UNIT_MAX.next_up(), Recurrence),
        (1.3, UnitInterval),
        (LN_GAMMA_RECURRENCE_MAX, Recurrence),
        (LN_GAMMA_RECURRENCE_MAX.next_up(), Lanczos),
        // The pole at the origin, approached from above only.
        (SMALL_SHAPE, SmallShape),
        (1e-30, SmallShape),
        // The negative axis reflects; the rest is Lanczos.
        (-0.5, Reflected),
        (-1.5, Reflected),
        (-1e5 - 0.5, Reflected),
        (5.0, Lanczos),
        (1e30, Lanczos),
    ] {
        assert_eq!(LnGammaMethod::for_argument(x), want, "at {x:e}");
    }
    // The neighbourhoods are closed, so one step outside each leaves the shifted series — and
    // now lands in the factored window rather than on Lanczos, which is the whole point of it.
    assert_eq!(
        LnGammaMethod::for_argument((1.0 + SMALL_SHAPE).next_up()),
        UnitInterval
    );
    assert_eq!(
        LnGammaMethod::for_argument((2.0 + SMALL_SHAPE).next_up()),
        UnitInterval
    );
    // Just above the small-shape cutoff the route is *Reflected*, not Lanczos: the reflection
    // applies to everything below `½`, which includes the small positives the shifted series
    // stops covering. That ordering is the original's and is preserved here.
    assert_eq!(
        LnGammaMethod::for_argument(SMALL_SHAPE.next_up()),
        Reflected
    );
    assert_eq!(LnGammaMethod::for_argument(0.25), Reflected);
    // A half is the first argument the factored window admits, and the reflection stops exactly
    // there — so the two thresholds meet without a gap or an overlap.
    assert_eq!(LnGammaMethod::for_argument(0.5), UnitInterval);
    // No continuity assertion across the zero-neighbourhood joins, and that is the point of
    // them: the shifted series and Lanczos *deliberately* disagree there by about an ulp of one
    // — measured `2.53e-16` at `x = 1.001` — because Lanczos carries an absolute error of
    // `−8.88e-16` beside a zero where the answer is itself near zero. The shifted arm is the
    // correct one, so a step at the join is the fix working rather than a defect. See
    // `LnGammaMethod::NearUnitZero`.
}

/// The five `Γ` routes, with the two poles kept apart.
///
/// The pole split is the load-bearing part: `Γ` has opposite signs across a negative integer so
/// no limit exists, while at the origin the sign of the zero says which side is meant. Routing
/// either one into the reflection is how `Γ(−20) = 5.27e-4` happened — a number that looks
/// entirely plausible.
#[test]
fn the_gamma_methods_keep_the_two_pole_kinds_apart() {
    use GammaMethod::{Lanczos, Reduced, Reflected, Saturated, SignedPole, Stirling, UnsignedPole};
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
        // Lanczos keeps only `[½, 1)` now: from one upward the reduction to `[1,2)` is ten to
        // twenty times more accurate, and above `STIRLING_MIN` the Stirling series already was.
        (0.5, Lanczos),
        (1.0_f64.next_down(), Lanczos),
        (1.0, Reduced),
        (1.9999, Reduced),
        (7.5, Reduced),
        // The reduction/Stirling handover, closed on the Stirling side. Its placement is an
        // accuracy decision, not a range one — see `math::gamma::STIRLING_MIN`.
        (crate::math::gamma::STIRLING_MIN.next_down(), Reduced),
        (crate::math::gamma::STIRLING_MIN, Stirling),
        (GAMMA_OVERFLOW_MIN, Stirling),
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

/// The factored window satisfies `lnΓ(x+1) = lnΓ(x) + ln x`, which needs no reference.
///
/// # Why a recurrence rather than a value table
///
/// [`LN_GAMMA_UNIT`] is thirty-nine machine-generated coefficients. A wrong one does not make
/// `lnΓ` ragged or obviously broken — it shifts a smooth function by a smooth amount, which a
/// handful of pinned values can easily straddle without noticing. The recurrence constrains the
/// *whole* window against itself: `x` and `x + 1` are evaluated at different points of the same
/// Chebyshev series and have to agree through an identity the series knows nothing about.
///
/// `x ∈ [½, 3/2]` is the range where both `x` and `x + 1` are inside the window, so this tests
/// the factored route against itself rather than against a neighbour.
///
/// # The grid is in sixty-fourths, and that is not cosmetic
///
/// `x + 1` has to be **exact**, and for a general `x` near one it is not: `x` there carries bits
/// at `2^-53` while `x + 1` can only hold `2^-52`, so the sum rounds and the identity is being
/// asked about two arguments that are not one apart. On a decimal grid this showed up as `55`
/// ulp at `x = 1.01`, which is `ψ(2.01)` times the `2.2e-16` by which `fl(2.01)` and `fl(1.01)+1`
/// differ — the kernel was right at both points. Sixty-fourths need six bits, so `x`, `x + 1`
/// and the subtraction back off them are all exact.
///
/// Measured worst over the arguments below: `3` ulp. The bar is `8`, which is the `3` ulp that
/// `ln x` and the addition can contribute on top of the two evaluations, rounded up.
#[test]
fn the_factored_window_satisfies_the_gamma_recurrence() {
    let mut worst = (0.0_f64, 0.0_f64);
    let mut checked = 0usize;
    for step in 0..=64 {
        let x = 0.5 + f64::from(step) / 64.0;
        let shifted = ln_gamma(x + 1.0);
        let expected = ln_gamma(x) + x.ln();
        // `lnΓ(x+1)` passes through zero at `x = 1`, where a relative comparison is meaningless.
        // Scaled against the size of the terms it is built from instead.
        let scale = ln_gamma(x).abs().max(x.ln().abs()).max(f64::MIN_POSITIVE);
        let ulps = (shifted - expected).abs() / crate::Real::ulp(scale);
        if ulps > worst.0 {
            worst = (ulps, x);
        }
        assert!(
            ulps <= 8.0,
            "at x = {x} the recurrence gives {expected} and lnΓ(x+1) is {shifted}, \
             {ulps} ulp apart"
        );
        checked += 1;
    }
    assert!(
        checked > 60,
        "only {checked} arguments reached the assertion"
    );
    assert!(
        worst.0 > 0.0,
        "the recurrence held bit-exactly everywhere, so it compared nothing at {}",
        worst.1
    );
}

/// The two zeros come out exactly zero, and the slope through each is right.
///
/// This is what factoring buys that refinement cannot: `(x−1)` and `(x−2)` are exact on the
/// window, so the product is exactly zero at `1` and `2` whatever the coefficients say. Beside
/// the zero the answer is then `−γ·h` and `(1−γ)·h`, which is the *relative* accuracy a log
/// density at shape one needs and which Lanczos could not give — it returned `−8.88e-16` at
/// `x = 1` where the answer is zero.
#[test]
fn the_factored_window_has_exact_zeros_and_the_right_slopes() {
    assert_eq!(ln_gamma(1.0), 0.0);
    assert_eq!(ln_gamma(2.0), 0.0);

    // Powers of two, so `1 + h` and `2 + h` are exact and the kernel sees the `h` this test
    // compares against. A decimal `h` loses its low bits to the addition — `1.0 + 1e-9` is not
    // `1 + 1e-9` — and the slope then misses by `1e-7` for reasons that have nothing to do with
    // `lnΓ`.
    let euler = crate::constants::f64::EULER_GAMMA;
    for &h in &[
        f64::from_bits(0x3E10_0000_0000_0000),
        1.0 / 1_048_576.0,
        1.0 / 1024.0,
    ] {
        // `lnΓ(1+h) ≈ −γh` and `lnΓ(2+h) ≈ (1−γ)h`, to first order in `h`.
        let near_one = ln_gamma(1.0 + h);
        let near_two = ln_gamma(2.0 + h);
        assert!(
            (near_one / (-euler * h) - 1.0).abs() <= 2.0 * h,
            "lnΓ(1+{h}) = {near_one}, which is not −γ·h to first order"
        );
        assert!(
            (near_two / ((1.0 - euler) * h) - 1.0).abs() <= 2.0 * h,
            "lnΓ(2+{h}) = {near_two}, which is not (1−γ)·h to first order"
        );
    }
    // And the sign is negative strictly between the zeros, positive outside.
    assert!(ln_gamma(1.5) < 0.0 && ln_gamma(0.75) > 0.0 && ln_gamma(2.5) > 0.0);
}

/// The reduced range reproduces the factorials exactly, and its own recurrence.
///
/// # Why integers, and why the recurrence as well
///
/// `Γ(n) = (n−1)!` is exact in `f64` up to `n = 23`, so these are reference-free assertions
/// about a route whose whole content is a product of exact subtractions times one `exp`. A
/// coefficient-level defect in the `lnΓ` window it rests on shows up here as a factorial that is
/// off by an ulp — which no amount of smoothness hides, because the target is an integer.
///
/// The recurrence then covers the non-integers between them: `Γ(x+1) = x·Γ(x)` constrains the
/// whole range against itself, and unlike the factorials it does not care whether the reduction
/// landed on a representable answer.
#[test]
fn the_reduced_range_reproduces_the_factorials_and_its_own_recurrence() {
    let mut factorial = 1.0_f64;
    for n in 1..=18_u32 {
        // Γ(n) = (n−1)!, with the factorial built exactly — every one of these fits in f64.
        let value = gamma(f64::from(n));
        assert_eq!(
            value, factorial,
            "Γ({n}) is {value}, and {n}−1 factorial is {factorial}"
        );
        factorial *= f64::from(n);
    }

    // The recurrence across the reduced range, on a grid where `x + 1` is exact.
    let mut worst = (0.0_f64, 0.0_f64);
    for step in 0..=120 {
        let x = 1.0 + f64::from(step) * 0.125;
        if x + 1.0 >= crate::math::gamma::STIRLING_MIN {
            break;
        }
        let shifted = gamma(x + 1.0);
        let expected = x * gamma(x);
        let ulps = (shifted - expected).abs() / crate::Real::ulp(shifted);
        if ulps > worst.0 {
            worst = (ulps, x);
        }
        assert!(
            ulps <= 8.0,
            "at x = {x} the recurrence gives {expected} and Γ(x+1) is {shifted}, {ulps} apart"
        );
    }
    assert!(
        worst.0 > 0.0,
        "the recurrence held exactly everywhere, so it compared nothing"
    );
}
