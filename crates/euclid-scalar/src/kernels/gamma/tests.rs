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
