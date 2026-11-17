//! `hypot3` and `sin_cos_degrees` — the two composites that **reduce their arguments** before
//! computing anything.
//!
//! See the [parent module](super) for why these need values rather than cross-precision budgets,
//! and [`super::elementary`] for the cancellation-prone forms this file was split from.
//!
//! What these two share is that the reduction is the algorithm. `hypot3` divides every component
//! by the largest before squaring, so that a representable answer is never reached through an
//! unrepresentable intermediate; `sin_cos_degrees` reduces modulo 360 *in degrees*, so that the
//! residual at a quadrant boundary is exactly zero. In both cases the naive expression is correct
//! in exact arithmetic and wrong in floating point, and in both cases the test has to visit the
//! argument where that difference appears — a mid-range sample exercises only the easy path.

use super::{F32_TOL, F64_TOL, agrees};
use crate::tests::*;
use core::fmt::Debug;

/// `hypot3(x, y, z) = sqrt(x² + y² + z²)`, scaled so no square overflows.
///
/// The Pythagorean quadruples are the reference — `1² + 2² + 2² = 3²`, `2² + 3² + 6² = 7²` —
/// exact in integers and therefore sourced from arithmetic rather than from the implementation.
///
/// # Which of them are exact in floating point, and why
///
/// Only those whose **largest component is a power of two**. The body divides each component by
/// the largest before squaring, and division by a power of two is exact, so `(1,2,2)`, `(1,4,8)`
/// and `(3,4,0)` survive the scaling with every bit intact and land on `3`, `9` and `5` exactly.
/// `(2,3,6)` divides by six, `2/6` is not dyadic, and `f32` returns `7.0000005`.
///
/// Worth separating rather than smoothing over with one tolerance: the exact group is a much
/// stronger assertion where it applies, and the reason it applies is a property of the algorithm
/// a reader should be able to see.
fn hypot3_laws<T>(tol: f64, what: &str)
where
    T: Real<Mask = bool> + Debug,
{
    let s = T::scalar;

    // Quadruples whose largest component is a power of two: exact at both precisions.
    for &(x, y, z, h) in &[
        (1.0_f64, 2.0_f64, 2.0_f64, 3.0_f64),
        (1.0, 4.0, 8.0, 9.0),
        (3.0, 4.0, 0.0, 5.0),
    ] {
        let got = s(x).hypot3(s(y), s(z));
        assert!(
            got.eq(&s(h)),
            "{what}: hypot3({x}, {y}, {z}) = {h} exactly, got {got:?}",
        );
        // Permutation-invariant: the `max` scaling must not depend on which argument is largest.
        assert!(
            s(z).hypot3(s(x), s(y)).eq(&s(h)),
            "{what}: hypot3 is symmetric"
        );
        assert!(
            s(y).hypot3(s(z), s(x)).eq(&s(h)),
            "{what}: hypot3 is symmetric"
        );
        // Sign-independent.
        assert!(
            s(-x).hypot3(s(-y), s(-z)).eq(&s(h)),
            "{what}: hypot3 ignores signs"
        );
    }

    // The rest of the quadruples, to tolerance — the integer target is still an independent
    // reference, it simply is not reached bit-for-bit once the scaling divides by a non-dyadic.
    for &(x, y, z, h) in &[(2.0_f64, 3.0_f64, 6.0_f64, 7.0_f64), (4.0, 4.0, 7.0, 9.0)] {
        agrees(
            &s(x).hypot3(s(y), s(z)),
            h,
            tol,
            what,
            "a Pythagorean quadruple",
        );
        agrees(
            &s(z).hypot3(s(x), s(y)),
            h,
            tol,
            what,
            "hypot3 is symmetric",
        );
        agrees(
            &s(-x).hypot3(s(-y), s(-z)),
            h,
            tol,
            what,
            "hypot3 ignores signs",
        );
    }

    // All-zero, the branch guarding the division by the maximum.
    assert!(
        s(0.0).hypot3(s(0.0), s(0.0)).eq(&T::zero()),
        "{what}: hypot3(0, 0, 0) = 0",
    );

    // The scaling that justifies the algorithm: each component squared would overflow, while the
    // answer is comfortably representable. Without the divide-by-maximum this is `+∞`.
    let big = s(1e30);
    let expected = 1e30 * 3.0_f64.sqrt();
    agrees(
        &big.clone().hypot3(big.clone(), big),
        expected,
        tol,
        what,
        "hypot3 does not overflow",
    );

    // And the other direction: components whose squares underflow to zero.
    let small = s(1e-30);
    let expected = 1e-30 * 3.0_f64.sqrt();
    agrees(
        &small.clone().hypot3(small.clone(), small),
        expected,
        tol,
        what,
        "hypot3 does not underflow",
    );

    // Non-finite components: an infinity dominates, but a NaN must still win over it.
    assert!(
        T::infinity().hypot3(s(1.0), s(2.0)).eq(&T::infinity()),
        "{what}: an infinite component gives ∞",
    );
    assert!(
        T::infinity().hypot3(T::nan(), s(2.0)).is_nan(),
        "{what}: a NaN component beats an infinite one",
    );
    assert!(
        s(1.0).hypot3(T::nan(), s(2.0)).is_nan(),
        "{what}: NaN propagates",
    );
}

/// `sin_cos_degrees` — GeographicLib's `sincosd`, which is **exact at every multiple of 90°**.
///
/// That exactness is the entire reason the function exists: reducing in degrees leaves a residual
/// of exactly zero at each quadrant boundary, which `to_radians()` first cannot do because `π/2`
/// is not representable. So the quadrant values are asserted with `==`, and that is also what
/// catches a deleted or reordered match arm — eleven of the survivors were in that `match`.
fn sin_cos_degrees_laws<T>(tol: f64, what: &str)
where
    T: Real<Mask = bool> + Debug,
{
    let s = T::scalar;
    let one = T::one();
    let zero = T::zero();

    // The four quadrant multiples, exact. A rotated `match` arm moves one of these.
    let quadrants: [(f64, T, T); 5] = [
        (0.0, zero.clone(), one.clone()),
        (90.0, one.clone(), zero.clone()),
        (180.0, zero.clone(), -one.clone()),
        (270.0, -one.clone(), zero.clone()),
        (360.0, zero.clone(), one.clone()),
    ];
    for (angle, want_sin, want_cos) in &quadrants {
        let (got_sin, got_cos) = s(*angle).sin_cos_degrees();
        assert!(
            got_sin.eq(want_sin) && got_cos.eq(want_cos),
            "{what}: sin_cos_degrees({angle}) must be exactly \
             ({want_sin:?}, {want_cos:?}), got ({got_sin:?}, {got_cos:?})",
        );
    }

    // The `% 360` reduction: full turns must not change the answer, in either direction.
    for &turns in &[1.0_f64, 2.0, 10.0, -1.0, -3.0] {
        let (got_sin, got_cos) = s(90.0 + 360.0 * turns).sin_cos_degrees();
        assert!(
            got_sin.eq(&one) && got_cos.eq(&zero),
            "{what}: 90° + {turns} turns must still be (1, 0), got ({got_sin:?}, {got_cos:?})",
        );
    }
    // Negative quadrants.
    for &(angle, want_sin, want_cos) in &[(-90.0_f64, -1.0_f64, 0.0_f64), (-180.0, 0.0, -1.0)] {
        let (got_sin, got_cos) = s(angle).sin_cos_degrees();
        assert!(
            got_sin.eq(&s(want_sin)) && got_cos.eq(&s(want_cos)),
            "{what}: sin_cos_degrees({angle}) = ({want_sin}, {want_cos})",
        );
    }

    // Off the quadrants, agreement with the radian pair — the ordinary case.
    for &angle in &[30.0_f64, 45.0, 60.0, 123.0, -37.0, 200.0] {
        let (got_sin, got_cos) = s(angle).sin_cos_degrees();
        let radians = angle.to_radians();
        agrees(
            &got_sin,
            radians.sin(),
            tol,
            what,
            "sin_cos_degrees agrees in sine",
        );
        agrees(
            &got_cos,
            radians.cos(),
            tol,
            what,
            "sin_cos_degrees agrees in cosine",
        );
    }
    // 45° is where sine and cosine coincide, which a swapped pair of arms would break.
    let (got_sin, got_cos) = s(45.0).sin_cos_degrees();
    agrees(
        &got_sin,
        core::f64::consts::FRAC_1_SQRT_2,
        tol,
        what,
        "sin 45° = 1/√2",
    );
    agrees(
        &got_cos,
        core::f64::consts::FRAC_1_SQRT_2,
        tol,
        what,
        "cos 45° = 1/√2",
    );

    // IEEE 754 F.10.1.12/13, which the last three lines of the body implement: a vanished cosine
    // is `+0`, and a vanished sine carries the argument's sign.
    let (sin_neg_zero, cos_neg_zero) = s(-0.0).sin_cos_degrees();
    assert!(
        sin_neg_zero.signbit(),
        "{what}: sin(−0°) must be −0, got {sin_neg_zero:?}",
    );
    assert!(cos_neg_zero.eq(&one), "{what}: cos(−0°) = 1");
    let (sin_180, cos_90) = (s(180.0).sin_cos_degrees().0, s(90.0).sin_cos_degrees().1);
    assert!(
        !cos_90.signbit(),
        "{what}: cos(90°) must be +0, got {cos_90:?}",
    );
    assert!(
        !sin_180.signbit(),
        "{what}: sin(180°) must be +0 for a positive argument, got {sin_180:?}",
    );
    let sin_neg_180 = s(-180.0).sin_cos_degrees().0;
    assert!(
        sin_neg_180.signbit(),
        "{what}: sin(−180°) must be −0, got {sin_neg_180:?}",
    );

    // Non-finite input: the residual is NaN by construction, so both outputs are.
    let (nan_sin, nan_cos) = T::infinity().sin_cos_degrees();
    assert!(
        nan_sin.is_nan() && nan_cos.is_nan(),
        "{what}: sin_cos_degrees(∞) must be (NaN, NaN)",
    );
}

#[test]
fn the_f64_carrier_reduces_before_computing() {
    hypot3_laws::<f64>(F64_TOL, "f64");
    sin_cos_degrees_laws::<f64>(F64_TOL, "f64");
}

#[test]
fn the_f32_carrier_reduces_before_computing() {
    hypot3_laws::<f32>(F32_TOL, "f32");
    sin_cos_degrees_laws::<f32>(F32_TOL, "f32");
}
