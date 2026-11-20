//! Resolving both sides of a remote pole.
//!
//! Part of the [`distribution_extremes`](super) suite. A pole far from the origin is where the
//! reflection formulae are at their worst: the distance to the nearest pole is the whole
//! magnitude of the answer, and rounding the angle before the sine destroys exactly that.

use super::*;

#[test]
fn digamma_resolves_both_sides_of_a_remote_pole() {
    // At -n + d, the leading pole is -1/d and the remaining finite part is O(ln(n)).
    let pole = -10_000_000_000_000.0_f64;
    for x in [pole.next_down(), pole.next_up()] {
        let value = x.digamma();
        let residual = value + 1.0 / (x - pole);
        assert!((29.0..31.0).contains(&residual), "digamma({x}) = {value}");
    }
    let pole = -10_000.0_f32;
    for x in [pole.next_down(), pole.next_up()] {
        let value = x.digamma();
        let residual = value + 1.0 / (x - pole);
        assert!((9.0..10.0).contains(&residual), "digamma({x}) = {value}");
    }
}

#[test]
fn trigamma_resolves_both_sides_of_a_remote_pole() {
    // Isolate the pole term in the series for ψ₁. As d tends to zero at
    // a remote negative integer, the other terms approach 2ζ(2) = π²/3.
    let pole = -10_000_000_000_000.0_f64;
    for x in [pole.next_down(), pole.next_up()] {
        let offset = x - pole;
        let residual = x.trigamma() - (offset * offset).recip();
        assert!(
            (3.28..3.30).contains(&residual),
            "trigamma({x}) residual {residual}"
        );
    }
    let pole = -10_000.0_f32;
    for x in [pole.next_down(), pole.next_up()] {
        let offset = f64::from(x - pole);
        let residual = f64::from(x.trigamma()) - (offset * offset).recip();
        assert!(
            (3.0..3.5).contains(&residual),
            "trigamma({x}) residual {residual}"
        );
    }
}

/// `ψ` must resolve its positive zero: three adjacent `f32` arguments, three distinct signs.
///
/// # The bound is `1e-11`, and why it is not `2e-14`
///
/// This is the regression for a native `f32` recurrence that returned **the wrong sign** at
/// `0x3fbb_16c3`. The fix at the time was to narrow the `f64` kernel, and the bound was then set
/// at `2e-14` — which is what an `f64` computation rounded to `f64` can deliver, not what an
/// `f32` kernel can.
///
/// `digamma` is native again, so the bound has to be one a native kernel can meet. It is no
/// longer the recurrence, though: `ψ(1+a) = (a − a₀)·h(a)` factors the zero out instead of
/// arriving at it by cancellation, which is what the old implementation could not do — see
/// `kernels::f32::gamma`'s `DIGAMMA_UNIT_H`. Measured, the three points come back at `2.6e-12`,
/// so `1e-11` holds with room while staying four orders tighter than the defect it guards
/// against. For scale, the plain recurrence returns **exactly `0.0`** at two of these three
/// arguments, which `1e-11` catches immediately.
///
/// The sign assertions are the point and are exact, so they do not depend on the bound at all.
#[test]
fn single_precision_digamma_resolves_its_positive_zero() {
    // References evaluated at the exact binary32 inputs by MPFR at 256 bits.
    for (bits, expected) in [
        (0x3fbb_16c2, -1.273_915_810_519_784_6e-7),
        (0x3fbb_16c3, -1.203_605_254_910_667e-8),
        (0x3fbb_16c4, 1.033_194_633_696_787_1e-7),
    ] {
        let actual = f64::from(f32::from_bits(bits).digamma());
        assert!((actual - expected).abs() < 1e-11, "{actual} vs {expected}");
        assert_eq!(
            actual < 0.0,
            expected < 0.0,
            "digamma({:e}) = {actual:e} has the wrong sign against {expected:e}",
            f32::from_bits(bits)
        );
    }
}

#[test]
fn digamma_handles_poles_and_signed_zero() {
    for pole in [-1.0, -f64::MAX, f64::NEG_INFINITY] {
        assert!(pole.digamma().is_nan());
    }
    for pole in [-1.0, -f32::MAX, f32::NEG_INFINITY] {
        assert!(pole.digamma().is_nan());
    }
    assert_eq!(0.0_f64.digamma(), f64::NEG_INFINITY);
    assert_eq!((-0.0_f64).digamma(), f64::INFINITY);
    assert_eq!(0.0_f32.digamma(), f32::NEG_INFINITY);
    assert_eq!((-0.0_f32).digamma(), f32::INFINITY);
}
