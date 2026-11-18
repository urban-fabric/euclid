//! The one-line delegations: `recip`, `log2`, `mul_add`, `ceil`, `atan2`, `to_radians`, the
//! finiteness predicates, and the rest of the surface that is a single call to the host float.
//!
//! See the [parent module](super) for why this module exists.
//!
//! # Why a one-line body still needs a test
//!
//! Because the line can be the wrong line, and nothing else notices. `impls/real_f32.rs` is
//! ninety methods of `f32::sqrt(*self)`-shaped delegation, and mutation testing found the whole
//! tail of it unpinned: `recip`, `log2`, `mul_add`, `ceil`, `to_radians`, `to_degrees` and
//! `ln_bessel_i0` could each return `Default::default()` — zero, for every input — and
//! `is_finite` and `is_infinite` could return a constant `true` or `false`, with the 1,200-test
//! workspace suite staying green.
//!
//! [`super::super::identities::cross_precision::delegating`] covers a dozen of these by
//! comparing f32 against f64, and that is the right instrument where it reaches. It does not
//! reach a *predicate* — `is_finite` returns a `Mask`, not a value to difference — and it does
//! not reach a method that needs two or three arguments.
//!
//! # Where the expectations come from
//!
//! These are the methods whose values are *specified* rather than approximated, so almost
//! everything here is an exact assertion against a number that is true by definition: `recip`
//! at a power of two, `log2` at a power of two, `ceil` at a half-integer, `atan2` at the axes.
//! Nothing here needs a tolerance, which is itself the reason these were easy to leave out and
//! cheap to put back.

use crate::tests::*;
use core::fmt::Debug;

/// The finiteness predicates, over the five classes of float.
///
/// Three predicates by five inputs, which is the whole truth table. `is_finite` and
/// `is_infinite` are not complements — NaN is neither — and that is the case a test built from
/// one of them and a negation would miss.
fn predicate_laws<T>(what: &str)
where
    T: Real<Mask = bool> + Debug,
{
    // (value, is_nan, is_infinite, is_finite)
    let cases: [(T, bool, bool, bool); 6] = [
        (T::zero(), false, false, true),
        (T::one(), false, false, true),
        (T::scalar(-2.5), false, false, true),
        (T::infinity(), false, true, false),
        (T::neg_infinity(), false, true, false),
        (T::nan(), true, false, false),
    ];
    for (value, nan, infinite, finite) in &cases {
        assert_eq!(value.is_nan(), *nan, "{what}: is_nan({value:?})");
        assert_eq!(
            value.is_infinite(),
            *infinite,
            "{what}: is_infinite({value:?})",
        );
        assert_eq!(value.is_finite(), *finite, "{what}: is_finite({value:?})");
    }
    // NaN is neither finite nor infinite — the reason all three exist separately.
    assert!(
        !T::nan().is_finite() && !T::nan().is_infinite(),
        "{what}: NaN is neither finite nor infinite",
    );
}

/// The methods whose results are exact at the arguments chosen, asserted with equality.
fn exact_value_laws<T>(what: &str)
where
    T: Real<Mask = bool> + Debug,
{
    let s = T::scalar;

    // `recip` at powers of two, where the reciprocal is exact.
    assert!(s(4.0).recip().eq(&s(0.25)), "{what}: recip(4) = 1/4");
    assert!(s(0.5).recip().eq(&s(2.0)), "{what}: recip(1/2) = 2");
    assert!(s(-8.0).recip().eq(&s(-0.125)), "{what}: recip(−8) = −1/8");
    assert!(
        s(1.0).recip().eq(&T::one()),
        "{what}: recip(1) = 1 — the mutation that returns zero dies here",
    );
    assert!(T::infinity().recip().eq(&T::zero()), "{what}: recip(∞) = 0",);

    // `log2` at powers of two, where the logarithm is an exact integer.
    assert!(s(1.0).log2().eq(&T::zero()), "{what}: log2(1) = 0");
    assert!(s(2.0).log2().eq(&T::one()), "{what}: log2(2) = 1");
    assert!(s(1024.0).log2().eq(&s(10.0)), "{what}: log2(1024) = 10");
    assert!(s(0.25).log2().eq(&s(-2.0)), "{what}: log2(1/4) = −2");

    // `mul_add(a, b)` is `self·a + b`, fused. Chosen so the exact answer is representable.
    assert!(
        s(3.0).mul_add(s(4.0), s(5.0)).eq(&s(17.0)),
        "{what}: 3·4 + 5 = 17",
    );
    assert!(
        s(0.5).mul_add(s(0.5), s(-0.25)).eq(&T::zero()),
        "{what}: 1/2·1/2 − 1/4 = 0",
    );
    // The argument order matters and is easy to transpose: `2·10 + 1` is 21, not `10·1 + 2`.
    assert!(
        s(2.0).mul_add(s(10.0), s(1.0)).eq(&s(21.0)),
        "{what}: mul_add takes (multiplier, addend) in that order",
    );

    // `ceil`, at half-integers either side of zero so the rounding direction is unambiguous.
    assert!(s(2.5).ceil().eq(&s(3.0)), "{what}: ceil(2.5) = 3");
    assert!(s(-2.5).ceil().eq(&-s(2.0)), "{what}: ceil(−2.5) = −2");
    assert!(s(3.0).ceil().eq(&s(3.0)), "{what}: ceil(3) = 3");
    assert!(
        s(0.1).ceil().eq(&T::one()),
        "{what}: ceil of a small positive is 1, not 0",
    );

    // `square`, which is `self * self` rather than a delegation.
    assert!(s(-3.0).square().eq(&s(9.0)), "{what}: (−3)² = 9");
    assert!(s(0.5).square().eq(&s(0.25)), "{what}: (1/2)² = 1/4");

    // `powi` with an integer exponent from the integer tier, lifted through `Integer` since
    // `f32` and `f64` pair with different widths there.
    let exponent = <T::Integer as Integer>::from_i64;
    assert!(
        s(2.0).powi(exponent(10)).eq(&s(1024.0)),
        "{what}: 2^10 = 1024",
    );
    assert!(s(-2.0).powi(exponent(3)).eq(&-s(8.0)), "{what}: (−2)³ = −8",);
    assert!(s(5.0).powi(exponent(0)).eq(&T::one()), "{what}: x^0 = 1",);
}

/// `atan2`, across all four quadrants and the axes.
///
/// A two-argument method, so the cross-precision pass cannot reach it, and it could return
/// `Default::default()` unchallenged. The axis values are exact multiples of `π/4` by
/// definition of the function, which makes them the natural reference.
fn atan2_laws<T>(tol: f64, what: &str)
where
    T: Real<Mask = bool> + Debug,
{
    let s = T::scalar;
    let quarter = core::f64::consts::FRAC_PI_2;
    let half = core::f64::consts::PI;

    // The four axes. `atan2(y, x)` takes `y` first, which is the transposition this catches.
    agrees(&s(0.0).atan2(s(1.0)), 0.0, tol, what, "atan2(0, 1) = 0");
    agrees(
        &s(1.0).atan2(s(0.0)),
        quarter,
        tol,
        what,
        "atan2(1, 0) = π/2",
    );
    agrees(&s(0.0).atan2(s(-1.0)), half, tol, what, "atan2(0, −1) = π");
    agrees(
        &s(-1.0).atan2(s(0.0)),
        -quarter,
        tol,
        what,
        "atan2(−1, 0) = −π/2",
    );
    // The diagonals, one per quadrant — where a sign error in either argument shows.
    agrees(
        &s(1.0).atan2(s(1.0)),
        core::f64::consts::FRAC_PI_4,
        tol,
        what,
        "atan2(1, 1) = π/4",
    );
    agrees(
        &s(1.0).atan2(s(-1.0)),
        3.0 * core::f64::consts::FRAC_PI_4,
        tol,
        what,
        "atan2(1, −1) = 3π/4",
    );
    agrees(
        &s(-1.0).atan2(s(-1.0)),
        -3.0 * core::f64::consts::FRAC_PI_4,
        tol,
        what,
        "atan2(−1, −1) = −3π/4",
    );
    agrees(
        &s(-1.0).atan2(s(1.0)),
        -core::f64::consts::FRAC_PI_4,
        tol,
        what,
        "atan2(−1, 1) = −π/4",
    );
    // `atan2(y, x) = atan(y/x)` in the right half-plane, tying it to its one-argument sibling.
    for &(y, x) in &[(1.0_f64, 2.0_f64), (-3.0, 4.0), (0.5, 0.25)] {
        let two_arg = s(y).atan2(s(x));
        let one_arg = s(y / x).atan();
        assert!(
            (two_arg - one_arg).abs().le(&s(tol)),
            "{what}: atan2({y}, {x}) must equal atan({y}/{x})",
        );
    }
}

/// The degree/radian conversions, and `ln_bessel_i0`.
fn conversion_laws<T>(tol: f64, what: &str)
where
    T: Real<Mask = bool> + Debug,
{
    let s = T::scalar;

    // Sourced from the definition of a degree, not from the constants the methods use.
    agrees(
        &s(180.0).to_radians(),
        core::f64::consts::PI,
        tol,
        what,
        "180° = π rad",
    );
    agrees(
        &s(90.0).to_radians(),
        core::f64::consts::FRAC_PI_2,
        tol,
        what,
        "90° = π/2 rad",
    );
    agrees(
        &core::f64::consts::PI.into_scalar::<T>().to_degrees(),
        180.0,
        tol,
        what,
        "π rad = 180°",
    );
    assert!(T::zero().to_radians().eq(&T::zero()), "{what}: 0° = 0 rad",);
    assert!(T::zero().to_degrees().eq(&T::zero()), "{what}: 0 rad = 0°",);
    // Round trip, which needs no reference at all.
    for &d in &[1.0_f64, 45.0, 123.5, -30.0] {
        let back = s(d).to_radians().to_degrees();
        agrees(&back, d, tol, what, "degrees round trip");
    }

    // `ln_bessel_i0` — the log of the modified Bessel `I₀`, whose value at zero is `ln 1 = 0`.
    assert!(
        T::zero().ln_bessel_i0().eq(&T::zero()),
        "{what}: ln I₀(0) = ln 1 = 0",
    );
    // Against its own definition, where `bessel_i0` is finite.
    for &x in &[0.5_f64, 1.0, 3.0, 8.0] {
        let direct = s(x).ln_bessel_i0();
        let composed = s(x).bessel_i0().ln();
        assert!(
            (direct - composed).abs().le(&s(tol * 10.0)),
            "{what}: ln_bessel_i0({x}) must agree with ln(bessel_i0({x}))",
        );
    }
    // And it must stay finite far past where `I₀` itself overflows — the reason it exists.
    assert!(
        s(200.0).ln_bessel_i0().is_finite(),
        "{what}: ln I₀ must survive where I₀ overflows",
    );
}

/// Lets a plain `f64` be lifted into the carrier under test without naming it twice.
trait IntoScalar {
    fn into_scalar<T: Real>(self) -> T;
}
impl IntoScalar for f64 {
    fn into_scalar<T: Real>(self) -> T {
        T::scalar(self)
    }
}

#[test]
fn the_f64_carrier_computes_the_delegated_surface() {
    predicate_laws::<f64>("f64");
    exact_value_laws::<f64>("f64");
    atan2_laws::<f64>(1e-15, "f64");
    conversion_laws::<f64>(1e-14, "f64");
}

#[test]
fn the_f32_carrier_computes_the_delegated_surface() {
    predicate_laws::<f32>("f32");
    exact_value_laws::<f32>("f32");
    atan2_laws::<f32>(1e-6, "f32");
    conversion_laws::<f32>(1e-6, "f32");
}
