//! The [`NormedDivisionAlgebra`] tier on the float carriers, which nothing was checking.
//!
//! # Why this file exists
//!
//! `Real` is `NormedDivisionAlgebra<RealPart = Self> + Compare`, so every float carrier inherits
//! five required methods from that tier: `conjugate`, `norm_sqr`, `real_part`, `from_real` and
//! `magnitude`. A full mutation run over `euclid-scalar` found **every one of them replaceable
//! with a constant** — `0.0`, `1.0`, `-1.0` — in both precisions, across sixteen survivors in
//! `impls/real_f64_support.rs` and sixteen more in its f32 peer.
//!
//! These tests cover the five inherited operations directly. The existing `real_laws` suite
//! exercises the real-analysis tier.
//!
//! It is the same finding as `euclid-algebra`'s `f32_satisfies_the_tower_it_claims`, which
//! records `div_rem` and `conjugate` being replaceable with `Default::default()` for the same
//! reason: the carrier list exercised the tiers it was written for and the inherited ones rode
//! along.
//!
//! # These are laws, not restatements
//!
//! Each assertion below relates two different operations, so none of them is the implementation
//! written twice. The important one is the third:
//!
//! - `norm_sqr(x) = x · conjugate(x)` — the defining relation of the norm, against `*`
//! - `magnitude(x)² = norm_sqr(x)` — ties the two norms together, against `*`
//! - **`|x·y| = |x|·|y|`** — multiplicativity, which is what the word *normed* means in
//!   "normed division algebra". A `magnitude` that returned a constant, or dropped a factor,
//!   fails this and nothing else here would notice.
//! - `from_real(real_part(x)) = x` — a round trip through the embedding
//! - `magnitude(x) ≥ 0`, and zero only at zero — the division-algebra requirement, which is why
//!   a reciprocal exists for every nonzero element
//!
//! # On the real line, two of the five are the identity — and that is the point
//!
//! `conjugate` and `real_part` are both `x` for a real carrier, and `from_real` is too. Asserting
//! that looks vacuous and is not: the trait's documentation promises exactly that ("the identity
//! for a real scalar"), it is the contract a generic algorithm relies on, and a carrier returning
//! `0.0` instead satisfies no law anywhere else in the crate. `conjugate` in particular is the
//! method that makes an adjoint or Hermitian algorithm correct over both ℝ and ℂ — getting it
//! wrong is silently real-only, which `euclid-algebra`'s `field_laws` makes the same argument
//! about.

use crate::traits::NormedDivisionAlgebra;
use euclid_utils::assert_exact;

/// Values spanning sign, magnitude and the awkward edges, exactly representable.
///
/// Powers of two and small integers only: the multiplicativity law forms `x·y` and compares it
/// against `|x|·|y|`, so a sample that rounds would be measuring the rounding rather than the
/// law. Same reasoning as `euclid-algebra`'s float carrier sample, which records the same
/// constraint.
const SAMPLE: &[f64] = &[0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 8.0, -0.125];

#[test]
fn the_f64_carrier_satisfies_the_normed_division_algebra_laws() {
    for &x in SAMPLE {
        assert_exact(
            x.conjugate(),
            x,
            "f64: conjugate is the identity on the reals",
        );
        assert_exact(
            x.real_part(),
            x,
            "f64: real_part is the identity on the reals",
        );
        assert_exact(
            <f64 as NormedDivisionAlgebra>::from_real(x.real_part()),
            x,
            "f64: from_real(real_part(x)) = x",
        );
        // The defining relation, against `*` rather than against `norm_sqr`'s own body.
        assert_exact(
            x.norm_sqr(),
            x * x.conjugate(),
            "f64: norm_sqr = x * conj(x)",
        );
        assert_exact(
            x.magnitude() * x.magnitude(),
            x.norm_sqr(),
            "f64: |x|^2 = norm_sqr(x)",
        );
        assert!(x.magnitude() >= 0.0, "f64: |{x}| must not be negative");
        assert_eq!(
            x.magnitude() == 0.0,
            x == 0.0,
            "f64: |x| vanishes exactly at zero, or there is no reciprocal"
        );
        // Even, which a signed `magnitude` would fail.
        assert_exact((-x).magnitude(), x.magnitude(), "f64: |-x| = |x|");
    }
    // Multiplicativity: the content of the word "normed".
    for &x in SAMPLE {
        for &y in SAMPLE {
            assert_exact(
                (x * y).magnitude(),
                x.magnitude() * y.magnitude(),
                "f64: |x*y| = |x| * |y|",
            );
        }
    }
}

#[test]
fn the_f32_carrier_satisfies_the_normed_division_algebra_laws() {
    // The f32 impl is a separate body, so it needs the same suite rather than riding along —
    // which is the whole reason these survivors came in pairs.
    #[allow(clippy::cast_possible_truncation)]
    for &wide in SAMPLE {
        let x = wide as f32;
        assert_eq!(
            x.conjugate(),
            x,
            "f32: conjugate is the identity on the reals"
        );
        assert_eq!(
            x.real_part(),
            x,
            "f32: real_part is the identity on the reals"
        );
        assert_eq!(
            <f32 as NormedDivisionAlgebra>::from_real(x.real_part()),
            x,
            "f32: from_real(real_part(x)) = x"
        );
        assert_eq!(
            x.norm_sqr(),
            x * x.conjugate(),
            "f32: norm_sqr = x * conj(x)"
        );
        assert_eq!(
            x.magnitude() * x.magnitude(),
            x.norm_sqr(),
            "f32: |x|^2 = norm_sqr(x)"
        );
        assert!(x.magnitude() >= 0.0, "f32: |{x}| must not be negative");
        assert_eq!(
            x.magnitude() == 0.0,
            x == 0.0,
            "f32: |x| vanishes exactly at zero"
        );
        assert_eq!((-x).magnitude(), x.magnitude(), "f32: |-x| = |x|");
    }
    #[allow(clippy::cast_possible_truncation)]
    for &wide_x in SAMPLE {
        for &wide_y in SAMPLE {
            let (x, y) = (wide_x as f32, wide_y as f32);
            assert_eq!(
                (x * y).magnitude(),
                x.magnitude() * y.magnitude(),
                "f32: |x*y| = |x| * |y|"
            );
        }
    }
}

/// The norm must also behave on the values a sample of powers of two cannot reach.
///
/// Separate from the laws above because these are *not* exact relations — `norm_sqr` squares, so
/// it overflows and underflows well before `magnitude` does, and that asymmetry is a property of
/// the tier worth stating rather than an edge to avoid.
#[test]
fn the_squared_norm_saturates_before_the_magnitude_does() {
    // `magnitude` is exact at every finite value, including the ones whose square is not.
    for &x in &[f64::MAX, f64::MIN, 1e200, -1e200, f64::MIN_POSITIVE, 5e-324] {
        assert_exact(x.magnitude(), x.abs(), "|x| is exact at every finite x");
    }
    // Its square is not representable there, and saying so is more useful than pretending.
    assert!(
        f64::MAX.norm_sqr().is_infinite(),
        "norm_sqr overflows where the magnitude does not"
    );
    assert_eq!(
        5e-324_f64.norm_sqr(),
        0.0,
        "norm_sqr underflows where the magnitude does not"
    );
    // Non-finite values pass through rather than being clamped.
    assert!(f64::NAN.magnitude().is_nan(), "NaN has no magnitude");
    assert_exact(
        f64::NEG_INFINITY.magnitude(),
        f64::INFINITY,
        "|-inf| = +inf",
    );
    // `assert_close` cannot be used here: it forms `|inf - inf|`, which is `NaN`, against a
    // bound that is also `NaN`. Infinities need an exact comparison or a predicate.
    assert_exact(
        f64::INFINITY.norm_sqr(),
        f64::INFINITY,
        "norm_sqr(inf) = inf",
    );
}
