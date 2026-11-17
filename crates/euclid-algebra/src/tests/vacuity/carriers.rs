//! Carriers contracts and rejection controls.

use super::*;

// ===== Surfaces nothing was exercising ====================================

/// `f32`'s place in the tower, which no suite was instantiating.
///
/// Mutation testing found `<f32 as EuclideanDomain>::div_rem` and `<f32 as Field>::conjugate`
/// both replaceable with `Default::default()` — returning `(0.0, 0.0)` and `0.0` — with the whole
/// workspace green. The reason is simple and worth stating: the carrier list in
/// [`laws`](super::laws) checks `i64`, `Rational64`, `Zp`, `Zn` and `GaussianRational`, and the
/// float impls ride along only through `f64`. `f32` has its own impl block and nothing reached
/// it.
///
/// For a field the remainder is always zero, which is what makes `div_rem` on a float look
/// trivial enough to skip.
///
/// # The sample is all powers of two, and it has to be
///
/// Writing this test turned up something worth recording: **no float carrier is run through
/// `field_laws` anywhere in the fleet**, and a first attempt with the sample
/// `[0.0, 1.0, -1.0, 2.5, -0.75]` failed on `b / a = b · a⁻¹` — `-3.3333333` against
/// `-3.3333335`. That is not a defect in `f32`'s impl. The law is stated with exact equality,
/// and float division is not exactly multiplication by the reciprocal, nor is float addition
/// associative.
///
/// So a float carrier can only satisfy these suites over a sample closed under *exact*
/// arithmetic. Every value here is a small power of two, which makes each sum, product,
/// quotient and reciprocal in play exactly representable. A consumer bringing a float carrier to
/// `semiring_laws` or `field_laws` needs to know that, and now there is a test that says so.
#[test]
fn f32_satisfies_the_tower_it_claims() {
    // Powers of two only — see the note above on exact equality.
    let sample = [0.0_f32, 1.0, -1.0, 2.0, -0.5];

    conformance::semiring_laws(&sample, "f32");
    conformance::ring_laws(&sample, "f32");
    conformance::field_laws(&sample, "f32");
    conformance::euclidean_laws(&sample, "f32");
    conformance::ref_magma_laws(&sample, "f32");

    // And the two operations spelled out, since a field's remainder being zero is the thing a
    // `Default::default()` body accidentally gets right.
    for &a in &sample {
        assert_eq!(Field::conjugate(a), a, "a real field conjugates to itself");
        for &b in &sample {
            if b != 0.0 {
                let (q, r) = EuclideanDomain::div_rem(a, b);
                assert_eq!(q, a / b, "f32: div_rem's quotient is a / b");
                assert_eq!(r, 0.0, "f32: a field leaves no remainder");
                assert_eq!(q * b + r, a, "f32: a = qb + r");
            }
        }
    }
}

/// `Integer::abs` and `Integer::signum`, which `gcd` depends on and no test pinned.
///
/// `gcd` is implemented as `gcd_abs(self.abs(), other.abs(), 0)`, so a broken `abs` is a broken
/// `gcd` for every negative input — and mutation testing found `abs` replaceable with
/// `Default::default()`, returning zero. The existing `gcd` tests evidently use non-negative
/// arguments, which is exactly the shape of input that cannot tell the two apart.
#[test]
fn the_integer_helpers_handle_sign() {
    for (x, want) in [(0_i64, 0), (7, 7), (-7, 7), (i64::MAX, i64::MAX)] {
        assert_eq!(Integer::abs(x), want, "abs({x})");
    }
    for (x, want) in [(0_i64, 0), (7, 1), (-7, -1)] {
        assert_eq!(Integer::signum(x), want, "signum({x})");
    }
    // `from_i64` is the way every other carrier is built from a literal, and it could be made to
    // return zero for every input.
    for x in [0_i64, 1, -1, 7, -7, i64::MAX, i64::MIN] {
        assert_eq!(<i64 as Integer>::from_i64(x), x, "from_i64({x})");
        assert_eq!(
            <i128 as Integer>::from_i64(x),
            i128::from(x),
            "from_i64({x})"
        );
    }

    // And `gcd` therefore works through the sign, in both operands and both orders.
    assert_eq!(Integer::gcd(12_i64, 18), 6);
    assert_eq!(Integer::gcd(-12_i64, 18), 6, "gcd must ignore sign");
    assert_eq!(Integer::gcd(12_i64, -18), 6);
    assert_eq!(Integer::gcd(-12_i64, -18), 6);
    assert_eq!(Integer::gcd(0_i64, -5), 5, "gcd(0, n) = |n|");
}

/// `Rational`'s ordering, including the comparison `PartialOrd` delegates.
///
/// `partial_cmp` could be made to return `None` for every pair — "these are incomparable" — and
/// nothing noticed, because `Rational` is only ever compared through `==` in the suites. A
/// rational that cannot be ordered silently breaks any sort, `min`/`max`, or binary search over
/// it.
#[test]
fn rationals_are_totally_ordered_through_partial_cmp() {
    use core::cmp::Ordering;

    let third = Rational::<i64>::new(1, 3);
    let half = Rational::<i64>::new(1, 2);
    let neg_half = Rational::<i64>::new(-1, 2);

    assert_eq!(third.partial_cmp(&half), Some(Ordering::Less));
    assert_eq!(half.partial_cmp(&third), Some(Ordering::Greater));
    assert_eq!(half.partial_cmp(&half), Some(Ordering::Equal));
    assert_eq!(neg_half.partial_cmp(&third), Some(Ordering::Less));

    // Never `None`: a rational is comparable to every other rational.
    for a in Rational::<i64>::sample() {
        for b in Rational::<i64>::sample() {
            assert!(
                a.partial_cmp(&b).is_some(),
                "{a:?} and {b:?} must be comparable"
            );
        }
    }

    // And the operators agree with it, which is what a consumer actually writes.
    assert!(third < half);
    assert!(neg_half < third);
    assert!(half >= half);
}

/// The `Display` impls, which are part of the published surface and were unexercised.
///
/// All three could be replaced with a body that writes nothing and returns `Ok(())`. A carrier
/// whose `Display` is empty is a diagnostic that says nothing — and these are the types that
/// appear in a conformance failure message, so the loss lands exactly where it hurts.
#[test]
fn the_carriers_display_their_values() {
    assert_eq!(Zp::<7>::new(3).to_string(), "3 (mod 7)");
    assert_eq!(Zn::<6>::new(4).to_string(), "4 (mod 6)");
    assert_eq!(
        GaussianRational::new(Rational::new(1, 2), Rational::new(-3, 4)).to_string(),
        "1/2 + -3/4i"
    );

    // Non-empty for every sampled value, so a future carrier cannot quietly render to nothing.
    for z in GaussianRational::sample() {
        assert!(!z.to_string().is_empty(), "{z:?} rendered to nothing");
    }
}
