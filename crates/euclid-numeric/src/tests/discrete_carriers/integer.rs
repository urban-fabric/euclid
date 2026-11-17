//! The [`Integer`] tier: `i32`, `i64`, `i128`.
//!
//! See the [parent module](super) for why this exists, where the expectations come from, and
//! which two mutants here are unkillable.
//!
//! The sign is what distinguishes this from [`super::natural`]. Three methods exist only
//! because the ring has negatives — `signum`, `abs`, and Euclidean `div_rem`, whose remainder
//! stays non-negative even when the quotient does not — and `gcd`/`lcm` have to pick a
//! representative among the signed associates. Every sample below therefore spans both signs,
//! which is the part a non-negative sample cannot reach.

use super::agrees;
use crate::tests::*;
use core::fmt::Debug;

/// The whole of [`Integer`] at one carrier.
///
/// `tol` is the carrier's real surface: `i32` casts into `f32` and gets `1e-5`, the two wider
/// carriers cast into `f64` and get `1e-12`. Magnitudes stay small enough to be exact at `i32`,
/// so the same assertions run at all three widths; the wide cases are separate tests below.
fn integer_carrier_laws<T>(tol: f64, what: &str)
where
    T: Integer + PartialEq + Debug + Copy + Ord,
    T::Real: Real<Mask = bool> + Debug,
{
    let i = |x: i64| T::from_i64(x);

    // --- the identities, from their algebraic definitions ------------------------------
    //
    // `one` is checked as a *multiplicative identity* rather than against the literal `1`,
    // which is what makes `one() -> Default::default()` fail. The corresponding
    // `zero() -> Default::default()` is unkillable and excluded — `Default` for a machine
    // integer is zero.
    assert_eq!(i(7) + T::zero(), i(7), "{what}: x + 0 = x");
    assert_eq!(i(7) * T::one(), i(7), "{what}: x · 1 = x");
    assert_eq!(i(7) * T::zero(), T::zero(), "{what}: x · 0 = 0");
    assert_ne!(T::zero(), T::one(), "{what}: 0 ≠ 1");
    // The ring's additive inverse, which the natural tier withholds.
    assert_eq!(i(7) + (T::zero() - i(7)), T::zero(), "{what}: x + (−x) = 0");

    // --- the casts, against integer literals -------------------------------------------
    agrees(&i(7).to_real(), 7.0, tol, what, "to_real");
    agrees(&i(-7).to_real(), -7.0, tol, what, "to_real(negative)");
    agrees(&i(0).to_real(), 0.0, tol, what, "to_real(0)");
    assert_eq!(
        T::from_i64(-7),
        T::zero() - i(7),
        "{what}: from_i64 carries the sign"
    );

    // `from_real` rounds rather than truncates, and does so away from zero on both sides —
    // truncation would send −7.6 to −7 and 7.6 to 7.
    assert_eq!(
        T::from_real(T::Real::scalar(7.6)),
        i(8),
        "{what}: from_real rounds up"
    );
    assert_eq!(
        T::from_real(T::Real::scalar(7.4)),
        i(7),
        "{what}: from_real rounds down"
    );
    assert_eq!(
        T::from_real(T::Real::scalar(-7.6)),
        i(-8),
        "{what}: from_real(−7.6) = −8"
    );
    assert_eq!(
        T::from_real(T::Real::scalar(-7.4)),
        i(-7),
        "{what}: from_real(−7.4) = −7"
    );
    assert_eq!(
        T::from_real(T::Real::scalar(0.0)),
        T::zero(),
        "{what}: from_real(0)"
    );

    // --- signum and abs, from their definitions ----------------------------------------
    for x in [-100_i64, -7, -1, 0, 1, 7, 100] {
        let signum = T::signum(i(x));
        let expected = T::from_i64(x.signum());
        assert_eq!(signum, expected, "{what}: signum({x})");
        // `|x| = x · signum(x)`, and `|x|` is non-negative — the two halves of what abs means.
        assert_eq!(
            T::abs(i(x)),
            i(x) * signum,
            "{what}: |x| = x · signum(x) at {x}"
        );
        assert!(T::abs(i(x)) >= T::zero(), "{what}: |{x}| ≥ 0");
    }

    // --- div_rem, against the Euclidean definition -------------------------------------
    //
    // The remainder is non-negative regardless of either sign, which is the convention that
    // distinguishes `div_euclid` from `/`. Checked structurally over all four sign pairs, so
    // the law holds everywhere rather than at the one triple a literal would pin.
    for a in [-100_i64, -7, -1, 0, 1, 7, 100] {
        for b in [-7_i64, -3, -1, 1, 3, 7] {
            let (q, r) = i(a).div_rem(i(b));
            assert_eq!(q * i(b) + r, i(a), "{what}: {a} = q·{b} + r");
            assert!(r >= T::zero(), "{what}: remainder {r:?} ≥ 0 at ({a}, {b})");
            assert!(r < T::abs(i(b)), "{what}: remainder {r:?} < |{b}|");
        }
    }
    // The four sign combinations as literals, matching the trait's documented examples.
    assert_eq!(i(7).div_rem(i(3)), (i(2), i(1)), "{what}: 7 = 2·3 + 1");
    assert_eq!(i(-7).div_rem(i(3)), (i(-3), i(2)), "{what}: −7 = −3·3 + 2");
    assert_eq!(i(7).div_rem(i(-3)), (i(-2), i(1)), "{what}: 7 = −2·−3 + 1");
    assert_eq!(i(-7).div_rem(i(-3)), (i(3), i(2)), "{what}: −7 = 3·−3 + 2");

    // --- gcd, from Euclid's worked example, on the magnitudes --------------------------
    //
    // The result is the non-negative representative, so the sign of either operand must not
    // reach it — four sign pairs, one answer.
    for (a, b) in [
        (1071_i64, 462_i64),
        (-1071, 462),
        (1071, -462),
        (-1071, -462),
    ] {
        assert_eq!(T::gcd(i(a), i(b)), i(21), "{what}: gcd({a}, {b}) = 21");
    }
    assert_eq!(T::gcd(i(462), i(1071)), i(21), "{what}: gcd commutes");
    assert_eq!(T::gcd(i(0), i(0)), T::zero(), "{what}: gcd(0, 0) = 0");
    assert_eq!(T::gcd(i(13), i(0)), i(13), "{what}: gcd(n, 0) = n");
    assert_eq!(T::gcd(i(-13), i(0)), i(13), "{what}: gcd(−n, 0) = n");
    assert_eq!(T::gcd(i(9), i(28)), T::one(), "{what}: gcd(9, 28) = 1");
    assert_eq!(T::gcd(i(17), i(51)), i(17), "{what}: gcd(17, 51) = 17");

    // --- lcm, and the theorem tying it to gcd ------------------------------------------
    assert_eq!(T::lcm(i(4), i(6)), i(12), "{what}: lcm(4, 6) = 12");
    assert_eq!(T::lcm(i(-4), i(6)), i(12), "{what}: lcm is non-negative");
    assert_eq!(T::lcm(i(4), i(-6)), i(12), "{what}: lcm(4, −6) = 12");
    assert_eq!(T::lcm(i(-4), i(-6)), i(12), "{what}: lcm(−4, −6) = 12");
    assert_eq!(T::lcm(i(0), i(5)), T::zero(), "{what}: lcm(0, n) = 0");
    assert_eq!(T::lcm(i(5), i(0)), T::zero(), "{what}: lcm(n, 0) = 0");
    assert_eq!(T::lcm(i(0), i(0)), T::zero(), "{what}: lcm(0, 0) = 0");

    // `gcd(a, b) · lcm(a, b) = |a · b|`. A statement about both functions at once, provable
    // without reference to either implementation, so it cross-checks them rather than
    // restating them — and it is what catches an `lcm` that divides where it should multiply.
    // Magnitudes kept inside `i32` so the product below is exact at every width.
    for (a, b) in [
        (4_i64, 6_i64),
        (-4, 6),
        (1071, 462),
        (9, 28),
        (17, 51),
        (36, -60),
        (1, 1),
        (-1, -1),
    ] {
        assert_eq!(
            T::gcd(i(a), i(b)) * T::lcm(i(a), i(b)),
            T::abs(i(a) * i(b)),
            "{what}: gcd·lcm = |a·b| at ({a}, {b})",
        );
    }
    // Divisibility, the other half of what the two mean: `gcd` divides both operands, and
    // `lcm` is divisible by both.
    for (a, b) in [(1071_i64, 462_i64), (36, -60), (-9, 28)] {
        let g = T::gcd(i(a), i(b));
        assert_eq!(i(a).div_rem(g).1, T::zero(), "{what}: gcd divides {a}");
        assert_eq!(i(b).div_rem(g).1, T::zero(), "{what}: gcd divides {b}");
        let l = T::lcm(i(a), i(b));
        assert_eq!(l.div_rem(i(a)).1, T::zero(), "{what}: {a} divides lcm");
        assert_eq!(l.div_rem(i(b)).1, T::zero(), "{what}: {b} divides lcm");
    }
}

#[test]
fn the_i32_carrier_honours_the_integer_contract() {
    // `f32`'s 24-bit significand sets the tolerance; every magnitude here is far inside its
    // exact-integer range, so this is headroom rather than a concession.
    integer_carrier_laws::<i32>(1e-5, "i32");
}

#[test]
fn the_i64_carrier_honours_the_integer_contract() {
    integer_carrier_laws::<i64>(1e-12, "i64");
}

#[test]
fn the_i128_carrier_honours_the_integer_contract() {
    integer_carrier_laws::<i128>(1e-12, "i128");
}

/// The three carriers must agree with one another, not merely each be self-consistent, since
/// [`Integer`] is one contract and a formula bounded on it is expected to give the same answer
/// whichever width a caller picks.
#[test]
fn the_integer_carriers_agree_with_one_another() {
    for a in [-1071_i64, -7, -1, 0, 1, 7, 462, 1071] {
        for b in [-462_i64, -3, 1, 3, 462] {
            assert_eq!(
                i64::from(Integer::gcd(a as i32, b as i32)),
                Integer::gcd(a, b),
                "i32 and i64 disagree on gcd({a}, {b})",
            );
            assert_eq!(
                i128::from(Integer::gcd(a, b)),
                Integer::gcd(a as i128, b as i128),
                "i64 and i128 disagree on gcd({a}, {b})",
            );
            assert_eq!(
                i128::from(Integer::lcm(a, b)),
                Integer::lcm(a as i128, b as i128),
                "i64 and i128 disagree on lcm({a}, {b})",
            );
            let (q, r) = Integer::div_rem(a, b);
            assert_eq!(
                (i128::from(q), i128::from(r)),
                Integer::div_rem(a as i128, b as i128),
                "i64 and i128 disagree on div_rem({a}, {b})",
            );
        }
    }
}

/// `lcm` reaches answers the naive `a · b / gcd` would overflow on the way to, which is why the
/// body divides first — the claim its doc comment makes.
#[test]
fn lcm_reaches_answers_a_naive_product_would_overflow() {
    // `lcm(2^62, 2^61) = 2^62`, representable; `a · b` is not.
    let a = 1_i64 << 62;
    let b = 1_i64 << 61;
    assert_eq!(Integer::lcm(a, b), a, "lcm(2^62, 2^61) = 2^62");
    assert_eq!(a.checked_mul(b), None, "the naive product overflows");

    let a = 1_i128 << 126;
    let b = 1_i128 << 125;
    assert_eq!(Integer::lcm(a, b), a, "i128: lcm(2^126, 2^125) = 2^126");
    assert_eq!(a.checked_mul(b), None, "i128: the naive product overflows");
}

/// The wide carriers hold magnitudes `i32` cannot, and `to_real` must round rather than wrap
/// once past `2^53` — the boundary where consecutive integers stop being distinguishable.
#[test]
fn the_wide_carriers_carry_magnitudes_i32_cannot() {
    // Exactly representable: `2^53` itself.
    let big = 1_i64 << 53;
    assert_eq!(Integer::to_real(&big), 9_007_199_254_740_992.0_f64);
    assert_eq!(i64::from_real(9_007_199_254_740_992.0), big);
    // `i128` past the `f64` grid entirely: `2^100` is exact in binary, so the cast is too.
    let wide = 1_i128 << 100;
    assert_eq!(Integer::to_real(&wide), 2.0_f64.powi(100));
    assert_eq!(Integer::abs(-wide), wide, "|−2^100| = 2^100");
    assert_eq!(Integer::signum(-wide), -1, "signum(−2^100) = −1");
    // The widest magnitudes each carrier holds, where `abs` has no positive counterpart.
    assert_eq!(Integer::signum(i64::MIN), -1);
    assert_eq!(Integer::signum(i128::MIN), -1);
    assert_eq!(Integer::signum(i32::MIN), -1);
}
