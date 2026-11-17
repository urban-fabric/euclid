//! Rational behavior at the boundaries of the backing integer.

use crate::{Integer, Rational64, Rational128};
use euclid_stability::{Config, check, generate, property};

property! {
    #![config(Config::default().cases(800))]

    fn rational64_order_agrees_with_widened_cross_products(
        a in generate::int(i64::MIN + 1, i64::MAX),
        b in generate::int(1, i64::MAX),
        c in generate::int(i64::MIN + 1, i64::MAX),
        d in generate::int(1, i64::MAX),
    ) {
        let left = Rational64::new(a, b);
        let right = Rational64::new(c, d);
        let expected = (i128::from(a) * i128::from(d)).cmp(&(i128::from(c) * i128::from(b)));
        check!(left.cmp(&right) == expected, "Rational64 order agrees with the exact products");
    }
}

property! {
    #![config(Config::default().cases(1500))]

    fn checked_rational64_arithmetic_matches_i128_reduction(
        a in generate::int(i64::MIN, i64::MAX),
        b in generate::int(1, i64::MAX),
        c in generate::int(i64::MIN, i64::MAX),
        d in generate::int(1, i64::MAX),
    ) {
        let left = Rational64::new(a, b);
        let right = Rational64::new(c, d);
        let (a, b, c, d) = (i128::from(a), i128::from(b), i128::from(c), i128::from(d));
        let reduce = |num, den| {
            let wide = Rational128::checked_new(num, den)?;
            Some((i64::try_from(wide.numerator()).ok()?, i64::try_from(wide.denominator()).ok()?))
        };
        for (actual, expected) in [
            (left.checked_add(right), reduce(a * d + c * b, b * d)),
            (left.checked_sub(right), reduce(a * d - c * b, b * d)),
            (left.checked_mul(right), reduce(a * c, b * d)),
            (left.checked_div(right), reduce(a * d, b * c)),
        ] {
            check!(actual.map(|r| (r.numerator(), r.denominator())) == expected, "checked arithmetic differs");
        }
    }
}

#[test]
fn adjacent_large_ratios_are_ordered_exactly() {
    // The cross-products differ by 1 and neither fits in i128.
    let max = i128::MAX;
    let lower = Rational128::new(max - 2, max - 1);
    let upper = Rational128::new(max - 1, max);
    assert!(lower < upper);
    assert!(upper > lower);
    assert!(-upper < -lower);
    assert!(Rational128::integer(i128::MIN) < -upper);
    assert!(Rational128::integer(i128::MAX) > upper);
    let one = Rational128::integer(1);
    let just_above_one = Rational128::new(max, max - 1);
    assert!(one < just_above_one);
    assert!(just_above_one > one);
}

#[test]
fn adding_or_subtracting_a_common_denominator_does_not_square_it() {
    let one = Rational64::new(1, i64::MAX);
    let two = Rational64::new(2, i64::MAX);
    assert_eq!(one + one, two);
    assert_eq!(two - one, one);
    assert_eq!(one - one, Rational64::integer(0));
}

#[test]
#[should_panic(expected = "rational division by zero")]
fn nonzero_rational_cannot_be_divided_by_zero() {
    let _ = Rational128::integer(1) / Rational128::integer(0);
}

#[test]
#[should_panic(expected = "rational division by zero")]
fn zero_rational_cannot_be_divided_by_zero() {
    let _ = Rational64::integer(0) / Rational64::integer(0);
}

#[test]
fn operators_cancel_before_narrowing() {
    let m = i64::MAX;
    let a = Rational64::new(1, m);
    let b = Rational64::new(1, m / 7);
    assert_eq!(a + b, Rational64::new(8, m));
    assert_eq!(a - b, Rational64::new(-6, m));
    let c = Rational64::new(i64::MIN, m);
    let d = Rational64::new(m, 2);
    assert_eq!(c * d, Rational64::integer(i64::MIN / 2));
    assert_eq!(c / c, Rational64::integer(1));
    assert_eq!(Rational64::integer(0) / c, Rational64::integer(0));
}

#[test]
fn rational128_reduces_a_numerator_wider_than_128_bits() {
    let s = 1_i128 << 62;
    let t = s + 1;
    let g = s + t;
    let a = Rational128::new(g * s - 1, g * s);
    let b = Rational128::new(g * t - 1, g * t);
    let exact = Rational128::new(2 * s * t - 1, s * t);
    assert_eq!(a + b, exact);
    assert_eq!(-a + -b, -exact);
    assert_eq!(a - -b, exact);
    assert_eq!(a.checked_sub(b), None);
}

#[test]
fn checked_operations_cover_signed_minima_and_overflow() {
    let min = i128::MIN;
    let zero = Rational128::integer(0);
    let one = Rational128::integer(1);
    let two = Rational128::integer(2);
    let smallest = Rational128::integer(min);
    let largest = Rational128::integer(i128::MAX);
    assert_eq!(Rational128::new(min, min), one);
    assert_eq!(Rational128::new(min, 1), smallest);
    assert_eq!(Rational128::new(2, min), Rational128::new(-1, -(min / 2)));
    assert_eq!(Rational128::checked_new(1, min), None);
    assert_eq!(Rational128::checked_new(min, -1), None);
    assert_eq!(Rational128::checked_new(1, 0), None);
    assert_eq!(largest.checked_add(one), None);
    assert_eq!(smallest.checked_sub(one), None);
    assert_eq!(largest.checked_mul(two), None);
    assert_eq!(smallest.checked_neg(), None);
    assert_eq!(smallest.checked_div(-one), None);
    assert_eq!(smallest.checked_div(zero), None);
}

#[test]
#[should_panic(expected = "rational overflow")]
fn operator_overflow_panics_in_every_profile() {
    let _ = Rational64::integer(i64::MAX) + Rational64::integer(1);
}

#[test]
fn primitive_gcd_accepts_a_signed_minimum_when_the_result_fits() {
    assert_eq!(Integer::gcd(i64::MIN, 12), 4);
    assert_eq!(Integer::gcd(12_i128, i128::MIN), 4);
}

#[test]
#[should_panic(expected = "integer gcd overflow")]
fn primitive_gcd_checks_i64_range() {
    let _ = Integer::gcd(i64::MIN, 0);
}

#[test]
#[should_panic(expected = "integer gcd overflow")]
fn primitive_gcd_checks_i128_range() {
    let _ = Integer::gcd(i128::MIN, 0);
}

#[test]
fn primitive_absolute_value_checks_the_signed_minimum() {
    assert_eq!(Integer::abs(-1_i64), 1);
    assert_eq!(Integer::abs(-1_i128), 1);
    assert!(std::panic::catch_unwind(|| Integer::abs(i64::MIN)).is_err());
    assert!(std::panic::catch_unwind(|| Integer::abs(i128::MIN)).is_err());
}
