//! Unit tests for counted quantities.
//!
//! A child module so implementation plus tests stay inside the crate 400-line budget, while
//! `use super::*` still reaches the private constructors and saturation helpers.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use super::{Count, ThreadCount};

#[test]
fn count_is_dimensionless_discrete_measure() {
    use euclid_units::{DiscreteQuantity, Measure};
    assert_eq!(
        <Count as Measure>::DIMENSION,
        euclid_units::Dimension::Dimensionless
    );
    assert_eq!(
        DiscreteQuantity::add(Count::new(5), Count::new(3)),
        Count::new(8)
    );
    assert_eq!(
        DiscreteQuantity::saturating_sub(Count::new(3), Count::new(5)),
        Count::ZERO
    );
}

#[test]
fn count_display_is_bare_number() {
    assert_eq!(Count::new(42).to_string(), "42");
}

#[test]
fn count_serde_is_bare_u64() {
    let c = Count::new(42);
    let json = serde_json::to_string(&c).unwrap();
    assert_eq!(json, "42");
    let back: Count = serde_json::from_str(&json).unwrap();
    assert_eq!(back, c);
}

#[test]
fn count_arithmetic_round_trips() {
    let mut c = Count::ZERO;
    c += Count::new(5);
    c += Count::new(3);
    assert_eq!(c, Count::new(8));
    c -= Count::new(1);
    assert_eq!(c, Count::new(7));
}

#[test]
fn thread_count_display_pluralizes() {
    assert_eq!(ThreadCount::new(1).to_string(), "1 thread");
    assert_eq!(ThreadCount::new(8).to_string(), "8 threads");
}

#[test]
fn thread_count_serde_is_bare_u32() {
    let t = ThreadCount::new(8);
    let json = serde_json::to_string(&t).unwrap();
    assert_eq!(json, "8");
    let back: ThreadCount = serde_json::from_str(&json).unwrap();
    assert_eq!(back, t);
}

/// The conversion, arithmetic and mixed-comparison surface for `Count` and `ThreadCount`.
///
/// Thirty-four survivors lived here, and the cause is the same split the tests above show:
/// `DiscreteQuantity::add` is a shared default and is checked, while `impl Add for Count`,
/// `impl Sub for Count`, the `From` conversions, `FromStr` and the mixed comparisons are
/// separate implementations that nothing reached.
#[test]
fn the_raw_accessors_agree_with_the_value() {
    for raw in [0_u64, 1, 42, u64::MAX / 2] {
        let c = Count::new(raw);
        assert_eq!(c.as_u64(), raw);
        #[allow(clippy::cast_precision_loss)]
        {
            assert_eq!(c.as_f64(), raw as f64);
            assert_eq!(c.as_f32(), raw as f32);
        }
    }
}

#[test]
fn the_operators_agree_with_the_trait_defaults() {
    for (a, b) in [(0_u64, 0_u64), (5, 3), (3, 5), (42, 42)] {
        assert_eq!(
            Count::new(a) + Count::new(b),
            euclid_units::DiscreteQuantity::add(Count::new(a), Count::new(b)),
            "+ must agree with DiscreteQuantity::add at ({a}, {b})"
        );
        assert_eq!(
            Count::new(a) - Count::new(b.min(a)),
            Count::new(a - b.min(a)),
            "- must subtract at ({a}, {b})"
        );
    }

    // `+=` against a bare u64 is the same addition.
    let mut c = Count::new(5);
    c += 3_u64;
    assert_eq!(c, Count::new(8));
}

#[test]
fn saturating_sub_clamps_at_zero() {
    assert_eq!(Count::new(3).saturating_sub(Count::new(5)), Count::new(0));
    assert_eq!(Count::new(5).saturating_sub(Count::new(3)), Count::new(2));
    assert_eq!(Count::new(4).saturating_sub(Count::new(4)), Count::new(0));
}

#[test]
fn the_integer_conversions_round_trip() {
    for raw in [0_u64, 1, 42, u64::MAX] {
        assert_eq!(u64::from(Count::from(raw)), raw, "u64 -> Count -> u64");
    }
    for raw in [0_u32, 1, 8, u32::MAX] {
        assert_eq!(
            u32::from(ThreadCount::from(raw)),
            raw,
            "u32 -> ThreadCount -> u32"
        );
    }
}

#[test]
fn parsing_a_count_reads_the_number() {
    use core::str::FromStr;
    assert_eq!(Count::from_str("0"), Ok(Count::new(0)));
    assert_eq!(Count::from_str("42"), Ok(Count::new(42)));
    // And a non-number is an error rather than a default.
    assert!(Count::from_str("not a number").is_err());
    assert!(Count::from_str("-1").is_err(), "a count is unsigned");
}

#[test]
fn mixed_comparisons_agree_with_comparing_the_raw_values() {
    for a in [0_u64, 1, 7, 42] {
        for b in [0_u64, 1, 7, 42] {
            let wrapped = Count::new(a);
            assert_eq!(wrapped == b, a == b, "Count == u64 at ({a}, {b})");
            assert_eq!(b == wrapped, b == a, "u64 == Count at ({a}, {b})");
            assert_eq!(wrapped.partial_cmp(&b), a.partial_cmp(&b));
            assert!(wrapped.partial_cmp(&b).is_some());
        }
    }
}

#[test]
fn the_thread_count_accessors_agree_with_the_value() {
    for raw in [0_u32, 1, 8, u32::MAX] {
        let t = ThreadCount::new(raw);
        assert_eq!(t.as_u32(), raw, "as_u32 must be the raw count");
        assert_eq!(
            t.as_usize(),
            raw as usize,
            "as_usize must widen the raw count"
        );
    }
}
