//! Comparisons checks for the relation module.

use super::*;

#[test]
fn exact_is_equality_and_describes_itself() {
    assert!(Exact.same(&3, &3));
    assert!(!Exact.same(&3, &4));
    assert!(Compare::<i32>::describe(&Exact).contains("exactly"));
}

/// The comparator that exists because `==` is wrong for a computed float: a value one ulp away
/// must be accepted at `Ulps(1)` and refused at `Ulps(0)`.
#[test]
fn ulps_accepts_a_neighbour_and_refuses_an_identical_demand() {
    let one = 1.0_f64;
    let next = f64::from_bits(one.to_bits() + 1);
    assert!(Ulps(1).same(&one, &next));
    assert!(!Ulps(0).same(&one, &next));
}

#[test]
fn relative_and_absolute_differ_on_scale() {
    // Two values 10% apart at a tiny scale: absolute accepts, relative does not.
    assert!(Absolute(1.0).same(&1e-9, &1.1e-9));
    assert!(!Relative(0.01).same(&1e-9, &1.1e-9));
}

#[test]
fn every_comparator_refuses_nan() {
    assert!(!Ulps(u64::MAX).same(&f64::NAN, &f64::NAN));
    assert!(!Relative(1.0).same(&f64::NAN, &f64::NAN));
    assert!(!Absolute(f64::INFINITY).same(&f64::NAN, &f64::NAN));
}

/// Every comparator must say **yes** to something.
///
/// The suite above checks each one's refusals thoroughly and its acceptances barely, and mutation
/// testing found the hole: `Relative::same` could be replaced with a function returning `false`
/// unconditionally and the whole workspace stayed green. A comparator that never agrees turns
/// every relation built on it into a vacuous pass-by-failure — or rather, it would make them all
/// fail, which is worse, and nothing was exercising the agreeing path to notice.
#[test]
fn every_comparator_accepts_a_value_that_matches() {
    assert!(Exact.same(&3_u32, &3_u32));
    assert!(Ulps(0).same(&1.0_f64, &1.0_f64));
    assert!(Relative(1e-9).same(&1.0_f64, &1.0_f64));
    assert!(Relative(0.01).same(&1.0_f64, &1.005_f64));
    assert!(Absolute(1e-9).same(&1.0_f64, &1.0_f64));
    assert!(Absolute(0.5).same(&1.0_f64, &1.25_f64));
}

/// A comparator's description reaches the reader on every failure, so it is part of the
/// deliverable rather than decoration.
///
/// Mutation testing replaced three of these four with `String::new()` and with `"xyzzy"`, and
/// nothing failed: only `Exact` had its description checked. A failure report that cannot say
/// *what* notion of sameness was being applied is the report that sends someone to read the
/// source of the relation they just used.
#[test]
fn every_comparator_describes_the_notion_it_applies() {
    let described = [
        <Exact as Compare<f64>>::describe(&Exact),
        <Ulps as Compare<f64>>::describe(&Ulps(3)),
        <Relative as Compare<f64>>::describe(&Relative(1e-9)),
        <Absolute as Compare<f64>>::describe(&Absolute(0.5)),
    ];
    for description in &described {
        assert!(
            !description.is_empty(),
            "a comparator described itself as nothing"
        );
    }
    // Each must name its own tolerance, or two failures at different tolerances read identically.
    assert!(
        <Ulps as Compare<f64>>::describe(&Ulps(3)).contains('3'),
        "Ulps did not name its tolerance: {}",
        described[1]
    );
    assert!(
        <Relative as Compare<f64>>::describe(&Relative(1e-9)).contains("e-9"),
        "Relative did not name its tolerance: {}",
        described[2]
    );
    // Scientific notation, like `Relative` — `0.5` reaches the reader as `5e-1`.
    assert!(
        <Absolute as Compare<f64>>::describe(&Absolute(0.5)).contains("5e-1"),
        "Absolute did not name its tolerance: {}",
        described[3]
    );
    // And they must be distinguishable from one another.
    let distinct: std::collections::BTreeSet<&String> = described.iter().collect();
    assert_eq!(
        distinct.len(),
        described.len(),
        "two comparators describe themselves identically: {described:?}"
    );
}

/// The `f32` ulp metric, pinned to exact counts.
///
/// `precision_agrees` is built entirely on this function and was the only thing exercising it —
/// through a tolerance comparison, which cannot distinguish a gap of 1 from a gap of 1,065,353,216
/// when the allowance is generous. Mutation testing left six survivors here, including flipping
/// the sign test that makes the mapping monotone across zero.
///
/// The constants are the point. `0.0` to `1.0` is `0x3F800000` steps because that *is* the bit
/// pattern of `1.0f32`, and a pair straddling zero is twice that — the property that a
/// sign-magnitude format does not give you for free and that this function exists to supply.
#[test]
fn f32_ulps_between_counts_representable_steps() {
    assert_eq!(f32_ulps_between(1.0, 1.0), Some(0));
    assert_eq!(f32_ulps_between(-1.0, -1.0), Some(0));

    let next = f32::from_bits(1.0_f32.to_bits() + 1);
    assert_eq!(f32_ulps_between(1.0, next), Some(1));
    assert_eq!(f32_ulps_between(next, 1.0), Some(1), "must be symmetric");

    // Signed zero is zero steps apart, not 2^31.
    assert_eq!(f32_ulps_between(-0.0, 0.0), Some(0));

    // Exact distances, which a tolerance comparison cannot pin.
    assert_eq!(f32_ulps_between(0.0, 1.0), Some(1_065_353_216));
    assert_eq!(f32_ulps_between(-1.0, 1.0), Some(2_130_706_432));

    // Non-finite on *either* side is unmeasurable, not large.
    assert_eq!(f32_ulps_between(f32::NAN, 1.0), None);
    assert_eq!(f32_ulps_between(1.0, f32::NAN), None);
    assert_eq!(f32_ulps_between(f32::INFINITY, 1.0), None);
    assert_eq!(f32_ulps_between(1.0, f32::NEG_INFINITY), None);
}
