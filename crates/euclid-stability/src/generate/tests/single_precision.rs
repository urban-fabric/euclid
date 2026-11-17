//! Single-precision representation coverage and safe interpolation across extreme ranges.

use super::*;
use crate::generate::float32;
use crate::{Config, run};

#[test]
fn native_single_precision_visits_subnormals_signed_zero_and_nonfinite_values() {
    let values = draw_many(&float32::any(), 4_000, 781);
    for expected in [
        0.0f32,
        -0.0,
        f32::from_bits(1),
        -f32::from_bits(1),
        f32::MIN_POSITIVE,
        f32::MAX,
        f32::MIN,
        f32::INFINITY,
        f32::NEG_INFINITY,
    ] {
        assert!(
            values.iter().any(|v| v.to_bits() == expected.to_bits()),
            "missing f32 bits {:x}",
            expected.to_bits()
        );
    }
    assert!(values.iter().any(|v| v.is_nan()));
    assert!(
        values
            .iter()
            .any(|v| v.is_finite() && *v > 100.0 && *v < 1e10)
    );
}

#[test]
fn finite_and_positive_domains_replace_rejections_and_cover_both_signs() {
    let values = draw_many(&float32::finite(), 800, 19);
    assert!(values.iter().all(|v| v.is_finite()));
    assert!(values.iter().any(|v| *v < 0.0));
    assert!(values.iter().any(|v| *v > 0.0));
    let values = draw_many(&float32::positive(), 800, 20);
    assert!(values.iter().all(|v| v.is_finite() && *v > 0.0));
    assert!(values.contains(&f32::from_bits(1)));
}

#[test]
fn full_finite_ranges_do_not_overflow_the_intermediate_width() {
    for (low, high) in [
        (-f32::MAX, f32::MAX),
        (f32::MAX / 2.0, f32::MAX),
        (-f32::MAX, -f32::MAX / 2.0),
    ] {
        let values = draw_many(&float32::in_range(low, high), 200, 21);
        assert!(
            values
                .iter()
                .all(|v| v.is_finite() && *v >= low && *v <= high)
        );
        assert!(values.contains(&low));
        assert!(values.contains(&high));
    }
    let values = draw_many(
        &crate::generate::float::in_range(-f64::MAX, f64::MAX),
        200,
        22,
    );
    assert!(values.iter().all(|v| v.is_finite()));
    assert!(values.contains(&f64::MAX));
    assert!(values.contains(&-f64::MAX));
    assert!(values.iter().any(|v| *v > -f64::MAX && *v < 0.0));
    assert!(values.iter().any(|v| *v > 0.0 && *v < f64::MAX));
}

#[test]
fn edge_first_single_precision_cases_are_exact_replayable_boundaries() {
    let mut seen = Vec::new();
    run(
        "f32 edges",
        &Config::default().cases(2).edges_first(),
        &float32::in_range(-10.0, 30.0),
        |value| {
            seen.push(value);
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(seen, [-10.0, 30.0]);
}

#[test]
fn single_precision_ranges_reach_the_interior_in_both_halves() {
    for (low, high) in [(-1.0f32, 1.0), (4.0, 8.0), (-8.0, -4.0)] {
        let values = draw_many(&float32::in_range(low, high), 200, 83);
        let middle = (low + high) / 2.0;
        assert!(values.iter().all(|v| *v >= low && *v <= high));
        assert!(values.iter().any(|v| *v > low && *v < middle));
        assert!(values.iter().any(|v| *v > middle && *v < high));
    }
}

#[test]
fn double_precision_edges_include_the_named_representation_boundaries() {
    let mut bits = std::collections::BTreeSet::new();
    run(
        "f64 boundaries",
        &Config::default().cases(17).edges_first(),
        &crate::generate::float::finite(),
        |value| {
            bits.insert(value.to_bits());
            Ok(())
        },
    )
    .unwrap();
    for value in [
        0.0f64,
        -0.0,
        f64::EPSILON,
        f64::MIN_POSITIVE,
        f64::from_bits(1),
        f64::MAX,
        f64::MIN,
    ] {
        assert!(
            bits.contains(&value.to_bits()),
            "missing boundary {value:?}"
        );
    }
}
