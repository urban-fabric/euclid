//! Independent distribution identities near the limits of the floating-point formats.

use super::*;

// Shared by `references` and `poles`, which both compare `f32`s by representable distance.

/// Compare two `f32`s by the number of representable values between them.
fn close_in_ulps(actual: f32, expected: f32, budget: i64, what: &str) {
    let gap = i64::from(actual.to_bits()) - i64::from(expected.to_bits());
    assert!(
        gap.abs() <= budget,
        "{what}: {actual:e} vs {expected:e} is {} ulp apart, budget {budget}",
        gap.abs()
    );
}

#[path = "distribution_extremes/limits.rs"]
mod limits;
#[path = "distribution_extremes/poles.rs"]
mod poles;
#[path = "distribution_extremes/references.rs"]
mod references;
#[path = "distribution_extremes/single_precision.rs"]
mod single_precision;
