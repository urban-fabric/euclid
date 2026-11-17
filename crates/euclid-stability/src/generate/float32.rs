//! Native single-precision domains, including every f32 representation and its boundaries.

use super::{Gen, sampled_from, uint, weighted};

/// Named f32 boundaries, including both signed zeros and the smallest subnormal.
pub fn notable() -> Gen<f32> {
    sampled_from(vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        f32::EPSILON,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        f32::from_bits(1),
        -f32::from_bits(1),
        f32::MAX,
        f32::MIN,
    ])
}

/// Every f32 bit pattern is reachable, with named edges mixed in deliberately.
pub fn any() -> Gen<f32> {
    weighted(vec![
        (1, notable()),
        (
            1,
            sampled_from(vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY]),
        ),
        (
            6,
            uint(0, u64::from(u32::MAX)).map(|bits| f32::from_bits(bits as u32)),
        ),
    ])
}

/// Finite f32 values, sampled in their own representation rather than narrowed from f64.
pub fn finite() -> Gen<f32> {
    any().filter("finite f32", |value| value.is_finite())
}

/// Strictly positive finite f32 values, including positive subnormals.
pub fn positive() -> Gen<f32> {
    finite()
        .map(f32::abs)
        .filter("positive f32", |value| *value > 0.0)
}

/// Finite values in the inclusive range, with exact endpoints and a stable interpolation.
pub fn in_range(low: f32, high: f32) -> Gen<f32> {
    assert!(
        low.is_finite() && high.is_finite() && low <= high,
        "invalid f32 range"
    );
    let spread = uint(0, 1 << 24).map(move |word| {
        let t = word as f32 / (1u32 << 24) as f32;
        if low.is_sign_negative() != high.is_sign_negative() {
            low * (1.0 - t) + high * t
        } else {
            low + (high - low) * t
        }
    });
    weighted(vec![(1, sampled_from(vec![low, high])), (4, spread)])
}
