//! The same conventions, checked on the f32 carrier.

use crate::Real;

// --- the same conventions on f32 ----------------------------------------

#[test]
fn f32_follows_the_same_conventions() {
    // A divergent f32 code path would show up here even though the values differ in precision.
    assert_eq!(Real::round(2.5f32), 3.0);
    assert_eq!(Real::round(-2.5f32), -3.0);
    assert_eq!(Real::trunc(-2.7f32), -2.0);
    assert_eq!(Real::floor(-2.7f32), -3.0);
    assert_eq!(Real::signum(-0.0f32), -1.0);
    assert_eq!(Real::copysign(1.0f32, -0.0), -1.0);
    assert_eq!(Real::min(f32::NAN, 1.0), 1.0);
    assert_eq!(Real::max(f32::NAN, 1.0), 1.0);
    assert_eq!(Real::rem_euclid(-1.0f32, 3.0), 2.0);
    assert!((Real::fract(-2.75f32) + 0.75).abs() < 1e-6);
}
