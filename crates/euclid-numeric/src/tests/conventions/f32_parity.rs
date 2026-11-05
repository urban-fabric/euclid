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

// --- the degree-trig and polynomial surface at f32 ----------------------

#[test]
fn degree_trig_and_polyval_run_at_f32() {
    // `f32` is the other carrier this crate implements, so both surfaces must
    // exist there too rather than only at `f64`.
    assert_eq!(90.0_f32.sin_cos_degrees(), (1.0, 0.0));
    assert_eq!(4.0_f32.polyval(&[2.0, -3.0, 1.0]), 21.0);
}
