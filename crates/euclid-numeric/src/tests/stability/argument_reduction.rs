//! Reducing an angle in its own unit, rather than converting first.
//!
//! `x.to_radians().sin_cos()` has to represent the angle in a unit where the
//! quadrant boundaries are irrational: `90f64.to_radians()` is the nearest `f64`
//! to `pi/2`, not `pi/2`, so its cosine is 6.1e-17 instead of zero. Reducing in
//! degrees first keeps the boundaries exact, because multiples of 90 are exact.
//!
//! As elsewhere in this module, each test asserts the failure as well as the
//! success — the `assert_ne!` against the radian route is the point, not decoration.

use crate::Real;

#[test]
fn sin_cos_degrees_is_exact_on_every_quadrant() {
    // The reason the function exists: these are exact, and the radian route is not.
    assert_eq!(0.0_f64.sin_cos_degrees(), (0.0, 1.0));
    assert_eq!(90.0_f64.sin_cos_degrees(), (1.0, 0.0));
    assert_eq!(180.0_f64.sin_cos_degrees(), (0.0, -1.0));
    assert_eq!(270.0_f64.sin_cos_degrees(), (-1.0, 0.0));
    assert_eq!((-90.0_f64).sin_cos_degrees(), (-1.0, 0.0));
    assert_eq!(720.0_f64.sin_cos_degrees(), (0.0, 1.0));
    assert_eq!((-720.0_f64).sin_cos_degrees(), (-0.0, 1.0));

    // The naive route misses by ~6e-17 at exactly the places that matter.
    assert_ne!(90.0_f64.to_radians().cos(), 0.0);
}

#[test]
fn sin_cos_degrees_agrees_with_the_radian_route_away_from_quadrants() {
    // Off the boundaries there is no exactness to preserve, only agreement.
    for deg in [1.0_f64, 17.5, 44.9, 45.0, 123.456, -78.9, 359.1] {
        let (s, c) = deg.sin_cos_degrees();
        let (rs, rc) = deg.to_radians().sin_cos();
        assert!(
            (s - rs).abs() < 1e-15 && (c - rc).abs() < 1e-15,
            "{deg} deg: ({s}, {c}) vs radian route ({rs}, {rc})"
        );
    }
}

#[test]
fn sin_cos_degrees_reduces_large_arguments_exactly() {
    // Reducing in degrees is not merely as good as the radian route for a large
    // argument — it is better, because 360 is exact and 2π is not. Periodicity
    // therefore holds exactly: a million degrees gives the same answer as its
    // residue, which `to_radians().sin_cos()` cannot manage.
    assert_eq!(
        (1_000_000.5_f64).sin_cos_degrees(),
        (280.5_f64).sin_cos_degrees()
    );
    let naive_big = (1_000_000.5_f64).to_radians().sin_cos();
    let naive_small = (280.5_f64).to_radians().sin_cos();
    assert_ne!(
        naive_big, naive_small,
        "the radian route must actually drift, or this proves nothing"
    );
}
