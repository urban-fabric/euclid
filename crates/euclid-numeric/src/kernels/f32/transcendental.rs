//! The `f32` peer of [`crate::kernels::transcendental`]: the Lambert `W` and the Riemann zeta.
//!
//! Evaluated in `f64` and narrowed once; see the [module header](super).

use crate::kernels::transcendental as reference;

/// The principal branch of the Lambert `W`: the `w ≥ −1` with `w·e^w = x`.
pub(crate) fn lambert_w(x: f32) -> f32 {
    reference::lambert_w(f64::from(x)) as f32
}

/// The Riemann zeta function `ζ(s)` for real `s`.
pub(crate) fn zeta(s: f32) -> f32 {
    reference::zeta(f64::from(s)) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use euclid_utils::assert_close_f32;

    #[test]
    fn the_narrowed_values_track_the_f64_reference() {
        for &x in &[-0.3f32, -0.1, 0.0, 0.5, 1.0, 10.0, 1e3, 1e20] {
            assert_close_f32(lambert_w(x), reference::lambert_w(f64::from(x)), 1.0, "W");
        }
        for &s in &[-5.5f32, -1.0, 0.0, 0.5, 1.5, 2.0, 3.0, 10.0] {
            assert_close_f32(zeta(s), reference::zeta(f64::from(s)), 1.0, "zeta");
        }
    }

    #[test]
    fn lambert_w_satisfies_its_defining_equation_in_f32() {
        // w e^w = x, to f32 resolution. Checked as a *relative* comparison because W spans
        // many decades across this sweep.
        for i in 0..40 {
            let x = -0.36f32 + (i as f32) * 0.5;
            let w = lambert_w(x);
            assert!(w.is_finite(), "W({x}) = {w}");
            let back = w * w.exp();
            assert!(
                (back - x).abs() <= 1e-5 * x.abs().max(1.0),
                "W({x}) = {w} does not invert: {back}"
            );
        }
    }

    #[test]
    fn the_f32_surface_keeps_the_f64_conventions() {
        assert_eq!(lambert_w(0.0), 0.0);
        assert!(lambert_w(-0.4).is_nan(), "below the branch point -1/e");
        assert!(lambert_w(f32::NAN).is_nan());
        assert_eq!(lambert_w(f32::INFINITY), f32::INFINITY);
        assert_eq!(zeta(1.0), f32::INFINITY, "the simple pole");
        assert!(zeta(f32::NAN).is_nan());
        // The trivial zeros survive narrowing exactly.
        for k in 1..6 {
            let s = -2.0 * (k as f32);
            assert_eq!(zeta(s), 0.0, "zeta({s}) must be exactly zero");
        }
        assert_close_f32(zeta(0.0), -0.5, 1.0, "zeta(0)");
        assert_close_f32(zeta(2.0), 1.644_934_066_848_226_4, 2.0, "zeta(2)");
    }
}
