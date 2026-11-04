//! The `f32` peer of the ordinary Bessel functions: `J₀`, `J₁`, `Y₀`, `Y₁`.
//!
//! Both kinds share one file here, unlike the `f64` side where each is a few hundred
//! lines of numerics: narrowed, every one of them is a single expression.
//!
//! Evaluated in `f64` and narrowed once; see the [module header](crate::kernels::f32). The
//! `f32` case is especially clear here: all three of the `f64` methods — a cancelling series,
//! an outward march, and a divergent asymptotic expansion — have error floors set by the
//! working precision, so a native `f32` port would be *less* accurate than narrowing, not more.

use crate::kernels::bessel::{ordinary_first, ordinary_second};

/// The Bessel function of the first kind `J₀(x)`, even in `x`.
pub(crate) fn j0(x: f32) -> f32 {
    ordinary_first::j0(f64::from(x)) as f32
}

/// The Bessel function of the first kind `J₁(x)`, odd in `x`.
pub(crate) fn j1(x: f32) -> f32 {
    ordinary_first::j1(f64::from(x)) as f32
}

/// The Bessel function of the second kind `Y₀(x)`, for `x > 0`.
pub(crate) fn y0(x: f32) -> f32 {
    ordinary_second::y0(f64::from(x)) as f32
}

/// The Bessel function of the second kind `Y₁(x)`, for `x > 0`.
pub(crate) fn y1(x: f32) -> f32 {
    ordinary_second::y1(f64::from(x)) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use euclid_utils::assert_close_f32;

    #[test]
    fn the_narrowed_values_track_the_f64_reference() {
        for i in 1..80 {
            let x = (i as f32) * 0.5;
            let wide = f64::from(x);
            assert_close_f32(j0(x), ordinary_first::j0(wide), 1.0, "J0");
            assert_close_f32(j1(x), ordinary_first::j1(wide), 1.0, "J1");
            assert_close_f32(y0(x), ordinary_second::y0(wide), 1.0, "Y0");
            assert_close_f32(y1(x), ordinary_second::y1(wide), 1.0, "Y1");
        }
    }

    #[test]
    fn the_wronskian_holds_at_f32_resolution() {
        // J0 Y1 - J1 Y0 = -2/(pi x). The f32 values carry ~1e-7, so the residual is judged
        // against that rather than against the f64 bound.
        for i in 1..60 {
            let x = (i as f32) * 0.5;
            let residual = j0(x) * y1(x) - j1(x) * y0(x) + 2.0 / (core::f32::consts::PI * x);
            let scale = 2.0 / (core::f32::consts::PI * x);
            assert!(
                residual.abs() <= 1e-5 * scale.max(1e-2),
                "Wronskian broken at x = {x}: {residual:e}"
            );
        }
    }

    #[test]
    fn the_f32_surface_keeps_the_f64_conventions() {
        assert_eq!(j0(0.0), 1.0);
        assert_eq!(j1(0.0), 0.0);
        // J0 is even and J1 odd, exactly.
        for &x in &[0.5f32, 1.0, 4.0, 11.0, 30.0] {
            assert_eq!(j0(-x), j0(x));
            assert_eq!(j1(-x), -j1(x));
        }
        assert!(j0(f32::NAN).is_nan());
        assert!(j1(f32::NAN).is_nan());
        assert!(y0(f32::NAN).is_nan());
        assert!(y1(f32::NAN).is_nan());
        assert_eq!(j0(f32::INFINITY), 0.0);
        assert_eq!(y0(f32::INFINITY), 0.0);
        // Y is real only above zero, with a singularity at the origin.
        assert!(y0(-1.0).is_nan());
        assert!(y1(-1.0).is_nan());
        assert_eq!(y0(0.0), f32::NEG_INFINITY);
        assert_eq!(y1(0.0), f32::NEG_INFINITY);
    }

    #[test]
    fn spot_values_match_to_f32_resolution() {
        assert_close_f32(j0(1.0), 7.651_976_865_579_666e-1, 2.0, "J0(1)");
        assert_close_f32(j1(1.0), 4.400_505_857_449_335e-1, 2.0, "J1(1)");
        assert_close_f32(y0(1.0), 8.825_696_421_567_696e-2, 4.0, "Y0(1)");
        assert_close_f32(y1(1.0), -7.812_128_213_002_887e-1, 2.0, "Y1(1)");
        assert_close_f32(j0(10.0), -2.459_357_644_513_483e-1, 2.0, "J0(10)");
        assert_close_f32(y1(20.0), -1.655_116_143_625_213e-1, 2.0, "Y1(20)");
    }
}
