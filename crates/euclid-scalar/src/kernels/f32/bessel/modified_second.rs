//! The `f32` peer of [`crate::kernels::bessel::modified_second`]: `K₀` and `K₁` with
//! their scaled forms.
//!
//! Unlike `I` above, these narrow the `f64` reference rather than being native. The `f64`
//! method is a trapezoidal rule whose step size is chosen from the strip width of the
//! integrand's analytic continuation; in `f32` that step would have to be retuned to a `1e-7`
//! target, giving a result no better than narrowing while doubling what has to be verified.
//! See the [module header](crate::kernels::f32).

/// The modified Bessel function of the second kind `K₀(x)`, for `x > 0`.
pub(crate) fn k0(x: f32) -> f32 {
    crate::kernels::bessel::modified_second::k0(f64::from(x)) as f32
}

/// The modified Bessel function of the second kind `K₁(x)`, for `x > 0`.
pub(crate) fn k1(x: f32) -> f32 {
    crate::kernels::bessel::modified_second::k1(f64::from(x)) as f32
}

/// The exponentially-scaled `K₀ᵉ(x) = e^{x}K₀(x)`.
pub(crate) fn k0e(x: f32) -> f32 {
    crate::kernels::bessel::modified_second::k0e(f64::from(x)) as f32
}

/// The exponentially-scaled `K₁ᵉ(x) = e^{x}K₁(x)`.
pub(crate) fn k1e(x: f32) -> f32 {
    crate::kernels::bessel::modified_second::k1e(f64::from(x)) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::f32::bessel::modified_first::{i0e, i1e};
    use euclid_utils::assert_close_f32;

    #[test]
    fn the_narrowed_values_track_the_f64_reference() {
        for i in 1..60 {
            let x = (i as f32) * 0.25;
            let wide = f64::from(x);
            assert_close_f32(
                k0(x),
                crate::kernels::bessel::modified_second::k0(wide),
                1.0,
                "K0",
            );
            assert_close_f32(
                k1(x),
                crate::kernels::bessel::modified_second::k1(wide),
                1.0,
                "K1",
            );
            assert_close_f32(
                k0e(x),
                crate::kernels::bessel::modified_second::k0e(wide),
                1.0,
                "K0e",
            );
        }
    }

    #[test]
    fn the_modified_wronskian_holds_at_f32_resolution() {
        // I0 K1 + I1 K0 = 1/x, in the scaled forms so it is checkable at any argument.
        for i in 1..40 {
            let x = (i as f32) * 0.5;
            let residual = i0e(x) * k1e(x) + i1e(x) * k0e(x) - 1.0 / x;
            assert!(
                (residual * x).abs() <= 1e-5,
                "I/K Wronskian broken at x = {x}: {residual:e}"
            );
        }
    }

    #[test]
    fn the_f32_surface_keeps_the_f64_conventions() {
        assert_eq!(k0(0.0), f32::INFINITY);
        assert_eq!(k1(0.0), f32::INFINITY);
        assert!(k0(-1.0).is_nan());
        assert!(k1(-1.0).is_nan());
        assert!(k0(f32::NAN).is_nan());
        assert!(k1e(f32::NAN).is_nan());
        assert_eq!(k0(f32::INFINITY), 0.0);
        assert_eq!(k0e(f32::INFINITY), 0.0);
        // K underflows f32 at a far smaller argument than f64; the scaled form does not.
        assert_eq!(
            k0(200.0),
            0.0,
            "the unscaled form underflows f32 by x = 200"
        );
        assert!(k0e(200.0) > 0.0 && k0e(200.0).is_finite());
        assert!(k0e(1e30) > 0.0, "the scaled form stays usable throughout");
    }

    #[test]
    fn spot_values_match_to_f32_resolution() {
        assert_close_f32(k0(1.0), 4.210_244_382_407_083e-1, 2.0, "K0(1)");
        assert_close_f32(k1(1.0), 6.019_072_301_972_346e-1, 2.0, "K1(1)");
        assert_close_f32(k0e(10.0), 3.916_319_344_365_987e-1, 2.0, "K0e(10)");
        assert_close_f32(k1e(50.0), 1.785_665_585_588_156e-1, 2.0, "K1e(50)");
    }
}
