//! The `f32` peer of [`crate::kernels::integrals`]: `E₁`, `Ei`, `Si` and `Ci`.
//!
//! Evaluated in `f64` and narrowed once. See the [module header](super) for why these four,
//! unlike their neighbours, are not ported natively — in short, the `Si`/`Ci` series cancels
//! against intermediates reaching `4e20`, which `f32` cannot absorb at all.

use crate::kernels::integrals as reference;

/// `Si(x) = ∫₀ˣ sin(t)/t dt`. Odd in `x`.
pub(crate) fn sin_int(x: f32) -> f32 {
    reference::sin_int(f64::from(x)) as f32
}

/// `Ci(x) = γ + ln x + ∫₀ˣ (cos t − 1)/t dt`, for `x > 0`.
pub(crate) fn cos_int(x: f32) -> f32 {
    reference::cos_int(f64::from(x)) as f32
}

/// `E₁(x) = ∫ₓ^∞ e^{−t}/t dt`, for `x > 0`.
pub(crate) fn exp_int_e1(x: f32) -> f32 {
    reference::exp_int_e1(f64::from(x)) as f32
}

/// `Ei(x)`, the exponential integral.
///
/// `Ei` grows like `e^x/x`, so the `f64` value overflows `f32` well before it overflows `f64`;
/// the `as` conversion saturates to infinity there, which is the right answer.
pub(crate) fn exp_int_ei(x: f32) -> f32 {
    reference::exp_int_ei(f64::from(x)) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use euclid_utils::assert_close_f32;

    #[test]
    fn the_narrowed_values_track_the_f64_reference() {
        // Narrowing is one rounding, so these should agree to well under an ulp of f32.
        for &x in &[0.25f32, 0.5, 1.0, 2.0, 4.0, 7.5, 20.0, 45.0, 100.0] {
            let wide = f64::from(x);
            assert_close_f32(sin_int(x), reference::sin_int(wide), 1.0, "Si");
            assert_close_f32(cos_int(x), reference::cos_int(wide), 1.0, "Ci");
            assert_close_f32(exp_int_e1(x), reference::exp_int_e1(wide), 1.0, "E1");
        }
        for &x in &[0.5f32, 1.0, 5.0, 20.0, 60.0] {
            assert_close_f32(
                exp_int_ei(x),
                reference::exp_int_ei(f64::from(x)),
                2.0,
                "Ei",
            );
        }
    }

    #[test]
    fn the_f32_surface_keeps_the_f64_conventions() {
        // The same edge behaviour, which is what a carrier swap must not change.
        assert_eq!(sin_int(0.0), 0.0);
        assert_eq!(sin_int(f32::INFINITY), core::f32::consts::FRAC_PI_2);
        assert_eq!(sin_int(f32::NEG_INFINITY), -core::f32::consts::FRAC_PI_2);
        assert!(sin_int(f32::NAN).is_nan());
        assert!(cos_int(-1.0).is_nan());
        assert_eq!(cos_int(0.0), f32::NEG_INFINITY);
        assert_eq!(cos_int(f32::INFINITY), 0.0);
        assert!(exp_int_e1(-1.0).is_nan());
        assert_eq!(exp_int_e1(0.0), f32::INFINITY);
        assert_eq!(exp_int_e1(f32::INFINITY), 0.0);
        assert_eq!(exp_int_ei(0.0), f32::NEG_INFINITY);
        // Ei overflows f32 far earlier than f64; saturating to infinity is correct.
        assert_eq!(exp_int_ei(100.0), f32::INFINITY);
        assert!(exp_int_ei(f32::NAN).is_nan());
    }

    #[test]
    fn si_is_odd_and_ci_matches_its_known_values() {
        for &x in &[0.5f32, 1.0, 3.0, 12.0] {
            assert_eq!(sin_int(-x), -sin_int(x));
        }
        // Spot values, to f32 resolution.
        assert_close_f32(sin_int(1.0), 9.460_830_703_671_83e-1, 2.0, "Si(1)");
        assert_close_f32(cos_int(1.0), 3.374_039_229_009_682e-1, 2.0, "Ci(1)");
        assert_close_f32(sin_int(45.0), 1.558_715_000_896_413e0, 2.0, "Si(45)");
    }
}
