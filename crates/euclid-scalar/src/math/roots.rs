//! Numerical steps shared by bracketed inverse CDFs.

use crate::{HostReal, Interpolate, Real};

/// Take a Newton step inside the bracket or bisect.
pub(crate) fn bracketed_newton_step<T>(x: T, residual: T, density: T, lo: T, hi: T) -> T
where
    T: HostReal + Interpolate<Scalar = T>,
{
    if density > T::ZERO {
        let candidate = x - residual / density;
        if candidate > lo && candidate < hi {
            return candidate;
        }
    }
    lo.midpoint(hi)
}

/// Compare the bracket's width with the upper endpoint's scale.
pub(crate) fn bracket_is_narrow<T: HostReal>(lo: T, hi: T, relative_tolerance: T) -> bool {
    hi - lo <= relative_tolerance * Real::max(hi.abs(), T::MIN_POSITIVE)
}

/// Transform a standard normal quantile into a gamma starting point.
pub(crate) fn wilson_hilferty_guess<T: HostReal>(z: T, shape: T) -> T {
    let cube_root =
        T::ONE - T::ONE / (T::from_f64(9.0) * shape) + z / (T::from_f64(3.0) * shape.sqrt());
    shape * cube_root * cube_root * cube_root
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn newton_acceptance_agrees_in_both_precisions() {
        // x, residual, density, lower endpoint, upper endpoint, expected next point.
        for [x, f, d, lo, hi, expected] in [
            [3.0_f64, 2.0, 2.0, 1.0, 3.0, 2.0],
            [6.0, 2.0, 2.0, 1.0, 6.0, 5.0],
            [6.0, 2.0, 0.0, 1.0, 6.0, 3.5],
            [6.0, 2.0, -0.0, 1.0, 6.0, 3.5],
            [6.0, -2.0, -2.0, 1.0, 6.0, 3.5],
            [6.0, 2.0, f64::NAN, 1.0, 6.0, 3.5],
            [6.0, 2.0, f64::INFINITY, 1.0, 6.0, 3.5],
            [6.0, 5.0, 1.0, 1.0, 6.0, 3.5],
            [1.0, -5.0, 1.0, 1.0, 6.0, 3.5],
        ] {
            assert_eq!(bracketed_newton_step(x, f, d, lo, hi), expected);
            let [x, f, d, lo, hi, expected] = [x, f, d, lo, hi, expected].map(|v| v as f32);
            assert_eq!(bracketed_newton_step(x, f, d, lo, hi), expected);
        }
    }

    #[test]
    fn width_check_includes_equality_and_scales_to_small_brackets() {
        let upper = 2.0_f64.powi(-800);
        assert!(bracket_is_narrow(0.875 * upper, upper, 0.25));
        assert!(bracket_is_narrow(0.75 * upper, upper, 0.25));
        assert!(!bracket_is_narrow(0.625 * upper, upper, 0.25));
        assert!(bracket_is_narrow(-0.25 * f32::MIN_POSITIVE, 0.0, 0.25));
        assert!(!bracket_is_narrow(-0.5 * f32::MIN_POSITIVE, 0.0, 0.25));
        assert!(bracket_is_narrow(f64::MAX, f64::MAX, f64::EPSILON));
        for hi in [f32::from_bits(1), f32::MIN_POSITIVE, 1.0, 1e37, f32::MAX] {
            assert!(bracket_is_narrow(hi.next_down(), hi, 4.0 * f32::EPSILON));
        }
    }

    #[test]
    fn wilson_hilferty_transform_matches_rational_inputs() {
        for (z, cube_root) in [(-1.5, 13.0 / 18.0), (0.0, 35.0 / 36.0), (1.5, 11.0 / 9.0)] {
            let expected = 4.0 * f64::powi(cube_root, 3);
            let f64_guess = wilson_hilferty_guess(z, 4.0_f64);
            let f32_guess = wilson_hilferty_guess(z as f32, 4.0_f32);
            assert!((f64_guess - expected).abs() <= 3e-16 * expected);
            assert!((f64::from(f32_guess) - expected).abs() <= 3e-7 * expected);
        }
    }
}
