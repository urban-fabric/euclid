//! Modified Lentz steps and convergence tests for scalar continued fractions.

use crate::HostReal;

pub(crate) const LENTZ_TINY_F64: f64 = 1e-300;
pub(crate) const LENTZ_TINY_F32: f32 = 1e-30;

pub(crate) fn nonzero_pivot<T: HostReal>(value: T, floor: T) -> T {
    if value.abs() < floor { floor } else { value }
}

/// Advance the forward pivot `c` and reciprocal pivot `d` by one coefficient pair.
/// Return the multiplier that updates the fraction's accumulated value.
pub(crate) fn lentz_step<T: HostReal>(
    numerator: T,
    denominator: T,
    c: &mut T,
    d: &mut T,
    floor: T,
) -> T {
    *d = nonzero_pivot(numerator * *d + denominator, floor).recip();
    *c = nonzero_pivot(denominator + numerator / *c, floor);
    *d * *c
}

/// Is the multiplier within a strict 1-norm tolerance of one?
/// A real fraction supplies zero for the imaginary component.
pub(crate) fn converged<T: HostReal>(real: T, imaginary: T, tolerance: T) -> bool {
    (real - T::ONE).abs() + imaginary.abs() < tolerance
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_coefficients_give_consecutive_fibonacci_ratios() {
        // 1 + 1/(1 + 1/(1 + ...)) has finite convergents 2/1, 3/2, 5/3, ...
        let (mut c, mut d, mut fraction) = (1.0, 0.0, 1.0);
        let (mut previous, mut next) = (1.0, 1.0);
        for _ in 0..20 {
            (previous, next) = (next, previous + next);
            fraction *= lentz_step(1.0, 1.0, &mut c, &mut d, LENTZ_TINY_F64);
            assert!((fraction - next / previous).abs() <= 2e-15);
        }
    }

    #[test]
    fn native_f32_step_matches_the_fibonacci_ratios() {
        let (mut c, mut d, mut fraction) = (1.0_f32, 0.0_f32, 1.0_f32);
        let (mut previous, mut next) = (1.0_f64, 1.0_f64);
        for _ in 0..20 {
            (previous, next) = (next, previous + next);
            fraction *= lentz_step(1.0, 1.0, &mut c, &mut d, LENTZ_TINY_F32);
            assert!((f64::from(fraction) - next / previous).abs() <= 3e-7);
        }
    }

    #[test]
    fn either_pivot_can_cancel_to_zero() {
        let (mut c, mut d) = (1.0, 0.5);
        assert_eq!(
            lentz_step(-1.0, 1.0, &mut c, &mut d, LENTZ_TINY_F64),
            2e-300
        );
        assert_eq!(c, 1e-300);
        assert_eq!(d, 2.0);

        let (mut c, mut d) = (2.0, 1.0);
        assert_eq!(
            lentz_step(-1.0, 1.0, &mut c, &mut d, LENTZ_TINY_F64),
            0.5 / 1e-300
        );
        assert_eq!(c, 0.5);
        assert_eq!(d, 1.0 / 1e-300);
    }

    #[test]
    fn pivot_floor_preserves_the_sign_at_its_boundary() {
        assert_eq!(nonzero_pivot(-1e-300, LENTZ_TINY_F64), -1e-300);
        assert_eq!(nonzero_pivot(-1e-301, LENTZ_TINY_F64), 1e-300);
        assert_eq!(nonzero_pivot(-1e-30, LENTZ_TINY_F32), -1e-30);
        assert_eq!(nonzero_pivot(-1e-31, LENTZ_TINY_F32), 1e-30);
    }

    #[test]
    fn convergence_requires_strict_agreement_on_both_sides_of_one() {
        let tolerance = 4.0 * f64::EPSILON;
        assert!(converged(1.0, 0.0, tolerance));
        assert!(converged(1.0 + tolerance / 2.0, 0.0, tolerance));
        assert!(converged(1.0 - tolerance / 2.0, 0.0, tolerance));
        assert!(!converged(1.0 + tolerance, 0.0, tolerance));
        assert!(!converged(1.0 - tolerance, 0.0, tolerance));
        assert!(!converged(f64::NAN, 0.0, tolerance));
        assert!(converged(1.0, -tolerance / 2.0, tolerance));
        assert!(!converged(1.0, -tolerance, tolerance));
        assert!(!converged(
            1.0 + tolerance / 2.0,
            -tolerance / 2.0,
            tolerance
        ));
    }
}
