//! Modified Lentz steps and convergence tests for scalar continued fractions.

use crate::HostReal;

pub(crate) const LENTZ_TINY_F64: f64 = 1e-300;
pub(crate) const LENTZ_TINY_F32: f32 = 1e-30;

pub(crate) fn nonzero_pivot<T: HostReal>(value: T, floor: T) -> T {
    if value.abs() < floor { floor } else { value }
}

/// Advance the forward pivot `c` and reciprocal pivot `d` by one coefficient pair.
/// Return the multiplier that updates the fraction's accumulated value.
/// `1/(scale·t)`, by whichever association keeps the product inside the format.
///
/// # The product is what overflows
///
/// The direct form is one rounding shorter and correct through the normal range, but at
/// `t > MAX/scale` the product becomes infinite while the *result* is an ordinary subnormal, and
/// `1/inf` flushes the answer to zero. Measured in `erfcx`, its only caller today:
/// `erfcx(1e308)` was correct at `5.6419e-309` and everything above `1.0142e308` returned `0`
/// where the true value is near `5.56e-309` — the boundary sitting exactly on `1/√π` of `MAX`,
/// which is what identified the product as the culprit.
///
/// Guarded rather than reassociated: dividing twice would re-round every value in the normal
/// range, where the direct form is correct. The fallback runs only where the product has already
/// left the format, so every value the callers return is unchanged to the bit.
///
/// # Mutation testing is what asked for this
///
/// Both divisions in the fallback survived the whole suite in both precisions, because nothing
/// reaches a `t` that large through a kernel: `erfcx`'s own argument has to exceed `1e308` to
/// produce one. Generic and named, the arm no argument reaches is reachable by passing `t`
/// directly, and it is tested once for both formats.
pub(crate) fn reciprocal_of_product<T: HostReal>(scale: T, t: T) -> T {
    let scaled = scale * t;
    if HostReal::is_finite(scaled) {
        T::ONE / scaled
    } else {
        (T::ONE / t) / scale
    }
}

/// Replace an exactly-zero *complex* intermediate with the floor, the complex analogue of
/// [`nonzero_pivot`].
///
/// Same purpose and same shape: a vanishing partial denominator is what makes Lentz's method
/// fragile, and nudging it off zero is what makes it robust. The floor is a parameter rather
/// than a constant here for the reason it is in `nonzero_pivot` — it is per format, and
/// [`LENTZ_TINY_F64`] and [`LENTZ_TINY_F32`] are the two values.
pub(crate) fn nudge<T: HostReal>(z: (T, T), floor: T) -> (T, T) {
    if super::complex::c_norm1(z) == T::ZERO {
        (floor, T::ZERO)
    } else {
        z
    }
}

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

    /// Both associations of `1/(scale·t)`, including the arm no kernel argument reaches.
    ///
    /// This test moved here with the function — it used to live beside the `f64` `erf` kernel,
    /// where it could only check one format. The fallback is selected above `MAX/scale`, which
    /// `erfcx` reaches only past `t = 1e308`, so passing `t` directly is the only way in.
    #[test]
    fn the_reciprocal_guards_the_product_rather_than_reassociating() {
        let root_pi = core::f64::consts::PI.sqrt();
        // Where the product fits, that is the form: `1/√π = 0.5641895835477563` at `t = 1`.
        assert_eq!(reciprocal_of_product(root_pi, 1.0), 1.0 / root_pi);
        for t in [1.0_f64, 2.5, 1e100, 1e300] {
            let divided = (1.0 / t) / root_pi;
            assert!(
                (reciprocal_of_product(root_pi, t) - divided).abs() <= 2.0 * f64::EPSILON * divided
            );
        }

        // Past `f64::MAX/√π = 1.0142e308` the product is gone and the divided arm is the answer —
        // an ordinary subnormal where the direct form would have returned zero.
        let beyond = 1.1e308_f64;
        assert!(!(root_pi * beyond).is_finite());
        assert_eq!(
            reciprocal_of_product(root_pi, beyond),
            (1.0 / beyond) / root_pi
        );
        assert!(reciprocal_of_product(root_pi, beyond) > 0.0);
        assert!(root_pi * beyond > f64::MAX);

        // The same two arms in `f32`, whose product leaves the format at `f32::MAX/√π = 1.92e38`.
        let root_pi = core::f32::consts::PI.sqrt();
        assert_eq!(reciprocal_of_product(root_pi, 1.0_f32), 1.0 / root_pi);
        let beyond = 2.0e38_f32;
        assert!(!(root_pi * beyond).is_finite());
        assert_eq!(
            reciprocal_of_product(root_pi, beyond),
            (1.0 / beyond) / root_pi
        );
        assert!(reciprocal_of_product(root_pi, beyond) > 0.0);

        // A unit scale makes it a plain reciprocal, which pins both divisions independently of
        // the constant the callers happen to pass.
        assert_eq!(reciprocal_of_product(1.0_f64, 4.0), 0.25);
        assert_eq!(reciprocal_of_product(2.0_f64, 4.0), 0.125);
    }
}
