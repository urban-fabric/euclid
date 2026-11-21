//! Complex arithmetic on `(re, im)` pairs, for the kernels that need a continued fraction in the
//! complex plane.
//!
//! No complex *type* — the two kernels that reach here carry their values as plain tuples, and
//! introducing one would put a public name on something only `E₁`'s Lentz fraction uses. These
//! are the three operations that fraction needs, and the reciprocal is the only one with any
//! numerical content.
//!
//! Centralised because both precisions of [`crate::kernels::integrals`] carried a
//! character-identical copy of each. The arithmetic has no format-dependent constant in it, so
//! there was nothing per-precision to rederive — unlike a crossover threshold, which belongs with
//! the kernel that measured it.

use crate::HostReal;

/// `(a + bi)(c + di)`.
pub(crate) fn c_mul<T: HostReal>((a, b): (T, T), (c, d): (T, T)) -> (T, T) {
    (a * c - b * d, a * d + b * c)
}

/// `1/(a + bi)`, by Smith's method.
///
/// The textbook `(a − bi)/(a² + b²)` squares both parts, which overflows or underflows for
/// perfectly representable inputs; dividing through by the larger component first keeps every
/// intermediate within range.
pub(crate) fn c_recip<T: HostReal>((a, b): (T, T)) -> (T, T) {
    if a.abs() >= b.abs() {
        let r = b / a;
        let d = a + b * r;
        (T::ONE / d, -r / d)
    } else {
        let r = a / b;
        let d = a * r + b;
        (r / d, -T::ONE / d)
    }
}

/// The 1-norm `|re| + |im|`, for Lentz's zero test.
///
/// A *sum* of magnitudes rather than a modulus: it needs only to detect an exact zero, and this
/// form cannot cancel two equal-magnitude opposite parts into one. See
/// [`super::continued_fraction::nudge`], which is the only caller.
pub(crate) fn c_norm1<T: HostReal>((a, b): (T, T)) -> T {
    a.abs() + b.abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_one_norm_is_the_sum_of_magnitudes() {
        assert_eq!(c_norm1((3.0_f64, -4.0)), 7.0);
        assert_eq!(c_norm1((-3.0_f64, 4.0)), 7.0);
        assert_eq!(c_norm1((0.0_f64, 0.0)), 0.0);
        assert_eq!(c_norm1((-0.0_f64, -0.0)), 0.0);
        // A sum, not a difference or a modulus: equal-magnitude opposite parts must not cancel,
        // which is the whole reason the zero test can trust it.
        assert_eq!(c_norm1((2.0_f64, -2.0)), 4.0);
        assert!(c_norm1((f64::MIN_POSITIVE, -f64::MIN_POSITIVE)) > 0.0);
        assert_eq!(c_norm1((3.0_f32, -4.0)), 7.0);
        assert!(c_norm1((f32::MIN_POSITIVE, -f32::MIN_POSITIVE)) > 0.0);
    }

    #[test]
    fn the_product_is_the_complex_product() {
        // `(1 + 2i)(3 + 4i) = −5 + 10i`.
        assert_eq!(c_mul((1.0_f64, 2.0), (3.0, 4.0)), (-5.0, 10.0));
        // `i² = −1`, which pins the sign of the real part's subtraction.
        assert_eq!(c_mul((0.0_f64, 1.0), (0.0, 1.0)), (-1.0, 0.0));
        // Multiplying by one is the identity, and by `i` is a quarter turn.
        assert_eq!(c_mul((2.0_f64, -3.0), (1.0, 0.0)), (2.0, -3.0));
        assert_eq!(c_mul((2.0_f64, -3.0), (0.0, 1.0)), (3.0, 2.0));
        assert_eq!(c_mul((1.0_f32, 2.0), (3.0, 4.0)), (-5.0, 10.0));
    }

    #[test]
    fn the_reciprocal_survives_a_component_that_would_square_out_of_range() {
        // `1/(2 + 0i) = 0.5`, and `1/i = −i`.
        assert_eq!(c_recip((2.0_f64, 0.0)), (0.5, -0.0));
        assert_eq!(c_recip((0.0_f64, 1.0)), (0.0, -1.0));
        // `z·(1/z) = 1` to a rounding, over both branches of the scaling.
        for z in [
            (3.0_f64, 4.0),
            (4.0, 3.0),
            (-3.0, 4.0),
            (1e300, 1.0),
            (1.0, 1e300),
            (1e-300, 1e-300),
        ] {
            let (re, im) = c_mul(z, c_recip(z));
            assert!((re - 1.0).abs() <= 4.0 * f64::EPSILON, "{z:?} -> {re}");
            assert!(im.abs() <= 4.0 * f64::EPSILON, "{z:?} -> {im}");
        }
        // Smith's method is what makes the extremes work: squaring either component of
        // `(1e300, 1.0)` would be `+∞`, and of `(1e-300, 1e-300)` would be `0`.
        assert!(!(1e300_f64 * 1e300).is_finite());
        assert_eq!(1e-300_f64 * 1e-300, 0.0);
        let (re, im) = c_recip((1e300_f64, 1.0));
        // `−r/d` is `−1e-600` here, so the imaginary part underflows to a *negative zero* — the
        // sign survives where the magnitude cannot, which is the most the format can say.
        assert!(re > 0.0 && re.is_finite());
        assert_eq!(im, 0.0);
        assert!(
            im.is_sign_negative(),
            "the sign of the imaginary part is lost"
        );
        let (re, im) = c_recip((1e-300_f64, 1e-300));
        assert!(re.is_finite() && im.is_finite() && re > 0.0);
    }
}
