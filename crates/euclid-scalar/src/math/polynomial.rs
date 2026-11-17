//! Polynomial evaluation shared by the primitive carriers.

use crate::Semiring;

/// Horner evaluation with the highest coefficient first and separate
/// multiply and add operations. An empty coefficient slice represents zero.
pub(crate) fn horner<T: Semiring>(x: T, coefficients: &[T]) -> T {
    let Some((leading, rest)) = coefficients.split_first() else {
        return T::zero();
    };
    let mut value = leading.clone();
    for coefficient in rest {
        value = value * x.clone() + coefficient.clone();
    }
    value
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn asymmetric_polynomial_matches_its_exact_dyadic_values() {
        for (x, expected) in [(-2.0_f64, -34.0), (-0.5, 0.5), (0.0, 4.0), (2.0, 18.0)] {
            assert_eq!(horner(x, &[2.0, -3.0, 5.0, 4.0]), expected);
            assert_eq!(horner(x as f32, &[2.0, -3.0, 5.0, 4.0]), expected as f32);
        }
        assert_eq!(horner(f64::INFINITY, &[3.0]), 3.0);
        assert_eq!(horner(f32::NAN, &[]), 0.0);
        assert_eq!(horner(-2_i128, &[2, -3, 5, 4]), -34);
    }
}
