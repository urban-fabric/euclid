//! Trigonometric formulas with a removable singularity.

use crate::{HostReal, Real};

use super::polynomial::horner;

// The coefficient of z^k, for z = x², is (-1)^k / (2k+1)!.
pub(crate) const SINC_F64: [f64; 4] = [-1.0 / 5040.0, 1.0 / 120.0, -1.0 / 6.0, 1.0];
pub(crate) const SINC_F32: [f32; 2] = [-1.0 / 6.0, 1.0];

fn use_sinc_series<T: HostReal>(x: T) -> bool {
    x.abs() < T::from_f64(1e-2)
}

fn sinc_series<T: Real>(x: T, coefficients: &[T]) -> T {
    horner(x.square(), coefficients)
}

/// `sin(x)/x` with the continuous value one at zero. For `|x| < 0.01`, the
/// first omitted Taylor term bounds truncation by `0.01^4/5!` for f32
/// and `0.01^8/9!` for f64, before rounding the arithmetic.
pub(crate) fn sinc<T: HostReal>(x: T, coefficients: &[T]) -> T {
    if use_sinc_series(x) {
        sinc_series(x, coefficients)
    } else {
        x.sin() / x
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use euclid_algebra::Rational64;

    fn exact_sinc_polynomial(denominator: i64, terms: i64) -> f64 {
        let squared = Rational64::new(1, denominator * denominator);
        let mut power = Rational64::new(1, 1);
        let mut sum = Rational64::new(0, 1);
        for k in 0..terms {
            let factorial = (1..=2 * k + 1).product();
            let sign = if k % 2 == 0 { 1 } else { -1 };
            sum += power * Rational64::new(sign, factorial);
            power *= squared;
        }
        sum.numerator() as f64 / sum.denominator() as f64
    }

    #[test]
    fn series_selection_is_strict_and_even() {
        let cutoff = 0.01_f64;
        for x in [0.0, cutoff.next_down(), -cutoff.next_down()] {
            assert!(use_sinc_series(x));
        }
        for x in [
            cutoff,
            -cutoff,
            cutoff.next_up(),
            -cutoff.next_up(),
            f64::NAN,
        ] {
            assert!(!use_sinc_series(x));
        }
        assert!(use_sinc_series(0.01_f32.next_down()));
        assert!(!use_sinc_series(-0.01_f32));
    }

    #[test]
    fn sinc_polynomials_agree_with_their_rational_sums() {
        // Check evaluation against the exact rational sum of signed factorial
        // terms. On this wider interval even the smallest coefficient matters.
        for denominator in [1, 2, 4, 8] {
            let x = 1.0 / denominator as f64;
            let want64 = exact_sinc_polynomial(denominator, 4);
            let want32 = exact_sinc_polynomial(denominator, 2) as f32;
            assert!((sinc_series(x, &SINC_F64) - want64).abs() <= f64::EPSILON);
            assert!((sinc_series(-x as f32, &SINC_F32) - want32).abs() <= f32::EPSILON);
        }
        assert_eq!(sinc_series(0.0, &SINC_F64), 1.0);
    }
}
