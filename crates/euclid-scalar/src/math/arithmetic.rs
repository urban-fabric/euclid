//! Error-free transforms for the host floating-point implementations.

use crate::HostReal;

/// Knuth's two-sum. Return the rounded sum and the discarded part.
pub(crate) fn two_sum<T: HostReal>(left: T, right: T) -> (T, T) {
    let sum = left + right;
    if !HostReal::is_finite(sum) {
        // The residual cannot reconstruct an overflowing or non-finite sum.
        return (sum, T::ZERO);
    }
    let virtual_right = sum - left;
    let error = (left - (sum - virtual_right)) + (right - virtual_right);
    (sum, error)
}

/// Return the rounded product and its FMA residual.
pub(crate) fn two_product<T: HostReal>(left: T, right: T) -> (T, T) {
    let product = left * right;
    if !HostReal::is_finite(product) {
        return (product, T::ZERO);
    }
    (product, left.mul_add(right, -product))
}

/// Test a series term's magnitude against a strict relative tolerance.
pub(crate) fn term_is_negligible<T: HostReal>(term: T, scale: T, tolerance: T) -> bool {
    term.abs() < scale * tolerance
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_sum_recovers_discarded_dyadic_terms() {
        let small = 2.0_f64.powi(-54);
        for (a, b, rounded, error) in [
            (1.0, small, 1.0, small),
            (small, 1.0, 1.0, small),
            (1.0, -small, 1.0, -small),
            (-small, -1.0, -1.0, -small),
            (-1.5, 0.5, -1.0, 0.0),
        ] {
            assert_eq!(two_sum(a, b), (rounded, error));
        }
        let small = 2.0_f32.powi(-25);
        assert_eq!(two_sum(1.0, small), (1.0, small));
        assert_eq!(two_sum(-small, -1.0), (-1.0, -small));
        assert_eq!(two_sum(1.0, 2.0 * small), (1.0, 2.0 * small));
    }

    #[test]
    fn two_product_recovers_the_cross_term() {
        let eps = 2.0_f64.powi(-28);
        assert_eq!(two_product(1.0 + eps, 1.0 - eps), (1.0, -eps * eps));
        assert_eq!(
            two_product(1.0 + eps, 1.0 + eps),
            (1.0 + 2.0 * eps, eps * eps)
        );
        let eps = 2.0_f32.powi(-13);
        assert_eq!(two_product(1.0 + eps, 1.0 - eps), (1.0, -eps * eps));
        assert_eq!(two_product(-1.0 - eps, 1.0 - eps), (-1.0, eps * eps));
    }

    #[test]
    fn nonfinite_results_carry_a_zero_residual() {
        for (rounded, residual) in [
            two_sum(f64::MAX, f64::MAX),
            two_sum(f64::NEG_INFINITY, -1.0),
            two_product(f64::MAX, 2.0),
            two_product(f64::INFINITY, -2.0),
        ] {
            assert!(rounded.is_infinite());
            assert_eq!(residual.to_bits(), 0);
        }
        for (rounded, residual) in [
            two_sum(f32::INFINITY, f32::NEG_INFINITY),
            two_sum(f32::NAN, 1.0),
            two_product(f32::INFINITY, 0.0),
            two_product(f32::NAN, 1.0),
        ] {
            assert!(rounded.is_nan());
            assert_eq!(residual.to_bits(), 0);
        }
    }

    #[test]
    fn relative_term_check_is_strict_even_at_the_subnormal_boundary() {
        for (term, expected) in [
            (0.0_f32, true),
            (0.125, true),
            (-0.125, true),
            (0.25, false),
            (-0.5, false),
        ] {
            assert_eq!(term_is_negligible(term, 2.0, 0.125), expected);
        }
        let min = f64::MIN_POSITIVE;
        assert!(term_is_negligible(0.125 * min, min, 0.25));
        assert!(!term_is_negligible(0.25 * min, min, 0.25));
        assert!(!term_is_negligible(0.0, f64::from_bits(1), 0.25));
    }
}
