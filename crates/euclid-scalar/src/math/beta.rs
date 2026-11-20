//! Numerical steps shared by the two precisions of the incomplete-beta kernels.
//!
//! The orientation choice and the fraction's prefactor are here; the fraction *itself* is not,
//! because its iteration cap, its convergence tolerance and its Lentz floor are three separate
//! per-format measurements, and `beta_fraction_cdf` exists in each kernel to bind them. That is
//! the split the tree's layout asks for: the arithmetic that is the same number in both formats
//! moves up, the orchestration that names each format's own constants stays down.

use crate::HostReal;

/// Which orientation of the continued fraction converges faster at this argument.
///
/// The fraction converges slowest at the distribution's mean, `a/(a+b)`, so the reflected form
/// `I_x(a,b) = 1 − I_{1−x}(b,a)` is taken on the far side of it. The split point is
/// `(a+1)/(a+b+2)` rather than the mean itself — the mode of a shifted beta, which is where the
/// two orientations cost the same.
///
/// At `(2, 4)` it is exactly `3/8`, a dyadic value, so the switch is exact; the kernels' own
/// tests pin that boundary.
pub(crate) fn use_lower_beta_fraction<T: HostReal>(a: T, b: T, x: T) -> bool {
    x < (a + T::ONE) / (a + b + T::TWO)
}

/// The common factor `x^a(1−x)^b/B(a,b)` of the direct and reflected fractions.
///
/// Formed in the log domain, because all three terms are `O((a+b)ln 2)` while their sum is
/// `O(1)`: at `a = b = 1e3` the exponent carries `1.386(a+b)·ε` of absolute error and `exp` hands
/// it back as relative error. That is the whole reason each format has a handover to an Edgeworth
/// approximation, and why the handover sits far lower in `f32`.
///
/// `ln B(a, b)` is a parameter rather than computed here: the two formats reach it through
/// different arms — a rising-factorial identity below their own Stirling threshold and a combined
/// Stirling expansion above it — and that threshold is a per-format measurement.
pub(crate) fn beta_fraction_prefactor<T: HostReal>(a: T, b: T, x: T, log_beta: T) -> T {
    (a * x.ln() + b * (-x).ln_1p() - log_beta).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_orientation_switches_at_the_shifted_mode() {
        // `(a+1)/(a+b+2)` is `3/8` at `(2,4)`, exactly representable, so the switch is exact.
        let x = 0.375_f64;
        assert!(use_lower_beta_fraction(2.0, 4.0, x.next_down()));
        assert!(!use_lower_beta_fraction(2.0, 4.0, x));
        assert!(!use_lower_beta_fraction(2.0, 4.0, x.next_up()));
        // Swapping the shapes reflects the split point, which is what makes the two orientations
        // cover the whole interval between them.
        assert!(use_lower_beta_fraction(4.0, 2.0, x));
        assert_eq!(
            use_lower_beta_fraction(2.0_f64, 4.0, 0.5),
            !use_lower_beta_fraction(4.0, 2.0, 0.5)
        );
        // Symmetric shapes split at one half.
        assert!(use_lower_beta_fraction(3.0_f64, 3.0, 0.5_f64.next_down()));
        assert!(!use_lower_beta_fraction(3.0_f64, 3.0, 0.5));
        let x = 0.375_f32;
        assert!(use_lower_beta_fraction(2.0, 4.0, x.next_down()));
        assert!(!use_lower_beta_fraction(2.0, 4.0, x));
    }

    #[test]
    fn the_prefactor_is_the_integer_shape_polynomial() {
        // `x^a(1−x)^b/B(a,b)` at `(2,3)` is `12x²(1−x)³`, since `B(2,3) = 1/12`.
        let log_beta = (1.0_f64 / 12.0).ln();
        for x in [2.0_f64.powi(-50), 0.125, 0.25, 0.75, 1.0_f64.next_down()] {
            let expected = 12.0 * x.powi(2) * (1.0 - x).powi(3);
            let actual = beta_fraction_prefactor(2.0, 3.0, x, log_beta);
            assert!(
                (actual / expected - 1.0).abs() <= 3e-14,
                "x={x}, actual={actual}"
            );
        }
        // Zero at both endpoints of the support, where one power vanishes.
        assert_eq!(beta_fraction_prefactor(2.0_f64, 3.0, 0.0, log_beta), 0.0);
        assert_eq!(beta_fraction_prefactor(2.0_f64, 3.0, 1.0, log_beta), 0.0);
        // The normaliser enters as a straight division, so doubling `B` halves the prefactor.
        let doubled = beta_fraction_prefactor(2.0_f64, 3.0, 0.25, log_beta + 2.0_f64.ln());
        let plain = beta_fraction_prefactor(2.0_f64, 3.0, 0.25, log_beta);
        assert!(
            (doubled / plain - 0.5).abs() < 1e-14,
            "{doubled} vs {plain}"
        );
        // And in `f32`, against the same polynomial.
        let log_beta = (1.0_f32 / 12.0).ln();
        let actual = beta_fraction_prefactor(2.0_f32, 3.0, 0.25, log_beta);
        assert!(
            (actual / (12.0 * 0.0625 * 0.421_875) - 1.0).abs() <= 2e-5,
            "{actual}"
        );
    }
}
