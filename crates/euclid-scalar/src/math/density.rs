//! The probability densities the quantile solvers differentiate, and the normaliser one needs.
//!
//! These are Newton's derivative in [`crate::kernels::quantile`], not part of any public surface
//! — `Real` exposes the CDFs and their inverses, never the densities. Both precisions carried a
//! character-identical copy of all three, and none of them holds a format-dependent constant: the
//! normaliser is three `lnΓ` calls and each density is one `exp` of a log-form, so there is
//! nothing per-precision to rederive.
//!
//! Each takes its normaliser as an argument rather than computing it. That is the hoist the
//! quantile kernels document at length: the solver asks for a density on *every* Newton step, and
//! the normaliser depends only on the shape, so computing it per step made each step pay for a
//! nine-term Lanczos evaluation it did not need. Passing it in leaves the arithmetic identical —
//! the same value is subtracted in the same place — so every density, every Newton trajectory and
//! every returned root is unchanged to the bit.

use crate::HostReal;

/// The gamma density `x^{a−1}e^{−x}/Γ(a)`, in log form.
///
/// Zero outside the support, which is what keeps a Newton step from walking off it.
pub(crate) fn gamma_pdf<T: HostReal>(x: T, a: T, log_normaliser: T) -> T {
    if x <= T::ZERO {
        return T::ZERO;
    }
    ((a - T::ONE) * x.ln() - x - log_normaliser).exp()
}

/// The beta density `x^{a−1}(1−x)^{b−1}/B(a,b)`, in log form.
///
/// Zero outside the open support at *both* ends: the lower guard keeps `ln 0` out of the
/// exponent and the upper one keeps `ln(1−1)` out of it.
pub(crate) fn beta_pdf<T: HostReal>(x: T, a: T, b: T, log_beta: T) -> T {
    if x <= T::ZERO || x >= T::ONE {
        return T::ZERO;
    }
    ((a - T::ONE) * x.ln() + (b - T::ONE) * (-x).ln_1p() - log_beta).exp()
}

/// `ln B(a, b)` as the three-term Lanczos form.
///
/// Deliberately **not** `kernels::beta::ln_beta`, whose rising-factorial and Stirling arms would
/// return a different value: this is the expression the density used to evaluate per step, so
/// keeping it verbatim is what makes every trajectory bit-identical to before the hoist.
///
/// # Mutation testing is what asked for this being named
///
/// Its operators survived as an expression inside `beta_reg_inv`, because the normaliser reaches
/// the answer only through Newton's derivative and the beta bracket is `[0, 1]`, where bisection
/// converges on its own. A wrong normaliser costs iterations and moves no root.
pub(crate) fn log_beta_normaliser<T: HostReal>(a: T, b: T) -> T {
    a.ln_gamma() + b.ln_gamma() - (a + b).ln_gamma()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close<T: HostReal>(got: T, want: T, tolerance: T) {
        assert!((got - want).abs() <= tolerance);
    }

    #[test]
    fn the_gamma_density_is_the_distributions_own() {
        // `Γ(2) = 1`, so the normaliser is zero and the density at `x = 1` is `e^{−1}`.
        assert_eq!(gamma_pdf(1.0_f64, 2.0, 0.0), (-1.0_f64).exp());
        close(
            gamma_pdf(2.0_f64, 3.0, crate::kernels::gamma::ln_gamma(3.0)),
            0.270_670_566_473_225_5,
            1e-15,
        );
        // Outside the support it is exactly zero.
        assert_eq!(gamma_pdf(0.0_f64, 2.0, 0.0), 0.0);
        assert_eq!(gamma_pdf(-1.0_f64, 2.0, 0.0), 0.0);
        assert_eq!(gamma_pdf(0.0_f32, 2.0, 0.0), 0.0);
        close(gamma_pdf(1.0_f32, 2.0, 0.0), (-1.0_f32).exp(), 1e-7);
    }

    #[test]
    fn the_beta_density_is_the_distributions_own() {
        // `B(2,3) = 1/12`, so the density is `12x(1−x)²`.
        let log_beta = log_beta_normaliser(2.0_f64, 3.0);
        close(beta_pdf(0.5_f64, 2.0, 3.0, log_beta), 1.5, 1e-14);
        close(beta_pdf(0.25_f64, 2.0, 3.0, log_beta), 1.6875, 1e-14);
        close(beta_pdf(0.75_f64, 2.0, 3.0, log_beta), 0.5625, 1e-14);
        // Both endpoints are outside the open support, and both guards are needed.
        assert_eq!(beta_pdf(0.0_f64, 2.0, 3.0, log_beta), 0.0);
        assert_eq!(beta_pdf(1.0_f64, 2.0, 3.0, log_beta), 0.0);
        assert_eq!(beta_pdf(-0.5_f64, 2.0, 3.0, log_beta), 0.0);
        assert_eq!(beta_pdf(1.5_f64, 2.0, 3.0, log_beta), 0.0);
        // The interior is strictly positive, so neither guard may swallow the support.
        for x in [0.01_f64, 0.1, 0.5, 0.9, 0.99] {
            assert!(beta_pdf(x, 2.0, 3.0, log_beta) > 0.0, "at {x}");
        }
        let log_beta = log_beta_normaliser(2.0_f32, 3.0);
        close(beta_pdf(0.5_f32, 2.0, 3.0, log_beta), 1.5, 1e-5);
        assert_eq!(beta_pdf(1.0_f32, 2.0, 3.0, log_beta), 0.0);
    }

    #[test]
    fn the_beta_normaliser_is_the_three_term_log_beta() {
        // `B(2,3) = 1/12`, `B(1,1) = 1`, `B(½,½) = π`. A few ulps rather than exact: the
        // three-term form is not the same rounding as a closed form. Any mutation of it moves
        // the value by whole units — flipping the sign of the `lnΓ(a+b)` term shifts `B(2,3)` by
        // `2lnΓ(5) = 6.4` — so this margin still separates them.
        close(
            log_beta_normaliser(2.0_f64, 3.0),
            (1.0_f64 / 12.0).ln(),
            1e-14,
        );
        close(log_beta_normaliser(1.0_f64, 1.0), 0.0, 1e-14);
        close(
            log_beta_normaliser(0.5_f64, 0.5),
            core::f64::consts::PI.ln(),
            1e-14,
        );
        // Symmetric in its arguments, which the `−lnΓ(a+b)` term is what delivers.
        for (a, b) in [(2.0_f64, 7.0_f64), (0.25, 3.5), (1e3, 1.0)] {
            assert_eq!(log_beta_normaliser(a, b), log_beta_normaliser(b, a));
            close(
                log_beta_normaliser(a, b),
                crate::kernels::beta::ln_beta(a, b),
                1e-12,
            );
        }
        close(
            log_beta_normaliser(2.0_f32, 3.0),
            (1.0_f32 / 12.0).ln(),
            1e-5,
        );
        assert_eq!(
            log_beta_normaliser(2.0_f32, 7.0),
            log_beta_normaliser(7.0_f32, 2.0)
        );
    }
}
