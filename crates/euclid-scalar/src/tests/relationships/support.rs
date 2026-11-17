//! Densities written purely through the trait, used by the edges in this suite.

use crate::Real;

// --- numerical machinery -------------------------------------------------

// --- densities, written only through the trait ---------------------------

/// `N(0,1)` density, normalized through [`Real::half_ln_tau`].
pub(super) fn normal_pdf(x: f64) -> f64 {
    (-0.5 * x * x - <f64 as Real>::half_ln_tau()).exp()
}

/// `Gamma(a, 1)` density, normalized through [`Real::ln_gamma`].
pub(super) fn gamma_pdf(x: f64, a: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    ((a - 1.0) * x.ln() - x - Real::ln_gamma(a)).exp()
}

/// `Beta(a, b)` density, normalized through [`Real::ln_beta`].
pub(super) fn beta_pdf(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 || x >= 1.0 {
        return 0.0;
    }
    ((a - 1.0) * x.ln() + (b - 1.0) * (1.0 - x).ln() - Real::ln_beta(a, b)).exp()
}
