//! The beta quantile: the `x` in `[0, 1]` with `I_x(a, b) = p`.
//!
//! A child module alongside [`super::gamma`], and much the shorter of the two for one reason:
//! the support is bounded, so the bracket is `[0, 1]` with no search required. That makes this
//! case unconditionally safe for any `a` and `b`, including the U-shaped densities whose mass
//! piles into both corners.

use super::*;

/// The beta quantile: the `x` in `[0, 1]` with `I_x(a, b) = p`.
///
/// The support is bounded, so the bracket is `[0, 1]` with no search required — this case is
/// unconditionally safe for any `a`, `b`, including the U-shaped densities (`a, b < 1`) whose
/// mass piles into both corners.
pub(crate) fn beta_reg_inv(p: f64, a: f64, b: f64) -> f64 {
    match BetaQuantileRoute::for_arguments(p, a, b) {
        BetaQuantileRoute::Undefined => f64::NAN,
        BetaQuantileRoute::AtZero => 0.0,
        BetaQuantileRoute::AtOne => 1.0,
        BetaQuantileRoute::Solve => beta_quantile_solve(p, a, b),
    }
}

/// Which route produces the beta quantile at this `(p, a, b)`.
///
/// One arm fewer than [`GammaQuantileRoute`], and the missing one is the point: the beta's
/// support is bounded, so both endpoints are attained and there is no unresolvable-shape case to
/// detect — the bracket `[0, 1]` is correct for every `a` and `b`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BetaQuantileRoute {
    /// Either shape not a positive finite number, or a `p` outside `[0, 1]` — `NaN` included,
    /// which the range check rejects by failing rather than by comparing.
    Undefined,
    /// `p = 0`: the left endpoint of the support, attained exactly.
    AtZero,
    /// `p = 1`: the right endpoint, which unlike the gamma's the beta does attain.
    AtOne,
    /// An interior probability: solve on `[0, 1]`.
    Solve,
}

impl BetaQuantileRoute {
    fn for_arguments(p: f64, a: f64, b: f64) -> Self {
        if !a.is_finite() || !b.is_finite() || a <= 0.0 || b <= 0.0 || !(0.0..=1.0).contains(&p) {
            Self::Undefined
        } else if p == 0.0 {
            Self::AtZero
        } else if p == 1.0 {
            Self::AtOne
        } else {
            Self::Solve
        }
    }
}

/// Solve for an interior beta quantile. The bracket is the whole support, known up front.
fn beta_quantile_solve(p: f64, a: f64, b: f64) -> f64 {
    let cdf = |x: f64| crate::kernels::beta::beta_reg(a, b, x);
    let log_beta = log_beta_normaliser(a, b);
    solve_increasing(
        cdf,
        |x| beta_pdf(x, a, b, log_beta),
        p,
        0.0,
        1.0,
        residual_tolerance(a.min(b)),
    )
}

#[cfg(test)]
mod tests;
