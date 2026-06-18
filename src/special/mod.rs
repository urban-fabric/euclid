//! Special functions — the concrete `f64` reference implementations behind the
//! [`LazyReal`](crate::numeric::LazyReal) special-function methods.
//!
//! Every entry here is an *elementwise* analytic function with a direct tensor-op
//! analogue (`Erf`/`Lgamma`/`Digamma`/`Igamma`/`RegularizedIncompleteBeta`, …), so
//! the same `LazyReal` method that runs `f64` on the CPU lowers to the matching op
//! on a tensor backend. The standard algorithms are used throughout: Lanczos for
//! the gamma function, Numerical-Recipes continued fractions for the incomplete
//! gamma/beta, and Giles's rational approximation for the inverse error function.
//!
//! Composite quantities built *on top of* these primitives (log-beta, factorials,
//! the multivariate gamma/beta, log-sum-exp) live with their consumers rather than
//! here.

pub mod bessel;
pub mod beta;
pub mod erf;
pub mod gamma;

pub use bessel::{i0, i0e, i1, i1_over_i0, i1e, ln_i0, ln_iv};
pub use beta::{beta, beta_reg, ln_beta};
pub use erf::{erf, erf_inv, erfc, erfcx};
pub use gamma::{digamma, gamma, gamma_lr, gamma_ur, ln_gamma, trigamma};
