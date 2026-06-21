//! Special functions — the `f64` reference implementations behind the
//! [`LazyReal`](crate::numeric::LazyReal) special-function methods, each with a direct
//! tensor-op analogue (`Erf`/`Lgamma`/`Igamma`/…). Standard algorithms throughout: Lanczos
//! for gamma, Numerical-Recipes continued fractions for the incomplete gamma/beta, Giles's
//! approximation for the inverse error function. Composite quantities (log-beta, factorials,
//! log-sum-exp) live with their consumers.

pub mod bessel;
pub mod beta;
pub mod erf;
pub mod gamma;

pub use bessel::{i0, i0e, i1, i1_over_i0, i1e, ln_i0, ln_iv};
pub use beta::{beta, beta_reg, ln_beta};
pub use erf::{erf, erf_inv, erfc, erfcx};
pub use gamma::{digamma, gamma, gamma_lr, gamma_ur, ln_gamma, trigamma};
