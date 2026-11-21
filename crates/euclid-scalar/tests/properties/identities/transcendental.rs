//! Lambert `W` and the Riemann zeta function.
//!
//! Part of the [`identities`](super) suite.
//!
//! Both had no generated property before this file, and `kernels/transcendental.rs` was measured
//! at **43 survivors of 194 mutants** — 35 of them inside `lambert_w`.
//!
//! # `W` has the best law in the crate: its own definition
//!
//! `W` is *defined* as the inverse of `w ↦ w e^w`, so `W(x)·e^{W(x)} = x` is an identity at every
//! argument in the domain, needs no reference value, and pins the function rather than a sample
//! of it. Nothing else here comes that cheap.
//!
//! It is checked in two forms, each chosen to avoid an overflow rather than a tolerance:
//!
//! - For `x > 0`, in logs: `ln W(x) + W(x) = ln x`. The direct form would need `e^{684}` at
//!   `x = 1e300` — representable, but only just, and `W(f64::MAX)·e^{W(f64::MAX)}` is within a
//!   factor of three of overflowing.
//! - For `−1/e ≤ x < 0`, directly: `W e^W = x`. Here `|x| ≤ 1/e`, so nothing is near a limit,
//!   and the log form is unavailable because `W` is negative.
//!
//! Measured at **one ulp** in both forms — `2.2e-16` across `[1e-300, 1e-1]`, `[1e1, 1e100]` and
//! `[1e100, 1e300]`, and `2.2e-16` on the negative branch.
//!
//! # Where the identity stops being a test, and why that is not a defect
//!
//! At the branch point the residual check goes **blind**, and it is worth saying so rather than
//! discovering it later. `d/dw(w e^w) = e^w(1 + w)` vanishes at `w = −1`, so an error in `W`
//! maps to a *smaller* error in the recovered `x`: at `1e-16` past the branch point the residual
//! is exactly zero while `1 + W` is `1.9e-8`. The function is not Lipschitz there — its own
//! docstring says accuracy is "inherently halved" — so the identity cannot see the error the
//! format cannot avoid. The property asserts the asymptotic shape there instead.
//!
//! # `ln x` passes through zero at `x = 1`
//!
//! The log form's reference is `ln x`, which is `0` at `x = 1`, so a relative comparison there is
//! undefined — measured as a relative error of `1.0` at exactly `x = 1`, which says nothing about
//! the kernel. The bound is mixed, the same correction the log-domain and normal suites both
//! needed. Three suites in a row have now hit this: **a reference that crosses zero needs an
//! absolute term, and the sweep will find the crossing.**

use super::*;

/// `−1/e`, the branch point, below which the real branches end.
fn branch_point() -> f64 {
    -1.0 / core::f64::consts::E
}

#[path = "transcendental/lambert.rs"]
mod lambert;
#[path = "transcendental/single_precision.rs"]
mod single_precision;
#[path = "transcendental/zeta.rs"]
mod zeta;
