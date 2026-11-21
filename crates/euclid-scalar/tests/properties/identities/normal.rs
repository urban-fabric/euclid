//! The standard normal family: CDF, log-CDF, quantile, and the inverse complementary error
//! function.
//!
//! Part of the [`identities`](super) suite.
//!
//! Four public methods — `ndtr`, `log_ndtr`, `ndtri`, `erfc_inv` — with real kernels behind them
//! and **no generated property** before this file. `kernels/normal.rs` and its f32 peer were
//! measured at 33 survivors, the largest cluster being the quantile's Newton seed.
//!
//! # The seed is the point of the deep-tail round trip
//!
//! `ndtri` solves `ln Φ(x) = ln p` by Newton from a seed, and eleven of those survivors are in
//! the seed function: perturb it and the iteration still converges, so the output cannot see it.
//! That is [cause 8](super::super) and the reason an accuracy band on a round trip is the wrong
//! instrument — a band wide enough for the real kernel is wide enough for the mutant.
//!
//! What a *generated* property adds is not a tighter tolerance but a **wider domain**. The fixed
//! suite checks five values of `p`; the property below sweeps `1e-300` upward in log space, and
//! six Newton steps from a bad seed do not converge everywhere. The seed also has a direct
//! accuracy test in the normal kernel.
//!
//! # Two identities that are true and untestable, and the saturation behind both
//!
//! Measured over 4,000-point sweeps before being written:
//!
//! - `log_ndtr(x) = ln(ndtr(x))` reaches a relative error of **0.11** at `x = 7.98`. `Φ(x)`
//!   rounds to `1.0` once `x` passes about `8.3`, so its log is `0` while the true value is
//!   `−8e-17`. That is the exact failure `log_ndtr`'s own docstring describes itself as
//!   existing to prevent, so the identity is asserted only on `x ≤ 0`, where `Φ ≤ ½` and
//!   nothing saturates.
//! - `ndtri(ndtr(x)) = x` reaches **1.5e-3** at `x = 7.97`, for the same reason from the other
//!   side: once `1 − Φ(x)` drops below `ulp(1)` the quantile cannot recover what the
//!   representation threw away.
//!
//! This is the same lesson as the log-domain suite's three rejected laws, and it is worth
//! stating once more as a rule: **where a stable kernel exists, the naive identity it replaces
//! is not available as a test.** The honest check is the one that avoids the regime the kernel
//! was written for, or an independent asymptotic — both are below.

use super::*;

/// `½ln(2π)`, the normal density's log normalizer.
fn half_ln_tau() -> f64 {
    0.5 * (2.0 * core::f64::consts::PI).ln()
}

#[path = "normal/cdf.rs"]
mod cdf;
#[path = "normal/erfc_inverse.rs"]
mod erfc_inverse;
#[path = "normal/quantile.rs"]
mod quantile;
#[path = "normal/single_precision.rs"]
mod single_precision;
