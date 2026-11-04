//! The Bessel functions of the **second kind**, `Y₀` and `Y₁`.
//!
//! # Method: a power series, then a march
//!
//! `Y` has no comparable quadrature — its integral representations are either oscillatory and
//! non-decaying, or carry an endpoint that destroys the trapezoidal rule's convergence. Its
//! power series carries a `ln(x/2)` and loses the same `ε·I₀(x)` the `J` series does, and the
//! Hankel expansion needs a large argument. That leaves a genuine middle gap.
//!
//! The gap is closed by **marching Bessel's equation outward**, which works here precisely
//! because of the property noted in the [parent module](super): `J` and `Y` both decay like
//! `x^{−1/2}`, so neither dominates and forward integration is stable — error introduced at
//! one step is not amplified at the next. Starting from the series where it is still exact, a
//! Taylor step of the ODE
//!
//! ```text
//! x²y″ + xy′ + (x² − n²)y = 0
//! ```
//!
//! has coefficients given by a four-term recurrence ([`taylor_step`]), with the nearest
//! singularity at the origin putting the radius of convergence at `x₀`. Stepping by `x₀/4`
//! therefore converges like `4^{−k}`, so a few dozen coefficients reach machine precision and
//! the whole march is a handful of steps.
//!
//! One march does double duty: `Y₀′ = −Y₁`, so integrating the `n = 0` equation returns `Y₀`
//! and `Y₁` together, mutually consistent by construction.
//!

use super::asymptotic::hankel;
use super::ln_half_x;
use super::ordinary_first::{HANKEL_MIN, j0, j1};
use crate::constants::f64::EULER_GAMMA;
use core::f64::consts::PI;

/// At or below this argument `Y₀`/`Y₁` use their power series. Chosen where the series is
/// still exact: its error is `≈ ε·ln(x)·I₀(x)`, which is `1e-15` here and `5e-14` by `x = 8`.
pub(super) const Y_SERIES_MAX: f64 = 4.0;

/// Fraction of the current abscissa used as a Taylor step. The radius of convergence is
/// `x₀`, so this makes the coefficients decay like `4^{−k}`.
const MARCH_RATIO: f64 = 0.25;

/// Taylor coefficients per march step. With a `4^{−k}` decay, `4^{−40} ≈ 8e-25`.
const MARCH_TERMS: usize = 40;

/// Iteration cap for the power series.
const MAX_TERMS: u32 = 1_000;

/// Relative size at which a series term stops contributing.
const TERM_EPS: f64 = 1e-20;

/// `Y₀(x)` by its power series, for small `x`.
///
/// `Y₀ = (2/π)(ln(x/2) + γ)J₀(x) + (2/π)Σ_{k≥1}(−1)^{k+1}H_k(x²/4)^k/(k!)²`.
pub(super) fn y0_series(x: f64) -> f64 {
    let q = x * x * 0.25;
    let mut term = 1.0f64; // (x²/4)^k/(k!)²
    let mut harmonic = 0.0f64;
    let mut sum = 0.0f64;
    for k in 1..MAX_TERMS {
        let kf = f64::from(k);
        term *= q / (kf * kf);
        harmonic += 1.0 / kf;
        let t = term * harmonic;
        sum += if k % 2 == 1 { t } else { -t };
        if t < sum.abs() * TERM_EPS {
            break;
        }
    }
    // `ln(x/2)` is formed as `ln x − ln 2` rather than `(x*0.5).ln()`. The product form
    // underflows for the smallest subnormals — `x*0.5` rounds to zero at `x = 5e-324`, making
    // `ln` return `−∞` where the true value is a perfectly representable `−474` — and the
    // difference is marginally more accurate everywhere else too.
    2.0 / PI * (ln_half_x(x) + EULER_GAMMA) * j0(x) + 2.0 / PI * sum
}

/// `Y₁(x)` by its power series, for small `x`.
///
/// `Y₁ = −2/(πx) + (2/π)ln(x/2)J₁(x) − (x/2π)Σ_{k≥0}(H_k + H_{k+1} − 2γ)(−x²/4)^k/(k!(k+1)!)`.
/// The `−2/(πx)` carries the pole and is kept exact.
pub(super) fn y1_series(x: f64) -> f64 {
    let q = x * x * 0.25;
    let mut term = 1.0f64; // (x²/4)^k/(k!(k+1)!)
    let mut h_k = 0.0f64;
    let mut sum = 0.0f64;
    for k in 0..MAX_TERMS {
        let kf = f64::from(k);
        if k > 0 {
            term *= q / (kf * (kf + 1.0));
            h_k += 1.0 / kf;
        }
        let h_k1 = h_k + 1.0 / (kf + 1.0);
        let t = term * (h_k + h_k1 - 2.0 * EULER_GAMMA);
        sum += if k % 2 == 0 { t } else { -t };
        if t.abs() < sum.abs() * TERM_EPS && k > 1 {
            break;
        }
    }
    -2.0 / (PI * x) + 2.0 / PI * ln_half_x(x) * j1(x) - x / (2.0 * PI) * sum
}

// --- second kind: the Taylor march --------------------------------------

/// One Taylor step of `x²y″ + xy′ + (x² − n²)y = 0` from `x₀` to `x₀ + δ`.
///
/// Substituting `x = x₀ + u` and collecting the coefficient of `uᵐ` gives
///
/// ```text
/// a_{m+2} = −[x₀(m+1)(2m+1)a_{m+1} + (m² + x₀² − n²)a_m + 2x₀a_{m−1} + a_{m−2}]
///           / (x₀²(m+2)(m+1))
/// ```
///
/// which is seeded by `a₀ = y(x₀)` and `a₁ = y′(x₀)` and needs no derivatives of the
/// coefficients. Both the value and the derivative are returned, by Horner on `Σaₖδᵏ` and
/// `Σk·aₖδ^{k−1}`.
fn taylor_step(x0: f64, y: f64, dy: f64, delta: f64, order_sq: f64) -> (f64, f64) {
    let mut a = [0.0f64; MARCH_TERMS];
    a[0] = y;
    a[1] = dy;
    let x0_sq = x0 * x0;
    for m in 0..MARCH_TERMS - 2 {
        let mf = m as f64;
        let prev1 = if m >= 1 { a[m - 1] } else { 0.0 };
        let prev2 = if m >= 2 { a[m - 2] } else { 0.0 };
        let numerator = x0 * (mf + 1.0) * (2.0 * mf + 1.0) * a[m + 1]
            + (mf * mf + x0_sq - order_sq) * a[m]
            + 2.0 * x0 * prev1
            + prev2;
        a[m + 2] = -numerator / (x0_sq * (mf + 2.0) * (mf + 1.0));
    }

    let mut value = 0.0f64;
    for k in (0..MARCH_TERMS).rev() {
        value = value * delta + a[k];
    }
    let mut derivative = 0.0f64;
    for k in (1..MARCH_TERMS).rev() {
        derivative = derivative * delta + (k as f64) * a[k];
    }
    (value, derivative)
}

/// `(Y₀(x), Y₁(x))` for `x > Y_SERIES_MAX`, by marching out from the series.
///
/// `Y₀′ = −Y₁`, so the `n = 0` march carries both.
pub(super) fn y_march(target: f64) -> (f64, f64) {
    let mut x = Y_SERIES_MAX;
    let mut value = y0_series(x);
    let mut derivative = -y1_series(x);
    while x < target {
        let step = (x * MARCH_RATIO).min(target - x);
        (value, derivative) = taylor_step(x, value, derivative, step, 0.0);
        x += step;
    }
    (value, -derivative)
}

/// The Bessel function of the second kind `Y₀(x)`, for `x > 0`.
pub(crate) fn y0(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x < 0.0 {
        return f64::NAN; // the real Y is defined for positive argument
    }
    if x == 0.0 {
        return f64::NEG_INFINITY; // logarithmic singularity
    }
    if x.is_infinite() {
        return 0.0;
    }
    if x <= Y_SERIES_MAX {
        return y0_series(x);
    }
    if x >= HANKEL_MIN {
        return hankel(0.0, x).1;
    }
    y_march(x).0
}

/// The Bessel function of the second kind `Y₁(x)`, for `x > 0`.
pub(crate) fn y1(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::NEG_INFINITY; // Y1 ~ -2/(pi x)
    }
    if x.is_infinite() {
        return 0.0;
    }
    if x <= Y_SERIES_MAX {
        return y1_series(x);
    }
    if x >= HANKEL_MIN {
        return hankel(1.0, x).1;
    }
    y_march(x).1
}
