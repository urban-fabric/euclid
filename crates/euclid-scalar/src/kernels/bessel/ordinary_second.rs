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
use crate::math::arithmetic::term_is_negligible;
use crate::math::pair::{Converged, Slope};
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
pub(super) const MAX_TERMS: u32 = 1_000;

/// Relative size at which a term stops contributing.
///
/// **`1e-19`, which is deliberately only a little below `f64`'s resolution.** `ε = 2.22e-16`, so
/// a term at `1e-19` of the sum moves the result by about a thousandth of an ulp, and every term
/// after it moves it by less. The value was `1e-20`, which asked the `Y` series and its march to converge
/// one order past the point where the format can represent the difference —
/// iterations bought and then discarded.
///
/// Raising it to `1e-19` is **free rather than a trade.** The outputs are bit-identical across a
/// 306,280-point sweep covering every kernel in this tree at both signs, while cutting `Y₀` by **2.9%** and `Y₁` by **2.7%**.9%** each.
///
/// Timings are the minimum of six interleaved A/B runs of two prebuilt binaries pinned to idle
/// cores, which cancels drift — this machine is shared, and a naive before/after on it reported
/// swings of 2x on *identical* code.
///
/// `1e-18` was measured too, and is another 1-3% faster, but it is no longer bit-identical: it
/// differs by up to 3 ulp on `Bi`. So `1e-19` is the last value that costs nothing at all, and
/// the margin over `ε` is what pays for a series whose tail is amplified by `largest/sum` where
/// it cancels.
const TERM_EPS: f64 = 1e-19;

/// `Y₀(x)` by its power series, for small `x`.
///
/// `Y₀ = (2/π)(ln(x/2) + γ)J₀(x) + (2/π)Σ_{k≥1}(−1)^{k+1}H_k(x²/4)^k/(k!)²`.
pub(super) fn y0_series(x: f64) -> f64 {
    y0_series_terms(x).value
}

/// `Y₀`'s series, with the number of terms it took.
///
/// Instrumented because the stopping rule is otherwise unassertable: `TERM_EPS` is `1e-19`,
/// below `f64`'s resolution, so stopping a term early or late moves no value and every mutation
/// of `t < sum·ε` survives. The count is what separates them. See [`Converged`].
pub(super) fn y0_series_terms(x: f64) -> Converged<f64> {
    let q = x * x * 0.25;
    let mut term = 1.0f64; // (x²/4)^k/(k!)²
    let mut harmonic = 0.0f64;
    let mut sum = 0.0f64;
    let mut terms = 0;
    for k in 1..MAX_TERMS {
        let kf = f64::from(k);
        term *= q / (kf * kf);
        harmonic += 1.0 / kf;
        let t = term * harmonic;
        sum += if k % 2 == 1 { t } else { -t };
        terms += 1;
        // `t == 0` is a *separate* exit from the tolerance, and it is not redundant: the
        // tolerance is relative, so when `x²/4` underflows to zero both `t` and `sum` are
        // exactly zero and `0 < 0·ε` is false. Without this the loop ran to `MAX_TERMS` for
        // every argument below about `3e-162`, summing nothing — 999 iterations to return the
        // leading term. `term` only ever shrinks, so a zero term means every later one is zero
        // too and stopping is exact rather than a truncation.
        if t == 0.0 || term_is_negligible(t, sum.abs(), TERM_EPS) {
            break;
        }
    }
    // `ln(x/2)` is formed as `ln x − ln 2` rather than `(x*0.5).ln()`. The product form
    // underflows for the smallest subnormals — `x*0.5` rounds to zero at `x = 5e-324`, making
    // `ln` return `−∞` where the true value is a perfectly representable `−474` — and the
    // difference is marginally more accurate everywhere else too.
    Converged::new(
        2.0 / PI * (ln_half_x(x) + EULER_GAMMA) * j0(x) + 2.0 / PI * sum,
        terms,
    )
}

/// `Y₁(x)` by its power series, for small `x`.
///
/// `Y₁ = −2/(πx) + (2/π)ln(x/2)J₁(x) − (x/2π)Σ_{k≥0}(H_k + H_{k+1} − 2γ)(−x²/4)^k/(k!(k+1)!)`.
/// The `−2/(πx)` carries the pole and is kept exact.
pub(super) fn y1_series(x: f64) -> f64 {
    y1_series_terms(x).value
}

/// `Y₁`'s series, with the number of terms it took.
///
/// Instrumented for the reason [`y0_series_terms`] records, and with one more comparison to
/// protect: the `k > 1` guard on the break forces at least two terms, and nothing but the count
/// can see it.
pub(super) fn y1_series_terms(x: f64) -> Converged<f64> {
    let q = x * x * 0.25;
    let mut term = 1.0f64; // (x²/4)^k/(k!(k+1)!)
    let mut h_k = 0.0f64;
    let mut sum = 0.0f64;
    let mut terms = 0;
    for k in 0..MAX_TERMS {
        let kf = f64::from(k);
        if k > 0 {
            term *= q / (kf * (kf + 1.0));
            h_k += 1.0 / kf;
        }
        let h_k1 = h_k + 1.0 / (kf + 1.0);
        let t = term * (h_k + h_k1 - 2.0 * EULER_GAMMA);
        sum += if k % 2 == 0 { t } else { -t };
        terms += 1;
        if term_is_negligible(t, sum.abs(), TERM_EPS) && k > 1 {
            break;
        }
    }
    Converged::new(
        -2.0 / (PI * x) + 2.0 / PI * ln_half_x(x) * j1(x) - x / (2.0 * PI) * sum,
        terms,
    )
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
pub(super) fn taylor_step(x0: f64, at: Slope<f64>, delta: f64, order_sq: f64) -> Slope<f64> {
    let mut a = [0.0f64; MARCH_TERMS];
    a[0] = at.value;
    a[1] = at.derivative;
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
    Slope::new(value, derivative)
}

/// `(Y₀(x), Y₁(x))` for `x > Y_SERIES_MAX`, by marching out from the series.
///
/// `Y₀′ = −Y₁`, so the `n = 0` march carries both.
pub(super) fn y_march(target: f64) -> Slope<f64> {
    y_march_steps(target).value
}

/// Cap on the outward march, in steps.
///
/// **Every other loop in this crate has a backstop and this one did not.** The march advances by
/// `x·MARCH_RATIO` from `Y_SERIES_MAX = 4` to at most `HANKEL_MIN = 18`, so a factor of `1.25`
/// per step reaches the far end in `log(18/4)/log(1.25) ≈ 6.7` — **seven** measured, the last
/// step being short because it is clamped to `target − x`.
///
/// Sixteen is a little over twice that, which is where this crate's other backstops sit:
/// `W_STEPS` is 2.6x its measured worst and the incomplete gamma's `MAX_ITER` 3.1x. The first
/// draft used sixty-four, and mutation testing showed why that is too loose — a mutated advance
/// then spends all sixty-four steps on a forty-term recurrence and merely becomes *slow*, which
/// surfaces as a timeout rather than a failed assertion. A cap close to the real requirement
/// turns the same mutation into a wrong answer the accuracy tests catch at once.
///
/// Mutation testing is what asked for it. Three mutants here — `x < target` widened, and `x +=
/// step` turned into `-=` or `*=` — stopped the march advancing and the suite **hung** instead of
/// failing. A hang is the worst outcome available: it is detected only by a timeout, it says
/// nothing about which mutation caused it, and in a library it would be an unkillable process
/// rather than a wrong number. With the cap those three produce a bounded wrong answer, which the
/// accuracy tests catch immediately.
pub(super) const MARCH_MAX_STEPS: u32 = 16;

/// The march, with the number of steps it took.
///
/// The count is the observable that `MARCH_MAX_STEPS` needs: a cap is only a backstop if
/// something asserts nothing reaches it, and the step count is the only way to see that.
pub(super) fn y_march_steps(target: f64) -> Converged<Slope<f64>> {
    let mut x = Y_SERIES_MAX;
    // `Y₀` and its derivative `Y₀′ = −Y₁` are what the equation is marched in; the caller wants
    // `Y₁`, so the sign comes back off at the end.
    let mut at = Slope::new(y0_series(x), -y1_series(x));
    let mut steps = 0;
    while x < target && steps < MARCH_MAX_STEPS {
        let step = (x * MARCH_RATIO).min(target - x);
        at = taylor_step(x, at, step, 0.0);
        x += step;
        steps += 1;
    }
    Converged::new(Slope::new(at.value, -at.derivative), steps)
}

/// The Bessel function of the second kind `Y₀(x)`, for `x > 0`.
/// Which route evaluates `Y_ν` at this argument.
///
/// One predicate for both orders, as with `K`: they differ in kind at the origin — `Y₀` has a
/// logarithmic singularity and `Y₁` a simple pole — but both diverge to `−∞`, so that is one arm.
/// The single-precision peer carries the same enum.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum YRoute {
    /// A value the function takes exactly: `NaN` off the domain, the singularity at zero, and the
    /// limit at infinity.
    Value(f64),
    /// `x ≤ `[`Y_SERIES_MAX`]: the power series.
    Series,
    /// `x ≥ `[`HANKEL_MIN`]: the Hankel expansion, shared with `J`.
    Hankel,
    /// The middle gap, by marching Bessel's equation outward from the series.
    March,
}

impl YRoute {
    pub(super) fn for_argument(x: f64) -> Self {
        if x.is_nan() || x < 0.0 {
            Self::Value(f64::NAN) // the real Y is defined for positive argument
        } else if x == 0.0 {
            Self::Value(f64::NEG_INFINITY)
        } else if x.is_infinite() {
            Self::Value(0.0)
        } else if x <= Y_SERIES_MAX {
            Self::Series
        } else if x >= HANKEL_MIN {
            Self::Hankel
        } else {
            Self::March
        }
    }
}

pub(crate) fn y0(x: f64) -> f64 {
    match YRoute::for_argument(x) {
        YRoute::Value(value) => value,
        YRoute::Series => y0_series(x),
        YRoute::Hankel => hankel(0.0, x).second_kind,
        YRoute::March => y_march(x).value,
    }
}

pub(crate) fn y1(x: f64) -> f64 {
    match YRoute::for_argument(x) {
        YRoute::Value(value) => value,
        YRoute::Series => y1_series(x),
        YRoute::Hankel => hankel(1.0, x).second_kind,
        // `y_march` carries `Y₀` and `Y₁` as a value and its derivative, so this is the same
        // march the `y0` arm runs — the field says which half is wanted.
        YRoute::March => y_march(x).derivative,
    }
}
