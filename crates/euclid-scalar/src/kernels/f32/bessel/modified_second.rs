//! The `f32` peer of [`crate::kernels::bessel::modified_second`]: `K₀` and `K₁` with their
//! scaled forms.
//!
//! Native single precision. This module used to narrow the `f64` reference, on the grounds that
//! its "trapezoidal rule whose step size is chosen from the strip width of the integrand's
//! analytic continuation" would in `f32` "have to be retuned to a `1e-7` target, giving a result
//! no better than narrowing while doubling what has to be verified."
//!
//! The first half of that is exactly right, and it is the reason this port is worth making. All
//! three quadrature constants are fixed by setting an analytic error bound equal to a target
//! accuracy, so retuning them is arithmetic rather than fitting — and every one of them relaxes:
//!
//! | constant | bound | `f64` | `f32` |
//! |---|---|---|---|
//! | [`K_STEP_MAX`] | `e^{−π²/h}` | `0.2` → `4e-22` | `0.5` → `3e-9` |
//! | [`K_STEP_COEFF`] | `e^{−2π²/c²}` | `0.5` → `5e-35` | `1.0` → `2.7e-9` |
//! | [`K_DECAY`] | `e^{−D}` | `45` → `3e-20` | `20` → `2e-9` |
//!
//! The second half — "no better than narrowing" — is true and beside the point. Narrowing is
//! always at least as accurate as a native path. The reason to choose `f32` is that it is
//! cheaper, and a narrowed kernel throws that away by doing the `f64` work plus two conversions.
//! A quadrature with a third of the nodes is the benefit the format exists to give. Measured
//! node counts bear that out: 16–19 with the `f64` constants against 5–7 with these, a ratio of
//! 2.6–3.2 across `x ∈ [2.5, 1e4]`.
//!
//! One constant moved the other way, and it mattered more than any of the three above:
//! [`K_SERIES_MAX`] had to come *down*, because the series' cancellation is a seventh of this
//! format's budget rather than a tenth of `f64`'s.

use super::modified_first::{i0, i1};
use crate::constants::f32::EULER_GAMMA;

/// Iteration cap for the power series. The `f64` peer allows 1,000.
///
/// The series runs only over `x ≤ `[`K_SERIES_MAX`], where the term ratio is `x²/(4k²) ≤ 1/k²`,
/// so it converges superexponentially. 32 is a backstop, not a schedule.
const MAX_TERMS: u32 = 32;

/// Relative size at which a series term is taken to be negligible.
const TERM_EPS: f32 = 1e-8;

/// At or below this argument `K₀`/`K₁` use their power series; above it, the quadrature.
///
/// **`1.5`, where the `f64` peer uses `2.0`**, and this is the one constant here that had to
/// come *down* rather than up.
///
/// The series' only subtraction is the leading `−(ln(x/2) + γ)·I₀(x)`, which the `f64` peer
/// describes as costing "about one digit" by `x = 2`. One digit is a tenth of `f64`'s budget and
/// a seventh of this one, and that difference is not a margin — it is the whole threshold.
/// Measured against the `f64` kernel, in `f32` ulps of `K₀`:
///
/// | `x` | 0.5 | 1.0 | **1.5** | 1.75 | 2.0 | 2.5 | 3.0 |
/// |---|---|---|---|---|---|---|---|
/// | series | `0` | `0` | **`2`** | `15` | `15` | `166` | `131` |
/// | quadrature | `2` | `1` | **`0`** | `1` | `0` | `2` | `1` |
///
/// Inheriting `2.0` would have shipped a `K₀` carrying 15 ulp at the join where the other arm
/// carries none, and the series is past 100 ulp only half a unit further on. `1.5` is where the
/// two cross.
///
/// The join cannot simply be pushed to zero, though the table above might suggest it: the
/// quadrature's node count *grows* like `ln(1/x)` towards the origin, so the series still earns
/// the small end on cost even where both are accurate.
const K_SERIES_MAX: f32 = 1.5;

/// Largest trapezoid step, set by the strip half-width `d → π/2` that bounds the small-`x`
/// regime: the error is `≈ e^{−2πd/h} = e^{−π²/h}`.
///
/// `f64` takes `0.2`, giving `e^{−π²/0.2} ≈ 4e-22`. This format needs the bound under half an
/// ulp, `6e-8`, which wants `h < π²/ln(2e7) ≈ 0.59`; `0.5` leaves margin and gives `3e-9`. That
/// is two and a half times the `f64` step, hence two and a half times fewer nodes across the
/// whole small-`x` regime.
const K_STEP_MAX: f32 = 0.5;

/// Step coefficient for the large-`x` regime, where the integrand is a Gaussian of width
/// `σ = 1/√x` and the trapezoid error is `≈ e^{−2π²σ²/h²}`.
///
/// With `h = c·σ` that bound is `e^{−2π²/c²}`, independent of `x`. `f64` takes `c = 0.5` for
/// `5e-35`; `f32` needs `2π²/c² > ln(2e7) ≈ 16.8`, so `c < 1.08`. `1.0` gives `2.7e-9` and
/// halves the node count against `f64`'s `0.5`.
const K_STEP_COEFF: f32 = 1.0;

/// Truncation level: integrate while `x(cosh t − 1) < K_DECAY`, so the discarded tail is below
/// `e^{−K_DECAY}` relative to an `O(1)` result.
///
/// `f64` takes `45` for `3e-20`. `f32` needs only `e^{−D} < 6e-8`, so `D > 16.6`; `20` gives
/// `2e-9`. The truncation time is `T = 2·asinh(√(D/2x))`, which for a large argument is
/// `≈ 2√(D/2x)` — so this shrinks the integration interval by `√(20/45) ≈ 0.67` on top of the
/// coarser step.
const K_DECAY: f32 = 20.0;

/// `ln(x/2)`, formed as `ln x − ln 2`.
///
/// The product form `(x*0.5).ln()` underflows for the smallest subnormals — in `f32`, `x*0.5`
/// rounds to zero at `x = 1.4e-45`, making `ln` return `−∞` where the true value is a perfectly
/// representable `−104` — and the difference is marginally more accurate everywhere else too.
fn ln_half_x(x: f32) -> f32 {
    x.ln() - core::f32::consts::LN_2
}

/// `Kνᵉ(x) = e^{x}Kν(x)` for `x > 0` and any `ν ≥ 0`, by the trapezoidal rule.
///
/// `pub(super)` for the Airy family, which reaches `Ai` through `K_{1/3}` and `K_{2/3}`. The
/// quadrature is naturally scaled — it *is* `e^{x}Kν(x)` — so an Airy caller forms no
/// exponential at all and the result stays finite however large the argument becomes.
///
/// `cosh t − 1` is formed as `2 sinh²(t/2)`: the direct subtraction cancels for small `t`, and
/// small `t` is precisely the regime the large-`x` branch lives in, where that error would then
/// be amplified by the factor `x` in the exponent.
pub(in crate::kernels::f32) fn k_nu_scaled(nu: f32, x: f32) -> f32 {
    debug_assert!(x > 0.0 && x.is_finite());
    let root_x = x.sqrt();
    let step = k_quadrature_step(root_x);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let nodes = (k_truncation_time(root_x) / step).ceil() as usize;

    // The t = 0 node has weight one-half and integrand one.
    let mut total = 0.5;
    for node in 1..=nodes {
        total += crate::math::bessel::k_scaled_integrand(nu, root_x, step * node as f32);
    }
    step * total
}

/// `(Kνᵉ(x), Kμᵉ(x))` for two orders at one argument, over a single sweep of nodes.
///
/// The `f32` peer of [`crate::kernels::bessel::modified_second::k_nu_pair_scaled`], and it
/// exists for the same reason: the nodes depend only on `x`, so Airy's `K_{1/3}`/`K_{2/3}` pair
/// was paying for two sweeps of the same `sinh` and `exp`. Bit-identical to two separate calls.
pub(in crate::kernels::f32) fn k_nu_pair_scaled(nu: f32, mu: f32, x: f32) -> (f32, f32) {
    debug_assert!(x > 0.0 && x.is_finite());
    let root_x = x.sqrt();
    let step = k_quadrature_step(root_x);
    let nodes = (k_truncation_time(root_x) / step).ceil() as usize;

    let mut total_nu = 0.5;
    let mut total_mu = 0.5;
    for node in 1..=nodes {
        let t = step * node as f32;
        let scaled_sinh = root_x * (t * 0.5).sinh();
        let decay = 2.0 * scaled_sinh * scaled_sinh;
        let envelope = (-decay).exp();
        total_nu += envelope * (nu * t).cosh();
        total_mu += envelope * (mu * t).cosh();
    }
    (step * total_nu, step * total_mu)
}

fn k_quadrature_step(root_x: f32) -> f32 {
    K_STEP_MAX.min(K_STEP_COEFF / root_x)
}

fn k_truncation_time(root_x: f32) -> f32 {
    // Solve `cosh T − 1 = K_DECAY/x`. Spelling that as `acosh(1 + K_DECAY/x)` would be a
    // cancellation bug rather than a mere loss of digits: once `x > K_DECAY/ε`, which in `f32`
    // is only `1.7e8`, the sum `1 + K_DECAY/x` rounds to exactly 1, `acosh` returns 0, and the
    // quadrature silently integrates nothing. The half-angle form `cosh T − 1 = 2 sinh²(T/2)`
    // gives `T = 2 asinh(√(K_DECAY/2x))`, exact across the whole range. Divide after taking the
    // roots so the radicand cannot overflow at a subnormal argument.
    2.0 * ((0.5 * K_DECAY).sqrt() / root_x).asinh()
}

/// `K₀(x)` by its power series, for small `x`.
///
/// `K₀(x) = −(ln(x/2) + γ)I₀(x) + Σ_{k≥1} (x²/4)^k/(k!)²·H_k`, with `H_k` the `k`th harmonic
/// number. Every term of the sum is positive; the only subtraction is the leading one.
fn k0_series(x: f32) -> f32 {
    k0_series_terms(x).0
}

/// `K₀` from its power series, with the number of terms it summed.
///
/// # Why the count is returned at all
///
/// The break is a convergence test, and the terms it stops after are below `TERM_EPS` of the
/// running sum — so narrowing it to `==`, which never fires and runs every call to
/// [`MAX_TERMS`], leaves the value unchanged and costs thirty-one iterations. The count is the
/// observable that separates them. `<=` against `<` differs only when a term lands *exactly* on
/// the tolerance and is an equivalent mutant; see the sibling test.
pub(super) fn k0_series_terms(x: f32) -> (f32, u32) {
    let q = x * x * 0.25;
    let mut term = 1.0f32; // (x²/4)^k/(k!)²
    let mut harmonic = 0.0f32;
    let mut sum = 0.0f32;
    let mut terms = 0;
    for k in 1..MAX_TERMS {
        let kf = k as f32;
        term *= q / (kf * kf);
        harmonic += 1.0 / kf;
        let t = term * harmonic;
        sum += t;
        terms += 1;
        if t < sum * TERM_EPS {
            break;
        }
    }
    (-(ln_half_x(x) + EULER_GAMMA) * i0(x) + sum, terms)
}

/// `K₁(x)` by its power series, for small `x`.
///
/// `K₁(x) = 1/x + ln(x/2)·I₁(x) − (x/4)Σ_{k≥0}(H_k + H_{k+1} − 2γ)(x²/4)^k/(k!(k+1)!)`.
/// The `1/x` carries the pole, so it is kept as a separate exact term rather than folded in.
fn k1_series(x: f32) -> f32 {
    k1_series_terms(x).0
}

/// `K₁` from its power series, with the number of terms it summed.
///
/// Instrumented for the reason [`k0_series_terms`] records: the break's `k > 1` guard admits
/// three mutants that each force the loop to [`MAX_TERMS`] without moving the value.
pub(super) fn k1_series_terms(x: f32) -> (f32, u32) {
    let q = x * x * 0.25;
    let mut term = 1.0f32; // (x²/4)^k/(k!(k+1)!)
    let mut h_k = 0.0f32; // H_k
    let mut sum = 0.0f32;
    let mut terms = 0;
    for k in 0..MAX_TERMS {
        let kf = k as f32;
        if k > 0 {
            term *= q / (kf * (kf + 1.0));
            h_k += 1.0 / kf;
        }
        let h_k1 = h_k + 1.0 / (kf + 1.0);
        let t = term * (h_k + h_k1 - 2.0 * EULER_GAMMA);
        sum += t;
        terms += 1;
        if t.abs() < sum.abs() * TERM_EPS && k > 1 {
            break;
        }
    }
    (1.0 / x + ln_half_x(x) * i1(x) - x * 0.25 * sum, terms)
}

/// Which route evaluates `K₀`/`K₁` at a given argument.
#[derive(Clone, Copy, Debug, PartialEq)]
enum KRoute {
    /// A value the function takes exactly: the singularity at zero and the limit at infinity.
    Value(f32),
    /// `x ≤ `[`K_SERIES_MAX`]: the power series, exact for small `x` and the cheaper route.
    Series,
    /// Above it, the trapezoidal quadrature — whose node count would *grow* like `ln(1/x)`
    /// towards zero, and is smallest exactly where the series stops being usable.
    Quadrature,
}

impl KRoute {
    /// One predicate serves both orders. They differ in kind at the origin — `K₀` has a
    /// logarithmic singularity and `K₁` a simple pole — but both diverge to `+∞`.
    fn for_argument(x: f32) -> Self {
        if x.is_nan() || x < 0.0 {
            Self::Value(f32::NAN) // the real K is defined for positive argument
        } else if x == 0.0 {
            Self::Value(f32::INFINITY)
        } else if x.is_infinite() {
            Self::Value(0.0)
        } else if x <= K_SERIES_MAX {
            Self::Series
        } else {
            Self::Quadrature
        }
    }
}

/// The modified Bessel function of the second kind `K₀(x)`, for `x > 0`.
///
/// Underflows to `0` for `x ≳ 92`, as the true value does; use [`k0e`] there. That is an order
/// of magnitude sooner than the `f64` peer's `746`, which is this format's range rather than
/// anything about the method.
pub(crate) fn k0(x: f32) -> f32 {
    match KRoute::for_argument(x) {
        KRoute::Value(value) => value,
        KRoute::Series => k0_series(x),
        KRoute::Quadrature => k_nu_scaled(0.0, x) * (-x).exp(),
    }
}

/// The modified Bessel function of the second kind `K₁(x)`, for `x > 0`.
pub(crate) fn k1(x: f32) -> f32 {
    match KRoute::for_argument(x) {
        KRoute::Value(value) => value,
        KRoute::Series => k1_series(x),
        KRoute::Quadrature => k_nu_scaled(1.0, x) * (-x).exp(),
    }
}

/// The exponentially-scaled `K₀ᵉ(x) = e^{x}K₀(x)`.
///
/// Finite across the whole positive axis, which is the point of it: `K₀` itself underflows from
/// about `x = 92` in this format.
pub(crate) fn k0e(x: f32) -> f32 {
    match KRoute::for_argument(x) {
        KRoute::Value(value) => value,
        KRoute::Series => k0_series(x) * x.exp(),
        KRoute::Quadrature => k_nu_scaled(0.0, x),
    }
}

/// The exponentially-scaled `K₁ᵉ(x) = e^{x}K₁(x)`.
pub(crate) fn k1e(x: f32) -> f32 {
    match KRoute::for_argument(x) {
        KRoute::Value(value) => value,
        KRoute::Series => k1_series(x) * x.exp(),
        KRoute::Quadrature => k_nu_scaled(1.0, x),
    }
}

#[cfg(test)]
mod tests;
