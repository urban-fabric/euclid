//! The modified Bessel functions of the **second kind**, `K₀` and `K₁`, with their
//! exponentially-scaled forms `Kₙᵉ(x) = e^{x}Kₙ(x)`.
//!
//! # Method: the trapezoidal rule on `Kν(x) = ∫₀^∞ e^{−x cosh t} cosh(νt) dt`
//!
//! `K` is the awkward member of the family. Its power series (below) carries a `ln(x/2)`
//! factor and is exact only for small argument; its asymptotic expansion needs `x ≳ 20` before
//! it reaches machine precision, because an optimally-truncated asymptotic series for `K`
//! errs by about `e^{−2x}`. That leaves a wide middle range — exactly the range most callers
//! use — where neither is good enough.
//!
//! The integral representation closes it, and does so *uniformly*. Three properties make the
//! trapezoidal rule the right quadrature for it:
//!
//! - **The integrand is even in `t`**, so `∫₀^∞ = ½∫_{−∞}^{∞}` and the rule inherits the
//!   exponential convergence the trapezoidal rule enjoys on the whole line — error `≈ e^{−2πd/h}`
//!   for an integrand analytic in the strip `|Im t| < d` — rather than the `O(h²)` an endpoint
//!   would impose.
//! - **The integrand is positive**, so there is no cancellation at any argument. The result is
//!   a sum of positive terms, which is the opposite of the `Si`/`Ci` situation.
//! - **Both the width and the decay scale with `x` in the same way**, so the node count stays
//!   bounded (~40) rather than growing. For small `x` the integrand is a broad plateau out to
//!   `t ≈ ln(2/x)` with double-exponential tails, and `d → π/2` bounds the step. For large `x`
//!   it is a narrow Gaussian of width `1/√x`, which instead bounds the step by `≈0.66/√x`;
//!   taking the smaller of the two is correct in both regimes.
//!
//! Writing it for the **scaled** function `Kνᵉ(x) = e^{x}Kν(x) = ∫₀^∞ e^{−x(cosh t − 1)}cosh(νt)dt`
//! makes the scaled forms primitive and overflow-free, with the unscaled ones recovered by a
//! single multiplication that is *allowed* to underflow — as the true `K₀(746)` does.
//!
//! `ν` is a parameter rather than fixed at 0 and 1 because the same quadrature is the
//! machine-precision route to the Airy functions on the positive axis, through
//! `Ai(x) = π^{−1}√(x/3)·K_{1/3}(⅔x^{3/2})`.

use super::ln_half_x;
use super::modified_first::{i0, i1};
use crate::constants::f64::EULER_GAMMA;

/// Iteration cap for the power series here.
const MAX_TERMS: u32 = 1_000;

/// Relative size at which a term stops contributing.
///
/// **`1e-19`, which is deliberately only a little below `f64`'s resolution.** `ε = 2.22e-16`, so
/// a term at `1e-19` of the sum moves the result by about a thousandth of an ulp, and every term
/// after it moves it by less. The value was `1e-20`, which asked the `K` power series to converge
/// one order past the point where the format can represent the difference —
/// iterations bought and then discarded.
///
/// Raising it to `1e-19` is **free rather than a trade.** The outputs are bit-identical across a
/// 306,280-point sweep covering every kernel in this tree at both signs, while leaving `K₀` and `K₁` unchanged to within noise — their cost is the quadrature, not the series.5%**, because the quadrature dominates their cost, not the series.
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

/// `pub(super)` rather than private: the unit tests live in a sibling `tests` module because
/// implementation plus tests exceeded this crate's 400-line file budget, and they compare the
/// two routes against each other directly.
///
/// At or below this argument `K₀`/`K₁` use their power series; above it, the quadrature.
///
/// The series is exact for small `x` and the quadrature's node count grows like `ln(1/x)` as
/// `x → 0`, so the series takes the small end. At `x = 2` the series has lost about one digit
/// to the `ln(x/2)·I₀` subtraction and the quadrature is already at machine precision.
pub(super) const K_SERIES_MAX: f64 = 2.0;

/// Largest trapezoid step, set by the strip half-width `d → π/2` that bounds the small-`x`
/// regime: the error is `≈ e^{−2πd/h} = e^{−π²/h}`, and `e^{−π²/0.2} ≈ 4e-22`.
const K_STEP_MAX: f64 = 0.2;

/// Step coefficient for the large-`x` regime, where the integrand is a Gaussian of width
/// `σ = 1/√x` and the trapezoid error is `≈ e^{−2π²σ²/h²}`. That needs `h < 0.66σ`; `0.5`
/// leaves margin.
///
/// # The margin is not slack, and has been measured
///
/// `e^{−2π²/c²}` is `5e-35` at `c = 0.5` and `2e-20` at the nominal limit `c = 0.66`, so the
/// bound looks like eighteen orders of headroom that a wider step could spend — and `K₀`/`K₁`
/// are among the slowest kernels here, so it is the obvious place to look. **It does not work.**
/// Taking `c = 0.66` with [`K_STEP_MAX`] at `0.22` to match moved 12,814 values of a
/// 306,280-point sweep, by up to **15 ulp on `K₁`** and 7 on `K₀`, while `K₁` got only 5% faster
/// and `K₀` got *slower*.
///
/// The bound is asymptotic in the node count: it describes how the discretization error decays
/// once the strip is well resolved, not its size at the few dozen nodes a moderate argument
/// actually uses. So the quantity `c` is really trading against is not the formula above, and
/// the formula's headroom is not available. Node count is also not where these two kernels spend
/// their time — see the [`TERM_EPS`] measurement, which cut the series' iterations and moved
/// `K₀`/`K₁` by under half a percent.
const K_STEP_COEFF: f64 = 0.5;

/// Truncation level: integrate while `x(cosh t − 1) < 45`, so the discarded tail is below
/// `e^{−45} ≈ 3e-20` relative to an `O(1)` result.
const K_DECAY: f64 = 45.0;

/// `Kνᵉ(x) = e^{x}Kν(x)` for `x > 0` and any `ν ≥ 0`, by the trapezoidal rule.
///
/// # Cost at small argument
///
/// Accurate at every argument — measured `3e-16` relative down to `x = 1e-300` — but *slow*
/// there: the integrand is a plateau out to `t ≈ ln(2/x)`, so the node count grows like
/// `ln(1/x)` and reaches ~6300 at `x = 1e-300`, about 390× the ~40 nodes a moderate argument
/// needs. Callers on this crate's surface never see that, because `k0`/`k1` take the power
/// series below [`K_SERIES_MAX`]; it is worth knowing before reaching for `ν` outside
/// `{0, 1}` at a tiny argument.
///
/// `ν` above 1 works and is exact — the half-integer closed forms `K_{1/2}`, `K_{3/2}` and
/// `K_{5/2}` are checked against it in the relationship suite — since `cosh(νt)` is always
/// dominated by `e^{−x cosh t}` at the truncation point.
///
/// See the discussion above. `cosh t − 1` is formed as `2 sinh²(t/2)`: the direct subtraction
/// cancels for small `t`, and small `t` is precisely the regime the large-`x` branch lives in,
/// where that error would then be amplified by the factor `x` in the exponent.
pub(crate) fn k_nu_scaled(nu: f64, x: f64) -> f64 {
    debug_assert!(x > 0.0 && x.is_finite());
    let root_x = x.sqrt();
    let step = k_quadrature_step(root_x);
    let nodes = (k_truncation_time(root_x) / step).ceil() as usize;

    // The t = 0 node has weight one-half and integrand one.
    let mut total = 0.5;
    for node in 1..=nodes {
        total += crate::math::bessel::k_scaled_integrand(nu, root_x, step * node as f64);
    }
    step * total
}

/// `(Kνᵉ(x), Kμᵉ(x))` for two orders at one argument, over a single sweep of nodes.
///
/// # Why this exists
///
/// Everything expensive about the quadrature depends on `x` alone. [`k_quadrature_step`] and
/// [`k_truncation_time`] fix the same nodes whichever order is asked for, and at each node the
/// integrand is `e^{−2(√x sinh(t/2))²}·cosh(νt)` — in which only the final `cosh` carries the
/// order. Two separate calls therefore evaluate `sinh` and `exp` twice over the identical node
/// list and throw one copy of each away.
///
/// Airy on the positive axis needs exactly such a pair, `K_{1/3}` and `K_{2/3}` at the same
/// argument, and was paying for two full sweeps — see [`super::super::airy`]. Per node this
/// goes from two `sinh`, two `exp` and two `cosh` to one, one and two, so the shared factor is
/// the `exp`, which is the dearest of the three.
///
/// # It is bit-identical, not merely close
///
/// Each term is `envelope * (nu * t).cosh()` with `envelope` formed by the same operations in
/// the same order as [`crate::math::bessel::k_scaled_integrand`], and the two accumulators are summed over the nodes
/// in the same order as before. So this returns exactly what two calls to [`k_nu_scaled`]
/// return, bit for bit, and is a cost change only — which is what makes it safe to apply to a
/// kernel whose error budget is already measured.
pub(crate) fn k_nu_pair_scaled(nu: f64, mu: f64, x: f64) -> (f64, f64) {
    debug_assert!(x > 0.0 && x.is_finite());
    let root_x = x.sqrt();
    let step = k_quadrature_step(root_x);
    let nodes = (k_truncation_time(root_x) / step).ceil() as usize;

    let mut total_nu = 0.5;
    let mut total_mu = 0.5;
    for node in 1..=nodes {
        let t = step * node as f64;
        // Apply √x before squaring so cosh(t) can leave the format while the decay stays finite.
        let scaled_sinh = root_x * (t * 0.5).sinh();
        let decay = 2.0 * scaled_sinh * scaled_sinh;
        let envelope = (-decay).exp();
        total_nu += envelope * (nu * t).cosh();
        total_mu += envelope * (mu * t).cosh();
    }
    (step * total_nu, step * total_mu)
}

fn k_quadrature_step(root_x: f64) -> f64 {
    K_STEP_MAX.min(K_STEP_COEFF / root_x)
}

fn k_truncation_time(root_x: f64) -> f64 {
    // Solve `cosh T − 1 = K_DECAY/x` for the truncation point. Spelling that as
    // `acosh(1 + K_DECAY/x)` would be a cancellation bug rather than a mere loss of digits:
    // once `x > K_DECAY/ε ≈ 2e17` the sum `1 + K_DECAY/x` rounds to exactly 1, `acosh` returns
    // 0, and the quadrature silently integrates nothing. Using the half-angle form
    // `cosh T − 1 = 2 sinh²(T/2)` instead gives `T = 2 asinh(√(K_DECAY/2x))`, which is exact
    // across the whole range.
    // Divide after taking the roots so the radicand does not overflow at subnormal x.
    2.0 * ((0.5 * K_DECAY).sqrt() / root_x).asinh()
}

/// `K₀(x)` by its power series, for small `x`.
///
/// `K₀(x) = −(ln(x/2) + γ)I₀(x) + Σ_{k≥1} (x²/4)^k/(k!)²·H_k`, with `H_k` the `k`th harmonic
/// number. Every term of the sum is positive; the only subtraction is the leading one, which
/// costs about a digit by `x = 2` and nothing below it.
pub(super) fn k0_series(x: f64) -> f64 {
    let q = x * x * 0.25;
    let mut term = 1.0f64; // (x²/4)^k/(k!)²
    let mut harmonic = 0.0f64;
    let mut sum = 0.0f64;
    for k in 1..MAX_TERMS {
        let kf = f64::from(k);
        term *= q / (kf * kf);
        harmonic += 1.0 / kf;
        let t = term * harmonic;
        sum += t;
        if t < sum * TERM_EPS {
            break;
        }
    }
    // `ln(x/2)` is formed as `ln x − ln 2` rather than `(x*0.5).ln()`. The product form
    // underflows for the smallest subnormals — `x*0.5` rounds to zero at `x = 5e-324`, making
    // `ln` return `−∞` where the true value is a perfectly representable `−474` — and the
    // difference is marginally more accurate everywhere else too.
    -(ln_half_x(x) + EULER_GAMMA) * i0(x) + sum
}

/// `K₁(x)` by its power series, for small `x`.
///
/// `K₁(x) = 1/x + ln(x/2)·I₁(x) − (x/4)Σ_{k≥0}(H_k + H_{k+1} − 2γ)(x²/4)^k/(k!(k+1)!)`.
/// The `1/x` carries the pole, so it is kept as a separate exact term rather than folded in.
pub(super) fn k1_series(x: f64) -> f64 {
    let q = x * x * 0.25;
    let mut term = 1.0f64; // (x²/4)^k/(k!(k+1)!)
    let mut h_k = 0.0f64; // H_k
    let mut sum = 0.0f64;
    for k in 0..MAX_TERMS {
        let kf = f64::from(k);
        if k > 0 {
            term *= q / (kf * (kf + 1.0));
            h_k += 1.0 / kf;
        }
        let h_k1 = h_k + 1.0 / (kf + 1.0);
        let t = term * (h_k + h_k1 - 2.0 * EULER_GAMMA);
        sum += t;
        if t.abs() < sum.abs() * TERM_EPS && k > 1 {
            break;
        }
    }
    1.0 / x + ln_half_x(x) * i1(x) - x * 0.25 * sum
}

/// The modified Bessel function of the second kind `K₀(x)`, for `x > 0`.
///
/// Underflows to `0` for `x ≳ 746`, as the true value does; use [`k0e`] there.
/// Which route evaluates `K_ν` at this argument.
///
/// One predicate serves all four entry points — `K₀`, `K₁` and their scaled peers — because they
/// differ only in what each arm returns, never in which arm applies. They differ in *kind* at the
/// origin, `K₀` having a logarithmic singularity and `K₁` a simple pole, but both diverge to `+∞`,
/// so even that is one arm. The single-precision peer carries the same enum.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum KRoute {
    /// A value the function takes exactly: `NaN` off the domain, the singularity at zero, and the
    /// limit at infinity — where the scaled forms decay like `√(π/2x)` and the unscaled ones
    /// faster still.
    Value(f64),
    /// `x ≤ `[`K_SERIES_MAX`]: the power series, exact for small `x` and the cheaper route.
    Series,
    /// Above it, the trapezoidal quadrature — whose node count would *grow* like `ln(1/x)`
    /// towards zero, and is smallest exactly where the series stops being usable.
    Quadrature,
}

impl KRoute {
    pub(super) fn for_argument(x: f64) -> Self {
        if x.is_nan() || x < 0.0 {
            Self::Value(f64::NAN) // the real K is defined for positive argument
        } else if x == 0.0 {
            Self::Value(f64::INFINITY)
        } else if x.is_infinite() {
            Self::Value(0.0)
        } else if x <= K_SERIES_MAX {
            Self::Series
        } else {
            Self::Quadrature
        }
    }
}

pub(crate) fn k0(x: f64) -> f64 {
    match KRoute::for_argument(x) {
        KRoute::Value(value) => value,
        KRoute::Series => k0_series(x),
        KRoute::Quadrature => k_nu_scaled(0.0, x) * (-x).exp(),
    }
}

pub(crate) fn k1(x: f64) -> f64 {
    match KRoute::for_argument(x) {
        KRoute::Value(value) => value,
        KRoute::Series => k1_series(x),
        KRoute::Quadrature => k_nu_scaled(1.0, x) * (-x).exp(),
    }
}

pub(crate) fn k0e(x: f64) -> f64 {
    match KRoute::for_argument(x) {
        KRoute::Value(value) => value,
        KRoute::Series => k0_series(x) * x.exp(),
        KRoute::Quadrature => k_nu_scaled(0.0, x),
    }
}

pub(crate) fn k1e(x: f64) -> f64 {
    match KRoute::for_argument(x) {
        KRoute::Value(value) => value,
        KRoute::Series => k1_series(x) * x.exp(),
        KRoute::Quadrature => k_nu_scaled(1.0, x),
    }
}

#[cfg(test)]
mod quadrature_tests {
    use super::*;

    #[test]
    fn node_spacing_is_capped_then_scales_with_the_inverse_root() {
        for (root_x, expected) in [(0.25, 0.2), (2.0, 0.2), (4.0, 0.125), (8.0, 0.0625)] {
            assert_eq!(k_quadrature_step(root_x), expected);
        }
    }

    #[test]
    fn cutoff_agrees_with_the_unscaled_hyperbolic_equation() {
        for x in [0.25_f64, 1.0, 4.0, 20.0] {
            let actual = k_truncation_time(x.sqrt());
            let expected = (1.0 + 45.0 / x).acosh();
            assert!((actual - expected).abs() < 2e-15 * expected);
        }
    }

    #[test]
    fn integrand_matches_a_node_with_exact_hyperbolic_values() {
        // cosh(ln 2) = (2 + 1/2)/2. At x = 4 the decay is therefore 1.
        let actual = crate::math::bessel::k_scaled_integrand(1.0, 2.0, core::f64::consts::LN_2);
        assert!((actual - 1.25 * (-1.0_f64).exp()).abs() < 1e-15);
    }

    #[test]
    fn half_order_agrees_with_its_closed_form_at_subnormal_arguments() {
        for x in [1e-308_f64, 1e-310, f64::from_bits(1)] {
            let expected = (core::f64::consts::PI * 0.5).sqrt() / x.sqrt();
            let actual = k_nu_scaled(0.5, x);
            assert!(
                (actual / expected - 1.0).abs() < 3e-13,
                "x={x:e}: {actual:e}"
            );
        }
    }
}
