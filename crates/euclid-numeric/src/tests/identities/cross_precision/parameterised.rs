//! The separately-implemented f32 kernels, against measured ulp budgets.
//!
//! See the [parent module](super) for the split and the measurement convention.
//!
//! # Why these needed a pass of their own
//!
//! [`super::delegating`] reaches a kernel through a one-argument `Real` method, which is all its
//! `agrees_to!` macro can express. Most of the f32 survivor population is not shaped like that:
//! `gamma_lr(x, a)`, `beta_reg(x, a, b)`, `gamma_lr_inv(p, a)` and `polygamma(x, order)` take a
//! shape or an order alongside the argument, and the f32 tree implements each of them
//! separately. `kernels/f32/quantile.rs` alone held 118 surviving mutants — the largest cluster
//! in the crate — and nothing reached it, because the quantile solvers are parameterised and the
//! existing pass was not.
//!
//! The parameters below are therefore part of the measurement, not decoration. Each kernel is
//! swept across its argument at several shapes, because the shape selects the branch: a series
//! below the crossover, a continued fraction above it, an asymptotic form past the threshold.
//! A single shape exercises one of those and leaves the rest as unmeasured as before.

use crate::Real;

/// Arguments swept per kernel per parameter set. Fixed, because the budgets below are the
/// maximum gap observed over *this* sweep — changing it changes which arguments are visited and
/// therefore the measurement.
const SWEEP: usize = 400;

/// Check a parameterised kernel's f32 implementation against its f64 one across a domain.
///
/// The closures take the swept argument only; the caller closes over the parameters. That keeps
/// the budget attached to a specific `(kernel, parameters)` pair, which is the granularity the
/// branches demand.
macro_rules! agrees_to {
    ($name:expr, $lo:expr, $hi:expr, $ulps:expr, $wide:expr, $narrow:expr) => {{
        for i in 0..=SWEEP {
            #[allow(clippy::cast_precision_loss)]
            let x = $lo + ($hi - $lo) * (i as f64) / (SWEEP as f64);
            let outcome =
                euclid_stability::relation::precision_agrees(&$name, x, $wide, $narrow, $ulps);
            assert!(
                outcome.is_ok(),
                "{}",
                outcome.err().map(|f| f.message).unwrap_or_default()
            );
        }
    }};
}

/// The regularised incomplete gamma, both tails, across the shapes that select its branches.
///
/// `gamma_ur` is consistently the looser of the two, by roughly a factor of four. That is the
/// tail subtraction: for small `x` the upper tail is `1 − gamma_lr`, so it inherits
/// `gamma_lr`'s absolute error against a result that is itself close to one, and relative
/// accuracy is what suffers.
#[test]
fn the_incomplete_gamma_tails_agree_across_precisions() {
    // (shape, lower-tail budget, upper-tail budget), measured.
    for &(a, lower, upper) in &[
        (0.5_f64, 9_u32, 72_u32),
        (1.0, 11, 43),
        (2.5, 10, 38),
        (10.0, 68, 51),
    ] {
        agrees_to!(
            format!("gamma_lr(a={a})"),
            0.05,
            30.0,
            lower,
            move |x: f64| Real::gamma_lr(x, a),
            #[allow(clippy::cast_possible_truncation)]
            move |x: f32| Real::gamma_lr(x, a as f32)
        );
        agrees_to!(
            format!("gamma_ur(a={a})"),
            0.05,
            30.0,
            upper,
            move |x: f64| Real::gamma_ur(x, a),
            #[allow(clippy::cast_possible_truncation)]
            move |x: f32| Real::gamma_ur(x, a as f32)
        );
    }
}

/// The regularised incomplete beta is **bit-exact** at every shape pair measured.
///
/// A budget of zero across four parameter pairs and four hundred arguments each is not luck: it
/// says the f32 path computes in f64 and rounds once. Pinning zero keeps that true and would
/// catch someone converting it into a genuine f32 path — the change that would move this kernel
/// into a budget rather than merely widen one.
#[test]
fn the_incomplete_beta_is_bit_exact_across_precisions() {
    for &(a, b) in &[(0.5_f64, 0.5_f64), (2.0, 3.0), (5.0, 1.0), (10.0, 10.0)] {
        agrees_to!(
            format!("beta_reg(a={a}, b={b})"),
            0.001,
            0.999,
            0,
            move |x: f64| Real::beta_reg(x, a, b),
            #[allow(clippy::cast_possible_truncation)]
            move |x: f32| Real::beta_reg(x, a as f32, b as f32)
        );
    }
}

/// The two quantile inverses — the iterative solvers, and the largest survivor cluster.
///
/// Both are Newton or bisection loops over a CDF, so their agreement is bounded by the CDF's
/// and then by how the iteration terminates. `beta_reg_inv` is the looser of the two despite
/// `beta_reg` itself being bit-exact, which locates the disagreement in the solver rather than
/// in the function it inverts — the two stop at different points.
#[test]
fn the_quantile_inverses_agree_across_precisions() {
    for &(a, budget) in &[(0.5_f64, 37_u32), (1.0, 30), (2.5, 18), (10.0, 15)] {
        agrees_to!(
            format!("gamma_lr_inv(a={a})"),
            0.001,
            0.999,
            budget,
            move |p: f64| Real::gamma_lr_inv(p, a),
            #[allow(clippy::cast_possible_truncation)]
            move |p: f32| Real::gamma_lr_inv(p, a as f32)
        );
    }
    for &(a, b, budget) in &[
        (0.5_f64, 0.5_f64, 4_u32),
        (2.0, 3.0, 50),
        (5.0, 1.0, 7),
        (10.0, 10.0, 15),
    ] {
        agrees_to!(
            format!("beta_reg_inv(a={a}, b={b})"),
            0.001,
            0.999,
            budget,
            move |p: f64| Real::beta_reg_inv(p, a, b),
            #[allow(clippy::cast_possible_truncation)]
            move |p: f32| Real::beta_reg_inv(p, a as f32, b as f32)
        );
    }
}

/// `polygamma` at each order, which is what selects its branch.
///
/// The budget grows with the order — 2, 3, 18, 24 — because each derivative adds a factor of the
/// recurrence's cancellation, and the f32 Hurwitz-zeta tail has fewer terms to spend on it.
#[test]
fn polygamma_agrees_across_precisions_at_every_order() {
    for &(order, budget) in &[(1_u32, 2_u32), (2, 3), (3, 18), (4, 24)] {
        agrees_to!(
            format!("polygamma(order={order})"),
            0.5,
            15.0,
            budget,
            move |x: f64| Real::polygamma(x, order),
            move |x: f32| Real::polygamma(x, order)
        );
    }
}

/// The separately-implemented single-argument kernels in `kernels/f32/gamma.rs`,
/// `kernels/f32/polygamma.rs` and `kernels/erf/f32.rs`.
///
/// `exprel` at 1 ulp is effectively exact. The rest are genuine f32 polynomials and sit where
/// the measurement puts them.
#[test]
fn the_separately_implemented_f32_kernels_agree_across_precisions() {
    agrees_to!("gamma", 0.1, 20.0, 37, Real::gamma, Real::gamma);
    agrees_to!("rgamma", -5.0, 10.0, 103, Real::rgamma, Real::rgamma);
    agrees_to!("exprel", -5.0, 5.0, 1, Real::exprel, Real::exprel);
    agrees_to!("erfcx", -5.0, 25.0, 47, Real::erfcx, Real::erfcx);
    agrees_to!("erf_inv", -0.99, 0.99, 15, Real::erf_inv, Real::erf_inv);
    agrees_to!("erf", -4.0, 4.0, 8, Real::erf, Real::erf);
    agrees_to!("erfc", -4.0, 10.0, 71, Real::erfc, Real::erfc);
}

/// Two kernels whose large budgets are a property of the *metric*, not of the code.
///
/// Both have the structure MUTANTS.md records for `ln_gamma` at 1,514 ulp: a zero or a pole
/// inside the swept range, where ulp spacing collapses towards zero while the function's
/// derivative does not. `euclid_stability::numeric::seam_ratio` documents the same effect and
/// divides by an expected step to remove it.
///
/// - `gamma` on the negative axis crosses a pole between every pair of consecutive negative
///   integers, and `−1.996` is about as close to one as this sweep lands.
/// - `ln_rising_factorial(a, 0.5)` passes through zero near `a = 1.23`.
///
/// They are pinned anyway, because a budget that is loose for a known reason still catches a
/// kernel that breaks for an unknown one. Tightening either would be a claim about the metric.
#[test]
fn the_kernels_with_a_zero_in_range_agree_to_their_measured_seam() {
    agrees_to!("gamma/negative", -4.9, -0.1, 387, Real::gamma, Real::gamma);
    for &(n, budget) in &[(0.5_f64, 301_u32), (2.0, 15), (7.0, 2)] {
        agrees_to!(
            format!("ln_rising_factorial(n={n})"),
            0.5,
            20.0,
            budget,
            move |a: f64| Real::ln_rising_factorial(a, n),
            #[allow(clippy::cast_possible_truncation)]
            move |a: f32| Real::ln_rising_factorial(a, n as f32)
        );
    }
}
