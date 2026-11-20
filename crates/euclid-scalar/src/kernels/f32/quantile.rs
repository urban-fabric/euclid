//! Single-precision peers of [`super::super::quantile`], with a shared `f32`
//! Newton/bisection loop and format-specific stopping rule.

use super::gamma::ln_gamma;
use super::normal::ndtri;
use crate::math::density::{beta_pdf, gamma_pdf, log_beta_normaliser};
use crate::math::roots::{bracket_is_narrow, bracketed_newton_step, wilson_hilferty_guess};

/// Iteration cap. Bisection alone halves the bracket each step, so this many steps is far more
/// than enough to reach the representable limit from any starting bracket.
const MAX_STEPS: u32 = 200;

/// Wilson–Hilferty starting guess: the cube-root transform of a gamma is nearly normal, so a
/// normal quantile maps across. Good to a few digits over most of the parameter range.
fn gamma_start(p: f32, a: f32) -> f32 {
    let guess = wilson_hilferty_guess(ndtri(p), a);
    if guess.is_finite() && guess > 0.0 {
        guess
    } else {
        // The transform degrades for very small shape; fall back to the small-x form
        // P(a,x) ~ x^a / Gamma(a+1).
        //
        // The floor is `f32::MIN_POSITIVE`, not the `1e-300` this line used to carry. That
        // literal came over from an `f64` context and **underflows to zero in `f32`**, so
        // `.max(1e-300)` was `.max(0.0)` — a no-op for a value already non-negative, and a floor
        // that floored nothing. Harmless in practice, because the caller applies
        // `.max(f32::MIN_POSITIVE)` to the result anyway, but a constant that silently means
        // something else in the narrower type is exactly the hazard this tree keeps producing.
        (p * (ln_gamma(a + 1.0)).exp())
            .powf(1.0 / a)
            .max(f32::MIN_POSITIVE)
    }
}

/// Whether every interior quantile of `Gamma(a, 1)` rounds to the shape at `f32`'s spacing.
///
/// Native, where this used to widen both arguments and call the `f64` predicate. The inequality
/// is pure arithmetic — there was never anything for the wider format to add — but the *spacing*
/// it tests is `f32`'s, so widening it was doubly beside the point: the quantity being compared
/// only exists in this format.
///
/// When half the spacing below `a` is at least 64 standard deviations, the Chernoff bounds for
/// `Gamma(a, 1)` put both outer tails below `exp(−1024)`, under the least positive `f32`
/// probability, so no interior `p` can resolve away from the mean. `√a` is the standard
/// deviation; the factor of 128 is 64 doubled to compare against the full spacing rather than
/// its half.
fn quantiles_round_to_shape(a: f32) -> bool {
    (a - a.next_down()) >= 128.0 * a.sqrt()
}

/// The gamma quantile: the `x` with `P(a, x) = p`.
///
/// # Native, and what that costs at a small shape
///
/// This used to widen to `f64` for `a ≤ 0.01`, with the note that "rounding the CDF before
/// inversion amplifies its error at small shapes". That note was exactly right, and the
/// amplification is worth stating as a number because it is a property of the *problem* rather
/// than of any implementation.
///
/// Near zero `P(a, x) ≈ x^a / Γ(1+a)`, so `∂P/∂x = aP/x` and therefore
///
/// ```text
/// δx/x = (δP/P) / a
/// ```
///
/// The inversion's condition number is `1/a`. At `a = 0.01` that is a hundredfold, so a CDF good
/// to one `f32` ulp yields a root good to about a hundred — and no choice of solver, bracket or
/// stopping rule changes it, because the forward function is genuinely that flat: `P` moves
/// through its whole range only as `x` crosses `e^{1/a}`. Solving for `ln x` instead does not
/// help either; `∂P/∂(ln x) = aP` carries the same factor.
///
/// Measured against the `f64` kernel rounded to `f32`, worst over `p ∈ (0, 1)`:
///
/// | `a` | `1e-3` | `0.01` | `0.05` | `0.5` | `1` | `10` | `100` |
/// |---|---|---|---|---|---|---|---|
/// | ulp | `111` | `115` | `59` | `9` | `8` | `12` | `50` |
///
/// which tracks `1/a` where that is large, and the CDF's own widening error at the other end.
/// So the `f64` route was not better by virtue of being better code — it had 29 more bits to
/// spend on a hundredfold amplification, and `f32` does not. A consumer needing a small-shape
/// gamma quantile to `f32`'s full resolution needs `f64`; this function cannot supply it, and
/// says so rather than quietly borrowing the wider format to look as though it can.
pub(crate) fn gamma_lr_inv(p: f32, a: f32) -> f32 {
    match GammaQuantileRoute::for_arguments(p, a) {
        GammaQuantileRoute::Undefined => f32::NAN,
        GammaQuantileRoute::AtZero => 0.0,
        GammaQuantileRoute::AtInfinity => f32::INFINITY,
        GammaQuantileRoute::RoundsToShape => a,
        GammaQuantileRoute::Solve => gamma_quantile_solve(p, a),
    }
}

/// Which tail of the gamma the quantile solves against.
///
/// The same two arms the `f64` peer in [`crate::kernels::quantile`] carries, and for the same
/// reason: they solve *different equations* for the same root, and both the CDF and the target
/// shift with the choice. The pair travels as one named thing because the two uses have to agree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GammaTail {
    /// `p ≤ ½`: solve `P(a, x) = p` directly.
    Lower,
    /// `p > ½`: solve `−Q(a, x) = p − 1` instead. `P` has a rounding plateau near one — wider
    /// here than in `f64`, with 24 bits of mantissa rather than 53 — where the residual carries
    /// no information about which `x` is the root. Negated so the function stays increasing.
    Upper,
}

impl GammaTail {
    fn for_probability(p: f32) -> Self {
        if p > 0.5 { Self::Upper } else { Self::Lower }
    }

    /// The increasing function whose root is the quantile.
    fn cdf(self, a: f32, x: f32) -> f32 {
        match self {
            Self::Lower => super::gamma::gamma_lr(a, x),
            Self::Upper => -super::gamma::gamma_ur(a, x),
        }
    }

    /// The target `p`, moved into the same frame as [`Self::cdf`].
    fn target(self, p: f32) -> f32 {
        match self {
            Self::Lower => p,
            Self::Upper => p - 1.0,
        }
    }
}

/// Which route produces the gamma quantile at this `(p, a)`.
///
/// The same five arms as the `f64` peer; [`quantiles_round_to_shape`] differs in taking the shape
/// alone, because this format's spacing is derived rather than passed in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GammaQuantileRoute {
    /// A shape that is not a positive finite number, or a `p` outside `[0, 1]` — `NaN` included,
    /// which the range check rejects by failing rather than by comparing.
    Undefined,
    /// `p = 0`: the infimum of the support, attained exactly.
    AtZero,
    /// `p = 1`: the supremum, which the gamma does not attain.
    AtInfinity,
    /// Every interior quantile at this shape rounds to the shape itself, so there is nothing to
    /// solve; see [`quantiles_round_to_shape`].
    RoundsToShape,
    /// An interior probability at a resolvable shape: bracket and solve.
    Solve,
}

impl GammaQuantileRoute {
    fn for_arguments(p: f32, a: f32) -> Self {
        if !a.is_finite() || a <= 0.0 || !(0.0..=1.0).contains(&p) {
            Self::Undefined
        } else if p == 0.0 {
            Self::AtZero
        } else if p == 1.0 {
            Self::AtInfinity
        } else if quantiles_round_to_shape(a) {
            Self::RoundsToShape
        } else {
            Self::Solve
        }
    }
}

/// Bracket and solve for an interior gamma quantile at a resolvable shape.
fn gamma_quantile_solve(p: f32, a: f32) -> f32 {
    let tail = GammaTail::for_probability(p);
    let cdf = |x: f32| tail.cdf(a, x);

    // Bracket: walk out from the guess until the CDF straddles p.
    let seed = gamma_start(p, a);
    let p = tail.target(p);
    let mut lo = 0.0f32;
    let mut hi = seed.max(f32::MIN_POSITIVE);
    let mut steps = 0;
    while cdf(hi) < p && steps < MAX_STEPS {
        lo = hi;
        hi *= 2.0;
        steps += 1;
    }
    if hi.is_infinite() {
        return f32::INFINITY;
    }
    // Every nonzero lower endpoint was accepted by `cdf(hi) < p` above.
    let log_normaliser = ln_gamma(a);
    solve_increasing(
        cdf,
        |x| gamma_pdf(x, a, log_normaliser),
        p,
        lo,
        hi,
        seed.clamp(lo, hi),
    )
}

/// Safeguarded Newton steps share the bracket-relative stopping rule for both distributions.
fn solve_increasing(
    cdf: impl Fn(f32) -> f32,
    pdf: impl Fn(f32) -> f32,
    p: f32,
    mut lo: f32,
    mut hi: f32,
    mut x: f32,
) -> f32 {
    for _ in 0..MAX_STEPS {
        let f = cdf(x) - p;
        if f == 0.0 {
            return x;
        }
        // Keep the bracket tight around the root.
        if f < 0.0 {
            lo = x;
        } else {
            hi = x;
        }
        let next = bracketed_newton_step(x, f, pdf(x), lo, hi);
        // The relative check includes adjacent endpoints, where bisection rounds to an edge.
        if bracket_is_narrow(lo, hi, 4.0 * f32::EPSILON) {
            return next;
        }
        x = next;
    }
    x
}

/// The beta quantile: the `x` in `[0, 1]` with `I_x(a, b) = p`.
pub(crate) fn beta_reg_inv(p: f32, a: f32, b: f32) -> f32 {
    match BetaQuantileRoute::for_arguments(p, a, b) {
        BetaQuantileRoute::Undefined => f32::NAN,
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
    fn for_arguments(p: f32, a: f32, b: f32) -> Self {
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
fn beta_quantile_solve(p: f32, a: f32, b: f32) -> f32 {
    // Note the internal argument order: (a, b, x), not (x, a, b).
    let cdf = |x: f32| super::beta::beta_reg(a, b, x);

    // The support is bounded, so the bracket is known up front — no expansion needed. That
    // makes this case unconditionally safe: bisection alone would converge.
    let log_beta = log_beta_normaliser(a, b);
    solve_increasing(
        cdf,
        |x| beta_pdf(x, a, b, log_beta),
        p,
        0.0,
        1.0,
        a / (a + b),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::f32::beta::beta_reg;
    use crate::kernels::f32::gamma::gamma_lr;

    fn close(a: f32, b: f32, tol: f32) {
        assert!(
            (a - b).abs() <= tol * b.abs().max(1.0),
            "{a:e} vs {b:e} (tol {tol:e})"
        );
    }

    #[test]
    fn gamma_quantile_round_trips_in_f32() {
        for &a in &[0.5f32, 1.0, 2.5, 10.0] {
            for i in 1..20 {
                let p = (i as f32) * 0.05;
                let x = gamma_lr_inv(p, a);
                assert!(x.is_finite() && x >= 0.0, "gamma_lr_inv({p},{a}) = {x}");
                close(gamma_lr(a, x), p, 1e-4);
            }
        }
    }

    #[test]
    fn beta_quantile_round_trips_in_f32() {
        for &(a, b) in &[(1.0f32, 1.0f32), (2.0, 5.0), (5.0, 2.0)] {
            for i in 1..20 {
                let p = (i as f32) * 0.05;
                let x = beta_reg_inv(p, a, b);
                assert!((0.0..=1.0).contains(&x), "beta_reg_inv({p},{a},{b}) = {x}");
                close(beta_reg(a, b, x), p, 1e-4);
            }
        }
    }

    /// A **small** root has to be resolved relative to itself, not to `1`.
    ///
    /// The regression for a defect this file's other tests could not see. `beta_reg_inv` stopped
    /// bisecting once `hi - lo` fell below a bare `4·ε = 4.8e-7`, an *absolute* bound — so for
    /// any root below that it returned the bracket midpoint. `I_x(0.1, 0.1) = 0.05` came back as
    /// `2.38e-7`, which is `2⁻²²` and visibly the bisection halting, against a true root of
    /// `8.66e-11`: **2,752 times too large**. Its sibling `gamma_lr_inv` and the f64 peer both
    /// scaled by the bracket; this one did not.
    ///
    /// Nothing above reaches it because the three shapes tested are `(1,1)`, `(2,5)` and `(5,2)`,
    /// whose roots all sit mid-interval where an absolute `4.8e-7` is harmless. A U-shaped beta
    /// — either parameter below 1 — piles its mass into the corners and pushes the root toward
    /// an endpoint, which is the same reason the f64 suite has a `quantiles::extremes` file.
    ///
    /// The reference is the `f64` kernel, four orders finer than `f32` can represent. The bound
    /// is `1e-5` against a measured worst of `2.98e-7` (about 2.5 f32 ulp), leaving room for the
    /// CDF's own narrowing error without admitting anything like the defect.
    #[test]
    fn the_beta_quantile_resolves_a_root_near_an_endpoint() {
        for &(a, b) in &[(0.1f32, 0.1f32), (0.5, 0.5), (0.2, 3.0), (3.0, 0.2)] {
            for i in 1..20 {
                let p = (i as f32) * 0.05;
                let got = beta_reg_inv(p, a, b);
                let truth = crate::kernels::quantile::beta_reg_inv(
                    f64::from(p),
                    f64::from(a),
                    f64::from(b),
                );
                // Skip where the true root is not a normal f32; there the format answers.
                if truth < f64::from(f32::MIN_POSITIVE) * 16.0 || truth > 1.0 - 1e-6 {
                    continue;
                }
                let relative = ((f64::from(got) - truth) / truth).abs();
                assert!(
                    relative < 1e-5,
                    "beta_reg_inv({p}, {a}, {b}) = {got:e}, f64 says {truth:e} \
                     (relative {relative:.3e}) — the bisection stopped on an absolute bound"
                );
            }
        }
    }

    /// The gamma quantile's five routes, asserted as decisions.
    #[test]
    fn the_gamma_quantile_routes_cover_the_probability_range() {
        use GammaQuantileRoute::{AtInfinity, AtZero, RoundsToShape, Solve, Undefined};

        for (p, a) in [
            (0.5, 0.0),
            (0.5, -1.0),
            (0.5, f32::NAN),
            (0.5, f32::INFINITY),
            (f32::NAN, 2.0),
            ((-0.0_f32).next_down(), 2.0),
            (1.0_f32.next_up(), 2.0),
            (-1.0, 2.0),
            (2.0, 2.0),
        ] {
            assert_eq!(
                GammaQuantileRoute::for_arguments(p, a),
                Undefined,
                "at p = {p:e}, a = {a:e}"
            );
        }
        assert_eq!(GammaQuantileRoute::for_arguments(0.0, 2.0), AtZero);
        assert_eq!(GammaQuantileRoute::for_arguments(1.0, 2.0), AtInfinity);
        // The snap needs `ulp(a) ≥ 128√a`, which for `a = 2^k` in this format is
        // `k − 23 ≥ 7 + k/2`, so `k ≥ 60` — against `k ≥ 118` in `f64`, the 29-bit mantissa
        // difference showing up directly in where the distribution stops being resolvable.
        for p in [f32::MIN_POSITIVE, 1e-12, 0.25, 0.5, 0.75, 1.0 - 1e-6] {
            assert_eq!(GammaQuantileRoute::for_arguments(p, 2.0), Solve);
            assert_eq!(GammaQuantileRoute::for_arguments(p, 2f32.powi(50)), Solve);
            assert_eq!(
                GammaQuantileRoute::for_arguments(p, 2f32.powi(64)),
                RoundsToShape
            );
        }
    }

    /// The two tails, and the fact that they solve the same root through different equations.
    #[test]
    fn the_gamma_tail_switches_at_the_median_and_agrees_with_itself() {
        use GammaTail::{Lower, Upper};

        // Half itself is the lower tail: the comparison is strict.
        assert_eq!(GammaTail::for_probability(0.5), Lower);
        assert_eq!(GammaTail::for_probability(0.5_f32.next_up()), Upper);
        assert_eq!(GammaTail::for_probability(0.5_f32.next_down()), Lower);
        assert_eq!(GammaTail::for_probability(0.0), Lower);
        assert_eq!(GammaTail::for_probability(1.0), Upper);

        // The target moves with the arm, and only for the upper one.
        assert_eq!(Lower.target(0.3), 0.3);
        assert_eq!(Upper.target(0.75), -0.25);

        // `P = 1 − Q`, so the two residuals agree at every `x`. That identity is what makes the
        // switch free, and it is the only thing tying the two arms together.
        let a = 3.0;
        for x in [0.5_f32, 1.0, 3.0, 7.0, 20.0] {
            let p = 0.75;
            let lower = Lower.cdf(a, x) - Lower.target(p);
            let upper = Upper.cdf(a, x) - Upper.target(p);
            assert!(
                (lower - upper).abs() <= 1e-6,
                "at x = {x}: {lower} vs {upper}"
            );
        }
        // And the arms are genuinely different functions of `x`.
        assert!(Lower.cdf(a, 3.0) > 0.0);
        assert!(Upper.cdf(a, 3.0) < 0.0);
    }

    /// The beta quantile's four routes. Its upper endpoint is attained, unlike the gamma's.
    #[test]
    fn the_beta_quantile_routes_attain_both_endpoints() {
        use BetaQuantileRoute::{AtOne, AtZero, Solve, Undefined};

        for (p, a, b) in [
            (0.5, 0.0, 1.0),
            (0.5, 1.0, 0.0),
            (0.5, -1.0, 1.0),
            (0.5, 1.0, -1.0),
            (0.5, f32::NAN, 1.0),
            (0.5, 1.0, f32::NAN),
            (0.5, f32::INFINITY, 1.0),
            (0.5, 1.0, f32::INFINITY),
            (f32::NAN, 1.0, 1.0),
            (-1.0, 1.0, 1.0),
            (2.0, 1.0, 1.0),
        ] {
            assert_eq!(
                BetaQuantileRoute::for_arguments(p, a, b),
                Undefined,
                "at p = {p:e}, a = {a:e}, b = {b:e}"
            );
        }
        assert_eq!(BetaQuantileRoute::for_arguments(0.0, 2.0, 3.0), AtZero);
        assert_eq!(BetaQuantileRoute::for_arguments(1.0, 2.0, 3.0), AtOne);
        // `I_x(a, b) = 1` at `x = 1` for every shape pair, which is the arm the gamma lacks.
        assert_eq!(beta_reg_inv(1.0, 0.1, 0.01), 1.0);
        for p in [f32::MIN_POSITIVE, 1e-12, 0.5, 1.0 - 1e-6] {
            assert_eq!(BetaQuantileRoute::for_arguments(p, 2.0, 3.0), Solve);
            // Including the U-shaped densities, where the mass piles into both corners.
            assert_eq!(BetaQuantileRoute::for_arguments(p, 0.1, 0.01), Solve);
        }
    }
}
