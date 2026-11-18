//! Single-precision peers of [`super::super::quantile`], with a shared `f32`
//! Newton/bisection loop and format-specific stopping rule.

use super::gamma::ln_gamma;
use super::normal::ndtri;
use crate::kernels::quantile::gamma_quantiles_round_to_shape;
use crate::math::roots::{bracket_is_narrow, bracketed_newton_step, wilson_hilferty_guess};

/// Iteration cap. Bisection alone halves the bracket each step, so this many steps is far more
/// than enough to reach the representable limit from any starting bracket.
const MAX_STEPS: u32 = 200;

/// `Gamma(a, 1)` density, from `ln_gamma`.
fn gamma_pdf(x: f32, a: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    ((a - 1.0) * x.ln() - x - ln_gamma(a)).exp()
}

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

/// The gamma quantile: the `x` with `P(a, x) = p`.
pub(crate) fn gamma_lr_inv(p: f32, a: f32) -> f32 {
    if a <= super::gamma::WIDE_SHAPE_MAX {
        // Rounding the CDF before inversion amplifies its error at small shapes.
        return super::super::quantile::gamma_lr_inv(f64::from(p), f64::from(a)) as f32;
    }
    if !a.is_finite() || !(0.0..=1.0).contains(&p) {
        return f32::NAN;
    }
    if p == 0.0 {
        return 0.0;
    }
    if p == 1.0 {
        return f32::INFINITY;
    }
    if gamma_quantiles_round_to_shape(f64::from(a), f64::from(a - a.next_down())) {
        return a;
    }

    let upper_tail = p > 0.5;
    let cdf = |x: f32| {
        if upper_tail {
            -super::gamma::gamma_ur(a, x)
        } else {
            super::gamma::gamma_lr(a, x)
        }
    };

    // Bracket: walk out from the guess until the CDF straddles p.
    let seed = gamma_start(p, a);
    let p = if upper_tail { p - 1.0 } else { p };
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
    solve_increasing(cdf, |x| gamma_pdf(x, a), p, lo, hi, seed.clamp(lo, hi))
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

/// `Beta(a, b)` density, from `ln_gamma` via the log beta function.
fn beta_pdf(x: f32, a: f32, b: f32) -> f32 {
    if x <= 0.0 || x >= 1.0 {
        return 0.0;
    }
    let ln_beta = ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b);
    ((a - 1.0) * x.ln() + (b - 1.0) * (-x).ln_1p() - ln_beta).exp()
}

/// The beta quantile: the `x` in `[0, 1]` with `I_x(a, b) = p`.
pub(crate) fn beta_reg_inv(p: f32, a: f32, b: f32) -> f32 {
    if !a.is_finite() || !b.is_finite() || a <= 0.0 || b <= 0.0 || !(0.0..=1.0).contains(&p) {
        return f32::NAN;
    }
    if p == 0.0 {
        return 0.0;
    }
    if p == 1.0 {
        return 1.0;
    }

    // Note the internal argument order: (a, b, x), not (x, a, b).
    let cdf = |x: f32| super::beta::beta_reg(a, b, x);

    // The support is bounded, so the bracket is known up front — no expansion needed. That
    // makes this case unconditionally safe: bisection alone would converge.
    solve_increasing(cdf, |x| beta_pdf(x, a, b), p, 0.0, 1.0, a / (a + b))
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
}
