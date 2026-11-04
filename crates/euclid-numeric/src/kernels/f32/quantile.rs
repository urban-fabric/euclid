//! Single-precision peers of [`super::super::quantile`]; identical algorithms, `f32`
//! throughout. See that module for the bracketed Newton/bisection design and why the
//! quantiles are solved rather than fitted.

use super::gamma::ln_gamma;
use super::normal::ndtri;

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
    let z = ndtri(p);
    let t = 1.0 - 1.0 / (9.0 * a) + z / (3.0 * a.sqrt());
    let guess = a * t * t * t;
    if guess.is_finite() && guess > 0.0 {
        guess
    } else {
        // The transform degrades for very small shape; fall back to the small-x form
        // P(a,x) ~ x^a / Gamma(a+1).
        (p * (ln_gamma(a + 1.0)).exp()).powf(1.0 / a).max(1e-300)
    }
}

/// The gamma quantile: the `x` with `P(a, x) = p`.
pub(crate) fn gamma_lr_inv(p: f32, a: f32) -> f32 {
    if p.is_nan() || a.is_nan() || a <= 0.0 || !(0.0..=1.0).contains(&p) {
        return f32::NAN;
    }
    if p == 0.0 {
        return 0.0;
    }
    if p == 1.0 {
        return f32::INFINITY;
    }

    let cdf = |x: f32| super::gamma::gamma_lr(a, x);

    // Bracket: walk out from the guess until the CDF straddles p.
    let mut lo = 0.0f32;
    let mut hi = gamma_start(p, a).max(f32::MIN_POSITIVE);
    let mut steps = 0;
    while cdf(hi) < p && steps < MAX_STEPS {
        lo = hi;
        hi *= 2.0;
        steps += 1;
    }
    if hi.is_infinite() {
        return f32::INFINITY;
    }
    // If the guess overshot, walk the lower end down instead.
    steps = 0;
    while lo > 0.0 && cdf(lo) > p && steps < MAX_STEPS {
        hi = lo;
        lo *= 0.5;
        steps += 1;
    }

    let mut x = hi.min(gamma_start(p, a).max(lo)).max(lo);
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
        let d = gamma_pdf(x, a);
        // Newton, but only if it lands strictly inside the bracket; otherwise bisect.
        let next = if d > 0.0 {
            let candidate = x - f / d;
            if candidate > lo && candidate < hi {
                candidate
            } else {
                0.5 * (lo + hi)
            }
        } else {
            0.5 * (lo + hi)
        };
        if next == x || (hi - lo) <= 4.0 * f32::EPSILON * hi.abs().max(f32::MIN_POSITIVE) {
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
    if p.is_nan() || a.is_nan() || b.is_nan() || a <= 0.0 || b <= 0.0 {
        return f32::NAN;
    }
    if !(0.0..=1.0).contains(&p) {
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
    let (mut lo, mut hi) = (0.0f32, 1.0f32);
    // Start from the mean, which is always interior.
    let mut x = a / (a + b);

    for _ in 0..MAX_STEPS {
        let f = cdf(x) - p;
        if f == 0.0 {
            return x;
        }
        if f < 0.0 {
            lo = x;
        } else {
            hi = x;
        }
        let d = beta_pdf(x, a, b);
        let next = if d > 0.0 {
            let candidate = x - f / d;
            if candidate > lo && candidate < hi {
                candidate
            } else {
                0.5 * (lo + hi)
            }
        } else {
            0.5 * (lo + hi)
        };
        if next == x || (hi - lo) <= 4.0 * f32::EPSILON {
            return next;
        }
        x = next;
    }
    x
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
}
