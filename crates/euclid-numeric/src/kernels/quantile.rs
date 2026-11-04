//! Quantiles of the gamma and beta distributions: the inverses of the regularized incomplete
//! gamma and beta functions.
//!
//! Both are **solved, not approximated**. A bracketed root-find on the forward CDF guarantees
//! convergence for every parameter combination, which a rational fit does not, and it inherits
//! the forward function's accuracy automatically — so the quantile can never drift away from
//! the CDF it is supposed to invert. The derivative needed by Newton is the density, available
//! in closed form from `ln_gamma`/`ln_beta`.
//!
//! The structure is the same in both cases:
//!
//! 1. A cheap starting guess, good to a few digits.
//! 2. Expand outward from it to a bracket with the root strictly inside.
//! 3. Newton steps, falling back to bisection whenever a step would leave the bracket or fail
//!    to reduce it. This is what makes the iteration unconditionally convergent: Newton gives
//!    the speed, the bracket gives the guarantee.

use super::gamma::ln_gamma;
use super::normal::ndtri;

/// Iteration cap for the root-find.
///
/// Sized to cross `f64`'s **exponent** range, not its mantissa. The ~60 steps that exhaust a
/// mantissa are enough only once the bracket is a single octave wide; starting from `[0, 1]`,
/// as the beta quantile does, bisection has to walk *down* one octave per step to reach a
/// small root, and there are 1075 octaves between 1 and the smallest subnormal.
///
/// At 200 this silently truncated. `beta_reg_inv(p, 1e-3, b)` stopped after 200 halvings at
/// `2^-200 ≈ 3.1e-61` and returned that for *every* `p` — the true median for `a = 1e-3` is
/// `0.5^1000 ≈ 9.3e-302`, perfectly representable — so the quantile was a constant function of
/// its own argument over the whole small-shape regime.
///
/// Raising it costs nothing in the ordinary case: [`solve_increasing`] exits as soon as the
/// bracket stops shrinking relative to its own magnitude, which Newton reaches in a handful of
/// steps.
const MAX_STEPS: u32 = 1_200;

/// Cap on the geometric bracket expansion, in octaves.
///
/// Separate from [`MAX_STEPS`] because it measures something different — how far the seed may
/// be from the root — and because getting it wrong is silent. At 200 the gamma quantile could
/// expand its bracket only `2^±200` from the Wilson–Hilferty seed, so for `a = 1e-3` it
/// returned `6.2e-64` for every `p` from `1e-12` to `0.5`, the root at `9.3e-302` lying far
/// below where the search gave up. 1100 octaves spans `f64` from `MIN_POSITIVE` to `MAX`.
const BRACKET_OCTAVES: u32 = 1_100;

/// Solve `cdf(x) == p` on a bracket known to contain the root, where `cdf` is non-decreasing.
///
/// Safeguarded Newton: a Newton step is taken only when it lands strictly inside the current
/// bracket, and bisection is used otherwise. That combination is unconditionally convergent —
/// the bracket can only shrink, so the worst case is plain bisection — while still getting
/// Newton's quadratic finish in the common case.
///
/// Iterating until the bracket stops shrinking, rather than until a fixed tolerance, is what
/// makes this parameter-independent: the useful precision differs enormously between a root
/// near `1e-13` and one near `1e4`, and the loop simply runs out of representable numbers in
/// either case.
/// Return whichever of a converged bracket's endpoints or midpoint comes closest to `p`.
///
/// The loop below converges to a bracket one or two ulps wide, and the midpoint of that
/// bracket is not necessarily the best of the three candidates. Usually it makes no
/// difference; where the CDF is steep it makes a large one. For `I_x(0.1, 0.01)` the whole
/// climb from `0.34` to `1` happens inside the last ulp below 1, so adjacent representable `x`
/// differ by `2.7e-4` of probability — and picking the wrong one of three throws that away.
///
/// Two extra evaluations buy the *best representable* answer, which is the strongest thing an
/// inverse can promise when the forward function is this ill-conditioned.
fn best_of(cdf: &impl Fn(f64) -> f64, p: f64, lo: f64, hi: f64) -> f64 {
    let mut best = 0.5 * (lo + hi);
    let mut err = (cdf(best) - p).abs();
    for candidate in [lo, hi] {
        let other = (cdf(candidate) - p).abs();
        if other < err {
            err = other;
            best = candidate;
        }
    }
    best
}

fn solve_increasing(
    cdf: impl Fn(f64) -> f64,
    pdf: impl Fn(f64) -> f64,
    p: f64,
    mut lo: f64,
    mut hi: f64,
) -> f64 {
    let mut x = 0.5 * (lo + hi);
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
        // Stop once the bracket cannot be subdivided any further.
        if hi <= lo || (hi - lo) <= f64::EPSILON * hi.abs().max(f64::MIN_POSITIVE) {
            return best_of(&cdf, p, lo, hi);
        }
        let d = pdf(x);
        let newton = x - f / d;
        x = if d > 0.0 && newton > lo && newton < hi {
            newton
        } else {
            0.5 * (lo + hi)
        };
    }
    best_of(&cdf, p, lo, hi)
}

/// `Gamma(a, 1)` density, from `ln_gamma`.
fn gamma_pdf(x: f64, a: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    ((a - 1.0) * x.ln() - x - ln_gamma(a)).exp()
}

/// Wilson–Hilferty starting guess: the cube-root transform of a gamma is nearly normal, so a
/// normal quantile maps across. Used only to seed the bracket search, so its accuracy affects
/// speed rather than the answer.
fn gamma_seed(p: f64, a: f64) -> f64 {
    let z = ndtri(p);
    let t = 1.0 - 1.0 / (9.0 * a) + z / (3.0 * a.sqrt());
    let guess = a * t * t * t;
    if guess.is_finite() && guess > 0.0 {
        guess
    } else {
        // The transform degrades for very small shape; fall back to the small-x form
        // P(a, x) ~ x^a / Gamma(a+1).
        a.max(1e-10)
    }
}

/// The gamma quantile: the `x` with `P(a, x) = p`.
pub(crate) fn gamma_lr_inv(p: f64, a: f64) -> f64 {
    if p.is_nan() || a.is_nan() || a <= 0.0 || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return 0.0;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }

    let cdf = |x: f64| super::gamma::gamma_lr(a, x);

    // Bracket by expanding geometrically from the seed in whichever direction is needed. This
    // does not assume the seed is on either particular side of the root, which is why it
    // survives shapes where Wilson-Hilferty is poor.
    let seed = gamma_seed(p, a);
    let (mut lo, mut hi) = (seed, seed);
    if cdf(seed) < p {
        for _ in 0..BRACKET_OCTAVES {
            lo = hi;
            hi *= 2.0;
            if !hi.is_finite() {
                return f64::INFINITY;
            }
            if cdf(hi) >= p {
                break;
            }
        }
    } else {
        for _ in 0..BRACKET_OCTAVES {
            hi = lo;
            lo *= 0.5;
            if lo == 0.0 || cdf(lo) <= p {
                break;
            }
        }
    }

    solve_increasing(cdf, |x| gamma_pdf(x, a), p, lo, hi)
}

/// `Beta(a, b)` density, from `ln_gamma` via the log beta function.
fn beta_pdf(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 || x >= 1.0 {
        return 0.0;
    }
    let ln_beta = ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b);
    ((a - 1.0) * x.ln() + (b - 1.0) * (-x).ln_1p() - ln_beta).exp()
}

/// The beta quantile: the `x` in `[0, 1]` with `I_x(a, b) = p`.
///
/// The support is bounded, so the bracket is `[0, 1]` with no search required — this case is
/// unconditionally safe for any `a`, `b`, including the U-shaped densities (`a, b < 1`) whose
/// mass piles into both corners.
pub(crate) fn beta_reg_inv(p: f64, a: f64, b: f64) -> f64 {
    if p.is_nan() || a.is_nan() || b.is_nan() || a <= 0.0 || b <= 0.0 {
        return f64::NAN;
    }
    if !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return 0.0;
    }
    if p == 1.0 {
        return 1.0;
    }
    let cdf = |x: f64| super::beta::beta_reg(a, b, x);
    solve_increasing(cdf, |x| beta_pdf(x, a, b), p, 0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::beta::beta_reg;
    use crate::kernels::gamma::gamma_lr;

    fn close(a: f64, b: f64, tol: f64) {
        assert!(
            (a - b).abs() <= tol * b.abs().max(1.0),
            "{a:e} vs {b:e} (tol {tol:e})"
        );
    }

    #[test]
    fn gamma_quantile_round_trips_over_a_wide_parameter_range() {
        for &a in &[0.1f64, 0.5, 1.0, 2.5, 10.0, 100.0, 1e4] {
            for i in 1..40 {
                let p = f64::from(i) * 0.025;
                let x = gamma_lr_inv(p, a);
                assert!(x.is_finite() && x >= 0.0, "gamma_lr_inv({p},{a}) = {x}");
                close(gamma_lr(a, x), p, 1e-10);
            }
        }
    }

    #[test]
    fn gamma_quantile_matches_the_exponential_closed_form() {
        // For a = 1 the CDF is 1 - e^-x, so the quantile is -ln(1-p).
        for i in 1..50 {
            let p = f64::from(i) * 0.02;
            close(gamma_lr_inv(p, 1.0), -(1.0 - p).ln(), 1e-10);
        }
    }

    #[test]
    fn beta_quantile_round_trips_over_a_wide_parameter_range() {
        for &(a, b) in &[
            (0.5f64, 0.5f64),
            (1.0, 1.0),
            (2.0, 5.0),
            (5.0, 2.0),
            (0.1, 0.1),
            (100.0, 50.0),
        ] {
            // A U-shaped beta (`a` or `b` below 1) piles its mass into the corners, so the
            // density at the endpoints is enormous — around 1e10 for (0.1, 0.1). The root
            // itself then sits within ~1e-13 of 0 or 1, where one ulp of `x` is worth about
            // 1e-6 of CDF. That is a conditioning limit of `f64`, not of the solver: no
            // root-finder can resolve the probability more finely than the representable
            // spacing of `x` allows. Measured worst case 4.4e-7 at (0.1, 0.1), which matches
            // that estimate; the tolerance sits just above it.
            let tol = if a < 1.0 || b < 1.0 { 1e-5 } else { 1e-10 };
            for i in 1..40 {
                let p = f64::from(i) * 0.025;
                let x = beta_reg_inv(p, a, b);
                assert!((0.0..=1.0).contains(&x), "beta_reg_inv({p},{a},{b}) = {x}");
                close(beta_reg(a, b, x), p, tol);
            }
        }
    }

    #[test]
    fn beta_quantile_matches_its_closed_forms() {
        // I_x(1,1) = x, so the quantile is p.
        for i in 1..50 {
            let p = f64::from(i) * 0.02;
            close(beta_reg_inv(p, 1.0, 1.0), p, 1e-12);
            // I_x(a,1) = x^a  =>  quantile is p^(1/a).
            close(beta_reg_inv(p, 3.0, 1.0), p.powf(1.0 / 3.0), 1e-10);
            // Arcsine law: I_x(1/2,1/2) = (2/pi) asin(sqrt x)  =>  x = sin^2(pi p / 2).
            let want = (core::f64::consts::PI * p / 2.0).sin().powi(2);
            close(beta_reg_inv(p, 0.5, 0.5), want, 1e-9);
        }
    }

    #[test]
    fn quantile_endpoints_and_domain_errors() {
        assert_eq!(gamma_lr_inv(0.0, 2.0), 0.0);
        assert!(gamma_lr_inv(1.0, 2.0).is_infinite());
        assert!(gamma_lr_inv(1.5, 2.0).is_nan());
        assert!(gamma_lr_inv(0.5, -1.0).is_nan());
        assert_eq!(beta_reg_inv(0.0, 2.0, 3.0), 0.0);
        assert_eq!(beta_reg_inv(1.0, 2.0, 3.0), 1.0);
        assert!(beta_reg_inv(-0.1, 2.0, 3.0).is_nan());
        assert!(beta_reg_inv(0.5, 0.0, 3.0).is_nan());
    }
}
