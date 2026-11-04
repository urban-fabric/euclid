//! Marching `y″ = xy` outward along the negative axis by its own Taylor series.
//!
//! # The gap this fills
//!
//! The negative axis has a band where neither of the two obvious methods reaches machine
//! precision, and they miss it from opposite directions:
//!
//! | `|x|` | 4 | 5 | 6 | 7 | 8 | 10 |
//! |-------|---|---|---|---|---|----|
//! | Maclaurin series ([`super::series`]) | `7e-15` | `3e-14` | `2e-13` | `4e-12` | `4e-11` | `1e-7` |
//! | Hankel asymptotic ([`crate::kernels::bessel::asymptotic`]) | `9e-6` | `2e-8` | `2e-10` | `4e-13` | `1e-14` | `1e-14` |
//!
//! The series loses `ε·e^{ζ}` to cancellation and the asymptotic's optimal truncation errs by
//! about `e^{−2ζ}`; setting those equal puts the crossing at `ζ = −⅓ln ε`, which is `|x| ≈ 6.9`,
//! and pins the best achievable error there at roughly `1e-12`. No choice of a single crossover
//! does better, because the two floors genuinely cross above machine precision.
//!
//! # The method
//!
//! Stepping the differential equation closes it. Expanding a solution about `x₀`,
//! `y(x₀ + h) = Σ cₙhⁿ`, and substituting into `y″ = xy` gives
//!
//! ```text
//! (n+2)(n+1)c_{n+2} = x₀cₙ + c_{n−1},     c_{−1} = 0
//! ```
//!
//! so the whole Taylor coefficient list follows from `c₀ = y(x₀)` and `c₁ = y′(x₀)` by a
//! two-term recurrence with *exact* rational weights. Nothing is transcribed, nothing is
//! truncated beyond the step length, and the same stepper transports any solution of the
//! equation — which is why it is applied to `f` and `g` rather than to `Ai` and `Bi`.
//!
//! Starting from [`super::SERIES_MAX`], where the series is still exact, and stepping outward
//! holds machine precision across the whole band:
//!
//! | `|x|` | 2.5 | 4 | 6 | 7.5 | 9 | 10 |
//! |-------|-----|---|---|-----|---|----|
//! | relative error in `Ai` | `6e-16` | `2e-16` | `0` | `2e-16` | `7e-15` | `2e-15` |
//!
//! # Why it is stable here, and only here
//!
//! This is the same device [`crate::kernels::bessel::ordinary_second`] uses to reach `Y`'s
//! middle range, and it is sound for the same reason. On the **negative** axis `Ai(−z)` and
//! `Bi(−z)` both oscillate with an amplitude decaying like `z^{−1/4}`: neither solution
//! dominates the other, so an error introduced at one step is transported, not amplified.
//!
//! On the **positive** axis the pair is maximally unequal — `Bi` grows like `e^{ζ}` while `Ai`
//! decays like `e^{−ζ}` — and marching `Ai` outward there would be catastrophic, since any
//! error picks up the growing solution and swamps the decaying one within a few steps. That is
//! why this module is reached only for `x < 0`, and why the positive axis goes through `K`
//! instead.

use super::series;

/// Taylor order. The step is bounded by [`MAX_STEP`], so the omitted tail is about
/// `|h|^{N+1}/(N+1)!`; at `N = 24` and `|h| = ½` that is below `1e-40`, which leaves the
/// truncation nowhere near the rounding.
const ORDER: usize = 24;

/// Largest step length.
///
/// Chosen from the cost side rather than the accuracy side: the local radius of convergence of
/// a solution of `y″ = xy` is unbounded, so `½` is far inside it, and halving it again would
/// double the work for no measurable gain. Each step costs `ORDER` recurrence passes plus two
/// Horner evaluations.
const MAX_STEP: f64 = 0.5;

/// Advance one solution of `y″ = xy` from `x₀` to `x₀ + h`, returning `(y, y′)` there.
fn step(x0: f64, y: f64, y_prime: f64, h: f64) -> (f64, f64) {
    let mut c = [0.0f64; ORDER + 2];
    c[0] = y;
    c[1] = y_prime;
    for n in 0..ORDER {
        let previous = if n >= 1 { c[n - 1] } else { 0.0 };
        c[n + 2] = (x0 * c[n] + previous) / (((n + 2) * (n + 1)) as f64);
    }
    // Horner, from the top down, for the value and for the derivative `Σ n·cₙh^{n−1}`.
    let mut value = 0.0f64;
    let mut derivative = 0.0f64;
    for n in (0..=ORDER + 1).rev() {
        value = value * h + c[n];
        if n >= 1 {
            derivative = derivative * h + (n as f64) * c[n];
        }
    }
    (value, derivative)
}

/// Transport `(y, y′)` from `x_start` to `x_end` in equal steps of at most [`MAX_STEP`].
fn transport(x_start: f64, x_end: f64, y: f64, y_prime: f64) -> (f64, f64) {
    let span = x_end - x_start;
    let count = (span.abs() / MAX_STEP).ceil().max(1.0);
    let h = span / count;
    let (mut value, mut derivative) = (y, y_prime);
    let mut x = x_start;
    let mut taken = 0.0f64;
    while taken < count {
        (value, derivative) = step(x, value, derivative, h);
        x += h;
        taken += 1.0;
    }
    (value, derivative)
}

/// `(Ai, Ai′, Bi, Bi′)` at `x < 0`, by marching out from `−`[`super::SERIES_MAX`].
///
/// `f` and `g` are transported separately and combined only at the end, through the same
/// [`series::combine`] the series branch uses. Starting the march exactly where that branch
/// stops makes the two agree at the join by construction.
pub(super) fn airy(x: f64) -> (f64, f64, f64, f64) {
    let start = -super::SERIES_MAX;
    let (f0, g0, f0_prime, g0_prime) = series::fg(start);
    let (f, f_prime) = transport(start, x, f0, f0_prime);
    let (g, g_prime) = transport(start, x, g0, g0_prime);
    series::combine(f, g, f_prime, g_prime)
}
