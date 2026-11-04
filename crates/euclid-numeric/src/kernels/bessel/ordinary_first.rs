//! The Bessel functions of the **first kind**, `J₀` and `J₁`.
//!
//! # Method: the aliased trapezoidal rule
//!
//! `J` has an elegant exact quadrature that seems to be underused. From the Jacobi–Anger
//! expansion `e^{ix sin θ} = Σₖ Jₖ(x)e^{ikθ}`,
//!
//! ```text
//! Jₙ(x) = (1/2π) ∫_{−π}^{π} e^{i(x sin θ − nθ)} dθ
//! ```
//!
//! The integrand is **periodic and entire**, which is the ideal case for the trapezoidal rule:
//! with `N` equally spaced nodes the rule is exact on every Fourier mode except those it
//! aliases, so the error is not an asymptotic estimate but an identity,
//!
//! ```text
//! (1/N) Σⱼ cos(x sin θⱼ − nθⱼ) = Jₙ(x) + J_{n+N}(x) + J_{n−N}(x) + J_{n+2N}(x) + ⋯
//! ```
//!
//! Since `Jₘ(x) ≈ (x/2)^m/m!` collapses once `m ≫ x`, choosing `N` a little above `x` makes
//! every aliased term vanish below rounding — and [`alias_nodes`] solves that condition
//! directly rather than guessing. Two further properties matter:
//!
//! - **It is an average of bounded cosines**, so the absolute rounding error is about `ε`
//!   whatever `x` is. There is no cancellation to lose, unlike the defining power series,
//!   whose terms sum in magnitude to `I₀(x)` and so give up `~ε·I₀(x)` — already `5e-14` by
//!   `x = 8`.
//! - **It needs no starting values and accumulates no error**, being a single finite sum.
//!
//! The cost is `O(x)` trigonometric evaluations, which is why the Hankel expansion still takes
//! over for large `x`.
//!
//!
//! # Accuracy, and the zeros
//!
//! All four are machine precision in **absolute** terms. None can be machine precision in
//! *relative* terms near its zeros, and no implementation can be: the zeros are irrational, so
//! `x` itself is already a rounded version of the argument the caller meant, and `J₀′ ≠ 0`
//! there carries that `~ε·x` input uncertainty straight into the result. The tests therefore
//! check *absolute* agreement near a zero and relative agreement away from it.

use super::asymptotic::hankel;
use core::f64::consts::{PI, TAU};

/// At or above this argument `J` uses the Hankel expansion; below it, the series or the
/// aliased trapezoid. Shared with [`super::ordinary_second`], and set by measurement.
pub(super) const HANKEL_MIN: f64 = 18.0;

/// Relative size below which an aliased term is taken to vanish.
const ALIAS_EPS: f64 = 1e-19;

/// Cap on the aliasing node count, which grows like `2x`; only reached above
/// [`HANKEL_MIN`], where this route is not used.
const ALIAS_MAX_NODES: u32 = 512;

/// Iteration cap for the power series.
const MAX_TERMS: u32 = 1_000;

/// Relative size at which a series term stops contributing.
const TERM_EPS: f64 = 1e-20;

/// Node count making the aliasing error of [`j_aliased`] negligible.
///
/// The error is `J_{N−n}(x) + J_{N+n}(x) + ⋯`, dominated by the first term, so this finds the
/// smallest `m` with `(x/2)^m/m! < ALIAS_EPS` and returns `N = m + n` rounded up to even (the
/// sum exploits a symmetry that needs an even node count).
///
/// The `m <= half` clause keeps the walk going until it is past the peak at `m ≈ x/2`: the
/// terms *grow* before they decay, so stopping at the first small value would otherwise
/// return `m = 1` for every argument above a few.
pub(super) fn alias_nodes(n: u32, x: f64) -> u32 {
    let half = x * 0.5;
    let mut m = 1.0f64;
    let mut term = half; // (x/2)^m / m!
    while m <= half || term > ALIAS_EPS {
        m += 1.0;
        term *= half / m;
        if m >= f64::from(ALIAS_MAX_NODES) {
            break;
        }
    }
    let needed = m as u32 + n + 2;
    needed + needed % 2
}

/// At or below this argument `J₀`/`J₁` use the power series rather than the quadrature.
///
/// Not an accuracy nicety but a correctness requirement. The quadrature's summand is
/// `cos(x·sin θ − nθ)`, and for small `x` the `x·sin θ` part is swallowed by `nθ`: at
/// `x = 2e-308`, `x − π/2` rounds to exactly `−π/2`, so the sum returns the rounding of
/// `cos(−π/2) ≈ 6e-17` instead of anything resembling `J₁(x) ≈ x/2`. The absolute error is
/// still tiny, but every digit of the relative answer is gone — and `J₁(x) ~ x/2` is an
/// exactly-conditioned limit that ought to be returned exactly.
///
/// The series has no such problem and no cancellation worth the name this small: its terms sum
/// in magnitude to `I₀(x)`, which is `2.28` at the cutoff.
pub(super) const J_SERIES_MAX: f64 = 2.0;

/// `Jₙ(x)` for small `x ≥ 0` and `n ∈ {0, 1}`, by the defining power series
/// `Σ (−1)^k (x/2)^{2k+n} / (k!(k+n)!)`.
pub(super) fn j_series(n: u32, x: f64) -> f64 {
    let h = x * 0.5;
    let order = f64::from(n);
    // The k = 0 term is (x/2)^n / n!, and n is 0 or 1, so n! is 1 either way.
    let mut term = if n == 0 { 1.0 } else { h };
    let mut total = term;
    for k in 1..MAX_TERMS {
        let kf = f64::from(k);
        term *= -h * h / (kf * (kf + order));
        total += term;
        if term.abs() < total.abs() * TERM_EPS {
            break;
        }
    }
    total
}

/// `Jₙ(x)` for `x ≥ 0` and `n ∈ {0, 1}`, by the aliased trapezoidal rule.
///
/// The summand is symmetric under `θ → 2π − θ`, so only half the period is walked.
pub(super) fn j_aliased(n: u32, x: f64) -> f64 {
    debug_assert!(x >= 0.0);
    let nodes = alias_nodes(n, x);
    let count = f64::from(nodes);
    let step = TAU / count;
    let order = f64::from(n);

    // The j = 0 and j = nodes/2 nodes are their own mirror images; everything between is
    // counted twice.
    let mut total = 1.0; // theta = 0, where the summand is cos(0) = 1
    total += (x * PI.sin() - order * PI).cos();
    let mut paired = 0.0f64;
    for j in 1..nodes / 2 {
        let theta = step * f64::from(j);
        paired += (x * theta.sin() - order * theta).cos();
    }
    (total + 2.0 * paired) / count
}

/// The Bessel function of the first kind `J₀(x)`, even in `x`.
pub(crate) fn j0(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    let ax = x.abs();
    if ax == 0.0 {
        return 1.0;
    }
    if ax.is_infinite() {
        return 0.0; // J0 decays like x^{-1/2}
    }
    if ax >= HANKEL_MIN {
        return hankel(0.0, ax).0;
    }
    if ax <= J_SERIES_MAX {
        return j_series(0, ax);
    }
    j_aliased(0, ax)
}

/// The Bessel function of the first kind `J₁(x)`, odd in `x`.
pub(crate) fn j1(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    let ax = x.abs();
    if ax == 0.0 {
        // Exactly, and preserving the sign of zero: the quadrature's mirror-image nodes
        // cancel only to rounding, leaving ~3e-17 where the true value is 0.
        return x;
    }
    if ax.is_infinite() {
        return 0.0;
    }
    let magnitude = if ax >= HANKEL_MIN {
        hankel(1.0, ax).0
    } else if ax <= J_SERIES_MAX {
        j_series(1, ax)
    } else {
        j_aliased(1, ax)
    };
    if x < 0.0 { -magnitude } else { magnitude }
}
