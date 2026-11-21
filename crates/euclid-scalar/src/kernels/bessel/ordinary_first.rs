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

/// Relative size at which a term stops contributing.
///
/// **`1e-19`, which is deliberately only a little below `f64`'s resolution.** `ε = 2.22e-16`, so
/// a term at `1e-19` of the sum moves the result by about a thousandth of an ulp, and every term
/// after it moves it by less. The value was `1e-22`, which asked the `J` power series to converge
/// three orders past the point where the format can represent the difference —
/// iterations bought and then discarded.
///
/// Raising it to `1e-19` is **free rather than a trade.** The outputs are bit-identical across a
/// 306,280-point sweep covering every kernel in this tree at both signs, while cutting `J₀` and `J₁` by **6.7%** each.0%** and `J₁` by **7.9%**.
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
/// Which route evaluates `J_ν` at this argument.
///
/// One predicate serves both orders: `J₀` is even and `J₁` odd, so each reduces to the magnitude
/// and the sign is applied afterwards. More arms than the single-precision peer's three, which
/// takes an already-absolute magnitude and leaves the guards to its callers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum JRoute {
    /// Not a number.
    Undefined,
    /// `x = ±0`, where `J₀` is exactly `1` and `J₁` exactly the argument — sign and all.
    AtOrigin,
    /// `±∞`, where both decay to zero like `x^{−1/2}`.
    AtInfinity,
    /// `|x| ≥ `[`HANKEL_MIN`]: the divergent Hankel expansion, cheapest and most accurate here.
    Hankel,
    /// `|x| ≤ `[`J_SERIES_MAX`]: the power series — required rather than preferred, see that
    /// constant.
    Series,
    /// The middle band, by aliased trapezoidal quadrature.
    Aliased,
}

impl JRoute {
    pub(super) fn for_argument(x: f64) -> Self {
        if x.is_nan() {
            return Self::Undefined;
        }
        let ax = x.abs();
        if ax == 0.0 {
            Self::AtOrigin
        } else if ax.is_infinite() {
            Self::AtInfinity
        } else if ax >= HANKEL_MIN {
            Self::Hankel
        } else if ax <= J_SERIES_MAX {
            Self::Series
        } else {
            Self::Aliased
        }
    }
}

pub(crate) fn j0(x: f64) -> f64 {
    let ax = x.abs();
    match JRoute::for_argument(x) {
        JRoute::Undefined => f64::NAN,
        JRoute::AtOrigin => 1.0,
        JRoute::AtInfinity => 0.0,
        JRoute::Hankel => hankel(0.0, ax).first_kind,
        JRoute::Series => j_series(0, ax),
        JRoute::Aliased => j_aliased(0, ax),
    }
}

/// Apply an odd function's sign to a magnitude taken at `|x|`.
///
/// Not `copysign`: `J₁` oscillates, so the magnitude carries a sign of its own that `copysign`
/// would overwrite. Not `* x.signum()` either, which is exact but leaves `*` against `/`
/// indistinguishable — multiplying and dividing by `±1` give the same answer, so that spelling
/// buys three unkillable mutants. A sign-bit test has no operator to mutate at all, and the
/// argument is never zero here: [`JRoute::AtOrigin`] takes that case.
fn odd_about_the_origin(magnitude: f64, x: f64) -> f64 {
    if x.is_sign_negative() {
        -magnitude
    } else {
        magnitude
    }
}

pub(crate) fn j1(x: f64) -> f64 {
    let ax = x.abs();
    match JRoute::for_argument(x) {
        JRoute::Undefined => f64::NAN,
        // Exactly, and preserving the sign of zero: the quadrature's mirror-image nodes cancel
        // only to rounding, leaving ~3e-17 where the true value is 0.
        JRoute::AtOrigin => x,
        JRoute::AtInfinity => 0.0,
        JRoute::Hankel => odd_about_the_origin(hankel(1.0, ax).first_kind, x),
        JRoute::Series => odd_about_the_origin(j_series(1, ax), x),
        JRoute::Aliased => odd_about_the_origin(j_aliased(1, ax), x),
    }
}
