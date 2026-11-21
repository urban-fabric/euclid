//! The `f32` peer of the ordinary Bessel functions: `J₀`, `J₁`, `Y₀`, `Y₁`.
//!
//! Native single precision throughout. The module header once grouped all four as needing `f64`
//! because "a cancelling series, an outward march, and a divergent asymptotic expansion" all
//! have error floors set by the working precision. Taken mechanism by mechanism:
//!
//! - **`J`** reaches its values through a series, an *aliased trapezoidal quadrature* and the
//!   Hankel expansion. The quadrature's node count and the expansion's handover are both fixed
//!   by equating an error bound to a target accuracy, so both retune to `f32` arithmetically —
//!   the same situation as `K`, and the same roughly threefold saving. That is this file's
//!   native half.
//! - **`Y`** has no comparable quadrature, and closes its middle gap by *marching Bessel's
//!   equation outward* from the series. A march is the one mechanism here whose floor is not a
//!   tunable: it accumulates a rounding of order `ε` per step, so the narrower format pays for
//!   every step and no retuning recovers it.
//!
//!   Measured, it costs less than that argument suggests, for a reason that only shows up once
//!   `J`'s handover has moved: the march runs from [`second::Y_SERIES_MAX`] to [`HANKEL_MIN`],
//!   and that interval is `[2, 7]` here against `f64`'s `[4, 18]` — about six steps either way.
//!   Forward integration is stable here because `J` and `Y` both decay like `x^{−1/2}`, so
//!   neither dominates and an error introduced at one step is not amplified at the next. The
//!   finished `Y` holds to `2.9e-7` of its envelope across `x ∈ (0, 300]`, under three ulp.
//!
//!   The concern was real and the conclusion drawn from it was not. A per-step `ε` matters when
//!   the step count is large; bounding the count is what makes the mechanism affordable, and
//!   that is a property of where the neighbouring routes hand over.

use crate::constants::f32::EULER_GAMMA;
use core::f32::consts::{PI, TAU};

/// At or above this argument `J` uses the Hankel expansion; below it, the series or the aliased
/// trapezoid.
///
/// **`7.0`, where the `f64` peer uses `18.0`.** The expansion is divergent and truncated at its
/// least term, so its floor is the size of that term — a function of `x` alone, not of `ε`. It
/// therefore becomes usable at a *smaller* argument in a coarser format, and the handover moves
/// down to meet it.
///
/// Measured as a fraction of the envelope `√(2/πx)`, which is the right scale here because `J`
/// has infinitely many zeros and an ulp count collapses at every one of them:
///
/// | `x` | 3 | 4 | 5 | 6 | **7** | 8 | 10 | 12 | 18 |
/// |---|---|---|---|---|---|---|---|---|---|
/// | Hankel | `5.1e-4` | `4.1e-5` | `2.0e-6` | `1.0e-6` | **`1.5e-7`** | `1.9e-7` | `1.4e-8` | `3.1e-7` | `3.2e-7` |
/// | aliased | `1.7e-7` | `3.2e-7` | `3.6e-7` | `4.9e-8` | **`2.5e-7`** | `5.5e-7` | `3.1e-7` | `9.0e-7` | `5.4e-8` |
///
/// They cross just under `7`. Confirmed end to end: the worst error of the finished `J` over
/// `x ∈ (0, 300]` is `9.1e-7` of the envelope with the handover at `6`, `5.4e-7` at `7`, and
/// `7.3e-7` at `8`.
///
/// Moving it down from `18` also saves the quadrature's most expensive band, since its node
/// count grows like `2x` — though the saving is smaller than it looks, because [`ALIAS_EPS`]
/// already cut the count: 16–50 nodes across `x ∈ [3, 25]` against the `f64` constants' 26–68,
/// a ratio of about `1.4` rather than the `3` the `K` quadrature gave.
const HANKEL_MIN: f32 = 7.0;

/// Relative size below which an aliased term is taken to vanish.
///
/// `1e-8` against the `f64` peer's `1e-19`. This sets the node count directly — the walk stops
/// at the smallest `m` with `(x/2)^m/m! < ALIAS_EPS` — so it is the main lever on the
/// quadrature's cost, and asking for eleven more orders than the format holds simply pays for
/// nodes whose contribution is never representable.
const ALIAS_EPS: f32 = 1e-8;

/// Cap on the aliasing node count, which grows like `2x`; only reached above [`HANKEL_MIN`],
/// where this route is not used.
const ALIAS_MAX_NODES: u32 = 64;

/// Iteration cap for the power series. The `f64` peer allows 1,000.
const MAX_TERMS: u32 = 32;

/// Relative size at which a series term stops contributing.
const TERM_EPS: f32 = 1e-8;

/// Term cap for the Hankel expansion; the minimum-term check normally stops it far sooner.
const HANKEL_TERMS: u32 = 32;

/// At or below this argument `J₀`/`J₁` use the power series rather than the quadrature.
///
/// **Unchanged from the `f64` peer's `2.0`**, because the reason for it is not an accuracy
/// trade. The quadrature's summand is `cos(x·sin θ − nθ)`, and for small `x` the `x·sin θ` part
/// is swallowed by `nθ`: at a subnormal `x`, `x − π/2` rounds to exactly `−π/2`, so the sum
/// returns the rounding of `cos(−π/2)` instead of anything resembling `J₁(x) ≈ x/2`. The
/// absolute error stays tiny while every digit of the relative answer is gone — and `J₁(x) ~ x/2`
/// is an exactly-conditioned limit that ought to come back exactly. That failure is about the
/// *representation* of the argument, so it happens in both formats and at the same place.
const J_SERIES_MAX: f32 = 2.0;

mod first;
mod second;

// Re-exported so the family keeps one path per function: every caller reached
// `kernels::f32::bessel::ordinary::j0` before the split and still does.
pub(in crate::kernels::f32) use first::hankel;
pub(crate) use first::{j0, j1};
pub(crate) use second::{y0, y1};

#[cfg(test)]
mod tests;
