//! Native single-precision special functions — the `f32` peer of the `f64` reference
//! implementations in [`super`]. Same algorithms (Lanczos gamma, Numerical-Recipes continued
//! fractions, Giles inverse-erf, peak-factored Bessel `I`), evaluated in `f32` throughout so
//! those parts of the `f32` [`Real`](crate::Real) surface never round-trip through `f64`. The
//! continued-fraction tolerances are loosened to the `f32` noise floor (~`1e-6`), and the
//! module follows [`super`]'s layout one module per function family. The one place it does not
//! mirror [`super`] exactly is [`bessel::ordinary`](crate::kernels::f32::bessel::ordinary):
//! `J` and `Y` get a file each there, but
//! share one here, because both merely narrow the `f64` march and neither carries an algorithm
//! of its own to separate.
//!
//! Native `f32` is marginally *less* accurate than evaluating in `f64` and narrowing (one
//! rounding versus two), typically within an ULP; the per-module tests cross-check against
//! the `f64` reference to bound that gap and guard against the two paths drifting apart.
//!
//! # The exception: the classical special functions
//!
//! [`bessel::ordinary`](crate::kernels::f32::bessel::ordinary) (`J`, `Y`),
//! [`bessel::modified_second`](crate::kernels::f32::bessel::modified_second) (`K`),
//! [`integrals`](crate::kernels::f32::integrals) (`Si`, `Ci`, `E₁`, `Ei`) and
//! [`transcendental`](crate::kernels::f32::transcendental) (`W`, `ζ`) **do** evaluate in `f64` and
//! narrow. This is a deliberate departure from the rule above, for two reasons:
//!
//! - **It is more accurate, not less.** The usual "one rounding versus two" argument assumes
//!   the native path is otherwise exact. It is not here: every one of these functions is
//!   reached through a cancelling series, a divergent asymptotic expansion, or an outward
//!   march, and each of those mechanisms has an error floor set by the working precision. In
//!   `f32` those floors are around `1e-7` *relative to the largest intermediate*, which for
//!   `Si(45)` means no correct digits at all. Computing in `f64` and rounding once gives a
//!   correctly-rounded `f32` instead.
//! - **It halves the surface where such a bug can hide.** A native `f32` port would need its
//!   own crossover thresholds, node counts and truncation points, all retuned and all
//!   separately verifiable. That is a second full implementation of the hardest numerics in
//!   the crate, carrying no benefit per the point above.
//!
//! The rule still holds for everything else, and for a reason worth keeping: those functions
//! are cheap, well-conditioned in `f32`, and a native path leaves the surface usable on a
//! carrier with no `f64` at all.

pub(crate) mod airy;
pub(crate) mod bessel;
pub(crate) mod beta;
pub(crate) mod gamma;
pub(crate) mod integrals;
pub(crate) mod normal;
pub(crate) mod polygamma;
pub(crate) mod quantile;
pub(crate) mod repr;
pub(crate) mod transcendental;

/// The single-precision `erf` family, which lives beside its `f64` peer in
/// [`crate::kernels::erf`] rather than in this tree.
///
/// Re-exported under the name it would have had here, so `kernels::f32::erf::erfc` resolves
/// whichever directory the file sits in and the call sites in [`crate::impls`] stay uniform
/// across all twelve families. As more of them are folded this list shrinks and the
/// re-exports below grow; when the last one moves, this module becomes re-exports only and
/// can go away.
pub(crate) use super::erf::f32 as erf;
