//! Native single-precision special functions — the `f32` peer of the `f64` reference
//! implementations in [`super`]. Same algorithms (Lanczos gamma, Numerical-Recipes continued
//! fractions, Giles inverse-erf, A&S Bessel), evaluated in `f32` throughout so the `f32`
//! [`Real`](crate::Real) surface never round-trips through `f64`. The
//! continued-fraction tolerances are loosened to the `f32` noise floor (~`1e-6`), and the
//! module mirrors [`super`]'s implementation layout.
//!
//! Native `f32` is marginally *less* accurate than evaluating in `f64` and narrowing (one
//! rounding versus two), typically within an ULP; the per-module tests cross-check against
//! the `f64` reference to bound that gap and guard against the two paths drifting apart.

pub(crate) mod bessel;
pub(crate) mod beta;
pub(crate) mod erf;
pub(crate) mod gamma;
pub(crate) mod normal;
