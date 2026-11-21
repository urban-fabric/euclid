//! Native single-precision special functions — the `f32` peer of the `f64` reference
//! implementations in [`super`]. Same algorithms (Lanczos gamma, Numerical-Recipes continued
//! fractions, Giles inverse-erf, peak-factored Bessel `I`), evaluated in `f32` throughout so
//! those parts of the `f32` [`Real`](crate::Real) surface never round-trip through `f64`. The
//! continued-fraction tolerances are loosened to the `f32` noise floor (~`1e-6`), and the
//! module follows [`super`]'s layout one module per function family. The one place it does not
//! mirror [`super`] exactly is [`bessel::ordinary`]:
//! `J` and `Y` get a file each there, but
//! share one here, because both merely narrow the `f64` march and neither carries an algorithm
//! of its own to separate.
//!
//! Native `f32` is marginally *less* accurate than evaluating in `f64` and narrowing (one
//! rounding versus two), typically within an ULP; the per-module tests cross-check against
//! the `f64` reference to bound that gap and guard against the two paths drifting apart.
//!
//! # There is no longer an exception, and the list is worth keeping as a record
//!
//! This module header used to carry one: [`bessel::ordinary`] (`J`, `Y`),
//! [`bessel::modified_second`] (`K`), [`integrals`] (`Si`, `Ci`, `E₁`, `Ei`),
//! [`transcendental`] (`W`, `ζ`) and [`airy`] evaluated in `f64` and narrowed, on the grounds
//! that each reaches its values through "a cancelling series, a divergent asymptotic expansion,
//! or an outward march", every one of which has an error floor set by the *working* precision —
//! so a native `f32` port would be less accurate rather than more.
//!
//! Every one of those five has since been measured, and every one has come off the list. The
//! reasons differ, and the differences are the useful part:
//!
//! - **`integrals`** and **`beta_reg`** (which carried the same argument in its own words): the
//!   cited catastrophe is real and sits on a branch the function does not take at those
//!   arguments. `Si`'s series does reach `4e20` intermediates — below `x = 4`, where it is not
//!   used; `beta_reg`'s prefactor does lose everything — above `min(a,b) = 1e3`, where a retuned
//!   handover does not send it.
//! - **`transcendental`**: no precision-dependent floor at all. `W` is a Halley iteration on a
//!   well-conditioned equation and `ζ` is a geometric series, an Euler–Maclaurin sum and a
//!   reflection.
//! - **`K`** and **`J`**: the floor is a *tunable*. Both reach their middle range by quadrature,
//!   whose cost is set by the accuracy asked of it, so retuning to this format's target made them
//!   cheaper — `K`'s node count fell by about three.
//! - **`Y`**: the floor is real and irreducible — a march accumulates `ε` per step — but the step
//!   count is bounded by where the *neighbouring* routes hand over, not by the march. Moving
//!   `J`'s Hankel handover shortened `Y`'s march with it.
//! - **`airy`**: all three mechanisms at once, and the strongest claim of the five. Its series
//!   really does leave `Ai(6)` with no correct digits in `f32` — measured at `2.4e1` relative.
//!   `AI_SERIES_MAX` is `1`, so `Ai(6)` comes from the `K` quadrature instead.
//!
//! The pattern across all five: **a mechanism's error floor and a function's error floor are
//! different things**, and only the second one decides. Measuring a series at an argument the
//! series is not used at produces a true statement and the wrong conclusion. What survived the
//! exercise is one genuine irreducible floor — the Airy phase `ζ = ⅔|x|^{3/2}`, which `f32` can
//! hold only to `ulp(ζ)` and which therefore degrades like `|x|^{3/2}ε` — and it is documented
//! where it lives rather than used to excuse a whole family.

pub(crate) mod airy;
pub(crate) mod bessel;
pub(crate) mod beta;
pub(crate) mod erf;
pub(crate) mod gamma;
pub(crate) mod integrals;
pub(crate) mod normal;
pub(crate) mod polygamma;
pub(crate) mod quantile;
pub(crate) mod repr;
pub(crate) mod transcendental;
