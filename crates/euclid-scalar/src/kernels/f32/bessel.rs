//! The `f32` peer of [`crate::kernels::bessel`], mirroring its split by kind.
//!
//! All four kinds are native `f32` implementations, and the Airy family now reaches `Ai` and
//! `Bi` through them — `K_{1/3}`, `K_{2/3}` and the scaled `I` asymptotic — so these are load
//! bearing for more than their own surface. The dividing line used to be whether the method has
//! a precision-dependent error floor; see the [module header](super) for how that turned out.
//!
//! `J` crossed it the same way `K` did, plus one genuine fix: its Hankel phase had to stop being
//! formed as a single `f32`, which had been costing `ulp(x)` of *angle*. `Y` crossed it last and
//! was the only real question, since a march's error is not a tunable — the answer is that its
//! step count is bounded by where the neighbouring routes hand over, and `J`'s handover moving
//! down to `7` shortened `Y`'s march with it. See [`ordinary`].
//!
//! `K` crossed that line by rederivation rather than by a new algorithm: its quadrature's three
//! constants are all set by equating an analytic error bound to a target accuracy, so retuning
//! them to `f32` is arithmetic, and it cut the node count by about three. Its series/quadrature
//! join had to move *down*, which is the opposite of every other threshold in this tree.

pub(crate) mod modified_first;
pub(crate) mod modified_second;
pub(crate) mod ordinary;

pub(crate) use modified_first::{i0, i0e, i1, i1_over_i0, i1e, ln_i0};
pub(crate) use modified_second::{k0, k0e, k1, k1e};
pub(crate) use ordinary::{j0, j1, y0, y1};
