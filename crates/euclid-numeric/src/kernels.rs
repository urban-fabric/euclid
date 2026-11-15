//! The concrete scalar algorithms behind the trait surface — crate-internal throughout.
//!
//! Everything a [`Real`] or [`Natural`] method needs to *compute* an answer lives here. Nothing
//! in this module is public: the contract is the trait layer, and [`crate::impls`] is the only
//! caller, forwarding each method to the routine below that implements it. A consumer never
//! names a path into this tree.
//!
//! A note for anyone editing this header: intra-doc links in a module's own `//!` block resolve
//! in the **parent's** scope, not this module's. That is why [`Real`] needs no path while the
//! children below are spelled absolutely — a bare `[bessel]` here would be looked up at the
//! crate root and silently fail, or worse, find something else.
//!
//! This was called `special` until the contents outgrew the name. It holds three kinds of
//! thing, and only the first are special functions:
//!
//! | | modules | what they are |
//! |---|---|---|
//! | **Special functions** | [`airy`](crate::kernels::airy), [`bessel`](crate::kernels::bessel), [`beta`](crate::kernels::beta), [`erf`](crate::kernels::erf), [`gamma`](crate::kernels::gamma), [`integrals`](crate::kernels::integrals), [`normal`](crate::kernels::normal), [`polygamma`](crate::kernels::polygamma), [`quantile`](crate::kernels::quantile), [`transcendental`](crate::kernels::transcendental) | the named transcendentals of classical analysis, plus the inverse CDFs built on them |
//! | **Format primitives** | [`repr`](crate::kernels::repr) | `frexp`, `ldexp`, `next_after`, `ulp` — bit-level, exact, and not approximations of anything |
//! | **Host combinatorics** | [`host`](crate::kernels::host) | integer `factorial`/`binomial` returning a float, for the `Natural` impls rather than the `Real` surface |
//!
//! # Precision
//!
//! Each family is written for `f64`. The `f32` peers live in [`f32`](mod@crate::kernels::f32), which mostly mirrors this
//! layout — see its header for which functions are evaluated natively in single precision and
//! which deliberately narrow the `f64` result, and why that distinction is not a shortcut.
//!
//! [`erf`](crate::kernels::erf) is the exception to the mirror: its two precisions sit together in one directory, as
//! `erf/mod.rs` and `erf/f32.rs`, with [`f32`](mod@crate::kernels::f32) re-exporting the latter so the call paths stay
//! uniform. That is the shape the rest of the tree is intended to take; it is being migrated
//! one family at a time rather than in a single sweep.
//!
//! # Where the tests are
//!
//! Split by what they can reach. A test that needs a private series, quadrature or crossover
//! constant is a `#[cfg(test)]` child of the module it tests — `bessel::tests` and
//! `airy::tests` are the fullest examples. Everything reachable through the trait lives in
//! the `crate::tests` tree instead, where it keeps passing when the algorithm here is replaced.

pub(crate) mod airy;
pub(crate) mod bessel;
pub(crate) mod beta;
pub(crate) mod erf;
pub(crate) mod gamma;
pub(crate) mod host;
pub(crate) mod integrals;
pub(crate) mod normal;
pub(crate) mod polygamma;
pub(crate) mod quantile;
pub(crate) mod repr;
pub(crate) mod transcendental;

/// Single-precision peers of the elementwise functions — same names, `f32` throughout.
pub(crate) mod f32;
