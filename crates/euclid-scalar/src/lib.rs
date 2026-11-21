//! # euclid-scalar
//!
//! Scalar contracts, host traits, primitive implementations and numerical kernels — the
//! scalar foundation of the euclid workspace. Depends on `euclid-algebra`, with no third-party
//! production dependencies.
//!
//! A formula written against these traits can run on ordinary `f64`/`f32` values and on
//! consumer-defined expression, graph, tensor, or unit-aware carriers. This crate defines the
//! operations a formula may use; it does not define storage layout, execution runtime, array
//! allocation, or a random-number engine.
//!
//! `f64`/`f32` implement the real and interpolation traits; `i32`/`i64`/`i128` implement
//! [`Integer`]; `u32`/`u64`/`u128` implement [`Natural`]; and `bool` implements [`Boolean`].
//!
//! ## Usage Rules
//!
//! - Choose the narrowest trait bound that contains the operations a formula needs.
//! - Treat generic values as `Clone`, not `Copy`.
//! - Lift host literals with methods such as [`Real::scalar`] and [`Real::scalar_like`].
//! - Comparisons return masks, consumed by [`Compare::select`] or [`Compare::if_then_else`].
//!   Nothing here returns a scalar `bool` from a generic comparison.
//!
//! ## The Workspace
//!
//! The sibling crates build on this one and never on each other:
//!
//! - `euclid-tensor` — the rank-graded `Tensor` contract (a vector is a rank-1 tensor, a matrix
//!   a rank-2 tensor) and backend-neutral dtype/spec metadata.
//! - `euclid-units` — physical-dimension contracts over any [`Real`] carrier.
//! - `euclid-random` — explicit-key random sampler contracts.
//!
//! ## Example
//!
//! ```
//! use euclid_scalar::Real;
//!
//! // Written once, against the trait — no mention of any concrete number type.
//! fn standard_normal_logpdf<T: Real>(x: T) -> T {
//!     let x2 = x.square();
//!     -(T::half() * x2) - T::half_ln_tau()
//! }
//!
//! // The same function, two different carriers.
//! let wide: f64 = standard_normal_logpdf(0.0_f64);
//! let narrow: f32 = standard_normal_logpdf(0.0_f32);
//!
//! let expected = -0.5 * (2.0 * std::f64::consts::PI).ln();
//! assert!((wide - expected).abs() < 1e-12);
//! assert!((narrow as f64 - expected).abs() < 1e-6);
//! ```
//!
//! Nothing in `standard_normal_logpdf` names `f64` or `f32`, so a carrier the consumer
//! defines — an expression handle, a graph node, a tensor element, a unit-aware wrapper —
//! is accepted on the same bound without editing the formula.
//!
//! ## Trait Tiers
//!
//! The traits are grouped by the structure they require:
//!
//! - [`Element`]: the canonical constants `0` and `1` every carrier defines directly — the
//!   tier booleans, naturals, integers, and the real/complex/quaternion division algebras all
//!   unify into, so any of them can be a `Tensor` element.
//! - [`Semiring`] and [`Natural`]: `+`, `*`, identities, natural-number lifting,
//!   Euclidean division, and count combinatorics.
//! - [`Ring`] and [`Integer`]: subtraction, negation, signed integer lifting,
//!   Euclidean division, and conversion to real carriers.
//! - [`Arithmetic`]: division on top of the ring operations.
//! - [`Real`]: real analysis, lifted constants, square/powers, rounding/sign
//!   operations, comparison-free selectors, masks, special functions, and stable log-domain
//!   helpers.
//! - [`NormedDivisionAlgebra`], [`Complex`], and [`Quaternion`]: conjugation,
//!   magnitude, real part, and complex/quaternionic construction.
//! - [`Boolean`] and [`Compare`]: mask algebra, comparisons, `select`, and carrier-level
//!   control flow.
//! - [`Interpolate`]: scalar-parameterized interpolation (`lerp`/`midpoint`), kept opt-in
//!   rather than blanket-implemented over the algebra tiers.
//! - [`ControlFlow`]: the explicit graph conditional/loop contract for carriers that can
//!   represent them.
//!
//! ## Host scalar traits
//!
//! [`HostReal`] and [`HostComplex`] add eager `Copy` requirements to [`Real`] and
//! [`Complex`]. They fuse the analysis surface with the `euclid-algebra` field traits,
//! associated constants and host predicates. Bind `Real` for backend-neutral formulas and
//! `HostReal` when a kernel requires ordinary scalar control flow and values.
//!
//! ## Constants
//!
//! [`constants`] holds the mathematical constants `core` does not provide — the
//! Euler–Mascheroni constant, the golden ratio, Apéry's and Catalan's constants, the omega and
//! lemniscate constants, and others. A carrier-generic formula lifts the common ones through
//! [`Real`] (`T::pi()`, `T::euler_gamma()`); concrete `f64`/`f32` code reaches into
//! [`constants`] for the full catalogue. Every value there is verified against an identity
//! rather than trusted as transcribed digits.
//!
//! ## Where Things Live
//!
//! The source is laid out by **role**, not by function family, because the distinction this
//! crate exists to maintain is the one between a contract and an implementation of it:
//!
//! | | what it holds | public? |
//! |---|---|---|
//! | this file | crate documentation and the public re-exports | yes |
//! | `traits.rs`, `traits/` | backend-neutral scalar contracts | via root re-exports |
//! | `host.rs`, `host/` | eager `HostReal` / `HostComplex` refinements | via root re-exports |
//! | [`constants`] | the constant catalogue | yes |
//! | `impls/` | the `f64`/`f32`/`i*`/`u*`/`bool` `impl` blocks | no — exports nothing |
//! | `kernels/` | the concrete scalar algorithms those impls forward to | no |
//! | `tests/` | the trait-surface test suites, organized by question asked | no |
//!
//! Two consequences worth knowing before navigating it.
//!
//! **The contracts are a small part of the source.** `kernels/` and `tests/` are four-fifths of the
//! crate between them, and that ratio is deliberate: a trait that promises `erf` is accurate to
//! `1e-15` is only worth as much as the evidence behind it. Someone reading for the *contract*
//! should stay in the files above and never need to open `kernels/`.
//!
//! **One function is therefore spread across several files.** `erf` is declared on [`Real`],
//! forwarded in `impls/real_f64.rs`, computed in `kernels/erf/`, and tested from four
//! directions in `tests/`. That is the cost of the role-based split, paid knowingly — the
//! alternative, a directory per function holding its own trait method, would scatter the trait
//! itself and make the contract impossible to read in one sitting.
//!
// Three public modules, and each is here for the same reason: it is a genuine *namespace*
// rather than a single item, so flattening it into the crate root would read worse.
// `constants::f64::CATALAN`, `compensated::sum` and `interval::contains` all say what they
// are; a bare `sum` or `contains` at the root would not.

// Plain comments, not `///`: a doc comment here would be concatenated with the module
// file's own `//!` header and the whole block resolved in *this* module's scope, which
// silently breaks every link that referred to an item inside the submodule. Each module
// documents itself in its own file.
//
//   compensated  accurate reductions: compensated summation and dot products
//   constants    the constant catalogue, `constants::f64::CATALAN` and friends
//   interval     one-dimensional closed-interval predicates over any `Real`
#[cfg(any(feature = "conformance", test))]
pub mod conformance;

pub mod compensated;
pub mod constants;
pub mod interval;

// Trait and host modules have one public path through the root re-exports.
mod host;
mod impls;
mod math;
#[cfg(test)]
mod tests;
mod traits;

pub(crate) mod kernels;

pub use host::{HostComplex, HostReal};
pub use traits::{
    Arithmetic, Boolean, Compare, Complex, ControlFlow, Element, Integer, Interpolate, Natural,
    NormedDivisionAlgebra, Quaternion, Real, Ring, Semiring,
};

// The README's examples, compiled.
//
// `cfg(doctest)` means this item exists only while rustdoc is collecting doctests: it is absent
// from every real build, adds nothing to the public surface, and never appears in the rendered
// docs. What it buys is that `cargo test --doc` compiles the code blocks in README.md itself.
//
// A crate whose deliverable is its trait surface has a README whose example *is* part of that
// surface, and a README example that no longer compiles is worse than none — it documents a
// signature the crate no longer has. Two crates used to check this by hand-copying the example
// into `tests/readme_check.rs`, which compiles the copy and lets the original drift. This checks
// the original.
//
// Illustrative fragments in the README are marked ```rust,ignore, which is the honest marking:
// they reference names that are deliberately undefined.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeExamples;
