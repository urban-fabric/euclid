//! # euclid-numeric
//!
//! Dependency-free scalar numeric contracts — the base of the euclid workspace, and the numeric
//! vocabulary the Urban Fabric projects share.
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
//! use euclid_numeric::Real;
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
//! | this file | the crate docs, and the five tier traits below | yes |
//! | `real.rs`, `boolean.rs`, `integer.rs`, `natural.rs`, `complex.rs`, `interpolate.rs`, `control_flow.rs` | one trait definition each | via the root re-exports |
//! | [`constants`] | the constant catalogue | yes |
//! | `impls/` | the `f64`/`f32`/`i*`/`u*`/`bool` `impl` blocks | no — exports nothing |
//! | `kernels/` | the concrete scalar algorithms those impls forward to | no |
//! | `tests/` | the trait-surface test suites, organized by question asked | no |
//!
//! Two consequences worth knowing before navigating it.
//!
//! **The contracts are about 6% of the lines.** `kernels/` and `tests/` are four-fifths of the
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
//! ## Trait Definitions in Two Places
//!
//! [`Element`], [`Semiring`], [`Ring`], [`Arithmetic`] and [`NormedDivisionAlgebra`] are
//! defined in this file; the other nine traits each get their own module. The split is not
//! arbitrary: those five are the *tier skeleton*, four of them blanket-implemented one-liners
//! over the operator bounds, and they have to sit below everything that refers to them. Keeping
//! them beside the tier table that explains them is more useful than a `tiers.rs` nobody would
//! open. Anything with real method bodies gets a file.

use core::ops::{Add, Div, Mul, Neg, Sub};

/// The carrier's canonical constants `0` and `1`.
///
/// Every element kind implements it — booleans (where `0`/`1` are `false`/`true`), naturals,
/// integers, and the real/complex/quaternion division algebras — so one bound names a
/// zero/one for all of them. It sits *below* the operators so booleans, which carry `0`/`1`
/// but not `+`/`*`, can share the identities without being a [`Semiring`].
///
/// Both are required rather than defaulted through a host-`f64` lift, so a carrier defines its
/// own exact representation. Arbitrary real lifting stays on [`Real::scalar`].
pub trait Element: Clone {
    /// The additive identity — `0` for numbers, `false` for booleans.
    fn zero() -> Self;
    /// The multiplicative identity — `1` for numbers, `true` for booleans.
    fn one() -> Self;
}

/// `+` and `*` over an [`Element`]'s identities, with subtraction and negation withheld.
/// Naturals sit at this tier.
pub trait Semiring: Element + Add<Output = Self> + Mul<Output = Self> {}
impl<T> Semiring for T where T: Element + Add<Output = T> + Mul<Output = T> {}

/// A [`Semiring`] plus `-` and negation, with division withheld. Signed integers sit at this
/// tier; unsigned carriers stop at [`Semiring`], having no `Neg`.
pub trait Ring: Semiring + Sub<Output = Self> + Neg<Output = Self> {}
impl<T> Ring for T where T: Semiring + Sub<Output = T> + Neg<Output = T> {}

/// A [`Ring`] plus division — the operator bound behind `+ - * /`. Constant lifting lives on
/// [`Real`].
pub trait Arithmetic: Ring + Div<Output = Self> {}

/// A normed division algebra over ℝ. By Frobenius/Hurwitz the associative ones are exactly
/// ℝ, ℂ, ℍ, so one kernel written against this trait runs over all three.
///
/// It names the real field that magnitudes and norms live in, the `conjugate` every Hermitian
/// formula needs, and real-axis embedding through [`from_real`](Self::from_real).
pub trait NormedDivisionAlgebra: Arithmetic {
    /// The real field of magnitudes / norms — `Self` for a real scalar, the real
    /// component for a complex one.
    type RealPart: Real;

    /// The complex conjugate; the identity for a real scalar.
    fn conjugate(&self) -> Self;
    /// The squared magnitude (`x²`, or `re² + im²`) — no `sqrt`.
    fn norm_sqr(&self) -> Self::RealPart;
    /// The real part (the value itself for a real scalar; `re` for complex).
    fn real_part(&self) -> Self::RealPart;
    /// Embed a real value (`re + 0i` for complex; the identity for a real scalar).
    fn from_real(re: Self::RealPart) -> Self;
    /// The magnitude `|x|`.
    fn magnitude(&self) -> Self::RealPart;
}

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
pub mod compensated;
pub mod constants;
pub mod interval;

// Every trait lives in a private module and is re-exported at the root below, so each has
// exactly one public path. `control_flow` and `interpolate` were `pub mod` for a while, which
// gave `ControlFlow` and `Interpolate` two paths apiece and made them look like namespaced
// sub-features rather than peers of `Real`. They are not: each file holds one trait and
// nothing else.
mod boolean;
mod complex;
mod control_flow;
mod impls;
mod integer;
mod interpolate;
mod natural;
mod real;
#[cfg(test)]
mod tests;

/// The concrete scalar algorithms the public trait methods forward to — special functions,
/// format primitives, and host combinatorics. Crate-internal; see [`kernels`] for the map.
pub(crate) mod kernels;

pub use boolean::{Boolean, Compare};
pub use complex::{Complex, Quaternion};
pub use control_flow::ControlFlow;
pub use integer::Integer;
pub use interpolate::Interpolate;
pub use natural::Natural;
pub use real::Real;
