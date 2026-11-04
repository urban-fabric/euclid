//! Concrete carrier implementations of the numeric traits, kept apart from their definitions.
//!
//! The trait *definitions* live in the [crate root](crate) and its sibling modules; this folder
//! holds the `f64`/`f32`/`i*`/`u*`/`bool` `impl` blocks, and nothing else. **These modules
//! export nothing** — they exist entirely for their `impl` blocks' side effects, which is why
//! there is no `pub use` below and no `impls::` path worth naming.
//!
//! The split is per carrier for [`Real`](crate::Real), much the largest surface at ~160
//! methods: `real_f64` and `real_f32`, each with a small `_support` module beside it for the
//! non-`Real` `impl` blocks and any helper a method body needs. The smaller tiers are grouped
//! instead — `integer_impl`, `natural_impl`, `boolean_impl` — since a file per carrier there
//! would be a dozen files of a few lines each.
//!
//! # What these files are *not*
//!
//! They are not where the numerics live. Almost every `Real` special-function body here is a
//! single forwarding expression into [`crate::kernels`]:
//!
//! ```ignore
//! fn erf(self) -> Self {
//!     crate::kernels::erf::erf(self)
//! }
//! ```
//!
//! That indirection is the point. It keeps the algorithm addressable and testable on its own —
//! `kernels` carries its own unit tests against its own private thresholds — while leaving this
//! layer a flat, auditable list of which carrier routes which method where. A method
//! implemented *inline* here is therefore a signal: it means the operation is a host intrinsic
//! or a short composite with no algorithm to speak of.

mod boolean_impl;
mod integer_impl;
mod natural_impl;
mod real_f32;
mod real_f32_support;
mod real_f64;
mod real_f64_support;
