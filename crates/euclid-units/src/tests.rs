//! Coverage over the `f64` carrier with toy unit carriers that implement the dimension
//! contracts. `f64: Real`, so this exercises the same generic code paths used by other real
//! carriers, and `<f64 as Compare>::Mask = bool`, so the mask-returning predicates collapse to
//! plain `bool` here. euclid ships no concrete unit type, so the carriers below play the role a
//! consumer crate's units would.
//!
//! Beyond spot-checking arithmetic, implementing *every* dimension trait for this family of
//! carriers is the real proof that the cross-dimensional algebra web is internally consistent:
//! the compiler only verifies that a coherent set of implementors can exist once they are
//! actually written, so the impl blocks below are themselves the test.

//!
//! ## Layout
//!
//! - [`toy_units`] — the toy carrier family and its dimension-trait implementations.
//! - [`algebra`] — value-level tests of the cross-dimensional algebra over those carriers.
//! - [`dim`] — the type-level `dim` markers and the [`Dimensioned`](crate::Dimensioned)
//!   bridge. The *compile-time* half of that contract is covered by `compile_fail` doctests
//!   on [`Dimensioned`](crate::Dimensioned) itself, since a test that must not compile cannot
//!   be expressed as a running test.

mod algebra;
mod dim;
mod toy_units;
