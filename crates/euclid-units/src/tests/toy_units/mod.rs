//! The toy unit-carrier family: one `f64`-backed carrier per dimension, plus every
//! dimension-trait implementation for them.
//!
//! euclid ships no concrete unit type, so these carriers play the role a consumer crate's
//! units would. Implementing *every* dimension trait for one coherent family is itself the
//! test: the compiler only confirms that a consistent set of implementors can exist once they
//! are actually written, so the impl modules below are the proof that the cross-dimensional
//! algebra web closes.
//!
//! Split to mirror the crate's own module layout, so `tests/toy_units/mechanics.rs` holds the
//! impls for the traits in `src/mechanics.rs`. The tests that exercise all of it live in
//! [`algebra`](super::algebra).

#![allow(clippy::float_cmp)]

mod carriers;
mod chemistry;
mod electromagnetism;
mod geometry;
mod kinematics;
mod mechanics;
mod photometry;
mod radiology;
mod thermal;

pub(super) use carriers::*;
