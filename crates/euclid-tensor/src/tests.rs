//! Contract tests over two reference carriers.
//!
//! The crate ships only traits, so these answer the question a contracts crate has to answer:
//! *is this surface implementable?* The two carriers divide the work, and the split is
//! deliberate:
//!
//! - [`toy_tensor`] is rank-0 and stops at `Ring`, which is enough to prove the arithmetic
//!   surface is implementable without the 98-method real surface. It is **not** a complete
//!   `Tensor` — its `reshape` ignores the shape it is given — so it does not satisfy
//!   [`conformance::shape_laws`](crate::conformance::shape_laws).
//! - [`dense`] is rank-n with the full real surface, and implements every capability extension
//!   in [`dense_capabilities`]. It is the carrier the conformance suites run against.
//!
//! | module | what it establishes |
//! |--------|---------------------|
//! | [`constructors`] | the constructor and structural surface is reachable |
//! | [`boolean_tensor`] | a boolean-element tensor reaches the mask tier |
//! | [`control_flow`] | the conditional/loop contract is implementable |
//! | [`capabilities`] | the eight capability extensions behave, over `dense` |
//! | [`conformance_suites`] | `dense` satisfies the public [`conformance`](crate::conformance) laws |
//!
//! Note that `dense` is a **data structure**, which is why it lives here rather than being
//! shipped: euclid implements over universal datatypes and ships no structure, so a reference
//! tensor is scaffolding rather than a deliverable.

mod boolean_tensor;
mod capabilities;
mod conformance_suites;
mod constructors;
mod control_flow;
mod dense;
mod dense_capabilities;
mod toy_tensor;

use toy_tensor::ToyTensor;
