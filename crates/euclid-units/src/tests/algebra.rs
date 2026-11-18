//! Value-level tests of the cross-dimensional algebra, over the toy carriers in
//! [`toy_units`](super::toy_units).
//!
//! `f64: Real`, so these exercise the same generic code paths any other real carrier would
//! take, and `<f64 as Compare>::Mask = bool`, so the mask-returning predicates collapse to
//! plain `bool` here.

#![allow(clippy::float_cmp)]

// Glob imports: the carrier set is large and each test touches a different slice of it, so
// naming them individually would mean a long, churn-prone list per file.
use super::toy_units::*;
use crate::*;

// --- tests ---------------------------------------------------------------------------------

mod dimensions;
mod physics;
mod vocabulary;
