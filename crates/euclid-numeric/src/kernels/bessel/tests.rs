//! Unit tests for the Bessel family, split by kind.
//!
//! Separated from the implementations because each pairing exceeded this crate's 400-line file
//! budget. These are children of [`super`], so `use super::super::*` reaches the private
//! series, quadratures and crossover constants that the tests need to compare against each
//! other.
//!
//! Tests that span *two* families — the Wronskians, the recurrences — live in
//! [`crate::tests::identities::bessel_family`] instead, where they can use the public trait
//! surface.
//!
//! The crossover probe these files lean on, `no_step_across`, now lives in
//! [`euclid_utils`] — it is not specific to this family, and the Airy branches need it too.

mod modified_first;
mod modified_second;
mod ordinary;
