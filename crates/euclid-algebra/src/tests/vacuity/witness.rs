//! Witness-layer rejection controls.
//!
//! The witness laws carry more weight than their element-layer peers, and the crate says so:
//! "a witness carries no property markers — the marker tower belongs to the element layer — so
//! these laws are the *only* thing standing between a witness and a wrong algebra". A vacuous
//! witness law therefore has nothing behind it, which is why both suites being replaceable with
//! an empty body mattered enough to fix.

use super::*;
use crate::conformance::{witness_dioid_laws, witness_semiring_laws};
use crate::witness::{DioidWitness, MinPlus, SemiringWitness};

// ===== A witness whose multiplication has no identity ======================

/// `⊗` is `ab + 1`, so `a ⊗ 1 = a + 1`.
///
/// The witness-layer twin of `semiring::NoMulIdentity`, and broken the same way — off by one in
/// the operation rather than in the constant, so `ONE` still *looks* right at the declaration.
struct OffByOneProduct;

impl SemiringWitness for OffByOneProduct {
    type Domain = i64;
    const ZERO: i64 = 0;
    const ONE: i64 = 1;
    fn add(a: i64, b: i64) -> i64 {
        a.wrapping_add(b)
    }
    fn mul(a: i64, b: i64) -> i64 {
        a.wrapping_mul(b).wrapping_add(1)
    }
}

#[test]
fn the_witness_semiring_laws_reject_a_broken_multiplicative_identity() {
    let sample = [0i64, 1, 2, -3];
    let message = rejects(|| witness_semiring_laws::<OffByOneProduct>(&sample, "OffByOneProduct"))
        .expect("witness_semiring_laws must reject a witness where a ⊗ 1 != a");
    assert!(
        message.contains("⊗ 1 = a") || message.contains("1 ⊗ a") || message.contains("⊗ 0 = 0"),
        "the failure must name the axiom that broke: {message}"
    );
}

#[test]
fn the_witness_semiring_laws_accept_a_correct_witness() {
    // Or every rejection above is satisfiable by a suite that always panics.
    let sample = [MinPlus::ZERO, MinPlus::ONE, 1, 7, -3];
    assert!(
        rejects(|| witness_semiring_laws::<MinPlus>(&sample, "MinPlus")).is_none(),
        "witness_semiring_laws must accept min-plus"
    );
}

// ===== A witness that claims idempotence it does not have ==================

/// Ordinary integer arithmetic, claiming [`DioidWitness`].
///
/// Every semiring axiom holds — this is `Arithmetic` by another name — and the single false
/// statement is the marker. That is what separates [`witness_dioid_laws`] from the
/// [`witness_semiring_laws`] it delegates to: without a carrier that passes the second and
/// fails the first, a `witness_dioid_laws` that forgot its idempotence check would be
/// indistinguishable from one that has it.
///
/// The consequence of getting this wrong is not cosmetic. Idempotent `⊕` is what makes a
/// closure iteration terminate, so a witness wrongly claiming the marker yields an algorithm
/// that runs forever or converges to the wrong fixed point.
struct NotADioid;

impl SemiringWitness for NotADioid {
    type Domain = i64;
    const ZERO: i64 = 0;
    const ONE: i64 = 1;
    fn add(a: i64, b: i64) -> i64 {
        a.wrapping_add(b)
    }
    fn mul(a: i64, b: i64) -> i64 {
        a.wrapping_mul(b)
    }
}
impl DioidWitness for NotADioid {}

#[test]
fn the_witness_dioid_laws_reject_an_addition_that_is_not_idempotent() {
    let sample = [0i64, 1, 2, -3];
    // The semiring tier must *accept* it, or this would not be isolating idempotence.
    assert!(
        rejects(|| witness_semiring_laws::<NotADioid>(&sample, "NotADioid")).is_none(),
        "plain arithmetic is a valid semiring; only the dioid marker is false"
    );
    let message = rejects(|| witness_dioid_laws::<NotADioid>(&sample, "NotADioid"))
        .expect("witness_dioid_laws must reject a witness where a ⊕ a != a");
    assert!(
        message.contains("idempotent"),
        "the failure must name idempotence: {message}"
    );
}

#[test]
fn the_witness_dioid_laws_accept_a_correct_dioid() {
    let sample = [MinPlus::ZERO, MinPlus::ONE, 1, 7, -3];
    assert!(
        rejects(|| witness_dioid_laws::<MinPlus>(&sample, "MinPlus")).is_none(),
        "witness_dioid_laws must accept min-plus"
    );
}
