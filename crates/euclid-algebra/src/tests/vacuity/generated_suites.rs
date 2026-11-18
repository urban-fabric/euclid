//! Rejection controls for the **generated** tier suites, and for the two knobs behind them.
//!
//! [`super::division`] and [`super::semiring`] pin the fixed-sample suites. These are their
//! generated peers, and all five were replaceable with an empty body: `cargo mutants` could
//! delete `semiring_laws_generated`, `commutative_semiring_laws_generated`,
//! `ring_laws_generated`, `field_laws_generated` and `euclidean_laws_generated` one at a time
//! and the workspace stayed green, because every caller ran them against a *correct* carrier.
//!
//! These are the suites a consumer runs to find out whether their own carrier is sound, so a
//! vacuous one reports success for a broken implementation — the same argument the parent module
//! makes, with the search attached.
//!
//! # The two knobs
//!
//! Beyond the five bodies, two smaller mutants survived, and they are the reason this file
//! asserts on numbers as well as on panics:
//!
//! - `GeneratedSample::cases` could return **1** instead of 300. A suite that runs one case is
//!   not a search; it is a single sample with extra machinery.
//! - `generated::settings` could return `Config::default()` instead of
//!   `Config::default().cases(T::cases())`, which silently substitutes `Config`'s own default of
//!   256 for whatever the carrier declared — so a carrier asking for 5,000 cases would quietly
//!   get 256.
//!
//! Neither is reachable by a rejection test: both change *how hard* the search looks, not
//! whether a broken carrier is caught, and a broken carrier here fails on its first case either
//! way. Counting generated draws would make the assertion statistical, which this repository
//! does not accept for a property. Asserting the two values directly is deterministic and says
//! exactly what the contract is, which is why `settings` is `pub(crate)`.

use super::division::Faulty;
use super::*;
use crate::conformance::generated::{self, GeneratedSample};
use euclid_stability::{Config, Gen, generate};

/// A small exact rational, bounded so a triple cannot overflow `i64`.
///
/// The laws run over triples — distributivity alone forms `(a ⊕ b) ⊗ c`, a product of three
/// denominators — and a debug-build overflow panics indistinguishably from a law failure, which
/// would make these tests pass for the wrong reason.
fn small_rational() -> Gen<Rational<i64>> {
    generate::pair(generate::int(-6, 6), generate::int(1, 6))
        .map(|(num, den)| Rational::<i64>::new(num, den))
}

impl GeneratedSample for Faulty<0> {
    fn generator() -> Gen<Self> {
        small_rational().map(Self)
    }
}

impl GeneratedSample for Faulty<4> {
    fn generator() -> Gen<Self> {
        small_rational().map(Self)
    }
}

#[test]
fn the_generated_suites_accept_a_sound_carrier() {
    // The control. Without it every rejection below is satisfiable by a suite that panics
    // unconditionally, which is the other way a law suite can be useless.
    generated::semiring_laws_generated::<Faulty<0>>("Faulty<0>");
    generated::commutative_semiring_laws_generated::<Faulty<0>>("Faulty<0>");
    generated::ring_laws_generated::<Faulty<0>>("Faulty<0>");
    generated::field_laws_generated::<Faulty<0>>("Faulty<0>");
    generated::euclidean_laws_generated::<Faulty<0>>("Faulty<0>");
}

#[test]
fn the_generated_semiring_suite_rejects_a_broken_carrier() {
    let message = rejects_engine(|| generated::semiring_laws_generated::<Faulty<4>>("Faulty<4>"))
        .expect("semiring_laws_generated must reject a carrier where a ⊗ 1 ≠ a");
    assert!(
        message.contains('⊗'),
        "the failure must come from a semiring axiom: {message}"
    );
}

#[test]
fn the_generated_commutative_semiring_suite_rejects_a_broken_carrier() {
    // `ab + 1` *is* commutative, so this rejection comes through the `semiring_laws` that
    // `commutative_semiring_laws` delegates to — which is the delegation the empty body removed.
    rejects_engine(|| generated::commutative_semiring_laws_generated::<Faulty<4>>("Faulty<4>"))
        .expect("commutative_semiring_laws_generated must reject a broken semiring trunk");
}

#[test]
fn the_generated_ring_suite_rejects_a_broken_carrier() {
    // `−(ab) = (−a)b` is where this lands: `−(ab + 1)` is `−ab − 1` while `(−a)b + 1` is
    // `−ab + 1`. The bridge law `⊗ is *` passes, because both layers are broken together.
    rejects_engine(|| generated::ring_laws_generated::<Faulty<4>>("Faulty<4>"))
        .expect("ring_laws_generated must reject a carrier where −(ab) ≠ (−a)b");
}

#[test]
fn the_generated_field_suite_rejects_a_broken_carrier() {
    rejects_engine(|| generated::field_laws_generated::<Faulty<4>>("Faulty<4>"))
        .expect("field_laws_generated must reject a carrier where a · a⁻¹ ≠ 1");
}

#[test]
fn the_generated_euclidean_suite_rejects_a_broken_carrier() {
    rejects_engine(|| generated::euclidean_laws_generated::<Faulty<4>>("Faulty<4>"))
        .expect("euclidean_laws_generated must reject a carrier where q·b + r ≠ a");
}

#[test]
fn a_carrier_that_declares_no_case_count_gets_the_documented_default() {
    // `GeneratedSample::cases` is defaulted, and its docstring commits to the number: "Defaulted
    // at 300: enough that the shrinker has something to work with, few enough that every tier
    // law over a triple stays quick." Dropping it to 1 turns the search into a single sample.
    assert_eq!(
        <Faulty<0> as GeneratedSample>::cases(),
        300,
        "the documented default case count"
    );
}

#[test]
fn the_generated_settings_run_the_case_count_the_carrier_declares() {
    // The contract `settings` exists for. `Config::default()` carries 256 cases of its own, so a
    // body that forgot to apply `T::cases()` would be wrong by a plausible-looking margin rather
    // than obviously — and every suite above would still pass, just searching less than asked.
    let declared = <Faulty<0> as GeneratedSample>::cases();
    assert_eq!(
        generated::settings::<Faulty<0>>().cases,
        declared,
        "settings must run the carrier's declared case count, not Config's default"
    );
    assert_ne!(
        declared,
        Config::default().cases,
        "this test is only meaningful while the two differ"
    );
}
