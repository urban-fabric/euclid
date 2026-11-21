//! Does the boolean and mask conformance suite notice a broken carrier?
//!
//! Part of the [`vacuity`](super) suite, whose preamble carries the `rejects` harness and the
//! panic-hook mutex every one of these needs. Each carrier here breaks exactly one law — an
//! `and` that is an `or`, a `le` that is not reflexive, a `select` with its branches swapped —
//! and the test is that the suite *fails*, which is the only way to show the suite is not
//! vacuous.

use super::*;

/// `bool` with `and` wired to `or`, and nothing else changed.
///
/// Chosen because it breaks the annihilation law `a ∧ 0 = 0` while leaving `a ∧ 1 = a` intact —
/// `true ∨ 1` is still `true` — so the carrier looks right on the identity a reader checks first.
/// It is also a plausible transcription error rather than a contrived one.
#[derive(Clone, Copy, PartialEq, Debug)]
struct OrIsAnd(bool);

impl Element for OrIsAnd {
    fn zero() -> Self {
        Self(false)
    }
    fn one() -> Self {
        Self(true)
    }
}

impl Boolean for OrIsAnd {
    fn and(&self, other: &Self) -> Self {
        Self(self.0 || other.0) // the defect
    }
    fn or(&self, other: &Self) -> Self {
        Self(self.0 || other.0)
    }
    fn xor(&self, other: &Self) -> Self {
        Self(self.0 ^ other.0)
    }
    fn not(&self) -> Self {
        Self(!self.0)
    }
}

impl ConformanceSample for OrIsAnd {
    fn sample() -> Vec<Self> {
        vec![Self(false), Self(true)]
    }
}

#[test]
fn the_boolean_laws_accept_a_correct_carrier() {
    // Without this, every rejection below is satisfiable by a suite that always panics.
    assert!(
        rejects(|| conformance::boolean_laws(&[false, true], "bool")).is_none(),
        "boolean_laws must accept bool"
    );
    assert!(
        rejects(|| conformance::boolean_laws_for::<bool>("bool")).is_none(),
        "boolean_laws_for must accept bool"
    );
}

#[test]
fn the_boolean_laws_reject_an_and_that_is_an_or() {
    let message = rejects(|| conformance::boolean_laws(&OrIsAnd::sample(), "OrIsAnd"))
        .expect("boolean_laws must reject a carrier where a ∧ 0 ≠ 0");
    assert!(
        message.contains('∧'),
        "the failure must name a conjunction law: {message}"
    );
}

#[test]
fn the_boolean_laws_for_wrapper_rejects_through_its_own_sample() {
    // The `*_for` wrappers are separate functions with their own empty-body mutant, so each
    // needs its own rejection — the suite it delegates to being checked is not enough.
    rejects(|| conformance::boolean_laws_for::<OrIsAnd>("OrIsAnd"))
        .expect("boolean_laws_for must reject the same carrier through ConformanceSample");
}

// ===== A Compare whose `le` is strict =====================================

/// An integer ordering with `le` wired to `lt`, so `a ≤ a` is false.
///
/// `le` is `lt ∨ eq` and every other relation here is correct, so this isolates exactly one of
/// `mask_laws`' clauses. Off-by-one in a non-strict comparison is the single most common way to
/// get a `Compare` impl wrong, and on a SIMD carrier it is invisible without a law: the mask is
/// the right *shape*, just wrong in one lane's worth of meaning.
#[derive(Clone, Copy, PartialEq, Debug)]
struct StrictLe(i32);

impl Compare for StrictLe {
    type Mask = bool;
    fn lt(&self, other: &Self) -> bool {
        self.0 < other.0
    }
    fn le(&self, other: &Self) -> bool {
        self.0 < other.0 // the defect: strict where the contract is non-strict
    }
    fn gt(&self, other: &Self) -> bool {
        self.0 > other.0
    }
    fn ge(&self, other: &Self) -> bool {
        self.0 >= other.0
    }
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
    fn ne(&self, other: &Self) -> bool {
        self.0 != other.0
    }
    fn select(mask: &bool, on_true: Self, on_false: Self) -> Self {
        if *mask { on_true } else { on_false }
    }
    fn while_loop<C, P, B>(init: C, cond: P, body: B) -> C
    where
        P: Fn(&C) -> bool,
        B: Fn(C) -> C,
    {
        let mut state = init;
        while cond(&state) {
            state = body(state);
        }
        state
    }
}

/// The same carrier with its branches swapped, and everything else correct.
///
/// `mask_laws`' own documentation calls this out as the defect the `select` laws exist for: "a
/// carrier with its branches reversed type-checks everywhere and silently inverts every
/// conditional written against the contract". A rejection control for a stated motivation is
/// worth more than one for an arbitrary break.
#[derive(Clone, Copy, PartialEq, Debug)]
struct SwappedSelect(i32);

impl Compare for SwappedSelect {
    type Mask = bool;
    fn lt(&self, other: &Self) -> bool {
        self.0 < other.0
    }
    fn le(&self, other: &Self) -> bool {
        self.0 <= other.0
    }
    fn gt(&self, other: &Self) -> bool {
        self.0 > other.0
    }
    fn ge(&self, other: &Self) -> bool {
        self.0 >= other.0
    }
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
    fn ne(&self, other: &Self) -> bool {
        self.0 != other.0
    }
    fn select(mask: &bool, on_true: Self, on_false: Self) -> Self {
        if *mask { on_false } else { on_true } // the defect
    }
    fn while_loop<C, P, B>(init: C, cond: P, body: B) -> C
    where
        P: Fn(&C) -> bool,
        B: Fn(C) -> C,
    {
        let mut state = init;
        while cond(&state) {
            state = body(state);
        }
        state
    }
}

#[test]
fn the_mask_laws_accept_a_correct_carrier() {
    assert!(
        rejects(|| conformance::mask_laws(&[0_i64, 1, -1, 7], "i64")).is_none(),
        "mask_laws must accept i64"
    );
}

#[test]
fn the_mask_laws_reject_a_non_reflexive_le() {
    let sample = [StrictLe(-2), StrictLe(0), StrictLe(5)];
    let message = rejects(|| conformance::mask_laws(&sample, "StrictLe"))
        .expect("mask_laws must reject a carrier where a ≤ a is false");
    assert!(
        message.contains("le") || message.contains('≤'),
        "the failure must name the comparison that broke: {message}"
    );
}

impl ConformanceSample for StrictLe {
    fn sample() -> Vec<Self> {
        vec![Self(-2), Self(0), Self(5)]
    }
}

#[test]
fn the_mask_laws_for_wrapper_rejects_through_its_own_sample() {
    // The third `*_for` wrapper, with its own empty-body mutant. Missed on the first pass
    // through this file, which is the argument for measuring rather than reasoning about which
    // wrappers are covered: all three look alike and only two had been written.
    rejects(|| conformance::mask_laws_for::<StrictLe>("StrictLe"))
        .expect("mask_laws_for must reject the same carrier through ConformanceSample");
}

#[test]
fn the_mask_laws_reject_a_select_with_its_branches_swapped() {
    let sample = [SwappedSelect(-2), SwappedSelect(0), SwappedSelect(5)];
    rejects(|| conformance::mask_laws(&sample, "SwappedSelect"))
        .expect("mask_laws must reject a select that takes the wrong branch");
}

// ===== The two defaulted knobs ============================================

/// A carrier that overrides neither [`ConformanceSample::tolerance`] nor `cases`.
///
/// Both defaults are unreachable from the existing suites — `f64` and `f32` override `tolerance`,
/// and every generated carrier is a float — so a mutation of either body changes nothing any
/// test observes. That is exactly how `tolerance -> -1.0` survived: a *negative* tolerance makes
/// every inexact bound negative and every inexact law fail, so it is not a subtle defect at all.
/// It simply never ran.
#[derive(Clone, Copy, PartialEq, Debug)]
struct Defaulted(i32);

impl ConformanceSample for Defaulted {
    fn sample() -> Vec<Self> {
        vec![Self(0), Self(1)]
    }
}

#[test]
fn a_carrier_that_declares_no_tolerance_gets_an_exact_one() {
    // The documented default: "zero, which is the right answer for an exact carrier". A negative
    // value is not a looser bound, it is an impossible one — `|err| ≤ −1 · scale` is never true,
    // so this default has to be pinned by value rather than by behaviour.
    assert_eq!(
        <Defaulted as ConformanceSample>::tolerance(),
        0.0,
        "the default tolerance is exact"
    );
}

// ===== The generated peers, and the case-count knob =======================

impl GeneratedSample for OrIsAnd {
    fn generator() -> euclid_stability::Gen<Self> {
        euclid_stability::generate::boolean().map(Self)
    }
}

impl GeneratedSample for StrictLe {
    fn generator() -> euclid_stability::Gen<Self> {
        #[allow(clippy::cast_possible_truncation)]
        euclid_stability::generate::int(-8, 8).map(|v| Self(v as i32))
    }
}

#[test]
fn the_generated_boolean_suite_rejects_a_broken_carrier() {
    // Each `generated::*` wrapper is its own function with its own empty-body mutant, so the
    // slice-taking suite being checked does not cover it.
    rejects_engine(|| gen_laws::boolean_laws_generated::<OrIsAnd>("OrIsAnd"))
        .expect("boolean_laws_generated must reject a carrier where a ∧ 0 ≠ 0");
}

#[test]
fn the_generated_mask_suite_rejects_a_broken_carrier() {
    rejects_engine(|| gen_laws::mask_laws_generated::<StrictLe>("StrictLe"))
        .expect("mask_laws_generated must reject a carrier where a ≤ a is false");
}

#[test]
fn the_generated_settings_run_the_case_count_the_carrier_declares() {
    // `Config::default()` carries 256 cases of its own against this crate's declared 400, so a
    // body that forgot to apply `T::cases()` would search less than asked while every suite
    // above still passed. Asserted by value for the reason given on `settings` itself.
    let declared = <OrIsAnd as GeneratedSample>::cases();
    assert_eq!(declared, 400, "the documented default case count");
    assert_eq!(
        gen_laws::settings::<OrIsAnd>().cases,
        declared,
        "settings must run the carrier's declared case count, not Config's default"
    );
    assert_ne!(
        declared,
        euclid_stability::Config::default().cases,
        "this test is only meaningful while the two differ"
    );
}

// ===== The `Real` tier, through a generated carrier =======================
//
// The tier the header above called "not yet written". It is written now, and generated: see
// [`carrier`] for why, and for the method-id table the tests below index into.
