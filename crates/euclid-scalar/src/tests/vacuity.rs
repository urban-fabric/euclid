//! That the conformance suites **reject** a carrier that breaks their laws.
//!
//! The peer of `euclid-algebra`'s `tests/vacuity.rs`, and it exists for the same reason, found
//! the same way: a full `cargo mutants` pass over this crate showed that `boolean_laws`,
//! `mask_laws`, `real_laws`, their three `*_for` wrappers and their three `generated::*` peers
//! could each be **replaced with an empty body** while the whole workspace stayed green. Every
//! caller runs them against a *correct* carrier — `bool`, `f64`, `f32`, the integer tiers — and
//! a law that asserts nothing passes a correct carrier too.
//!
//! For a crate whose deliverable is the contract a backend implements, that is the worst
//! possible failure mode. `real_laws` is the single most load-bearing function in euclid: a
//! consumer writing a SIMD or GPU carrier runs it to find out whether their implementation is
//! sound, and a vacuous version tells them yes.
//!
//! # Why the three tiers cost such different amounts
//!
//! Worth stating, because it is why this file closes two tiers and not three.
//!
//! | Trait | Required methods | A deliberately-broken carrier is |
//! |---|---|---|
//! | `Boolean` | 4 | four lines |
//! | `Compare` | 8 | a dozen lines |
//! | `Real` | **153** | a generated delegation — not yet written, see below |
//!
//! `Real` had no non-primitive implementation anywhere in the workspace, so there was nothing to
//! wrap: a broken carrier has to carry all 153 methods before it can be wrong in one of them.
//! That is the same shape as `euclid-algebra`'s `Faulty<WHICH>` — "a deliberately-wrong field has
//! to carry the whole tower before it can misbehave in one corner of it" — an order of magnitude
//! larger.
//!
//! **That tier is now closed too, by generating the delegation.** [`carrier`] holds a
//! `Faulty<WHICH>` whose 160 methods delegate to `f64` except for method number `WHICH`, emitted
//! by `bin/carrier.py` from the trait definitions themselves. Writing it by hand would have meant
//! a second surface to keep in step with `Real`, and it would have rotted the first time a method
//! was added; generating it means `python3 -B bin/carrier.py --check` fails instead.

// A test asserts by panicking, and these tests assert on a panic *message*, which `expect` is
// the clearest way to extract.
#![allow(clippy::expect_used)]

use std::panic::{self, AssertUnwindSafe};
use std::sync::{Mutex, PoisonError};

/// Serialises everything in this file that touches, or depends on, the global panic hook.
///
/// libtest runs these on a thread pool, and two of the things they do are process-global: this
/// file swaps the panic hook out for a silent one so an expected panic does not print a
/// backtrace, and `euclid-stability` installs its own hook to record panic *locations*, which it
/// uses as a failure's identity when confirming that a shrunk counterexample still reproduces
/// the same failure.
///
/// Those two interact badly and the symptom is not a failure but a **flake**: a rejection test
/// that happened to overlap another's hook swap reported `unstable property: counterexample did
/// not reproduce the same case and failure` instead of the law that broke. It passed in
/// isolation and failed in the full-workspace run, which is the worst way to find out.
///
/// One lock around both kinds keeps the quiet output and removes the race. These tests are
/// microseconds each, so serialising them costs nothing worth measuring.
static PANIC_HOOK: Mutex<()> = Mutex::new(());

use crate::conformance::generated::{self as gen_laws, GeneratedSample};
use crate::conformance::{self, ConformanceSample};
use crate::traits::{Boolean, Compare, Element};
use carrier::Faulty;

mod carrier;

/// Run `body` and report whether it panicked, with the panic message.
///
/// The panic hook is silenced for the duration, because a law suite reports by panicking and
/// these tests expect it — the default hook would print a backtrace per case.
fn rejects(body: impl FnOnce()) -> Option<String> {
    let _guard = PANIC_HOOK.lock().unwrap_or_else(PoisonError::into_inner);
    let previous = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));
    let outcome = panic::catch_unwind(AssertUnwindSafe(body));
    panic::set_hook(previous);

    match outcome {
        Ok(()) => None,
        Err(payload) => Some(
            payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_string()))
                .unwrap_or_else(|| "non-string panic".to_string()),
        ),
    }
}

/// [`rejects`], but **without touching the panic hook** — for a body that runs the
/// `euclid-stability` engine.
///
/// The engine installs a global panic hook to record panic locations, and uses those as a
/// failure's identity when verifying that a shrunk counterexample reproduces the same failure.
/// Swapping the hook out from under it loses that, and a concurrent engine-backed test then
/// reports `unstable property: counterexample did not reproduce the same case and failure`
/// rather than the law that broke — a race that shows up only once enough tests run at once.
/// Anything calling `check` therefore keeps the engine's hook and accepts the stderr noise.
fn rejects_engine(body: impl FnOnce()) -> Option<String> {
    let _guard = PANIC_HOOK.lock().unwrap_or_else(PoisonError::into_inner);
    match panic::catch_unwind(AssertUnwindSafe(body)) {
        Ok(()) => None,
        Err(payload) => Some(
            payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_string()))
                .unwrap_or_else(|| "non-string panic".to_string()),
        ),
    }
}

// ===== A Boolean whose AND is an OR =======================================

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

/// The samples the laws are run on: finite, strictly positive, straddling one.
///
/// `real_laws`' own docstring states the requirement — "values must be **finite** and **strictly
/// positive** — the logarithm laws need `x > 0`, and the trig laws stay on the principal branch,
/// so keep values in `(0, π/2)`".
fn real_samples<const WHICH: u32>() -> Vec<Faulty<WHICH>> {
    [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
        .into_iter()
        .map(Faulty)
        .collect()
}

/// The tolerance the `Real` laws run at here: tight enough that a coarse perturbation cannot
/// hide inside it, loose enough for a faithful `f64` delegation to pass every inverse pair.
const REAL_TOLERANCE: f64 = 1e-12;

impl<const WHICH: u32> ConformanceSample for Faulty<WHICH> {
    fn sample() -> Vec<Self> {
        real_samples::<WHICH>()
    }
    fn tolerance() -> f64 {
        REAL_TOLERANCE
    }
}

impl<const WHICH: u32> GeneratedSample for Faulty<WHICH> {
    fn generator() -> euclid_stability::Gen<Self> {
        euclid_stability::generate::float::in_range(0.25, 1.5).map(Self)
    }
}

/// A faithful delegation satisfies every law, which is the premise the rejections rest on.
///
/// A suite that rejects *everything* is as useless as one that accepts everything, and it is the
/// easier mistake to make — so this is checked first and separately for all three entry points.
#[test]
fn the_real_laws_accept_a_faithful_carrier() {
    conformance::real_laws(&real_samples::<0>(), REAL_TOLERANCE, "Faulty<0>");
    conformance::real_laws_for::<Faulty<0>>("Faulty<0> via real_laws_for");
    gen_laws::real_laws_generated::<Faulty<0>>(REAL_TOLERANCE, "Faulty<0> via real_laws_generated");
}

/// Every law must reject a carrier whose one broken method that law depends on.
///
/// This is what the generated carrier exists for. Each entry names a method `real_laws`
/// exercises and the law that should notice it; if any were accepted, that law is absent or
/// vacuous.
///
/// # The perturbation has to break a fixed point, not just move a value
///
/// Worth recording, because the first attempt failed here and the failure was informative. With
/// an affine perturbation — `v ↦ 1.5v + 0.25` — the broken `floor` is **still idempotent**: its
/// image is `{0.25, 1.75}` and it maps that set to itself. Idempotence is the *only* law
/// `real_laws` states about `floor`, so the carrier passed, and the honest conclusion was not
/// that the test needed loosening but that the perturbation sat in the law's kernel. Negating as
/// well breaks the fixed point.
///
/// That is also a real limitation of the suite, now documented rather than discovered later: a
/// `floor` that is wrong but idempotent satisfies every law `real_laws` states.
#[test]
fn the_real_laws_reject_each_broken_method_they_depend_on() {
    macro_rules! reject {
        ($id:literal, $method:literal, $law:literal) => {{
            let outcome = rejects(|| {
                conformance::real_laws(
                    &real_samples::<$id>(),
                    REAL_TOLERANCE,
                    concat!("Faulty<", stringify!($id), ">"),
                )
            });
            assert!(
                outcome.is_some(),
                "real_laws accepted a carrier whose `{}` is wrong — the law `{}` is missing \
                 or vacuous",
                $method,
                $law
            );
        }};
    }

    reject!(
        14,
        "one",
        "the identity the tolerance scale and sin²+cos² both rest on"
    );
    reject!(16, "scalar_like", "scalar and scalar_like must agree");
    reject!(48, "exp", "ln(exp(x)) = x");
    reject!(49, "ln", "exp(ln(x)) = x");
    reject!(52, "sqrt", "sqrt(x)² = x");
    reject!(53, "abs", "|·| is idempotent");
    reject!(54, "square", "sqrt(x²) = |x|");
    reject!(70, "sin_cos", "sin²(x) + cos²(x) = 1");
    reject!(78, "floor", "floor is idempotent");
    reject!(79, "ceil", "ceil is idempotent");
    reject!(81, "trunc", "trunc is idempotent");
    reject!(88, "min", "min(x, x) = x");
    reject!(89, "max", "max(x, x) = x");
}

/// The `_for` and generated wrappers must reject too, since each is its own function with its
/// own empty-body mutant.
#[test]
fn the_real_wrappers_reject_a_broken_carrier() {
    rejects(|| conformance::real_laws_for::<Faulty<48>>("Faulty<48>"))
        .expect("real_laws_for must reject a carrier whose exp is wrong");
    rejects_engine(|| gen_laws::real_laws_generated::<Faulty<48>>(REAL_TOLERANCE, "Faulty<48>"))
        .expect("real_laws_generated must reject a carrier whose exp is wrong");
}

/// The *tolerance* mutants, which a coarse fault cannot reach.
///
/// `real_laws` bounds each inexact law by `scale · tolerance`, and mutating that `*` to `+` makes
/// the bound about `1` instead of about `1e-12` — vacuous, while every test above still passes.
/// The reason they pass is worth stating: a coarsely broken carrier mostly produces `NaN`, and
/// `NaN ≤ anything` is false, so it is rejected under either bound and says nothing about which
/// bound is in force.
///
/// `Faulty<1000 + k>` is wrong by `1e-9` relative in method `k` — three orders above the
/// tolerance and nine below one. So it must be rejected, and is accepted exactly when the bound
/// has been loosened.
#[test]
fn the_real_laws_reject_an_error_that_only_just_exceeds_the_tolerance() {
    for (identifier, method) in [(1048u32, "exp"), (1049, "ln"), (1052, "sqrt")] {
        let outcome = match identifier {
            1048 => rejects(|| {
                conformance::real_laws(&real_samples::<1048>(), REAL_TOLERANCE, "Faulty<1048>")
            }),
            1049 => rejects(|| {
                conformance::real_laws(&real_samples::<1049>(), REAL_TOLERANCE, "Faulty<1049>")
            }),
            _ => rejects(|| {
                conformance::real_laws(&real_samples::<1052>(), REAL_TOLERANCE, "Faulty<1052>")
            }),
        };
        assert!(
            outcome.is_some(),
            "real_laws accepted a `{method}` that is wrong by 1e-9 relative at a tolerance of \
             {REAL_TOLERANCE:e} — the bound is not `scale * tolerance`"
        );
    }
}

/// A carrier that is **lawful but unusual**, which is the only way to see the two
/// `delete match arm` mutants.
///
/// `real_laws` checks that each of `floor`, `ceil` and `trunc` is idempotent, dispatching on the
/// law's name. Delete the `"floor"` arm and the check falls through to `_ => trunc`, so it
/// becomes `trunc(floor(x)) == floor(x)` — which holds for every ordinary carrier, because
/// `floor(x)` is an integer and `trunc` is the identity on integers. The mutant is therefore
/// invisible to any carrier that rounds normally.
///
/// `Faulty<2000>` rounds `trunc` **down to an even integer**. That is idempotent, so every law
/// the suite states still holds and the real suite accepts it — while `trunc(floor(1.5))` is
/// `trunc(1)` is `0`, not `1`, so the altered suite rejects it. Assert acceptance of the
/// lawful carrier to catch a check that imposes a stronger rule than the contract.
#[test]
fn the_real_laws_accept_a_lawful_carrier_whose_trunc_is_unusual() {
    // The premise: this carrier's `trunc` really is both idempotent and not the identity on
    // `floor`'s output, or the test below would prove nothing.
    let unusual = Faulty::<2000>(1.5);
    let floored = crate::Real::floor(unusual);
    let truncated_floor = crate::Real::trunc(floored);
    assert_eq!(floored.0, 1.0, "floor is still the real floor");
    assert_eq!(
        truncated_floor.0, 0.0,
        "while trunc rounds down to even, so trunc(floor(1.5)) != floor(1.5)"
    );
    let twice = crate::Real::trunc(crate::Real::trunc(unusual));
    assert_eq!(
        twice.0,
        crate::Real::trunc(unusual).0,
        "and it is idempotent, which is the only law the suite states about it"
    );

    // The point: every law holds, so the suite must accept it. With either match arm deleted it
    // does not.
    conformance::real_laws(&real_samples::<2000>(), REAL_TOLERANCE, "Faulty<2000>");
}
