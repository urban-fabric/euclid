//! That the conformance suites **reject** a carrier that breaks their laws.
//!
//! # Why this file exists
//!
//! Every other test of `conformance` runs a law suite against a *correct* carrier and checks
//! that it does not complain. That is necessary and it is not sufficient, and mutation testing
//! showed exactly how insufficient: **eighteen of this crate's law suites could be replaced with
//! an empty body** and the entire workspace suite stayed green. A law that asserts nothing
//! passes every correct implementation too.
//!
//! For a crate whose deliverable *is* its conformance suites — a consumer implements the trait
//! tower, runs `semiring_laws_for::<Theirs>()`, and is told whether their implementation is
//! sound — a vacuous law is worse than a missing one. It reports success.
//!
//! So each test below builds a carrier that is deliberately wrong in one specific way, claims
//! the marker trait it violates, and requires the matching suite to **panic**. A suite that
//! asserts nothing fails these tests; a suite that asserts the wrong thing fails them too,
//! because each carrier is wrong in exactly one law and correct in all the others.
//!
//! # The markers are what make this cheap
//!
//! The tower derives its structures from marker traits — `AssociativeMultiplication`,
//! `Distributive`, and so on — which a carrier *claims* rather than proves. A broken carrier is
//! therefore a handful of one-line impls that claim a marker they do not satisfy, which is also
//! a fair description of the mistake a real consumer makes.

// A test asserts by panicking, and these tests assert on a panic *message*, which `expect` is
// the clearest way to extract. Same exemption the other suites in the fleet take.
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

use crate::conformance::{
    self, ConformanceSample, commutative_semiring_laws_for, ring_laws_for, semiring_laws_for,
};
use crate::*;

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
/// The distinction is not cosmetic, and getting it wrong produced a flake rather than a failure.
/// The engine installs its own global panic hook to record panic *locations*, which it then uses
/// as a failure's identity when it checks that a shrunk counterexample still reproduces the same
/// failure. [`rejects`] swaps that hook out for a silent one — so with several engine-backed
/// tests in flight on libtest's thread pool, a property could lose the location recording it
/// depends on and report `unstable property: counterexample did not reproduce the same case and
/// failure` instead of the law that actually broke.
///
/// That is a race, and it only appeared once enough tests ran concurrently: the generated-suite
/// rejections passed in isolation and failed in the full-workspace run.
///
/// So anything that calls `check` keeps the engine's hook. The cost is a backtrace on stderr for
/// a panic the test expects, which is noise; the alternative was a test that fails one run in
/// several for a reason unrelated to what it checks.
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

mod carriers;
mod division;
mod generated_suites;
mod order_star_module;
mod reference;
mod semiring;
mod witness;
