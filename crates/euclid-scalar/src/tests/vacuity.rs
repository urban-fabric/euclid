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

mod boolean;
mod real;
