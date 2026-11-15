//! The shrinker, checked on its own terms: does it actually minimise, and does it stay sound?
//!
//! These tests drive [`minimise`] directly with a hand-written probe, so they test the
//! minimiser rather than the generators on top of it. The end-to-end question — "does a real
//! property report a small counterexample" — is in `crate::tests`.

use super::{MAX_PROBES, Probe, minimise};

/// A probe that keeps failing while `predicate` holds of the tape.
fn probe_on(predicate: impl Fn(&[u64]) -> bool) -> impl Fn(&[u64]) -> Probe {
    move |tape| {
        if predicate(tape) {
            Probe::StillFails
        } else {
            Probe::Discard
        }
    }
}

/// The basic job: a tape that only needs one large entry must come back with that entry at its
/// smallest still-failing value, not merely smaller.
#[test]
fn an_entry_is_minimised_to_its_threshold() {
    let tape = vec![9_999_999, 5, 7];
    let (minimal, _) = minimise(tape, probe_on(|t| !t.is_empty() && t[0] >= 42));
    assert_eq!(
        minimal[0], 42,
        "binary search must land exactly on the bound"
    );
}

/// Binary search, not a decrement loop: reaching 42 from a 64-bit value must cost tens of
/// probes, not billions. This is the test that would catch someone replacing the search with
/// `value - 1`.
#[test]
fn minimising_a_large_entry_is_logarithmic() {
    let tape = vec![u64::MAX];
    let (minimal, probes) = minimise(tape, probe_on(|t| !t.is_empty() && t[0] >= 42));
    assert_eq!(minimal[0], 42);
    assert!(
        probes < 300,
        "took {probes} probes, which is not a binary search"
    );
}

/// Irrelevant entries must be deleted outright, not just zeroed. A tape of a hundred entries
/// where one matters should come back at length one.
#[test]
fn irrelevant_entries_are_deleted() {
    let mut tape = vec![0u64; 100];
    tape[60] = 500;
    // Fails while *some* entry is >= 7. Only one entry is needed to witness that.
    let (minimal, _) = minimise(tape, probe_on(|t| t.iter().any(|v| *v >= 7)));
    assert_eq!(minimal.len(), 1, "got {minimal:?}");
    assert_eq!(minimal[0], 7);
}

/// A tape whose failure needs a specific *length* must not be shortened past it. This is the
/// soundness direction: over-eager deletion would report a case that does not fail.
#[test]
fn a_required_length_is_preserved() {
    let tape = vec![1u64; 20];
    let (minimal, _) = minimise(tape, probe_on(|t| t.len() >= 5));
    assert_eq!(minimal.len(), 5, "must stop at the length that still fails");
    assert!(
        minimal.iter().all(|v| *v == 0),
        "entries themselves were irrelevant and should have been zeroed: {minimal:?}"
    );
}

/// Two entries that must *both* be large: neither can be deleted or zeroed, and both must be
/// minimised. Catches a shrinker that stops at the first entry it cannot improve.
#[test]
fn several_required_entries_are_all_minimised() {
    let tape = vec![1_000, 1_000, 1_000];
    let (minimal, _) = minimise(
        tape,
        probe_on(|t| t.len() >= 3 && t[0] >= 10 && t[1] >= 20 && t[2] >= 30),
    );
    assert_eq!(minimal, vec![10, 20, 30]);
}

/// The minimiser must never return a tape the probe rejects. Every accepted candidate is
/// re-probed here, which is the invariant that makes a shrunk counterexample trustworthy.
#[test]
fn the_result_always_still_fails() {
    let predicate = |t: &[u64]| t.len() >= 2 && t[0] % 7 == 3 && t[1] >= 9;
    let tape = vec![31, 44, 12, 88, 5];
    assert!(predicate(&tape), "precondition: the start must fail");

    let (minimal, _) = minimise(tape, probe_on(predicate));
    assert!(
        predicate(&minimal),
        "minimiser returned a passing tape: {minimal:?}"
    );
}

/// A probe that always discards leaves the tape alone rather than returning something degenerate
/// — the case where the "failure" was a flake and nothing can be reproduced.
#[test]
fn an_unreproducible_failure_shrinks_to_nothing_useful_but_stays_valid() {
    let tape = vec![5, 6, 7];
    let (minimal, _) = minimise(tape.clone(), |_| Probe::Discard);
    assert_eq!(
        minimal, tape,
        "nothing was reproducible, so nothing changed"
    );
}

/// Sorting is a canonicalisation: when order does not matter to the failure, the reported tape
/// must be the sorted one, so the same bug reports identically regardless of the seed that
/// found it.
#[test]
fn equivalent_tapes_canonicalise_to_the_sorted_one() {
    let predicate = |t: &[u64]| t.len() >= 3 && t.iter().filter(|v| **v >= 5).count() >= 2;

    let (from_one, _) = minimise(vec![9, 0, 8], probe_on(predicate));
    let (from_other, _) = minimise(vec![8, 9, 0], probe_on(predicate));
    assert_eq!(
        from_one, from_other,
        "two orderings of the same failure reported differently"
    );
    assert!(
        from_one.windows(2).all(|w| w[0] <= w[1]),
        "result is not sorted: {from_one:?}"
    );
}

/// The probe budget must be respected, so a pathological probe cannot hang a test run.
#[test]
fn the_probe_budget_is_respected() {
    let mut calls = 0usize;
    let huge: Vec<u64> = (0..200).map(|i| u64::MAX - i).collect();
    let (_, probes) = minimise(huge, |_| {
        calls += 1;
        Probe::StillFails
    });
    assert!(probes <= MAX_PROBES, "spent {probes}");
    assert!(calls <= MAX_PROBES, "probe called {calls} times");
}

/// An empty tape is a legitimate input and must not panic — it is what a generator that reads
/// nothing produces.
#[test]
fn an_empty_tape_is_handled() {
    let (minimal, _) = minimise(Vec::new(), |_| Probe::StillFails);
    assert!(minimal.is_empty());
}
