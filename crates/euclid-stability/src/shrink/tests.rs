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
    let (minimal, cost) = minimise(tape, probe_on(|t| !t.is_empty() && t[0] >= 42));
    assert_eq!(minimal[0], 42);
    assert!(
        cost.probes < 300,
        "took {} probes, which is not a binary search",
        cost.probes
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
    let (_, cost) = minimise(huge, |_| {
        calls += 1;
        Probe::StillFails
    });
    assert!(cost.probes <= MAX_PROBES, "spent {}", cost.probes);
    assert!(calls <= MAX_PROBES, "probe called {calls} times");
}

/// An empty tape is a legitimate input and must not panic — it is what a generator that reads
/// nothing produces.
#[test]
fn an_empty_tape_is_handled() {
    let (minimal, _) = minimise(Vec::new(), |_| Probe::StillFails);
    assert!(minimal.is_empty());
}

// ===== The memo ===========================================================

/// The closure must see each distinct candidate **once**. Everything else is answered from the
/// table, which is the whole point: a repeat re-runs the generator and the property, and for a
/// real suite that is the dominant cost of a failure.
#[test]
fn a_repeated_candidate_never_reaches_the_probe_closure() {
    use std::cell::RefCell;
    use std::collections::HashSet;

    let seen: RefCell<HashSet<Vec<u64>>> = RefCell::new(HashSet::new());
    let repeats = RefCell::new(0usize);

    let tape = vec![500, 400, 300, 200, 100, 50, 25, 12];
    let (_, cost) = minimise(tape, |candidate| {
        if !seen.borrow_mut().insert(candidate.to_vec()) {
            *repeats.borrow_mut() += 1;
        }
        if candidate.len() >= 3 && candidate[0] >= 7 {
            Probe::StillFails
        } else {
            Probe::Discard
        }
    });

    assert_eq!(
        *repeats.borrow(),
        0,
        "the probe closure was handed a candidate it had already answered"
    );
    assert_eq!(
        seen.borrow().len(),
        cost.probes,
        "distinct candidates must equal the reported probe count"
    );
}

/// The fixpoint loop re-tries candidates by design — the final round exists to prove nothing
/// more can be done — so a minimisation with any structure to it must record memo hits. A run
/// reporting zero would mean the memo is never consulted and the whole table is dead weight.
#[test]
fn a_fixpoint_round_produces_memo_hits() {
    let tape = vec![900, 800, 700, 600, 500];
    let (_, cost) = minimise(tape, probe_on(|t| t.len() >= 2 && t[0] >= 5));
    assert!(
        cost.reused > 0,
        "no candidate was answered from the memo, so the fixpoint round is re-probing nothing"
    );
}

/// A memo hit must not spend budget. That is what turns [`MAX_PROBES`] from a bound on loop
/// iterations into a bound on *distinct candidates tried* — the same cap now explores strictly
/// more of the search space, so a case that used to exhaust its budget re-proving a fixpoint can
/// reach an actually smaller tape.
///
/// Tested on the [`Prober`](super::Prober) directly rather than through `minimise`, because a
/// repeat has to be *arranged* to be certain of it. The first attempt at this test drove
/// `minimise` with an always-failing probe over a long tape, assuming churn — but an
/// always-failing probe means the widest deletion succeeds immediately, the tape empties in one
/// probe, and there is nothing left to repeat. The property is about the memo, so it is checked
/// where the memo lives.
#[test]
fn memo_hits_do_not_consume_the_probe_budget() {
    let calls = std::cell::Cell::new(0usize);
    let mut prober = super::Prober::new(|_: &[u64]| {
        calls.set(calls.get() + 1);
        Probe::Discard
    });

    assert_eq!(prober.ask(&[1, 2, 3]), Probe::Discard);
    assert_eq!(prober.ask(&[1, 2, 3]), Probe::Discard);
    assert_eq!(prober.ask(&[1, 2, 3]), Probe::Discard);
    assert_eq!(prober.ask(&[9]), Probe::Discard);

    assert_eq!(
        calls.get(),
        2,
        "only the two distinct candidates may be run"
    );
    assert_eq!(
        prober.cost.probes, 2,
        "budget is spent per distinct candidate"
    );
    assert_eq!(prober.cost.reused, 2, "the two repeats must be memo hits");
}

/// Whatever the tape, the cap bounds the distinct probes actually spent.
#[test]
fn the_cap_bounds_distinct_probes() {
    for tape in [
        vec![u64::MAX; 40],
        (0..60).map(|i| 1_000 - i).collect::<Vec<u64>>(),
        vec![0; 80],
    ] {
        let (_, cost) = minimise(tape, probe_on(|t| t.len() >= 2));
        assert!(
            cost.probes <= MAX_PROBES,
            "distinct probes must respect the cap: {}",
            cost.probes
        );
    }
}

#[test]
fn hitting_the_cap_is_reported() {
    let cheap = super::Cost {
        probes: 10,
        reused: 0,
    };
    assert!(!cheap.hit_the_cap());
    let exhausted = super::Cost {
        probes: MAX_PROBES,
        reused: 0,
    };
    assert!(exhausted.hit_the_cap());
}

/// The hasher has to consume a `u64` run through its bulk `write` path, because that is how
/// `Hash for [u64]` feeds it — the length through `write_usize`, then every element as one byte
/// slice. A byte-at-a-time implementation would be correct and slow, and a chunked one that
/// dropped the remainder would be fast and wrong.
#[test]
fn the_tape_hasher_distinguishes_tapes_through_its_bulk_path() {
    use std::hash::{BuildHasher, BuildHasherDefault, Hasher};

    let hash_of = |tape: &[u64]| BuildHasherDefault::<super::TapeHasher>::default().hash_one(tape);

    // Different contents, same length.
    assert_ne!(hash_of(&[1, 2, 3]), hash_of(&[1, 2, 4]));
    // Same contents, different length — the length prefix is what separates these.
    assert_ne!(hash_of(&[1, 2]), hash_of(&[1, 2, 0]));
    // Order matters, or a permutation pass would collide with its own input.
    assert_ne!(hash_of(&[1, 2]), hash_of(&[2, 1]));
    // The empty tape is hashable and does not panic on the remainder path.
    let _ = hash_of(&[]);

    // And a byte slice whose length is not a multiple of eight reaches the remainder branch.
    let mut hasher = BuildHasherDefault::<super::TapeHasher>::default().build_hasher();
    hasher.write(&[1, 2, 3, 4, 5]);
    let odd = hasher.finish();
    let mut hasher = BuildHasherDefault::<super::TapeHasher>::default().build_hasher();
    hasher.write(&[1, 2, 3, 4, 6]);
    assert_ne!(
        odd,
        hasher.finish(),
        "the remainder bytes must affect the hash"
    );
}
