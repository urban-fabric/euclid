//! Minimising a failing tape.
//!
//! Part of the [`shrink`](super) tests. The invariant that matters is in
//! `the_result_always_still_fails`: a shrunk tape that no longer reproduces the failure is worse
//! than no shrinking at all, because it reports a case that was never broken.

mod budget;

use super::{Probe, Prober, minimise};

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

/// `zero_chunks` has to earn its place: a case only it can minimise.
///
/// Mutation testing found that **deleting the whole pass** left every test here green, so the
/// suite was paying for a pass nothing depended on — or, read the other way, had no evidence the
/// pass worked. `a_required_length_is_preserved` looks like it covers this and does not: there
/// the entries are individually irrelevant, so `shrink_entries` zeroes them one at a time and
/// reaches the same answer.
///
/// This probe makes one-at-a-time impossible. The failure needs the length pinned at four and
/// the entries either *all* zero or *all* at least 1000 — a valley between two basins:
///
/// - `delete_chunks` cannot help, because every deletion changes the length.
/// - `shrink_entries` cannot cross the valley. Lowering any entry below 1000 leaves the tape
///   neither all-zero nor all-above-1000, so every probe is discarded and each entry sticks at
///   exactly 1000.
/// - `zero_chunks` zeroes all four in a single probe, landing in the other basin, which still
///   fails and is strictly smaller.
///
/// The gap in each basin is what makes this test work where a first attempt did not. With a
/// `sum >= 1000` condition instead, `shrink_entries` could zero three of the four entries, and
/// the *next* fixpoint round then handed `zero_chunks` a tape with zeros already in it — enough
/// for even a broken pass to finish the job. A mutant has to be unable to recover across rounds,
/// not merely stall within one.
///
/// Both of that pass's own comparisons are pinned here too. The entries start all-nonzero and
/// stay that way under any mutation, so inverting the `== 0` skip test — which turns "this chunk
/// is already zeroed, do not waste a probe" into "skip exactly the chunks that need zeroing" —
/// makes the pass a no-op for the whole run.
#[test]
fn zeroing_a_run_reaches_a_case_entry_by_entry_shrinking_cannot() {
    let predicate =
        |t: &[u64]| t.len() == 4 && (t.iter().all(|v| *v == 0) || t.iter().all(|v| *v >= 1_000));
    let tape = vec![2_000u64; 4];
    assert!(predicate(&tape), "precondition: the start must fail");

    let (minimal, _) = minimise(tape, probe_on(predicate));
    assert_eq!(
        minimal,
        vec![0, 0, 0, 0],
        "a whole run had to be zeroed at once; entry-by-entry shrinking stalls at [1000; 4]"
    );
}

/// An empty tape is a legitimate input and must not panic — it is what a generator that reads
/// nothing produces.
#[test]
fn an_empty_tape_is_handled() {
    let (minimal, _) = minimise(Vec::new(), |_| Probe::StillFails);
    assert!(minimal.is_empty());
}

#[test]
fn nonzero_chunk_widths_descend_to_one_by_halving() {
    for len in (0..1_024).chain([usize::MAX]) {
        let mut remaining = len;
        let mut widths = super::chunk_widths(len);
        for _ in 0..usize::BITS {
            let Some(expected) = std::num::NonZeroUsize::new(remaining) else {
                break;
            };
            assert_eq!(widths.next(), Some(expected));
            remaining /= 2;
        }
        assert_eq!(widths.next(), None);
    }
}
