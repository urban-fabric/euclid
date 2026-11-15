//! Tape minimisation.
//!
//! Every pass here edits a `Vec<u64>` and asks one question: does the property still fail? It
//! never looks at the generated value, never knows its type, and has no per-type shrink rules
//! to keep in step with the generators. That is the whole payoff of the tape design — there is
//! exactly one shrinker, and adding a generator cannot introduce a shrinking bug.
//!
//! # The passes, and why each exists
//!
//! Ordered cheapest-and-most-effective first, and run to a fixpoint because each enables the
//! others: deleting a chunk often makes a value newly reducible, and reducing a length entry
//! often makes a chunk newly deletable.
//!
//! | pass | what it minimises |
//! |---|---|
//! | `delete_chunks` | list length, expression depth — the big structural wins |
//! | `zero_chunks` | runs of entries at once, where one-at-a-time would stall |
//! | `shrink_entries` | individual magnitudes, by binary search |
//! | `sort_pairs` | canonical ordering, so equivalent cases report identically |
//!
//! `delete_chunks` runs first and biggest-first. A failing case is usually *mostly*
//! irrelevant — eight nodes where two suffice — and deleting a long run is one probe that
//! removes what a hundred single-entry reductions would chip at.
//!
//! # Soundness
//!
//! A candidate is accepted only if the property still fails on it, so the shrunk case is
//! always a genuine counterexample rather than a different test that happens to fail.
//! Candidates that overrun the tape are discarded, which is what the `Fixed` mode in
//! [`Source`](crate::source::Source) is for: topping a short tape up with fresh randomness
//! would silently change the case under examination.

/// How many candidate tapes a single minimisation may try.
///
/// A bound rather than a true fixpoint, because the passes can in principle keep finding
/// microscopic improvements for a long time on a large tape, and a shrink that takes a minute
/// is worse than one that stops slightly early. Reaching the cap is reported rather than
/// hidden, so a case that shrank badly says so.
pub const MAX_PROBES: usize = 4_096;

/// The outcome of one candidate tape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Probe {
    /// The property still fails; keep this tape.
    StillFails,
    /// The property passes, was rejected, or the tape overran; discard this tape.
    Discard,
}

/// Minimise `tape`, keeping only candidates for which `probe` reports [`Probe::StillFails`].
///
/// Returns the smallest tape found and how many probes were spent.
///
/// # Convergence is detected by comparing tapes, not by trusting the passes
///
/// Each pass returns whether it improved anything, and the obvious loop is "repeat while some
/// pass improved". That loop does not converge. `shrink_entries` lowering an entry can put the
/// tape out of order, `sort_pairs` then swaps it back, and the two report progress forever
/// while the tape cycles between two states — which is how an eleven-entry tape burned the
/// entire probe budget and reported itself unminimised.
///
/// Snapshotting the tape and stopping when a full round leaves it unchanged is the actual
/// fixpoint, and it costs one clone per round.
pub fn minimise(tape: Vec<u64>, mut probe: impl FnMut(&[u64]) -> Probe) -> (Vec<u64>, usize) {
    let mut best = tape;
    let mut spent = 0usize;

    while spent < MAX_PROBES {
        let before = best.clone();

        delete_chunks(&mut best, &mut probe, &mut spent);
        zero_chunks(&mut best, &mut probe, &mut spent);
        shrink_entries(&mut best, &mut probe, &mut spent);
        sort_pairs(&mut best, &mut probe, &mut spent);

        if best == before {
            break;
        }
    }

    (best, spent)
}

/// Try removing contiguous runs, longest first.
///
/// This is the pass that turns "eight nodes and eleven edges" into "two nodes and one edge",
/// because a list's length lives in one entry and its elements in the entries after it —
/// deleting a suffix of those elements and lowering the length is a single structural edit.
fn delete_chunks(
    tape: &mut Vec<u64>,
    probe: &mut impl FnMut(&[u64]) -> Probe,
    spent: &mut usize,
) -> bool {
    let mut improved = false;
    let mut width = tape.len().max(1);

    while width > 0 {
        let mut start = 0;
        while start + width <= tape.len() {
            if *spent >= MAX_PROBES {
                return improved;
            }
            let mut candidate = tape.clone();
            candidate.drain(start..start + width);
            *spent += 1;
            if probe(&candidate) == Probe::StillFails {
                *tape = candidate;
                improved = true;
                // Do not advance `start`: the window now holds different entries, which are
                // often deletable too. Re-probing the same offset is how a long irrelevant
                // run collapses in a few probes instead of one probe per entry.
            } else {
                start += 1;
            }
        }
        width /= 2;
    }
    improved
}

/// Try zeroing contiguous runs, longest first.
///
/// Distinct from deletion: zeroing keeps the tape's *length* and so keeps every later draw at
/// the same position, which is what lets a structurally required entry be minimised without
/// disturbing anything after it. Deleting it would shift the whole remainder and almost always
/// fail the probe.
fn zero_chunks(
    tape: &mut [u64],
    probe: &mut impl FnMut(&[u64]) -> Probe,
    spent: &mut usize,
) -> bool {
    let mut improved = false;
    let mut width = tape.len().max(1);

    while width > 0 {
        let mut start = 0;
        while start + width <= tape.len() {
            if *spent >= MAX_PROBES {
                return improved;
            }
            if tape[start..start + width].iter().all(|entry| *entry == 0) {
                start += 1;
                continue;
            }
            let mut candidate = tape.to_vec();
            for entry in &mut candidate[start..start + width] {
                *entry = 0;
            }
            *spent += 1;
            if probe(&candidate) == Probe::StillFails {
                tape.copy_from_slice(&candidate);
                improved = true;
            }
            start += 1;
        }
        width /= 2;
    }
    improved
}

/// Binary-search each entry towards zero.
///
/// Per entry this is `O(log n)` probes rather than the `O(n)` a decrement loop would take, which
/// matters because tape entries start as full 64-bit values. An entry that only needs to reach
/// `3` gets there in about six probes from `2^64`.
fn shrink_entries(
    tape: &mut [u64],
    probe: &mut impl FnMut(&[u64]) -> Probe,
    spent: &mut usize,
) -> bool {
    let mut improved = false;

    for position in 0..tape.len() {
        if tape[position] == 0 {
            continue;
        }
        // Invariant: `low` is known to be insufficient (or untried at 0), `high` is known to
        // still fail. Narrow until they meet.
        let mut low = 0u64;
        let mut high = tape[position];

        while low < high {
            if *spent >= MAX_PROBES {
                return improved;
            }
            let mid = low + (high - low) / 2;
            if mid == high {
                break;
            }
            let previous = tape[position];
            tape[position] = mid;
            *spent += 1;
            if probe(tape) == Probe::StillFails {
                high = mid;
                improved = true;
            } else {
                tape[position] = previous;
                low = mid + 1;
            }
        }
        tape[position] = high;
    }
    improved
}

/// Swap adjacent out-of-order pairs where doing so preserves the failure.
///
/// A canonicalisation rather than a size reduction: two tapes that describe the same graph with
/// its vertices permuted are equally small, and without this pass which one gets reported
/// depends on the seed. Sorting makes the reported counterexample a function of the *bug*
/// rather than of the search, so the same defect produces the same minimal case on every
/// machine — which is what makes a persisted regression meaningful.
fn sort_pairs(
    tape: &mut [u64],
    probe: &mut impl FnMut(&[u64]) -> Probe,
    spent: &mut usize,
) -> bool {
    let mut improved = false;

    for position in 1..tape.len() {
        if tape[position - 1] <= tape[position] {
            continue;
        }
        if *spent >= MAX_PROBES {
            return improved;
        }
        tape.swap(position - 1, position);
        *spent += 1;
        if probe(tape) == Probe::StillFails {
            improved = true;
        } else {
            tape.swap(position - 1, position);
        }
    }
    improved
}

#[cfg(test)]
mod tests;
