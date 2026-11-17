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

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::num::NonZeroUsize;

/// How many **distinct** candidate tapes a single minimisation may try.
///
/// A bound rather than a true fixpoint, because the passes can in principle keep finding
/// microscopic improvements for a long time on a large tape, and a shrink that takes a minute
/// is worse than one that stops slightly early. Reaching the cap is reported rather than
/// hidden, so a case that shrank badly says so.
///
/// Distinct, because a repeated candidate is answered from the memo below and does not spend
/// budget. That makes the cap a measure of *search effort* rather than of loop iterations, which
/// is what it was always meant to be.
pub const MAX_PROBES: usize = 4_096;

/// How many answers to remember. Beyond this, probing continues without memoising.
///
/// A tape is a `Vec<u64>` and a minimisation is bounded at [`MAX_PROBES`] distinct candidates,
/// so the table is small in practice — a 40-entry tape at the cap is about 1.5 MB, inside a test
/// binary that already holds the whole suite. The bound exists so a pathological tape cannot turn
/// a shrink into an allocation problem, which would trade a slow failure report for a worse one.
const MAX_MEMO: usize = MAX_PROBES;

/// The outcome of one candidate tape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Probe {
    /// The property still fails; keep this tape.
    StillFails,
    /// The property passes, was rejected, or the tape overran; discard this tape.
    Discard,
}

/// What a minimisation cost.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cost {
    /// Distinct candidate tapes actually run through the generator and the property.
    pub probes: usize,
    /// Candidates answered from the memo instead of being re-run.
    pub reused: usize,
}

impl Cost {
    /// Whether the search stopped on its budget rather than at a fixpoint, so the reported case
    /// may not be minimal.
    #[must_use]
    pub fn hit_the_cap(&self) -> bool {
        self.probes >= MAX_PROBES
    }
}

/// A hasher for tapes: fast, not adversary-resistant.
///
/// The default `SipHash` is the right default for a `HashMap` keyed on untrusted input and the
/// wrong one here. These keys are tapes the shrinker just built for itself — there is no attacker
/// and no collision-flooding risk — and they are hashed on *every* probe, which is the one place
/// in the crate where hashing is hot. This is the usual multiply-xor-rotate construction, which
/// is several times quicker on a run of `u64`s.
///
/// Note the `write_u64` and bulk `write` paths both matter: `Hash for [u64]` writes the length
/// through `write_usize` and then the elements as one bulk byte slice, so a byte-at-a-time
/// `write` would undo the point of being here.
#[derive(Default)]
struct TapeHasher(u64);

impl TapeHasher {
    #[inline]
    fn add(&mut self, value: u64) {
        const K: u64 = 0x517c_c1b7_2722_0a95;
        self.0 = (self.0 ^ value).wrapping_mul(K).rotate_left(26);
    }
}

impl Hasher for TapeHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let (words, remainder) = bytes.as_chunks::<8>();
        for word in words {
            self.add(u64::from_le_bytes(*word));
        }
        let mut tail = [0; 8];
        let start = tail.len() - remainder.len();
        tail[start..].copy_from_slice(remainder);
        self.add(u64::from_be_bytes(tail));
    }

    #[inline]
    fn write_u64(&mut self, value: u64) {
        self.add(value);
    }

    #[inline]
    fn write_usize(&mut self, value: usize) {
        self.add(value as u64);
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
}

/// The probe closure, plus the memory of what it has already answered.
///
/// # Why a memo is worth its hash
///
/// A third of the candidates a minimisation tries are ones it has already tried. That is not a
/// flaw in the passes, it is what a fixpoint loop *is*: the final round exists to prove nothing
/// more can be done, and it proves it by re-probing candidates earlier rounds rejected. Measured
/// on the benchmark in `benches/engine.rs`, 194 of 572 probes — 33.9% — were repeats.
///
/// Each of those re-runs the generator and the property. For the scalar suites that is cheap; for
/// a property whose body runs `tensor_laws` over three rank-4 tensors it is milliseconds, and
/// shrinking is already the dominant cost of a failure. Hashing a 40-entry tape is tens of
/// nanoseconds against that.
///
/// The second effect matters more than the speed. A memo hit **does not spend budget**, so the
/// [`MAX_PROBES`] cap now bounds distinct candidates rather than loop iterations — the same
/// budget explores strictly more of the search space, and a case that used to exhaust its budget
/// re-proving a fixpoint can now reach an actually smaller tape. Shrinking got better as well as
/// faster, and that is the part worth keeping.
struct Prober<F> {
    probe: F,
    memo: HashMap<Vec<u64>, Probe, BuildHasherDefault<TapeHasher>>,
    cost: Cost,
}

impl<F: FnMut(&[u64]) -> Probe> Prober<F> {
    fn new(probe: F) -> Self {
        Self {
            probe,
            memo: HashMap::default(),
            cost: Cost {
                probes: 0,
                reused: 0,
            },
        }
    }

    /// The verdict on `candidate`, from the memo if it is known.
    ///
    /// Queried by slice so a repeat costs no allocation — `Vec<u64>: Borrow<[u64]>` makes that
    /// work, and the `to_vec` happens only on a miss, where a probe is about to be spent anyway.
    fn ask(&mut self, candidate: &[u64]) -> Probe {
        if let Some(known) = self.memo.get(candidate) {
            self.cost.reused += 1;
            return *known;
        }

        let verdict = (self.probe)(candidate);
        self.cost.probes += 1;
        if self.memo.len() < MAX_MEMO {
            self.memo.insert(candidate.to_vec(), verdict);
        }
        verdict
    }

    fn exhausted(&self) -> bool {
        self.cost.probes >= MAX_PROBES
    }
}

/// Minimise `tape`, keeping only candidates for which `probe` reports [`Probe::StillFails`].
///
/// Returns the smallest tape found and how many probes were spent.
///
/// Each round must shorten the tape or lower it lexicographically. Memoized probes do
/// not spend budget, so the progress check also applies to rounds made entirely of memo hits.
pub fn minimise(tape: Vec<u64>, probe: impl FnMut(&[u64]) -> Probe) -> (Vec<u64>, Cost) {
    let mut best = tape;
    let mut prober = Prober::new(probe);
    // One buffer reused by every pass, so a candidate costs a `copy_from_slice` rather than an
    // allocation. Only a memo *miss* allocates, and only to own the key it is about to store.
    let mut scratch: Vec<u64> = Vec::with_capacity(best.len());

    while !prober.exhausted() {
        let before = best.clone();

        delete_chunks(&mut best, &mut prober, &mut scratch);
        zero_chunks(&mut best, &mut prober, &mut scratch);
        shrink_entries(&mut best, &mut prober);
        sort_pairs(&mut best, &mut prober);

        match (best.len(), &best).cmp(&(before.len(), &before)) {
            std::cmp::Ordering::Less => {}
            std::cmp::Ordering::Equal | std::cmp::Ordering::Greater => break,
        }
    }

    (best, prober.cost)
}

/// The nonzero widths obtained by repeatedly halving a tape's starting length.
fn chunk_widths(len: usize) -> impl Iterator<Item = NonZeroUsize> {
    std::iter::successors(NonZeroUsize::new(len), |width| {
        NonZeroUsize::new(width.get() >> 1)
    })
}

/// Try removing contiguous runs, longest first.
///
/// This is the pass that turns "eight nodes and eleven edges" into "two nodes and one edge",
/// because a list's length lives in one entry and its elements in the entries after it —
/// deleting a suffix of those elements and lowering the length is a single structural edit.
fn delete_chunks<F: FnMut(&[u64]) -> Probe>(
    tape: &mut Vec<u64>,
    prober: &mut Prober<F>,
    scratch: &mut Vec<u64>,
) {
    for width in chunk_widths(tape.len()) {
        let mut starts = (0..tape.len()).peekable();
        while let Some(&start) = starts.peek() {
            let (prefix, rest) = tape.split_at(start);
            let Some(suffix) = rest.get(width.get()..) else {
                break;
            };
            if prober.exhausted() {
                return;
            }
            scratch.clear();
            scratch.extend_from_slice(prefix);
            scratch.extend_from_slice(suffix);

            if prober.ask(scratch) == Probe::StillFails {
                tape.clear();
                tape.extend_from_slice(scratch);
                // Do not advance `start`: the window now holds different entries, which are
                // often deletable too. Re-probing the same offset is how a long irrelevant
                // run collapses in a few probes instead of one probe per entry.
            } else {
                starts.next();
            }
        }
    }
}

/// Try zeroing contiguous runs, longest first.
///
/// Distinct from deletion: zeroing keeps the tape's *length* and so keeps every later draw at
/// the same position, which is what lets a structurally required entry be minimised without
/// disturbing anything after it. Deleting it would shift the whole remainder and almost always
/// fail the probe.
fn zero_chunks<F: FnMut(&[u64]) -> Probe>(
    tape: &mut [u64],
    prober: &mut Prober<F>,
    scratch: &mut Vec<u64>,
) {
    for width in chunk_widths(tape.len()) {
        let width = width.get();
        for start in 0..=tape.len() - width {
            if prober.exhausted() {
                return;
            }
            let range = start..start + width;
            if tape[range.clone()].iter().all(|entry| *entry == 0) {
                continue;
            }
            scratch.clear();
            scratch.extend_from_slice(tape);
            scratch[range].fill(0);

            if prober.ask(scratch) == Probe::StillFails {
                tape.copy_from_slice(scratch);
            }
        }
    }
}

/// Binary-search each entry towards zero.
///
/// Per entry this is `O(log n)` probes rather than the `O(n)` a decrement loop would take, which
/// matters because tape entries start as full 64-bit values. An entry that only needs to reach
/// `3` gets there in about six probes from `2^64`.
fn shrink_entries<F: FnMut(&[u64]) -> Probe>(tape: &mut [u64], prober: &mut Prober<F>) {
    for position in 0..tape.len() {
        if tape[position] == 0 {
            continue;
        }
        // Invariant: `low` is known to be insufficient (or untried at 0), `high` is known to
        // still fail. Narrow until they meet.
        let mut low = 0u64;
        let mut high = tape[position];

        for _ in 0..u64::BITS {
            if low == high {
                break;
            }
            if prober.exhausted() {
                return;
            }
            let mid = low.midpoint(high);
            let previous = tape[position];
            tape[position] = mid;
            if prober.ask(tape) == Probe::StillFails {
                high = mid;
            } else {
                tape[position] = previous;
                low = mid.saturating_add(1);
            }
        }
        tape[position] = high;
    }
}

/// Swap adjacent out-of-order pairs where doing so preserves the failure.
///
/// A canonicalisation rather than a size reduction: two tapes that describe the same graph with
/// its vertices permuted are equally small, and without this pass which one gets reported
/// depends on the seed. Sorting makes the reported counterexample a function of the *bug*
/// rather than of the search, so the same defect produces the same minimal case on every
/// machine — which is what makes a persisted regression meaningful.
fn sort_pairs<F: FnMut(&[u64]) -> Probe>(tape: &mut [u64], prober: &mut Prober<F>) {
    for position in 1..tape.len() {
        if tape[position - 1] <= tape[position] {
            continue;
        }
        if prober.exhausted() {
            return;
        }
        tape.swap(position - 1, position);
        if prober.ask(tape) != Probe::StillFails {
            tape.swap(position - 1, position);
        }
    }
}

#[cfg(test)]
mod tests;
