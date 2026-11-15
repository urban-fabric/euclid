//! The choice tape — the one idea the rest of this crate rests on.
//!
//! A generator never draws from a random number generator directly. It draws from a
//! [`Source`], which records every choice it hands out onto a **tape** of `u64`s. A generated
//! value is therefore a pure function of its tape: replay the tape and you rebuild the value
//! exactly, with no need to store the value itself or to know its type.
//!
//! That single property is what collapses shrinking from a per-type problem into a generic one.
//! Instead of every type explaining how to make itself smaller, the shrinker edits the *tape* —
//! shortening it, lowering entries — and re-runs the generator. Whatever comes out is a
//! structurally valid value by construction, because it came from the generator rather than
//! from a shrink rule that has to re-derive the generator's invariants and might get them
//! wrong.
//!
//! # Why this and not proptest's design
//!
//! proptest pairs each value with a `ValueTree` that knows how to shrink it. That works well
//! for independent values and badly for *dependent* ones: when a bound is computed from an
//! earlier draw — a node count that determines the legal range of edge endpoints — the shrink
//! tree has to re-run the dependent half every time the first half moves. proptest's own docs
//! say nested `prop_flat_map` combinators "risk exponential run time" and cap the work with a
//! `max_flat_map_regens` counter, after which shrinking simply stops improving.
//!
//! A tape has no such problem. Lowering the entry that produced the node count means the
//! generator re-reads the following entries under the *new* bound, so the dependency is
//! re-derived for free and every candidate stays in range. Dependent generation is the common
//! case for graphs, which is why the trade is worth making here.
//!
//! The cost is that a tape is opaque: a failing case is reported by replaying it, not by
//! pretty-printing a stored shrink path. In exchange, a persisted failure is 30 bytes of hex
//! that reproduce exactly, on any machine, forever.

use crate::rng::SplitMix64;

/// How a [`Source`] behaves when a generator reads past the end of the tape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    /// Extend the tape with fresh randomness. Used while searching for a failure.
    Grow,
    /// Refuse, by returning [`Overrun`]. Used while shrinking, so a candidate tape that no
    /// longer feeds the generator is discarded rather than silently topped up with new
    /// randomness — which would make the candidate a different test case and the shrink
    /// unsound.
    Fixed,
}

/// A generator read past the end of a fixed tape.
///
/// Not a test failure and not a rejection: it means this candidate tape does not describe a
/// complete value, so the shrinker discards it and tries another edit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Overrun;

/// The tape a generator draws from, and the record of what it drew.
#[derive(Clone, Debug)]
pub struct Source {
    tape: Vec<u64>,
    position: usize,
    rng: SplitMix64,
    mode: Mode,
    max_len: usize,
}

impl Source {
    /// A source that invents its tape as it goes, seeded from `seed`.
    #[must_use]
    pub fn generating(seed: u64, max_len: usize) -> Self {
        Self {
            tape: Vec::new(),
            position: 0,
            rng: SplitMix64::new(seed),
            mode: Mode::Grow,
            max_len,
        }
    }

    /// A source that replays `tape` exactly and overruns past its end.
    #[must_use]
    pub fn replaying(tape: Vec<u64>) -> Self {
        let max_len = tape.len();
        Self {
            tape,
            position: 0,
            rng: SplitMix64::new(0),
            mode: Mode::Fixed,
            max_len,
        }
    }

    /// The tape as recorded — the reproducible identity of the value that was generated.
    #[must_use]
    pub fn tape(&self) -> &[u64] {
        &self.tape
    }

    /// The tape truncated to what was actually consumed.
    ///
    /// A `Grow` source can leave unread entries when a generator takes fewer draws than a
    /// previous run; those are not part of this value and would otherwise be handed to the
    /// shrinker as dead weight to minimise.
    #[must_use]
    pub fn consumed(&self) -> Vec<u64> {
        self.tape[..self.position.min(self.tape.len())].to_vec()
    }

    /// Draw the next raw choice.
    fn next_raw(&mut self) -> Result<u64, Overrun> {
        if self.position < self.tape.len() {
            let value = self.tape[self.position];
            self.position += 1;
            return Ok(value);
        }
        match self.mode {
            Mode::Fixed => Err(Overrun),
            Mode::Grow => {
                if self.tape.len() >= self.max_len {
                    return Err(Overrun);
                }
                let value = self.rng.next_u64();
                self.tape.push(value);
                self.position += 1;
                Ok(value)
            }
        }
    }

    /// Draw a value in `0..bound`.
    ///
    /// # Why multiply-shift and not modulo
    ///
    /// This mapping is **monotone**: a smaller tape entry can never produce a larger value. That
    /// is not a nicety, it is the property the whole shrinker depends on, and getting it wrong
    /// is the single most expensive mistake available in this file.
    ///
    /// The obvious implementation is `entry % bound`, and it was the first one here. Under
    /// modulo, whether a case fails is *periodic* in the tape entry rather than monotone — for a
    /// node count drawn as `entry % 8`, the failing inputs are a residue class scattered across
    /// the whole `u64` range. [`shrink_entries`](crate::shrink) binary-searches an entry
    /// downward on the assumption that if a smaller value still fails then smaller values are
    /// worth trying, and periodicity breaks that assumption completely: the search probes the
    /// midpoint, lands on a different residue, sees the case pass, and concludes the entry
    /// cannot be lowered. The observable symptom was a tape that spent its entire probe budget
    /// and came back with its first two entries still full-width garbage.
    ///
    /// Multiply-shift fixes it by construction. `(entry × bound) >> 64` is non-decreasing in
    /// `entry`, so lowering an entry lowers the drawn value, and binary search works as
    /// intended: a node count shrinks to 1, a list length to 0, an index to 0.
    ///
    /// Still exactly one tape entry per draw, and still no rejection loop. A rejection loop
    /// would make the number of entries consumed depend on the *values* drawn, so an edit to one
    /// entry could shift every later read — turning a local shrink into a global
    /// reinterpretation of the tape. The residual non-uniformity (some outputs are reachable
    /// from one more tape value than others) is around one part in 2^64 for the bounds used
    /// here, and buys both of the properties above.
    pub fn below(&mut self, bound: u64) -> Result<u64, Overrun> {
        if bound <= 1 {
            // Still consume an entry, so the tape layout does not depend on the bound being
            // degenerate. A generator whose bound collapses to 1 on a shrunk tape must not
            // shift every subsequent read by one position.
            let _ = self.next_raw()?;
            return Ok(0);
        }
        let raw = self.next_raw()?;
        Ok(((u128::from(raw) * u128::from(bound)) >> 64) as u64)
    }

    /// Draw a value in the inclusive range `low..=high`.
    ///
    /// # Panics
    /// If `low > high`.
    pub fn in_range(&mut self, low: i64, high: i64) -> Result<i64, Overrun> {
        assert!(low <= high, "empty range {low}..={high}");
        let span = high.wrapping_sub(low) as u64;
        let offset = self.below(span.wrapping_add(1))?;
        Ok(low.wrapping_add(offset as i64))
    }

    /// Draw a `bool`, true with probability one half.
    pub fn boolean(&mut self) -> Result<bool, Overrun> {
        Ok(self.below(2)? == 1)
    }

    /// Draw an index in `0..len` weighted by `weights`.
    ///
    /// # Panics
    /// If `weights` is empty or sums to zero.
    pub fn weighted(&mut self, weights: &[u32]) -> Result<usize, Overrun> {
        let total: u64 = weights.iter().map(|w| u64::from(*w)).sum();
        assert!(total > 0, "weights must sum to a positive value");
        let mut pick = self.below(total)?;
        for (index, weight) in weights.iter().enumerate() {
            let weight = u64::from(*weight);
            if pick < weight {
                return Ok(index);
            }
            pick -= weight;
        }
        Ok(weights.len() - 1)
    }
}

#[cfg(test)]
mod tests;
