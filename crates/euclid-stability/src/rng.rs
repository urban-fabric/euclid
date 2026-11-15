//! SplitMix64 — the one place this crate makes a random number.
//!
//! Thirteen lines of arithmetic, chosen because the requirements here are unusually weak and
//! unusually specific. A property search needs a stream that is *varied* and *reproducible from
//! a `u64`*. It does not need cryptographic strength, long-period guarantees, or multiple
//! streams, because every failure is persisted as a tape rather than as a seed to re-derive.
//!
//! This is also why the engine does not reach for [`euclid_random`]: that crate is a
//! *contract* — `RandomKey`, `RandomSampler` — and deliberately ships no PRNG, on the grounds
//! that a PRNG algorithm is a backend's choice. It would be the right dependency for generating
//! *statistical* samples; it is the wrong one for turning a seed into a tape, which is all that
//! happens here.
//!
//! SplitMix64 is the mixing function from Java 8's `SplittableRandom`, published by Steele,
//! Lea and Flood. It is in the public domain and reproduced from the published constants.
//!
//! [`euclid_random`]: https://github.com/urban-fabric/euclid

/// A reproducible `u64` stream from a `u64` seed.
#[derive(Clone, Copy, Debug)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Seed the stream. Every seed gives a different sequence, including zero — the increment
    /// below is applied before mixing, so a zero state is not a fixed point.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// The next value in the stream.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

#[cfg(test)]
mod tests {
    use super::SplitMix64;
    use std::collections::BTreeSet;

    #[test]
    fn a_seed_reproduces_its_stream() {
        let mut a = SplitMix64::new(12_345);
        let mut b = SplitMix64::new(12_345);
        for _ in 0..64 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_seeds_diverge_immediately() {
        let mut a = SplitMix64::new(1);
        let mut b = SplitMix64::new(2);
        assert_ne!(a.next_u64(), b.next_u64());
    }

    /// A zero seed must not be a fixed point, which is the failure mode of a naive xorshift and
    /// the reason the increment is applied before mixing rather than after.
    #[test]
    fn a_zero_seed_still_varies() {
        let mut rng = SplitMix64::new(0);
        let drawn: BTreeSet<u64> = (0..32).map(|_| rng.next_u64()).collect();
        assert_eq!(drawn.len(), 32, "a zero seed produced a repeat");
        assert!(!drawn.contains(&0), "zero stayed zero");
    }

    /// The low bits have to vary too: generators take `value % bound` with small bounds, so a
    /// stream with a stuck low bit would silently halve the search space.
    #[test]
    fn the_low_bits_vary() {
        let mut rng = SplitMix64::new(7);
        let mut ones = 0;
        const DRAWS: usize = 512;
        for _ in 0..DRAWS {
            ones += usize::from(rng.next_u64() & 1 == 1);
        }
        assert!(
            (DRAWS / 4..3 * DRAWS / 4).contains(&ones),
            "low bit was {ones}/{DRAWS} ones, which is not plausibly balanced"
        );
    }
}
