//! A counter-based reference sampler, used to prove the contracts in this crate are
//! implementable end to end.
//!
//! `ToySampler` is an eager scalar backend: [`Real`] is `f64`, [`Bool`] is `bool`, and
//! [`Int`] is `i64`. It is a *reference*, not a production PRNG — the mixing function is a
//! splitmix64 variant and the rejection loops are iteration-bounded, which is enough to
//! exercise every method on [`RandomKey`], [`RandomSampler`], [`GammaSampler`], and
//! [`CountSampler`] without pulling a dependency.
//!
//! Being counter-based rather than state-based is what makes the explicit-key contract
//! natural: a draw hashes its key, so it is a pure function of that key, and
//! [`split`](RandomKey::split)/[`fold_in`](RandomKey::fold_in) are just further mixing.
//!
//! [`Real`]: RandomSampler::Real
//! [`Bool`]: RandomSampler::Bool
//! [`Int`]: RandomSampler::Int

use crate::{CountSampler, GammaSampler, RandomKey, RandomSampler};

/// An eager scalar reference backend. Carries no state; every draw is a hash of its key.
pub(super) struct ToySampler;

/// Rejection/accumulation loops are bounded so a pathological parameter can never hang the
/// test suite. Well above the handful of iterations the algorithms actually need.
const MAX_ITERS: u32 = 10_000;

/// Domain separators keeping `split` and `fold_in` from deriving the same child for the same
/// subordinate value.
const SPLIT_DOMAIN: u64 = 0x5F11_7000_0000_0001;
const FOLD_DOMAIN: u64 = 0xF01D_1000_0000_0002;

/// splitmix64's finalizer — cheap, dependency-free, and good enough to decorrelate keys.
fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A finite draw sequence derived from one key. Independent lanes within a single draw (the
/// two uniforms Box–Muller needs, the trials a binomial sums) come from successive counter
/// values, so they never reuse the same hash output.
struct Stream(u64);

impl Stream {
    fn new(key: u64) -> Self {
        Self(key)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        mix(self.0)
    }

    /// A uniform in `[0, 1)`, taking the top 53 bits — the exactly-representable `f64` range.
    fn next_u01(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// A uniform in `(0, 1]`, for use inside a logarithm.
    fn next_u01_positive(&mut self) -> f64 {
        1.0 - self.next_u01()
    }

    /// A standard normal via Box–Muller, consuming two uniforms.
    fn next_standard_normal(&mut self) -> f64 {
        let u1 = self.next_u01_positive();
        let u2 = self.next_u01();
        (-2.0 * u1.ln()).sqrt() * (core::f64::consts::TAU * u2).cos()
    }

    /// `Gamma(concentration, rate = 1)` by Marsaglia–Tsang, with Johnk's boost for `a < 1`.
    fn next_standard_gamma(&mut self, concentration: f64) -> f64 {
        if concentration <= 0.0 || concentration.is_nan() {
            return 0.0;
        }
        if concentration < 1.0 {
            let boosted = self.next_standard_gamma(concentration + 1.0);
            let u = self.next_u01_positive();
            return boosted * u.powf(1.0 / concentration);
        }

        let d = concentration - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        for _ in 0..MAX_ITERS {
            let x = self.next_standard_normal();
            let v = (1.0 + c * x).powi(3);
            if v <= 0.0 {
                continue;
            }
            let u = self.next_u01_positive();
            let x2 = x * x;
            if u < 1.0 - 0.0331 * x2 * x2 {
                return d * v;
            }
            if u.ln() < 0.5 * x2 + d * (1.0 - v + v.ln()) {
                return d * v;
            }
        }
        // Unreachable for valid input: acceptance probability per iteration is > 0.9.
        d
    }
}

/// Combine a key with a subordinate value (a child index, folded-in data, a domain tag).
///
/// The two arguments must not be interchangeable. Composing two `mix` outputs with a
/// *commutative* operator (`a + b`, `a ^ b`) is the obvious implementation and it is wrong:
/// since `key(n)` is itself `mix(n)`, `derive(key(1), 2)` and `derive(key(2), 1)` would
/// collapse to the same value, making independently-seeded key trees share children. Passing
/// the key through its own mixing stage first puts the two operands in structurally different
/// positions, so no such identity exists.
fn derive(key: u64, subordinate: u64) -> u64 {
    mix(mix(key).wrapping_add(subordinate).wrapping_add(1))
}

/// Derive the stream for a draw. The per-distribution domain tag keeps two different
/// distributions drawn from the *same* key independent, which is what callers expect when they
/// reuse a key across call sites.
fn stream(key: &u64, domain: u64) -> Stream {
    Stream::new(derive(*key, domain))
}

impl RandomKey for ToySampler {
    type Key = u64;

    fn key(seed: u64) -> Self::Key {
        mix(seed)
    }

    fn split(key: &Self::Key, count: usize) -> Vec<Self::Key> {
        // Domain-separated from `fold_in` so splitting to index `i` and folding in `i` are
        // different derivations.
        (0..count)
            .map(|i| derive(*key ^ SPLIT_DOMAIN, i as u64))
            .collect()
    }

    fn fold_in(key: &Self::Key, data: u64) -> Self::Key {
        derive(*key ^ FOLD_DOMAIN, data)
    }
}

impl RandomSampler for ToySampler {
    type Real = f64;
    type Bool = bool;
    type Int = i64;

    fn uniform(key: &Self::Key, _shape: &[usize], low: Self::Real, high: Self::Real) -> Self::Real {
        low + (high - low) * stream(key, 1).next_u01()
    }

    fn normal(
        key: &Self::Key,
        _shape: &[usize],
        mean: Self::Real,
        stddev: Self::Real,
    ) -> Self::Real {
        mean + stddev * stream(key, 2).next_standard_normal()
    }

    fn bernoulli(key: &Self::Key, _shape: &[usize], p: Self::Real) -> Self::Bool {
        stream(key, 3).next_u01() < p
    }

    fn randint(key: &Self::Key, _shape: &[usize], low: Self::Int, high: Self::Int) -> Self::Int {
        if high <= low {
            return low;
        }
        let span = high.wrapping_sub(low) as u64;
        low.wrapping_add((stream(key, 4).next_u64() % span) as i64)
    }
}

impl GammaSampler for ToySampler {
    fn standard_gamma(key: &Self::Key, _shape: &[usize], concentration: Self::Real) -> Self::Real {
        stream(key, 5).next_standard_gamma(concentration)
    }
}

impl CountSampler for ToySampler {
    fn poisson(key: &Self::Key, _shape: &[usize], rate: Self::Real) -> Self::Int {
        if rate <= 0.0 || rate.is_nan() {
            return 0;
        }
        // Knuth's product-of-uniforms method, in the log domain so a large rate cannot
        // underflow the running product to zero.
        let mut s = stream(key, 6);
        let mut accumulated = 0.0_f64;
        for k in 0..MAX_ITERS {
            accumulated -= s.next_u01_positive().ln();
            if accumulated > rate {
                return i64::from(k);
            }
        }
        i64::from(MAX_ITERS)
    }

    fn binomial(key: &Self::Key, _shape: &[usize], trials: Self::Real, p: Self::Real) -> Self::Int {
        if trials < 1.0 || trials.is_nan() || p <= 0.0 || p.is_nan() {
            return 0;
        }
        let n = (trials.floor() as u64).min(u64::from(MAX_ITERS));
        let mut s = stream(key, 7);
        let mut successes = 0_i64;
        for _ in 0..n {
            if s.next_u01() < p {
                successes += 1;
            }
        }
        successes
    }

    fn beta_binomial(
        key: &Self::Key,
        shape: &[usize],
        trials: Self::Real,
        a: Self::Real,
        b: Self::Real,
    ) -> Self::Int {
        // p ~ Beta(a, b), formed from the gamma pair, then Binomial(trials, p).
        let mut s = stream(key, 8);
        let ga = s.next_standard_gamma(a);
        let gb = s.next_standard_gamma(b);
        let total = ga + gb;
        let p = if total > 0.0 { ga / total } else { 0.5 };
        Self::binomial(&Self::fold_in(key, 0xB37A), shape, trials, p)
    }
}
