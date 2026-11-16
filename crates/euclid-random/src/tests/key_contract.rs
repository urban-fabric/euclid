//! The [`RandomKey`] contract: keys are immutable, and splitting/folding yields independent
//! children deterministically.

use std::collections::HashSet;

use super::toy_sampler::ToySampler;
use crate::RandomKey;

type S = ToySampler;

#[test]
fn key_is_reproducible_from_a_seed() {
    assert_eq!(S::key(7), S::key(7));
    assert_ne!(S::key(7), S::key(8));
}

#[test]
fn split_is_deterministic_and_yields_distinct_children() {
    let root = S::key(42);
    let a = S::split(&root, 8);
    let b = S::split(&root, 8);

    assert_eq!(a, b, "split must be a pure function of its key");
    assert_eq!(a.len(), 8);

    let distinct: HashSet<_> = a.iter().collect();
    assert_eq!(distinct.len(), 8, "children must be pairwise distinct");
    assert!(
        !a.contains(&root),
        "no child may collide with its parent key"
    );
}

#[test]
fn split_of_zero_is_empty() {
    assert!(S::split(&S::key(1), 0).is_empty());
}

#[test]
fn split_children_differ_between_parents() {
    let left = S::split(&S::key(1), 4);
    let right = S::split(&S::key(2), 4);
    assert!(
        left.iter().all(|k| !right.contains(k)),
        "independent parents must not share children"
    );
}

#[test]
fn key_derivation_is_not_commutative() {
    // Regression: combining the parent key with the child index using a commutative operator
    // (`+` or `^`) makes the two interchangeable. Because `key(n)` is itself a hash of `n`,
    // `split(key(a))[b]` then equals `split(key(b))[a]` — independently-seeded trees share
    // children. Derivation must put key and index in structurally different positions.
    for a in 0..6_u64 {
        for b in 0..6_u64 {
            if a == b {
                continue;
            }
            let from_a = S::split(&S::key(a), 8);
            let from_b = S::split(&S::key(b), 8);
            assert_ne!(
                from_a[b as usize], from_b[a as usize],
                "split(key({a}))[{b}] collided with split(key({b}))[{a}]"
            );
            assert_ne!(
                S::fold_in(&S::key(a), b),
                S::fold_in(&S::key(b), a),
                "fold_in(key({a}), {b}) collided with fold_in(key({b}), {a})"
            );
        }
    }
}

#[test]
fn split_and_fold_in_are_different_derivations() {
    // Taking child `i` and folding in `i` are distinct operations and must not alias.
    let root = S::key(11);
    let children = S::split(&root, 16);
    for (i, child) in children.iter().enumerate() {
        assert_ne!(*child, S::fold_in(&root, i as u64));
    }
}

#[test]
fn fold_in_is_deterministic_and_data_dependent() {
    let root = S::key(99);
    assert_eq!(S::fold_in(&root, 3), S::fold_in(&root, 3));
    assert_ne!(S::fold_in(&root, 3), S::fold_in(&root, 4));
    assert_ne!(S::fold_in(&root, 0), root, "folding must move the key");
}

#[test]
fn fold_in_over_a_loop_index_gives_distinct_keys() {
    // The documented use: deriving per-iteration randomness from one root key.
    let root = S::key(5);
    let keys: HashSet<_> = (0..64).map(|i| S::fold_in(&root, i)).collect();
    assert_eq!(keys.len(), 64);
}

// --- the public conformance suites ----------------------------------------------------------
//
// The laws themselves are `crate::conformance`, which is public API a backend calls. Running
// them here proves the suites are satisfiable, and that the reference backend satisfies them.
// The tests above remain because they check properties of *this* key type — it happens to be
// `Eq + Hash`, which the contract does not require and the generic suites cannot assume.

impl crate::conformance::ConformanceSample for ToySampler {
    /// `0` first, because a multiplicative mixing function that forgets to handle it
    /// degenerates there and every derivation law would then pass vacuously.
    fn seeds() -> Vec<u64> {
        vec![0, 1, 7, 42, 0xDEAD_BEEF]
    }
}

#[test]
fn toy_sampler_satisfies_the_key_laws() {
    crate::conformance::key_laws_for::<ToySampler>("ToySampler");
}

#[test]
fn toy_sampler_satisfies_the_sampler_bound_laws() {
    crate::conformance::sampler_bound_laws::<ToySampler>(&[0, 1, 7, 42], "ToySampler");
}

/// `sampler_bound_laws` must **reject** a sampler that leaves its range.
///
/// [`toy_sampler_satisfies_the_sampler_bound_laws`] runs the suite against a correct sampler and
/// checks it stays quiet, which cannot tell a law that holds from a law that is never checked.
/// Mutation testing found both: the whole suite was replaceable with an empty body, and the `two`
/// it builds as `one + one` could be made `one - one` — collapsing the test range to `[0, 0]`, so
/// every bound check became trivially true while still *looking* like a range check.
///
/// The sampler below draws outside the interval it is handed, which is the defect the suite's
/// first assertion exists to catch and the most consequential thing a backend can get wrong: a
/// `uniform(0, 1)` that occasionally returns `1.4` silently corrupts every rejection sampler
/// built on it.
#[test]
fn the_sampler_bound_laws_reject_a_sampler_that_escapes_its_range() {
    /// A sampler identical to [`ToySampler`] except that `uniform` overshoots its upper bound.
    struct Escapes;

    impl crate::RandomKey for Escapes {
        type Key = <ToySampler as crate::RandomKey>::Key;
        fn key(seed: u64) -> Self::Key {
            <ToySampler as crate::RandomKey>::key(seed)
        }
        fn split(key: &Self::Key, count: usize) -> Vec<Self::Key> {
            <ToySampler as crate::RandomKey>::split(key, count)
        }
        fn fold_in(key: &Self::Key, data: u64) -> Self::Key {
            <ToySampler as crate::RandomKey>::fold_in(key, data)
        }
    }

    impl crate::RandomSampler for Escapes {
        type Real = f64;
        type Bool = bool;
        type Int = i64;

        fn uniform(
            key: &Self::Key,
            shape: &[usize],
            low: Self::Real,
            high: Self::Real,
        ) -> Self::Real {
            // Overshoots by a fifth of the range, so a non-degenerate interval is escaped.
            let drawn = <ToySampler as crate::RandomSampler>::uniform(key, shape, low, high);
            drawn + (high - low) * 0.2
        }
        fn normal(
            key: &Self::Key,
            shape: &[usize],
            mean: Self::Real,
            stddev: Self::Real,
        ) -> Self::Real {
            <ToySampler as crate::RandomSampler>::normal(key, shape, mean, stddev)
        }
        fn bernoulli(key: &Self::Key, shape: &[usize], p: Self::Real) -> Self::Bool {
            <ToySampler as crate::RandomSampler>::bernoulli(key, shape, p)
        }
        fn randint(key: &Self::Key, shape: &[usize], low: Self::Int, high: Self::Int) -> Self::Int {
            <ToySampler as crate::RandomSampler>::randint(key, shape, low, high)
        }
    }

    /// Whether `body` panicked, with the hook silenced so an expected panic prints nothing.
    fn rejects(body: impl FnOnce()) -> bool {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(body));
        std::panic::set_hook(previous);
        outcome.is_err()
    }

    assert!(
        rejects(|| crate::conformance::sampler_bound_laws::<Escapes>(&[0, 1, 7, 42], "Escapes")),
        "sampler_bound_laws must reject a uniform that draws outside its interval"
    );
}
