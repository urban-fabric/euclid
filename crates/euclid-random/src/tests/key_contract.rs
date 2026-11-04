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
