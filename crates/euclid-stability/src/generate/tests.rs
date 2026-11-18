//! Generators: that they stay in their declared bounds, compose, and reject rather than lie.
//!
//! Basic combinators live here. `composition` checks wide values, sizing and replay;
//! `single_precision` checks native floats and extreme ranges; `exploration_metadata` checks
//! finite-domain completeness and bounded allocation.

use super::{
    DrawError, Gen, boolean, distinct, index, int, just, one_of, pair, permutation, recursive,
    triple, vec, weighted,
};
use crate::source::Source;

fn draw_many<T: 'static>(generator: &Gen<T>, count: usize, seed: u64) -> Vec<T> {
    let mut source = Source::generating(seed, 65_536);
    (0..count)
        .filter_map(|_| generator.draw(&mut source).ok())
        .collect()
}

#[test]
fn just_is_constant() {
    let drawn = draw_many(&just(42u32), 16, 1);
    assert!(drawn.iter().all(|v| *v == 42));
}

#[test]
fn int_stays_in_range() {
    let drawn = draw_many(&int(-5, 5), 400, 2);
    assert!(drawn.iter().all(|v| (-5..=5).contains(v)));
    assert!(drawn.iter().any(|v| *v < 0), "never produced a negative");
    assert!(drawn.iter().any(|v| *v > 0), "never produced a positive");
}

#[test]
fn index_stays_in_range() {
    let drawn = draw_many(&index(2, 6), 400, 3);
    assert!(drawn.iter().all(|v| (2..=6).contains(v)));
}

#[test]
fn boolean_produces_both() {
    let drawn = draw_many(&boolean(), 200, 4);
    assert!(drawn.contains(&true) && drawn.contains(&false));
}

#[test]
fn vec_respects_its_length_bounds() {
    let drawn = draw_many(&vec(int(0, 9), 2, 5), 200, 5);
    assert!(drawn.iter().all(|xs| (2..=5).contains(&xs.len())));
    assert!(
        drawn.iter().any(|xs| xs.len() == 2),
        "never hit the minimum"
    );
    assert!(
        drawn.iter().any(|xs| xs.len() == 5),
        "never hit the maximum"
    );
}

#[test]
fn an_empty_vec_is_reachable_when_allowed() {
    let drawn = draw_many(&vec(int(0, 9), 0, 4), 200, 6);
    assert!(drawn.iter().any(std::vec::Vec::is_empty));
}

#[test]
fn map_transforms_every_value() {
    let drawn = draw_many(&int(0, 10).map(|v| v * 2), 100, 7);
    assert!(drawn.iter().all(|v| v % 2 == 0 && (0..=20).contains(v)));
}

/// The dependent case the whole design exists for: the second draw's legal range is computed
/// from the first. Every produced pair must respect it.
#[test]
fn and_then_threads_the_dependency() {
    let generator = index(1, 6).and_then(|n| vec(index(0, n - 1), 0, 4).map(move |xs| (n, xs)));
    let drawn = draw_many(&generator, 300, 8);
    assert!(!drawn.is_empty());
    for (n, xs) in drawn {
        assert!(
            xs.iter().all(|x| *x < n),
            "index escaped its dependent bound: n={n}, xs={xs:?}"
        );
    }
}

#[test]
fn filter_rejects_rather_than_looping() {
    let generator = int(0, 100).filter("must be even", |v| v % 2 == 0);
    let mut source = Source::generating(9, 4_096);
    let mut rejected = 0;
    let mut accepted = 0;
    for _ in 0..200 {
        match generator.draw(&mut source) {
            Ok(value) => {
                assert_eq!(value % 2, 0);
                accepted += 1;
            }
            Err(DrawError::Rejected(reason)) => {
                assert_eq!(reason.reason, "must be even");
                rejected += 1;
            }
            Err(DrawError::Overrun) => break,
        }
    }
    assert!(accepted > 0 && rejected > 0, "expected a mix of both");
}

#[test]
fn pair_and_triple_draw_each_component() {
    let pairs = draw_many(&pair(int(0, 3), int(10, 13)), 100, 10);
    assert!(
        pairs
            .iter()
            .all(|(a, b)| (0..=3).contains(a) && (10..=13).contains(b))
    );

    let triples = draw_many(&triple(int(0, 1), int(2, 3), int(4, 5)), 100, 11);
    assert!(
        triples
            .iter()
            .all(|(a, b, c)| (0..=1).contains(a) && (2..=3).contains(b) && (4..=5).contains(c))
    );
}

#[test]
fn one_of_reaches_every_branch() {
    let generator = one_of(vec![just(1u8), just(2), just(3)]);
    let drawn = draw_many(&generator, 300, 12);
    for wanted in 1..=3 {
        assert!(drawn.contains(&wanted), "branch {wanted} never chosen");
    }
}

#[test]
fn weighted_never_picks_a_zero_weight_branch() {
    let generator = weighted(vec![(3, just('a')), (0, just('b')), (1, just('c'))]);
    let drawn = draw_many(&generator, 300, 13);
    assert!(!drawn.contains(&'b'), "a zero-weight branch was chosen");
    assert!(drawn.contains(&'a') && drawn.contains(&'c'));
}

/// A recursive generator must terminate and must actually recurse — producing only leaves would
/// pass a naive test while testing nothing nested.
#[test]
fn recursive_terminates_and_still_nests() {
    #[derive(Debug)]
    enum Tree {
        Leaf,
        Node(Box<Tree>, Box<Tree>),
    }

    fn depth(tree: &Tree) -> usize {
        match tree {
            Tree::Leaf => 1,
            Tree::Node(l, r) => 1 + depth(l).max(depth(r)),
        }
    }

    let leaf: Gen<Tree> = Gen::new(|_| Ok(Tree::Leaf));
    let generator = recursive(leaf, 4, |inner| {
        pair(inner.clone(), inner).map(|(l, r)| Tree::Node(Box::new(l), Box::new(r)))
    });

    let drawn = draw_many(&generator, 200, 14);
    assert!(!drawn.is_empty(), "recursion produced nothing");

    let deepest = drawn.iter().map(depth).max().unwrap_or(0);
    assert!(deepest > 1, "never recursed past a leaf");
    assert!(deepest <= 5, "exceeded the depth budget: {deepest}");
}

/// A generator must be reproducible from its tape — the invariant the shrinker's probes rely on.
#[test]
fn a_generated_value_is_a_function_of_its_tape() {
    let generator = pair(index(1, 8), vec(int(0, 20), 0, 6));

    let mut generating = Source::generating(21, 4_096);
    let first = generator.draw(&mut generating).expect("room on the tape");
    let tape = generating.consumed();

    let mut replaying = Source::replaying(tape);
    let second = generator.draw(&mut replaying).expect("tape is complete");

    assert_eq!(first, second);
}

// ===== distinct ===========================================================

/// The point of `distinct` is that no draw contains a repeat, even when the element generator is
/// narrow enough that collisions are the norm.
#[test]
fn distinct_never_repeats_a_value() {
    // Three possible elements and a requested length of eight, so a plain `vec` would repeat in
    // essentially every draw.
    let generator = distinct(int(0, 2), 8, 8);
    for (case, drawn) in draw_many(&generator, 64, 7).into_iter().enumerate() {
        let mut seen = drawn.clone();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), drawn.len(), "case {case} repeated: {drawn:?}");
    }
}

/// The documented consequence: deduplication shortens the result, so the length is a ceiling
/// rather than a guarantee. Asserted so the behaviour cannot drift into rejection later without
/// someone noticing — rejection is what this function exists to avoid.
#[test]
fn distinct_comes_out_short_rather_than_rejecting() {
    let generator = distinct(int(0, 2), 8, 8);
    let drawn = draw_many(&generator, 32, 11);
    assert!(!drawn.is_empty(), "nothing was drawn at all");
    assert!(
        drawn.iter().any(|v| v.len() < 8),
        "no draw was shortened, so this is not testing the dedup path"
    );
    assert!(
        drawn.iter().all(|v| !v.is_empty() && v.len() <= 3),
        "every draw must be non-empty and within the element range's cardinality"
    );
}

#[test]
fn distinct_is_a_function_of_its_tape() {
    let generator = distinct(int(0, 50), 2, 6);
    let mut generating = Source::generating(5, 4_096);
    let first = generator.draw(&mut generating).expect("room on the tape");
    let mut replaying = Source::replaying(generating.consumed());
    assert_eq!(first, generator.draw(&mut replaying).expect("complete"));
}

// ===== permutation ========================================================

/// Every draw must be a genuine bijection of `0..n`. A generator that produced a repeated or
/// out-of-range axis would be worse than useless: `transpose_axes` would be handed an invalid
/// permutation and whatever it did with it would be blamed on the backend.
#[test]
fn every_permutation_is_a_bijection() {
    for n in 0..8usize {
        let generator = permutation(n);
        for drawn in draw_many(&generator, 32, 1_000 + n as u64) {
            assert_eq!(drawn.len(), n, "wrong length for n = {n}: {drawn:?}");
            let mut sorted = drawn.clone();
            sorted.sort_unstable();
            assert_eq!(
                sorted,
                (0..n).collect::<Vec<_>>(),
                "not a permutation of 0..{n}: {drawn:?}"
            );
        }
    }
}

/// `n = 0` is the rank-0 case, and must give the empty permutation rather than failing.
#[test]
fn the_empty_permutation_is_valid() {
    let drawn = draw_many(&permutation(0), 4, 3);
    assert!(drawn.iter().all(Vec::is_empty));
}

/// Argsort was chosen over "draw indices and fix up the duplicates" because the fix-up biases
/// towards the identity — and the identity is the one permutation a broken `transpose` survives.
/// So the generator has to actually produce non-identity permutations.
#[test]
fn permutations_are_not_all_the_identity() {
    let identity: Vec<usize> = (0..5).collect();
    let drawn = draw_many(&permutation(5), 64, 17);
    assert!(
        drawn.iter().any(|p| *p != identity),
        "every draw was the identity, so the symmetric group is not being covered"
    );
}

#[test]
fn permutation_is_a_function_of_its_tape() {
    let generator = permutation(6);
    let mut generating = Source::generating(9, 4_096);
    let first = generator.draw(&mut generating).expect("room on the tape");
    let mut replaying = Source::replaying(generating.consumed());
    assert_eq!(first, generator.draw(&mut replaying).expect("complete"));
}

mod composition;
mod exploration_metadata;
mod single_precision;
