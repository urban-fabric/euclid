//! Generators and the combinators over them.
//!
//! A [`Gen<T>`] is a recipe for reading a `T` off a [`Source`]. It is one boxed closure behind
//! an [`Rc`], which is a deliberate choice: proptest gives every combinator its own type so the
//! whole pipeline monomorphises, which is the right call for a crate that must be fast and
//! generic over user strategies. Here the work per case is dominated by the property under
//! test, a graph closure or a matrix product, so one indirection per draw is free — and in
//! exchange the entire combinator set is a few lines each and `Gen<T>` is nameable in a
//! signature without `impl Strategy<Value = T>` spreading through every helper.
//!
//! # Rejection
//!
//! [`filter`](Gen::filter) produces [`Rejected`]; `assume!` returns a
//! [`FailureKind::Discard`](crate::FailureKind::Discard). Both are distinct from a failed
//! assertion and an [`Overrun`]. A rejected case is not evidence of anything — it is an input
//! the property does not claim to cover — so the runner discards it and draws again, up to a
//! budget. Hitting that budget is itself reported, because a generator that rejects nine cases
//! in ten is testing far less than its case count suggests.

mod compound;
mod explore;
pub mod float;
pub mod float32;

pub use compound::{array, i128, option, result, sampled_from, sized, u64, u128, uint};
use explore::{Metadata, product, range_tapes, sequence, weighted_tapes};

use std::rc::Rc;

use crate::source::{Overrun, Source};

/// A generator rejected a case through a filter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rejected {
    /// What was being assumed, for the diagnostic when the rejection budget runs out.
    pub reason: &'static str,
}

/// Why a draw produced no value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DrawError {
    /// The tape ran out. The candidate is incomplete; discard it silently.
    Overrun,
    /// The case was filtered out. Draw another.
    Rejected(Rejected),
}

impl From<Overrun> for DrawError {
    fn from(_: Overrun) -> Self {
        DrawError::Overrun
    }
}

/// The closure behind a [`Gen`], named so the type is readable in a signature.
type Draw<T> = Rc<dyn Fn(&mut Source) -> Result<T, DrawError>>;

/// A recipe for reading a `T` off a [`Source`].
pub struct Gen<T> {
    draw: Draw<T>,
    metadata: Rc<dyn Fn() -> Metadata>,
}

impl<T> Clone for Gen<T> {
    fn clone(&self) -> Self {
        Self {
            draw: Rc::clone(&self.draw),
            metadata: self.metadata.clone(),
        }
    }
}

impl<T: 'static> Gen<T> {
    /// Build a generator from a closure over a [`Source`].
    pub fn new(draw: impl Fn(&mut Source) -> Result<T, DrawError> + 'static) -> Self {
        Self {
            draw: Rc::new(draw),
            metadata: Rc::new(|| (None, Rc::new(Vec::new()))),
        }
    }

    pub(crate) fn exploration(&self) -> Metadata {
        (self.metadata)()
    }

    fn with_metadata(mut self, build: impl Fn() -> Metadata + 'static) -> Self {
        self.metadata = Rc::new(build);
        self
    }

    /// Read one value.
    pub fn draw(&self, source: &mut Source) -> Result<T, DrawError> {
        (self.draw)(source)
    }

    /// Transform every generated value.
    pub fn map<U: 'static>(self, f: impl Fn(T) -> U + 'static) -> Gen<U> {
        let metadata = self.metadata.clone();
        let mut result = Gen::new(move |source| self.draw(source).map(&f));
        result.metadata = metadata;
        result
    }

    /// Choose the next generator based on the value just drawn.
    ///
    /// This is the dependent case — a node count that bounds the edge endpoints after it — and
    /// the one the tape design exists to serve. There is no shrinking penalty for using it,
    /// and no counterpart to proptest's `max_flat_map_regens`, because the shrinker edits the
    /// tape and lets this closure re-derive the bound.
    pub fn and_then<U: 'static>(self, f: impl Fn(T) -> Gen<U> + 'static) -> Gen<U> {
        Gen::new(move |source| {
            let first = self.draw(source)?;
            f(first).draw(source)
        })
    }

    /// Keep only values satisfying `predicate`; reject the rest.
    pub fn filter(self, reason: &'static str, predicate: impl Fn(&T) -> bool + 'static) -> Self {
        let metadata = self.metadata.clone();
        let mut result = Gen::new(move |source| {
            let value = self.draw(source)?;
            if predicate(&value) {
                Ok(value)
            } else {
                Err(DrawError::Rejected(Rejected { reason }))
            }
        });
        result.metadata = metadata;
        result
    }

    /// Inject explicit edge values. This wrapper changes the generator's tape format.
    /// Random runs mix them in; `Config::edges_first` runs their complete tapes first.
    pub fn with_examples(self, values: Vec<T>) -> Self
    where
        T: Clone,
    {
        weighted(vec![(1, sampled_from(values)), (3, self)])
    }
}

/// The same value every time.
pub fn just<T: Clone + 'static>(value: T) -> Gen<T> {
    Gen::new(move |_| Ok(value.clone()))
        .with_metadata(|| (Some(Rc::new(vec![vec![]])), Rc::new(vec![vec![]])))
}

/// An integer in the inclusive range `low..=high`.
///
/// # Panics
/// If `low > high`.
pub fn int(low: i64, high: i64) -> Gen<i64> {
    assert!(low <= high, "empty range {low}..={high}");
    Gen::new(move |source| Ok(source.in_range(low, high)?)).with_metadata(move || {
        range_tapes(
            (core::primitive::i128::from(high) - core::primitive::i128::from(low) + 1) as u128,
        )
    })
}

/// A `usize` in the inclusive range `low..=high`.
///
/// # Panics
/// If `low > high`.
pub fn index(low: usize, high: usize) -> Gen<usize> {
    assert!(low <= high, "empty index range");
    uint(low as u64, high as u64).map(|value| value as usize)
}

/// A `bool`.
pub fn boolean() -> Gen<bool> {
    int(0, 1).map(|value| value == 1)
}

/// A `Vec<T>` whose length is in `min..=max`.
///
/// The length is drawn first, so lowering its tape entry shortens the list and the shrinker
/// gets list minimisation for free.
///
/// # Panics
/// If `min > max`.
pub fn vec<T: 'static>(element: Gen<T>, min: usize, max: usize) -> Gen<Vec<T>> {
    assert!(min <= max, "empty length range {min}..={max}");
    let metadata_element = element.clone();
    let result = Gen::new(move |source| {
        let len = source.in_unsigned_range(min as u64, max as u64)? as usize;
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            out.push(element.draw(source)?);
        }
        Ok(out)
    });
    result.with_metadata(move || sequence(&metadata_element, min, max))
}

/// A `Vec<T>` of pairwise-distinct values, with length in `min..=max`.
///
/// Distinct **by construction, not by rejection.** The obvious spelling is
/// `vec(element, min, max).filter("distinct", …)`, and it is a trap: the shrinker's whole job is
/// to make values smaller and more alike, so it walks straight into collisions and spends its
/// probe budget on candidates the filter throws away. Deduplicating instead means every draw and
/// every shrink candidate is valid, and the length simply comes out shorter than requested when
/// duplicates appeared.
///
/// That shortening is the documented behaviour rather than a wart: the result may be shorter than
/// `min`, down to one element. A caller needing a guaranteed length cannot have one — there is no
/// way to honour it without rejection — so the law under test has to tolerate a short list, which
/// every law over a sample already does.
///
/// The motivating case is a law with an inequality in it. `euclid-random`'s `key_laws` asserts
/// that every *pair* of seeds draws differently, so a sample containing the same seed twice fails
/// it — and that failure says nothing about the backend, because the same seed is *required* to
/// draw the same value. Distinctness there is a precondition on the sample, which a hand-written
/// list satisfies by inspection and a generator has to be told about.
///
/// # Panics
/// If `min > max`.
pub fn distinct<T: PartialEq + 'static>(element: Gen<T>, min: usize, max: usize) -> Gen<Vec<T>> {
    assert!(min <= max, "empty length range {min}..={max}");
    let metadata_element = element.clone();
    let result = Gen::new(move |source| {
        let len = source.in_unsigned_range(min as u64, max as u64)? as usize;
        let mut out: Vec<T> = Vec::with_capacity(len);
        for _ in 0..len {
            let candidate = element.draw(source)?;
            // Quadratic, and deliberately so: these samples are a handful of elements, and
            // `PartialEq` is a far weaker bound than the `Ord` or `Hash` a cleverer structure
            // would need. A sample large enough for that to matter is too large for the `O(n³)`
            // laws it exists to feed.
            if !out.contains(&candidate) {
                out.push(candidate);
            }
        }
        Ok(out)
    });
    result.with_metadata(move || sequence(&metadata_element, min, max))
}

/// A permutation of `0..n`, as a `Vec<usize>` of length `n`.
///
/// Every draw is a genuine bijection — built by sorting the indices against generated keys, so
/// there is nothing to reject and no way to produce a repeated or out-of-range axis. `n == 0`
/// gives the empty permutation, which is correct and is the rank-0 case a tensor law has to skip
/// rather than fail on.
///
/// Written because the alternative is wrong in a way that is hard to see. Generating `n` indices
/// and hoping they form a permutation fails almost always for any interesting `n`; drawing them
/// and then "fixing up" the duplicates biases the result towards the identity, which is the one
/// permutation that passes a broken `transpose`. Argsort has neither problem: the keys are
/// unconstrained, so the permutations come out spread across the symmetric group.
///
/// The motivating case is `euclid-tensor`, where `structural_laws` can only name the three
/// permutations it can write down without knowing the rank — the identity, a swap, and a
/// 3-cycle. A generated permutation covers the rest of the group, which is what caught that
/// crate's 3-cycle being malformed at rank 4.
pub fn permutation(n: usize) -> Gen<Vec<usize>> {
    Gen::new(move |source| {
        let mut keys = Vec::with_capacity(n);
        for _ in 0..n {
            keys.push(source.in_range(0, 1 << 20)?);
        }
        let mut axes: Vec<usize> = (0..n).collect();
        // Ties broken by index, so the comparison is a total order and the result is a genuine
        // bijection even when two keys collide.
        axes.sort_by_key(|&i| (keys[i], i));
        Ok(axes)
    })
}

/// One of `choices`, uniformly.
///
/// # Panics
/// If `choices` is empty.
pub fn one_of<T: 'static>(choices: Vec<Gen<T>>) -> Gen<T> {
    assert!(!choices.is_empty(), "one_of needs at least one choice");
    weighted(
        choices
            .into_iter()
            .map(|generator| (1, generator))
            .collect(),
    )
}

/// One of `choices`, by relative weight.
///
/// # Panics
/// If `choices` is empty or every weight is zero.
pub fn weighted<T: 'static>(choices: Vec<(u32, Gen<T>)>) -> Gen<T> {
    assert!(!choices.is_empty(), "weighted needs at least one choice");
    let weights: Vec<u32> = choices.iter().map(|(w, _)| *w).collect();
    let metadata_choices = choices.clone();
    let result = Gen::new(move |source| {
        let pick = source.weighted(&weights)?;
        choices[pick].1.draw(source)
    });
    result.with_metadata(move || weighted_tapes(&metadata_choices))
}

/// Two generators as a pair.
pub fn pair<A: 'static, B: 'static>(first: Gen<A>, second: Gen<B>) -> Gen<(A, B)> {
    let metadata_first = first.clone();
    let metadata_second = second.clone();
    Gen::new(move |source| Ok((first.draw(source)?, second.draw(source)?))).with_metadata(
        move || {
            let (finite_a, edges_a) = metadata_first.exploration();
            let (finite_b, edges_b) = metadata_second.exploration();
            (
                product(finite_a.as_deref(), finite_b.as_deref()).map(Rc::new),
                Rc::new(product(Some(&edges_a), Some(&edges_b)).unwrap_or_default()),
            )
        },
    )
}

/// Three generators as a triple.
pub fn triple<A: 'static, B: 'static, C: 'static>(
    first: Gen<A>,
    second: Gen<B>,
    third: Gen<C>,
) -> Gen<(A, B, C)> {
    pair(pair(first, second), third).map(|((a, b), c)| (a, b, c))
}

/// A self-referential generator, for trees and expressions.
///
/// `leaf` is the base case; `build` receives a generator for the next level down and returns
/// one for this level. `depth` bounds the recursion.
///
/// The depth budget is spent per *level*, not per node, and the recursion stops dead at zero by
/// falling back to `leaf`. That bound is what keeps a recursive generator from producing a
/// value whose size explodes with the tape length — and because the depth is read from the tape
/// like everything else, the shrinker collapses a deep expression to a shallow one by lowering
/// those entries.
pub fn recursive<T: 'static>(
    leaf: Gen<T>,
    depth: usize,
    build: impl Fn(Gen<T>) -> Gen<T> + 'static,
) -> Gen<T> {
    let mut current = leaf.clone();
    for _ in 0..depth {
        let deeper = build(current.clone());
        // At each level, either stop (leaf) or go deeper. Weighted towards stopping so the
        // expected size stays small; an unweighted choice makes the average case the maximum
        // depth, which is slow and no better at finding bugs.
        current = weighted(vec![(1, leaf.clone()), (2, deeper)]);
    }
    current
}

#[cfg(test)]
mod tests;
