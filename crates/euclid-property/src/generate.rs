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
//! [`filter`](Gen::filter) and `assume!` both produce [`Rejected`], which is distinct from both
//! a failure and an [`Overrun`]. A rejected case is not evidence of anything — it is an input
//! the property does not claim to cover — so the runner discards it and draws again, up to a
//! budget. Hitting that budget is itself reported, because a generator that rejects nine cases
//! in ten is testing far less than its case count suggests.

pub mod float;

use std::rc::Rc;

use crate::source::{Overrun, Source};

/// A generated case was rejected by a filter or an `assume!`.
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
}

impl<T> Clone for Gen<T> {
    fn clone(&self) -> Self {
        Self {
            draw: Rc::clone(&self.draw),
        }
    }
}

impl<T: 'static> Gen<T> {
    /// Build a generator from a closure over a [`Source`].
    pub fn new(draw: impl Fn(&mut Source) -> Result<T, DrawError> + 'static) -> Self {
        Self {
            draw: Rc::new(draw),
        }
    }

    /// Read one value.
    pub fn draw(&self, source: &mut Source) -> Result<T, DrawError> {
        (self.draw)(source)
    }

    /// Transform every generated value.
    pub fn map<U: 'static>(self, f: impl Fn(T) -> U + 'static) -> Gen<U> {
        Gen::new(move |source| self.draw(source).map(&f))
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
        Gen::new(move |source| {
            let value = self.draw(source)?;
            if predicate(&value) {
                Ok(value)
            } else {
                Err(DrawError::Rejected(Rejected { reason }))
            }
        })
    }
}

/// The same value every time.
pub fn just<T: Clone + 'static>(value: T) -> Gen<T> {
    Gen::new(move |_| Ok(value.clone()))
}

/// An integer in the inclusive range `low..=high`.
///
/// # Panics
/// If `low > high`.
pub fn int(low: i64, high: i64) -> Gen<i64> {
    assert!(low <= high, "empty range {low}..={high}");
    Gen::new(move |source| Ok(source.in_range(low, high)?))
}

/// A `usize` in the inclusive range `low..=high`.
///
/// # Panics
/// If `low > high`.
pub fn index(low: usize, high: usize) -> Gen<usize> {
    assert!(low <= high, "empty range {low}..={high}");
    #[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
    Gen::new(move |source| Ok(source.in_range(low as i64, high as i64)? as usize))
}

/// A `bool`.
pub fn boolean() -> Gen<bool> {
    Gen::new(|source| Ok(source.boolean()?))
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
    Gen::new(move |source| {
        #[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
        let len = source.in_range(min as i64, max as i64)? as usize;
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            out.push(element.draw(source)?);
        }
        Ok(out)
    })
}

/// One of `choices`, uniformly.
///
/// # Panics
/// If `choices` is empty.
pub fn one_of<T: 'static>(choices: Vec<Gen<T>>) -> Gen<T> {
    assert!(!choices.is_empty(), "one_of needs at least one choice");
    Gen::new(move |source| {
        let pick = source.below(choices.len() as u64)? as usize;
        choices[pick].draw(source)
    })
}

/// One of `choices`, by relative weight.
///
/// # Panics
/// If `choices` is empty or every weight is zero.
pub fn weighted<T: 'static>(choices: Vec<(u32, Gen<T>)>) -> Gen<T> {
    assert!(!choices.is_empty(), "weighted needs at least one choice");
    let weights: Vec<u32> = choices.iter().map(|(w, _)| *w).collect();
    Gen::new(move |source| {
        let pick = source.weighted(&weights)?;
        choices[pick].1.draw(source)
    })
}

/// Two generators as a pair.
pub fn pair<A: 'static, B: 'static>(first: Gen<A>, second: Gen<B>) -> Gen<(A, B)> {
    Gen::new(move |source| Ok((first.draw(source)?, second.draw(source)?)))
}

/// Three generators as a triple.
pub fn triple<A: 'static, B: 'static, C: 'static>(
    first: Gen<A>,
    second: Gen<B>,
    third: Gen<C>,
) -> Gen<(A, B, C)> {
    Gen::new(move |source| {
        Ok((
            first.draw(source)?,
            second.draw(source)?,
            third.draw(source)?,
        ))
    })
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
