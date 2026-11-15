//! Metamorphic relations, named.
//!
//! A metamorphic relation says that two executions must agree, without either needing a known
//! answer. That is what makes it checkable over generated input: `f(f(x)) = x` can be verified
//! for a random `x` even when nobody knows what `f(x)` should be. It is the only kind of
//! property available for code with no oracle — a compiler, a path solver, a special function —
//! and it is most of what the fleet actually tests.
//!
//! The lineage is separate from property testing proper. T.Y. Chen named metamorphic testing in
//! 1998; the engineering landmarks are EMI for compilers (mutate dead code, demand identical
//! behaviour — hundreds of GCC and LLVM bugs) and SQLancer's TLP and NoREC for databases. What
//! those have in common is that the relation is *designed*, not stumbled on, and the same dozen
//! shapes recur. This module is those shapes.
//!
//! # Why they are here rather than written out each time
//!
//! Three reasons, in increasing order of importance.
//!
//! The relation gets a name, so a reader sees "this is an involution check" rather than
//! reconstructing it from two calls and an assertion. The failure message is written once and
//! says which law broke and on what. And — the real reason — **the degenerate cases are handled
//! in one place.** A hand-written commutativity check compares with `==`, which is wrong for a
//! float and silently wrong for a float that happens to be `NaN`; a hand-written round-trip
//! check forgets that the backward direction may be lossy. Each of those is a mistake that
//! looks like a passing test.
//!
//! # Comparison is a parameter
//!
//! Every relation takes a [`Compare`], because the right notion of "the same" is not a property
//! of the relation. `f(f(x)) = x` over a graph means structural equality; over an `f64` it means
//! within a couple of ulps, and asserting `==` there is the single most common way to write a
//! numerical test that cannot pass.
//!
//! ```
//! use euclid_property::relation::{Exact, Ulps, involution};
//! use euclid_property::{generate, property};
//!
//! property! {
//!     fn negation_is_an_involution(x in generate::int(-1000, 1000)) {
//!         involution("negate", x, |v| -v, &Exact)?;
//!     }
//!
//!     fn reciprocal_is_an_involution(x in generate::float::positive()) {
//!         // Two round trips through a division: a couple of ulps, not equality.
//!         involution("reciprocal", x, |v| 1.0 / v, &Ulps(2))?;
//!     }
//! }
//! ```
//!
//! Each returns [`Outcome`], so `?` inside a `property!` body reports the failure through the
//! same machinery as [`check!`](macro@crate::check).

use core::fmt::Debug;

use crate::runner::{Failure, Outcome};
use crate::tolerance;

/// What it means for two results to be "the same".
///
/// Implement this for a domain with its own notion of closeness. The four provided cover
/// everything in the fleet so far.
pub trait Compare<T> {
    /// Whether `a` and `b` count as the same result.
    fn same(&self, a: &T, b: &T) -> bool;

    /// How to describe this comparison in a failure message.
    fn describe(&self) -> String;
}

/// Bit-for-bit (well, [`PartialEq`]) equality.
///
/// Correct for integers, graphs, and anything discrete. **Wrong for a computed float**, where
/// it is the comparison that makes a true property look false.
pub struct Exact;

impl<T: PartialEq> Compare<T> for Exact {
    fn same(&self, a: &T, b: &T) -> bool {
        a == b
    }
    fn describe(&self) -> String {
        "exactly equal".to_string()
    }
}

/// Equal to within `n` representable steps. The right default for floats.
pub struct Ulps(pub u64);

impl Compare<f64> for Ulps {
    fn same(&self, a: &f64, b: &f64) -> bool {
        tolerance::within_ulps(*a, *b, self.0)
    }
    fn describe(&self) -> String {
        format!("within {} ulp", self.0)
    }
}

/// Equal to within a relative error.
pub struct Relative(pub f64);

impl Compare<f64> for Relative {
    fn same(&self, a: &f64, b: &f64) -> bool {
        tolerance::within_relative(*a, *b, self.0)
    }
    fn describe(&self) -> String {
        format!("within {:e} relative", self.0)
    }
}

/// Equal to within an absolute difference. Only for quantities with a known scale.
pub struct Absolute(pub f64);

impl Compare<f64> for Absolute {
    fn same(&self, a: &f64, b: &f64) -> bool {
        tolerance::within_absolute(*a, *b, self.0)
    }
    fn describe(&self) -> String {
        format!("within {:e} absolute", self.0)
    }
}

fn broke<T: Debug>(law: &str, name: &str, comparison: &str, left: &T, right: &T) -> Failure {
    Failure::new(
        format!(
            "{name}: {law} does not hold ({comparison})\n    left:  {left:?}\n    right: {right:?}"
        ),
        "a metamorphic relation",
    )
}

/// `f(f(x)) = x` — applying `f` twice returns the original.
///
/// Transpose, negate, reverse, complement, reciprocal. The reason to check it rather than
/// `f(x) != x` is that a one-sided test passes for a function that does nothing at all.
pub fn involution<T: Clone + Debug>(
    name: &str,
    x: T,
    f: impl Fn(T) -> T,
    compare: &impl Compare<T>,
) -> Outcome {
    let once = f(x.clone());
    let twice = f(once);
    if compare.same(&twice, &x) {
        Ok(())
    } else {
        Err(broke("f(f(x)) = x", name, &compare.describe(), &twice, &x))
    }
}

/// `f(f(x)) = f(x)` — applying `f` again changes nothing.
///
/// What makes something a *closure* or a normalisation rather than merely a transformation:
/// sorting, deduplicating, taking a transitive closure, canonicalising.
pub fn idempotent<T: Clone + Debug>(
    name: &str,
    x: T,
    f: impl Fn(T) -> T,
    compare: &impl Compare<T>,
) -> Outcome {
    let once = f(x);
    let twice = f(once.clone());
    if compare.same(&twice, &once) {
        Ok(())
    } else {
        Err(broke(
            "f(f(x)) = f(x)",
            name,
            &compare.describe(),
            &twice,
            &once,
        ))
    }
}

/// `back(forward(x)) = x` — encoding then decoding is the identity.
///
/// Serialisation, interchange between layouts, `frexp`/`ldexp`, unit conversion. Note the
/// direction: this checks that `forward` loses nothing, which is the claim worth making.
/// The other direction (`forward(back(y)) = y`) is a *different* property and only holds when
/// `forward` is surjective, so it needs its own call.
pub fn round_trip<A: Clone + Debug, B>(
    name: &str,
    x: A,
    forward: impl Fn(A) -> B,
    back: impl Fn(B) -> A,
    compare: &impl Compare<A>,
) -> Outcome {
    let returned = back(forward(x.clone()));
    if compare.same(&returned, &x) {
        Ok(())
    } else {
        Err(broke(
            "back(forward(x)) = x",
            name,
            &compare.describe(),
            &returned,
            &x,
        ))
    }
}

/// `op(a, b) = op(b, a)`.
pub fn commutative<T: Clone + Debug>(
    name: &str,
    a: T,
    b: T,
    op: impl Fn(T, T) -> T,
    compare: &impl Compare<T>,
) -> Outcome {
    let forward = op(a.clone(), b.clone());
    let reversed = op(b, a);
    if compare.same(&forward, &reversed) {
        Ok(())
    } else {
        Err(broke(
            "op(a, b) = op(b, a)",
            name,
            &compare.describe(),
            &forward,
            &reversed,
        ))
    }
}

/// `op(op(a, b), c) = op(a, op(b, c))`.
///
/// Worth checking on floats even though it is **false** for floating-point addition — which is
/// exactly why: the right call there is to assert it holds to a stated tolerance, and discover
/// at what magnitude the tolerance has to be absurd. That is a conditioning measurement, not a
/// law check.
pub fn associative<T: Clone + Debug>(
    name: &str,
    a: T,
    b: T,
    c: T,
    op: impl Fn(T, T) -> T,
    compare: &impl Compare<T>,
) -> Outcome {
    let left = op(op(a.clone(), b.clone()), c.clone());
    let right = op(a, op(b, c));
    if compare.same(&left, &right) {
        Ok(())
    } else {
        Err(broke(
            "op(op(a, b), c) = op(a, op(b, c))",
            name,
            &compare.describe(),
            &left,
            &right,
        ))
    }
}

/// `mul(a, add(b, c)) = add(mul(a, b), mul(a, c))` — left distributivity.
pub fn distributes<T: Clone + Debug>(
    name: &str,
    a: T,
    b: T,
    c: T,
    mul: impl Fn(T, T) -> T,
    add: impl Fn(T, T) -> T,
    compare: &impl Compare<T>,
) -> Outcome {
    let left = mul(a.clone(), add(b.clone(), c.clone()));
    let right = add(mul(a.clone(), b), mul(a, c));
    if compare.same(&left, &right) {
        Ok(())
    } else {
        Err(broke(
            "a*(b+c) = a*b + a*c",
            name,
            &compare.describe(),
            &left,
            &right,
        ))
    }
}

/// `one(x) = two(x)` — two implementations of the same thing agree.
///
/// Differential testing, and the workhorse. A reference implementation against an optimised
/// one, a sparse kernel against a dense one, a closed form against a series. Neither side needs
/// to be known correct for the disagreement to be informative.
pub fn agree<T: Clone, R: Debug>(
    name: &str,
    x: T,
    one: impl Fn(T) -> R,
    two: impl Fn(T) -> R,
    compare: &impl Compare<R>,
) -> Outcome {
    let first = one(x.clone());
    let second = two(x);
    if compare.same(&first, &second) {
        Ok(())
    } else {
        Err(broke(
            "two implementations agree",
            name,
            &compare.describe(),
            &first,
            &second,
        ))
    }
}

/// `a <= b` implies `f(a) <= f(b)` — `f` preserves order.
///
/// A CDF, a cumulative cost, `erf`, `ln`. Takes the two inputs in either order and sorts them
/// itself, so the caller does not have to.
pub fn monotone<T: PartialOrd + Clone + Debug, R: PartialOrd + Debug>(
    name: &str,
    a: T,
    b: T,
    f: impl Fn(T) -> R,
) -> Outcome {
    let (low, high) = if a <= b { (a, b) } else { (b, a) };
    let (fl, fh) = (f(low.clone()), f(high.clone()));
    if fl <= fh {
        Ok(())
    } else {
        Err(Failure::new(
            format!(
                "{name}: f is not monotone\n    f({low:?}) = {fl:?}\n    f({high:?}) = {fh:?}\n    \
                 but {low:?} <= {high:?}"
            ),
            "a metamorphic relation",
        ))
    }
}

/// `f(-x) = -f(x)` — `f` is odd.
///
/// `sin`, `tan`, `asin`, `sinh`, `erf`, and the sign-handling of anything that claims to be a
/// smooth extension through zero. Catches a kernel that takes `abs` too early.
pub fn odd(name: &str, x: f64, f: impl Fn(f64) -> f64, compare: &impl Compare<f64>) -> Outcome {
    let negated_input = f(-x);
    let negated_output = -f(x);
    if compare.same(&negated_input, &negated_output) {
        Ok(())
    } else {
        Err(broke(
            "f(-x) = -f(x)",
            name,
            &compare.describe(),
            &negated_input,
            &negated_output,
        ))
    }
}

/// `f(-x) = f(x)` — `f` is even.
///
/// `cos`, `cosh`, `abs`, a Gaussian, a norm.
pub fn even(name: &str, x: f64, f: impl Fn(f64) -> f64, compare: &impl Compare<f64>) -> Outcome {
    let negated = f(-x);
    let plain = f(x);
    if compare.same(&negated, &plain) {
        Ok(())
    } else {
        Err(broke(
            "f(-x) = f(x)",
            name,
            &compare.describe(),
            &negated,
            &plain,
        ))
    }
}

/// `low <= f(x) <= high` — the result stays in its stated range.
///
/// A probability, a correlation, `erf`, a normalised vector component. The cheapest possible
/// property and one of the most effective, because a kernel that goes wrong usually goes
/// *far* wrong.
pub fn bounded<T: Clone + Debug, R: PartialOrd + Debug>(
    name: &str,
    x: T,
    f: impl Fn(T) -> R,
    low: R,
    high: R,
) -> Outcome {
    let value = f(x.clone());
    if value >= low && value <= high {
        Ok(())
    } else {
        Err(Failure::new(
            format!("{name}: f({x:?}) = {value:?} is outside [{low:?}, {high:?}]"),
            "a metamorphic relation",
        ))
    }
}

/// `invariant(f(x)) = invariant(x)` — `f` changes the value but not this property of it.
///
/// The relation for a transformation that is supposed to preserve something: a permutation
/// preserves a multiset, a rebalance preserves the element set, a change of layout preserves
/// the edge count, a normalisation preserves a direction. Often the only thing that *can* be
/// stated about `f` without reimplementing it.
pub fn preserves<T: Clone + Debug, I: PartialEq + Debug>(
    name: &str,
    x: T,
    f: impl Fn(T) -> T,
    invariant: impl Fn(&T) -> I,
) -> Outcome {
    let before = invariant(&x);
    let transformed = f(x);
    let after = invariant(&transformed);
    if before == after {
        Ok(())
    } else {
        Err(Failure::new(
            format!(
                "{name}: f did not preserve the invariant\n    before: {before:?}\n    \
                 after:  {after:?}"
            ),
            "a metamorphic relation",
        ))
    }
}

/// `f(x) = x` — `x` is a fixed point of `f`.
pub fn fixed_point<T: Clone + Debug>(
    name: &str,
    x: T,
    f: impl Fn(T) -> T,
    compare: &impl Compare<T>,
) -> Outcome {
    let image = f(x.clone());
    if compare.same(&image, &x) {
        Ok(())
    } else {
        Err(broke("f(x) = x", name, &compare.describe(), &image, &x))
    }
}

#[cfg(test)]
mod tests;
