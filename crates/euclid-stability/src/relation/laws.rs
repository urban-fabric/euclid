//! The algebraic laws: involution, idempotence, round trips, commutativity, associativity,
//! distribution and order independence.
//!
//! A child module so the file stays inside the crate's 400-line budget. These share a shape the
//! stability laws in [`super::stability`] do not: each is an *identity* between two ways of
//! evaluating the same thing, so it needs no reference value and no budget — only a comparator.

use super::*;

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

/// A reduction over `xs` does not depend on the order of `xs` — to within `compare`.
///
/// Sum, product, min, max, a norm, a dot product, `logsumexp`. Mathematically these are all
/// order-independent and in floating point **none of them is**, which is exactly why the relation
/// takes a [`Compare`] rather than asserting equality: `(a + b) + c` and `a + (b + c)` differ by
/// rounding, and a reduction that reassociates — pairwise, blocked, vectorised, or run across
/// threads — gives a different answer from the naive left fold by design.
///
/// So the property is not "the order does not matter". It is **"the order does not matter more
/// than this much"**, and the bound is the interesting part:
///
/// - `Exact` is right for an integer, a `min`/`max`, or a reduction over a dioid, where
///   reassociation genuinely changes nothing.
/// - For a float sum, the honest bound grows with the condition number of the sum — `Σ|xᵢ| /
///   |Σxᵢ|` — which [`crate::tolerance::condition_number`] computes.
///   A sample that cancels has no tight fixed bound and asserting one is asserting something
///   false.
///
/// The failure this catches in practice is not a reassociation that lost an ulp. It is a
/// reduction that is order-*dependent* in a way nobody intended: an accumulator initialised from
/// the first element rather than the identity, a running `max` that mishandles the first
/// comparison, a Kahan compensation term that is reset inside the loop. Each of those is wrong by
/// much more than rounding, and each passes a test over a sorted fixture.
///
/// Pairs with [`generate::permutation`](crate::generate::permutation), which supplies `order`.
///
/// # Panics
/// Never. A malformed `order` — wrong length, repeated or out-of-range index — is reported as a
/// failure of the *caller* rather than indexed into, since a panic here would be attributed to
/// the code under test.
pub fn order_independent<T: Clone + Debug, R: Debug>(
    name: &str,
    xs: &[T],
    order: &[usize],
    reduce: impl Fn(&[T]) -> R,
    compare: &impl Compare<R>,
) -> Outcome {
    if order.len() != xs.len() || order.iter().any(|&i| i >= xs.len()) {
        return Err(Failure::new(
            format!(
                "{name}: order {order:?} is not a permutation of 0..{} — this is a bug in the \
                 test, not in the reduction",
                xs.len()
            ),
            "a metamorphic relation",
        ));
    }

    let permuted: Vec<T> = order.iter().map(|&i| xs[i].clone()).collect();
    let direct = reduce(xs);
    let reordered = reduce(&permuted);
    if compare.same(&direct, &reordered) {
        Ok(())
    } else {
        Err(Failure::new(
            format!(
                "{name}: the reduction depends on input order by more than {}\n    \
                 in order:  {direct:?}\n    permuted:  {reordered:?}\n    order:     {order:?}",
                compare.describe()
            ),
            "a metamorphic relation",
        ))
    }
}
