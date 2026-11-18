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
//! use euclid_stability::relation::{Exact, Ulps, involution};
//! use euclid_stability::{generate, property};
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
    .with_identity(format!("{name}: {law}"))
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

/// `f` is **forward stable** at `x`: its relative error is no worse than the problem's own
/// conditioning allows.
///
/// The central claim of numerical stability, and the one a tolerance literal cannot express.
/// Asserting `within 8 ulp` says "within a number I picked"; this says "no worse than the problem
/// itself", which stays true when someone rewrites the kernel and stays honest where the problem
/// is genuinely hard.
///
/// `reference` is a more accurate computation of the same quantity — a wider type, a series
/// summed further, an independent implementation. The bound is
/// [`crate::numeric::forward_error_bound`], which is
/// `safety · κ(x) · u`: the error a forward-stable implementation is *entitled* to at `x`.
///
/// # It distinguishes a bad kernel from a hard problem
///
/// That distinction is the reason to use this rather than a fixed tolerance, and it is the one
/// numerical reviewers actually argue about. An error of `1e-9` is a scandal at `κ = 1` and
/// excellent at `κ = 1e7`. A fixed tolerance conflates them, so it is either too loose to catch
/// an unstable kernel in the benign region or too tight to pass a well-written one near a
/// singularity — and the usual resolution is to loosen it until the suite goes green, which
/// discards the finding. The failure message reports `κ` alongside the error so the reader can
/// tell which situation they are in.
///
/// Returns `Ok` where the bound is undefined — at `x = 0`, where relative conditioning has no
/// meaning, or where `f` is non-finite nearby. Those are not passes so much as abstentions, and a
/// suite that cares should classify them.
///
/// # Reporting the margin
///
/// The ratio of observed error to admissible bound is the natural score for
/// [`target`](crate::target()) and for a [`budget`](crate::budget) baseline: `1.0` means
/// "exactly at the limit", it is comparable across kernels and inputs, and it is continuous in
/// `x`, which is what the hill climb needs. [`forward_stable_margin`] returns it.
pub fn forward_stable(
    name: &str,
    x: f64,
    f: impl Fn(f64) -> f64,
    reference: impl Fn(f64) -> f64,
    safety: f64,
) -> Outcome {
    let Some(margin) = forward_stable_margin(x, &f, &reference, safety) else {
        return Ok(());
    };
    if margin <= 1.0 {
        return Ok(());
    }

    let condition = crate::numeric::condition_at(&f, x).unwrap_or(f64::NAN);
    let (computed, expected) = (f(x), reference(x));
    let relative = ((computed - expected) / expected).abs();
    Err(Failure::new(
        format!(
            "{name}: not forward stable at x = {x:e}\n    \
             computed:   {computed:e}\n    \
             reference:  {expected:e}\n    \
             relative:   {relative:e}\n    \
             admissible: {:e} (safety {safety} x condition {condition:e} x unit roundoff)\n    \
             over by:    {margin:.2}x\n    \
             {}",
            relative / margin,
            if condition > 1e6 {
                "The condition number is large, so the *problem* is ill conditioned here. If this \
                 input is one callers really hit, the kernel needs a different algorithm in this \
                 region, not a looser tolerance."
            } else {
                "The condition number is small, so the problem is benign and the error is the \
                 implementation's own. This is an unstable kernel rather than a hard input."
            }
        ),
        "a stability relation",
    ))
}

/// How much of its admissible error `f` used at `x`, as a ratio: `1.0` is exactly at the limit.
///
/// The score to feed [`target`](crate::target()) and [`budget`](crate::budget). `None` where the
/// bound is undefined — see [`forward_stable`].
#[must_use]
pub fn forward_stable_margin(
    x: f64,
    f: impl Fn(f64) -> f64,
    reference: impl Fn(f64) -> f64,
    safety: f64,
) -> Option<f64> {
    let bound = crate::numeric::forward_error_bound(&f, x, safety)?;
    let expected = reference(x);
    if expected == 0.0 || !expected.is_finite() {
        return None;
    }
    let computed = f(x);
    if !computed.is_finite() {
        return None;
    }
    let relative = ((computed - expected) / expected).abs();
    (bound > 0.0).then(|| relative / bound)
}

/// A narrow-precision implementation agrees with its wide-precision counterpart, to narrow
/// precision.
///
/// euclid ships an `f32` kernel beside almost every `f64` one, and the `f32` one is the easier to
/// get wrong: it is written second, exercised less, and its constants are usually the `f64`
/// coefficients rounded — which is the right starting point and not always the right answer,
/// because a polynomial fitted for `f64` can be the wrong polynomial at `f32`.
///
/// The relation is "two implementations agree" with the only bound that makes sense across a
/// precision change: the wide result, rounded to narrow, is the best the narrow one could
/// possibly do, so the comparison is in **`f32` ulps against that**. Comparing in `f64` ulps
/// would fail every correct narrow kernel; comparing with a relative tolerance picked by hand
/// re-introduces the number nobody can justify.
///
/// `ulps` is the slack allowed on top, in `f32` ulps — `2` to `4` for a single evaluation.
pub fn precision_agrees(
    name: &str,
    x: f64,
    wide: impl Fn(f64) -> f64,
    narrow: impl Fn(f32) -> f32,
    ulps: u32,
) -> Outcome {
    #[allow(clippy::cast_possible_truncation)]
    let narrow_input = x as f32;
    // Evaluate the wide version at the *narrowed* input, not at `x`. Otherwise the difference
    // includes the input's own rounding, which is not the narrow kernel's fault and swamps
    // everything else near a steep region.
    let expected = wide(f64::from(narrow_input));
    let computed = narrow(narrow_input);

    #[allow(clippy::cast_possible_truncation)]
    let rounded = expected as f32;
    if rounded == computed || (rounded.is_nan() && computed.is_nan()) {
        return Ok(());
    }

    let gap = f32_ulps_between(computed, rounded);
    if gap.is_some_and(|gap| gap <= u64::from(ulps)) {
        Ok(())
    } else {
        Err(Failure::new(
            format!(
                "{name}: the narrow kernel disagrees with the wide one at x = {x:e}\n    \
                 narrow:        {computed:e}\n    \
                 wide, rounded: {rounded:e}\n    \
                 gap:           {} f32 ulp (allowed {ulps})\n    \
                 The wide result rounded to f32 is the best the narrow kernel could return, so \
                 this gap is the narrow implementation's own — most often an f64 coefficient set \
                 reused at f32, where it is no longer the right polynomial.",
                gap.map_or_else(|| "non-finite".to_string(), |gap| gap.to_string())
            ),
            "a stability relation",
        ))
    }
}

/// Representable `f32` steps between `a` and `b`, or `None` if either is non-finite.
///
/// The `f32` peer of [`crate::tolerance::ulps_between`], kept here
/// because it exists for exactly one caller. Sign-magnitude bits are mapped to a monotone
/// ordering so a pair straddling zero measures correctly rather than wrapping.
fn f32_ulps_between(a: f32, b: f32) -> Option<u64> {
    if !a.is_finite() || !b.is_finite() {
        return None;
    }
    let order = |x: f32| -> i64 {
        let bits = i64::from(x.to_bits() as i32);
        if bits < 0 {
            i64::from(i32::MIN) - bits
        } else {
            bits
        }
    };
    Some(order(a).abs_diff(order(b)))
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

/// `f(perturb(x)) = f(x)` — the output does **not** depend on what `perturb` changes.
///
/// The one shape missing from the list above, and the only one here that asserts an
/// *independence* rather than a correspondence. Its uses are the places where a function is
/// documented to ignore something: periodicity (`sin_cos_degrees` must ignore a whole turn),
/// invariance under a common positive scale, insensitivity to a representation detail that
/// carries no value.
///
/// # Why a named relation rather than two calls and an `assert_eq!`
///
/// Because the claim is easy to make in prose and easy to leave untested, and when it holds it
/// is *invisible to mutation testing* — which is how this gap was found. `gamma_lr_inv` is a
/// safeguarded Newton solve: a cheap seed, a bracket, and bisection whenever a Newton step would
/// leave it. Its answer is therefore independent of both the seed and the derivative, and
/// measuring that against a 120-digit reference confirms it — with the seed removed and the
/// densities forced to `0`, so no Newton step can fire at all, the worst relative error moves
/// from `1.1e-13` to `7.4e-14`. Unchanged, marginally better.
///
/// Fifty-six mutants survive in that kernel for exactly that reason, and no accuracy assertion
/// can kill them because there is no accuracy difference to find. The honest response is to
/// state the guarantee instead of chasing them, and this is the function that states it.
///
/// # What it cannot reach
///
/// Recorded because the motivating example is the one case this does *not* serve. The quantile
/// solver's insensitivity is **internal** — the seed is a local, not an argument — so no
/// relation taking `f` from outside can perturb it. Expressing that would mean making the seed
/// injectable, which is production surface added for a test, and the measurement in the
/// paragraph above is the better evidence. Use this where the ignored thing is part of the
/// *input*; where it is an implementation detail, measure against truth with the detail
/// disabled and write down what you found.
///
/// ```
/// use euclid_stability::relation::{self, Exact};
///
/// // A turn is a turn: reducing an angle must ignore whole revolutions.
/// relation::insensitive_to(
///     "angle reduction is periodic",
///     90.0_f64,
///     |x| x + 360.0,
///     |x| x.rem_euclid(360.0),
///     &Exact,
/// )
/// .unwrap();
/// ```
///
/// Note `Exact` there. An insensitivity claim is usually either exact or false — the ignored
/// thing is ignored or it is not — so reach for a tolerance only when the *computation* after
/// the reduction is inexact, and say which of the two you mean.
pub fn insensitive_to<T: Clone + Debug, R: Debug>(
    name: &str,
    x: T,
    perturb: impl Fn(T) -> T,
    f: impl Fn(T) -> R,
    compare: &impl Compare<R>,
) -> Outcome {
    let direct = f(x.clone());
    let perturbed = f(perturb(x));
    if compare.same(&perturbed, &direct) {
        Ok(())
    } else {
        Err(broke(
            "f(perturb(x)) = f(x)",
            name,
            &compare.describe(),
            &perturbed,
            &direct,
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
