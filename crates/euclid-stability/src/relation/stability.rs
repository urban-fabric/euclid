//! The numerical-stability laws: forward stability, cross-precision agreement and monotonicity.
//!
//! A child module alongside [`super::laws`]. These differ in kind: each measures a *quantity*
//! rather than asserting an identity, so each carries a budget, and the budget is a number at an
//! effort rather than a worst case — see the crate README on soak levels.

use super::*;

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

/// `f` is forward stable at `x` against a reference that is **itself a cancelling sum**.
///
/// [`forward_stable`] bounds the disagreement by `safety · κ(f) · u`, which is what the *left*
/// side is entitled to. That is the whole bound only when the reference is more accurate than
/// `f` — a wider type, a closed form, an independent implementation. It is not the whole bound
/// when the reference is an expression assembled from terms that cancel, and then the shortfall
/// is reported against the kernel.
///
/// # Two ways that went wrong, which is why this exists
///
/// A recurrence is the usual source. `ψ(x+1) = ψ(x) + 1/x` makes the reference a difference of
/// two `1/x`-sized terms: at `x = 0.5` it is `−1.9635 + 2 = 0.0365`, a 54-fold cancellation, and
/// `κ(ψ(·+1))` is `12.8` and knows nothing about it. The property exceeded its budget by `1.5×`
/// on an error that was the reference's, and
/// [`forward_stable`]'s message said "the condition number is small, so the problem is benign and
/// the error is the implementation's own" — a correct inference from a premise that did not hold.
///
/// The same shape, worse, with a finite difference: there the reference's floor is `ε^{2/3}`,
/// about `4e-11`, which `κ·u` cannot express at any safety factor worth having.
///
/// # What it measures instead
///
/// The caller hands over the reference's **terms** rather than its value. Their sum is the
/// reference; the ratio `Σ|tᵢ| / |Σtᵢ|` is how much relative accuracy forming it threw away.
/// So each side declares what it may contribute and the bound is the sum of the two
/// entitlements:
///
/// ```text
/// admissible = safety · (κ(f) + Σ|tᵢ|/|Σtᵢ|) · u
/// ```
///
/// At `ψ`'s cancelling orientation that is `12.8 + 54`, which passes; at the benign orientation
/// the second term is `1.02` and the bound is essentially [`forward_stable`]'s, so moving a
/// property onto this relation does not quietly loosen it where there is nothing to forgive.
///
/// **Reorienting beats forgiving where it is available.** `ψ(x) = ψ(x+1) − 1/x` has no
/// cancellation at all and is the better property; this relation is for the references that have
/// no good orientation, like a difference of two values of the same function.
pub fn forward_stable_sum(
    name: &str,
    x: f64,
    f: impl Fn(f64) -> f64,
    terms: &[f64],
    safety: f64,
) -> Outcome {
    let reference: f64 = terms.iter().sum();
    let magnitude: f64 = terms.iter().map(|t| t.abs()).sum();
    let computed = f(x);
    if reference == 0.0 || !reference.is_finite() || !computed.is_finite() {
        return Ok(());
    }
    let cancellation = magnitude / reference.abs();
    let Some(condition) = crate::numeric::condition_at(&f, x) else {
        return Ok(());
    };
    let admissible = safety * (condition + cancellation) * crate::numeric::UNIT_ROUNDOFF;
    let relative = ((computed - reference) / reference).abs();
    if relative <= admissible || admissible == 0.0 {
        return Ok(());
    }
    Err(Failure::new(
        format!(
            "{name}: not forward stable at x = {x:e}\n    \
             computed:   {computed:e}\n    \
             reference:  {reference:e}  (sum of {} terms)\n    \
             relative:   {relative:e}\n    \
             admissible: {admissible:e} (safety {safety} x [condition {condition:e} + \
             cancellation {cancellation:e}] x unit roundoff)\n    \
             over by:    {:.2}x\n    \
             {}",
            terms.len(),
            relative / admissible,
            if cancellation > 4.0 * condition.max(1.0) {
                "The reference's own cancellation dominates the bound, so this is a statement \
                 about the comparison rather than about the kernel. Check whether the identity \
                 has an orientation that computes the larger side instead."
            } else {
                "The cancellation is not what is binding here, so the error is the \
                 implementation's own."
            }
        ),
        "a stability relation",
    ))
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
pub(crate) fn f32_ulps_between(a: f32, b: f32) -> Option<u64> {
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
