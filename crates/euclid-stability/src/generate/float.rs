//! Float generators, built for finding numerical bugs rather than for uniformity.
//!
//! A uniform draw over `f64` is almost useless for testing numerics. It never produces a
//! subnormal, a negative zero, a NaN, or a value adjacent to a power of two, and it is
//! overwhelmingly a number around `10^300` — because that is where most of the exponent range
//! is. None of those are the cases where an identity breaks.
//!
//! So these generators do two things differently.
//!
//! # Magnitude is drawn in log space
//!
//! [`magnitude`] picks an *exponent* uniformly and then a mantissa, so a draw is as likely to be
//! near `10^-200` as near `10^200`. That is the distribution a numerical property needs: the
//! interesting behaviour of `log1p`, `expm1` and friends is concentrated near zero, and a
//! uniform draw over the reals visits that region with probability about zero.
//!
//! # Notable values are injected deliberately
//!
//! [`finite`] and [`any`] mix a weighted choice of *named* values — zero, both signs of it,
//! one, the subnormal boundary, `MAX`, `EPSILON`, the integers either side of `2^53` — into the
//! random stream, rather than hoping to land on them. jqwik and Hypothesis both do this, and the
//! reason is arithmetic: the probability of randomly drawing exactly `1.0` out of `2^64`
//! bit patterns is not small, it is nil in practice.
//!
//! The weighting also helps the shrinker, because these values sit at the *low* end of the
//! choice. A failure involving a subnormal shrinks to the subnormal rather than to whatever
//! random neighbour happened to fail first.
//!
//! # Tolerance is the caller's problem, and it is a real one
//!
//! Nothing here chooses a comparison. `0.1 + 0.2 != 0.3` is not a bug and `==` is not the
//! check; see [`crate::tolerance`] for ULP and relative-error comparisons, and prefer them.

use super::{Gen, int, sampled_from, weighted};

/// Values worth drawing on purpose, in roughly increasing order of magnitude.
///
/// Ordered deliberately: the shrinker lowers tape entries, so an earlier entry in this list is
/// a *smaller* choice and a failure reported against `0.0` rather than against `f64::MAX` is a
/// failure a reader can reason about.
const NOTABLE: &[f64] = &[
    0.0,
    -0.0,
    1.0,
    -1.0,
    0.5,
    2.0,
    f64::EPSILON,
    -f64::EPSILON,
    1.0 + f64::EPSILON,
    1.0 - f64::EPSILON / 2.0,
    f64::MIN_POSITIVE, // the smallest normal
    -f64::MIN_POSITIVE,
    5e-324,                  // the smallest subnormal
    9_007_199_254_740_992.0, // 2^53, where consecutive integers stop being representable
    9_007_199_254_740_993.0, // not representable; rounds to 2^53
    f64::MAX,
    f64::MIN,
];

/// The non-finite values, which are legal `f64`s and the ones most code forgets.
const NON_FINITE: &[f64] = &[f64::INFINITY, f64::NEG_INFINITY, f64::NAN];

/// Bits in an `f64` mantissa.
const MANTISSA_BITS: u32 = 52;

/// The largest mantissa bit pattern [`magnitude`] may draw.
///
/// Named rather than written inline, because it is the one bound in this module that **no amount
/// of sampling can check**. [`magnitude`] builds `fraction = 1 + bits / 2^52`, so the interval is
/// half-open only while `bits` stops one short of `2^52` — and a bound one too high is reachable
/// with probability `2^-52`, which no test that draws values will ever witness. Mutation testing
/// found it surviving a four-thousand-draw assertion that every mantissa lands in `[1, 2)`.
///
/// So the constant is asserted directly instead, by
/// `the_mantissa_bound_keeps_the_interval_half_open`. The lesson generalises: a bound whose
/// violation is astronomically rare has to be pinned as arithmetic, not as behaviour.
const MANTISSA_MAX: i64 = (1i64 << MANTISSA_BITS) - 1;

/// `2^52`, as the divisor that maps [`MANTISSA_MAX`] onto `[1, 2)`.
#[allow(clippy::cast_precision_loss)]
const MANTISSA_SCALE: f64 = (1i64 << MANTISSA_BITS) as f64;

/// One of the notable values listed in this module's source.
#[must_use]
pub fn notable() -> Gen<f64> {
    sampled_from(NOTABLE.to_vec())
}

/// One of `±∞` or `NaN`.
#[must_use]
pub fn non_finite() -> Gen<f64> {
    sampled_from(NON_FINITE.to_vec())
}

/// A finite `f64` with its exponent drawn uniformly from `min_exp..=max_exp` (base two).
///
/// `magnitude(-20, 20)` covers roughly `1e-6` to `1e6` with equal weight per octave, which is
/// what you want when the property under test behaves differently at different scales. The sign
/// is drawn separately, so both are reached.
///
/// # The bounds are binary, and the top one is off by an octave
///
/// Two traps, and both have already cost a debugging session.
///
/// The exponents are **base two**, not base ten. `magnitude(120, 300)` is about `1e36` to
/// `1e90`, not `1e120` to `1e300` — which matters when the property is about `f64` overflow,
/// since squaring `1e90` is perfectly finite and squaring `1e300` is not.
///
/// And the largest value produced is `2^(max_exp + 1)`, not `2^max_exp`, because the mantissa
/// is drawn from `[1, 2)` and multiplies the power. So `magnitude(-80, -6)` reaches `2^-5`.
/// That one octave is enough to break a property whose reference is a truncated series chosen
/// for the nominal bound.
///
/// # Panics
/// If `min_exp > max_exp`, or either is outside `f64`'s exponent range.
#[must_use]
pub fn magnitude(min_exp: i32, max_exp: i32) -> Gen<f64> {
    assert!(min_exp <= max_exp, "empty exponent range");
    assert!(
        min_exp >= -1_074 && max_exp <= 1_023,
        "exponent range {min_exp}..={max_exp} is outside f64"
    );

    let exponent = int(i64::from(min_exp), i64::from(max_exp));
    // A mantissa in [1, 2) built from 52 bits, so the result is a full-precision float rather
    // than a round number — round numbers hide alignment bugs.
    let mantissa = int(0, MANTISSA_MAX);
    let negative = super::boolean();

    super::triple(exponent, mantissa, negative).map(|(exp, bits, neg)| {
        #[allow(clippy::cast_precision_loss)]
        let fraction = 1.0 + (bits as f64) / MANTISSA_SCALE;
        let value = fraction * power_of_two(exp as i32);
        if neg { -value } else { value }
    })
}

/// A finite `f64`: mostly [`magnitude`] over the full range, with [`notable`] values mixed in.
///
/// Weighted three-to-one towards the random draw. Heavier would starve the magnitude sweep;
/// lighter would make the named values rare enough not to count as covered.
#[must_use]
pub fn finite() -> Gen<f64> {
    weighted(vec![(1, notable()), (3, magnitude(-1_020, 1_020))])
        .filter("finite", |v: &f64| v.is_finite())
}

/// Any `f64`, including `NaN` and both infinities.
///
/// Use this where the contract covers them — a `select`, a comparison, a classification — and
/// [`finite`] where it does not. Handing `NaN` to a property that assumes an ordering is a
/// failure of the test, not of the code.
#[must_use]
pub fn any() -> Gen<f64> {
    weighted(vec![(1, non_finite()), (2, notable()), (6, finite())])
}

/// A finite `f64` in `low..=high`, with the endpoints reached deliberately.
///
/// # Panics
/// If `low > high` or either bound is not finite.
#[must_use]
pub fn in_range(low: f64, high: f64) -> Gen<f64> {
    assert!(low <= high, "empty range {low}..={high}");
    assert!(
        low.is_finite() && high.is_finite(),
        "range {low}..={high} must be finite"
    );

    // Boundaries first in the weighting, so they shrink to and are never missed.
    let boundaries = sampled_from(vec![low, high, interpolate(low, high, 0.5)]);
    let spread = int(0, 1 << 30).map(move |t| {
        #[allow(clippy::cast_precision_loss)]
        let fraction = (t as f64) / f64::from(1u32 << 30);
        interpolate(low, high, fraction)
    });

    weighted(vec![(1, boundaries), (4, spread)])
}

fn interpolate(low: f64, high: f64, fraction: f64) -> f64 {
    let width = high - low;
    if width.is_finite() {
        low + width * fraction
    } else {
        low * (1.0 - fraction) + high * fraction
    }
}

/// A `f64` strictly positive and finite — the domain of `ln`, `sqrt` and most of the gamma
/// family, so the bound most numerical properties actually need.
#[must_use]
pub fn positive() -> Gen<f64> {
    finite()
        .map(f64::abs)
        .filter("strictly positive", |v: &f64| *v > 0.0)
}

/// Exactly a power of two, with both signs — the binade boundary, which [`magnitude`] cannot
/// reach.
///
/// # The one input a magnitude sweep structurally cannot produce
///
/// A power of two is where the exponent field increments and the gap between representable
/// values doubles. Any code that chooses a *direction* or a *scale* is correct everywhere inside
/// a binade and can be wrong only on its edge, because that is the only place the step up and
/// the step down have different widths.
///
/// [`magnitude`] will not take you there, by design: it draws a full 52-bit mantissa so that
/// "the result is a full-precision float rather than a round number — round numbers hide
/// alignment bugs". That is the right call for alignment bugs and it makes `mantissa == 0` a
/// `2^-52` event, so a property sweeping six hundred decades visits a boundary approximately
/// never.
///
/// This cost a real defect. `euclid-scalar`'s `ulp` measured the step *toward* zero for negative
/// arguments instead of away from it — wrong by a factor of two at every binade boundary, and
/// invisible at every other argument. It survived a 1,000-decade generated property and was
/// found by mutation testing. `-2.0` shows it immediately; `-3.0` cannot.
///
/// Both signs are drawn, and that is not symmetry for its own sake: a direction error is a
/// *sign* error, so the negative side is the half that distinguishes it.
///
/// # Unlike [`magnitude`], the top of the range is exactly `2^max_exp`
///
/// There is no extra octave here. `magnitude` reaches `2^(max_exp + 1)` because its mantissa
/// multiplies the power; this returns the power itself, so `binade_boundary(-4, 4)` produces
/// values in `±{2^-4 … 2^4}` and nothing above `16`.
///
/// # Panics
/// If `min_exp > max_exp`, or either is outside `f64`'s exponent range. The subnormal powers are
/// included — `min_exp` may go to `-1074`, which is the smallest subnormal.
#[must_use]
pub fn binade_boundary(min_exp: i32, max_exp: i32) -> Gen<f64> {
    assert!(min_exp <= max_exp, "empty exponent range");
    assert!(
        min_exp >= -1_074 && max_exp <= 1_023,
        "exponent range {min_exp}..={max_exp} is outside f64"
    );

    let exponent = int(i64::from(min_exp), i64::from(max_exp));
    let negative = super::boolean();

    super::pair(exponent, negative).map(|(exp, neg)| {
        let value = power_of_two(exp as i32);
        if neg { -value } else { value }
    })
}

/// The exact value of `2^exp` for `-1074..=1023`, assembled from its bits.
///
/// `2.0_f64.powi(exp)` is the obvious spelling and it is wrong at the bottom of the range:
/// `powi` reaches a negative exponent through the reciprocal of a positive one, so
/// `powi(-1074)` forms `2^1074` first, overflows to infinity, and returns `0.0` — losing the
/// subnormal powers entirely, which are half the reason this generator exists.
///
/// Assembling the field is exact by construction and has no range to get wrong.
fn power_of_two(exp: i32) -> f64 {
    /// Exponent bias, and the mantissa width in bits.
    const BIAS: i32 = 1_023;
    const MANTISSA: i32 = 52;

    let bits = if exp >= -1_022 {
        // Normal: a biased exponent field over a zero mantissa.
        #[allow(clippy::cast_sign_loss)]
        let biased = (exp + BIAS) as u64;
        biased << MANTISSA
    } else {
        // Subnormal: a zero exponent field with the single mantissa bit that carries `2^exp`.
        // `2^-1023` is the top subnormal power and sits in the highest mantissa bit.
        #[allow(clippy::cast_sign_loss)]
        let bit = (exp + BIAS + MANTISSA - 1) as u32;
        1u64 << bit
    };
    f64::from_bits(bits)
}

/// A pair whose second element is drawn from the edge cases a two-argument guard must handle.
///
/// # A guard over two arguments needs the test at each argument
///
/// `if x.is_nan() || y.is_nan()` reads symmetrically and is not. Weaken that `||` to `&&` and a
/// property that passes NaN as the *first* argument may still pass, because the fall-through path
/// can reconstruct the right answer from `x` by accident — while the same weakening on the
/// *second* argument returns a plausible, wrong, finite value.
///
/// That is exactly what happened to `euclid-scalar`'s `next_after`: its generated property swept
/// six hundred decades but always passed `±INFINITY` as the target, so a NaN target was never
/// tried and `next_after(1.0, NaN)` returned `1.0 - 1ulp` instead of NaN. No number of cases
/// over the first argument could have found it.
///
/// So the second element here is not another ordinary draw. It is weighted towards the values
/// that make a guard fire — both NaNs, both infinities, both zeros — and it includes `x`
/// *itself*, which is the case an equality guard turns on and a random pair never hits.
///
/// The selector is ordered so the shrinker reports the simplest target: a failure involving NaN
/// shrinks to NaN rather than to whichever random neighbour failed first, the same reasoning as
/// the ordering of the notable values.
///
/// # Panics
/// If `min_exp > max_exp`, or either is outside `f64`'s exponent range.
#[must_use]
pub fn pair_with_edge_cases(min_exp: i32, max_exp: i32) -> Gen<(f64, f64)> {
    super::triple(
        magnitude(min_exp, max_exp),
        int(0, 6),
        magnitude(min_exp, max_exp),
    )
    .map(|(x, which, other)| {
        let target = match which {
            0 => f64::NAN,
            1 => f64::INFINITY,
            2 => f64::NEG_INFINITY,
            3 => 0.0,
            4 => -0.0,
            // The equality case, which a guard like `x == y` turns on.
            5 => x,
            _ => other,
        };
        (x, target)
    })
}

/// A pair that cancels catastrophically: `(x, y)` with `x - y` tiny relative to `x`.
///
/// The generator for conditioning tests. A naive difference of two nearby large values loses
/// every significant digit, and the whole point of a cancellation-safe form is to not — so the
/// property needs inputs where the naive version is actually bad. Random pairs essentially never
/// cancel, which is why this is built rather than hoped for.
///
/// The perturbation moves `y` toward zero so that both operands stay under the upper magnitude
/// bound of `x`. Replaying a tape produces the same direction for either sign.
#[must_use]
pub fn cancelling_pair() -> Gen<(f64, f64)> {
    super::pair(magnitude(-40, 40), int(1, 52)).map(|(x, shed)| {
        // y differs from x only in the low bits, so x - y is around x * 2^-shed.
        let delta = x * power_of_two(-(shed as i32));
        (x, x - delta)
    })
}

#[cfg(test)]
mod tests;
