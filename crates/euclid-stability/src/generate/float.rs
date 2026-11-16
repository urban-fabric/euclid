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

use super::{Gen, int, weighted};

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
    int(0, NOTABLE.len() as i64 - 1).map(|i| NOTABLE[i as usize])
}

/// One of `±∞` or `NaN`.
#[must_use]
pub fn non_finite() -> Gen<f64> {
    int(0, NON_FINITE.len() as i64 - 1).map(|i| NON_FINITE[i as usize])
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
        let value = fraction * 2.0_f64.powi(exp as i32);
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
    let boundaries = int(0, 2).map(move |i| match i {
        0 => low,
        1 => high,
        _ => low + (high - low) / 2.0,
    });
    let spread = int(0, 1 << 30).map(move |t| {
        #[allow(clippy::cast_precision_loss)]
        let fraction = (t as f64) / f64::from(1u32 << 30);
        low + (high - low) * fraction
    });

    weighted(vec![(1, boundaries), (4, spread)])
}

/// A `f64` strictly positive and finite — the domain of `ln`, `sqrt` and most of the gamma
/// family, so the bound most numerical properties actually need.
#[must_use]
pub fn positive() -> Gen<f64> {
    finite()
        .map(f64::abs)
        .filter("strictly positive", |v: &f64| *v > 0.0)
}

/// A pair that cancels catastrophically: `(x, y)` with `x - y` tiny relative to `x`.
///
/// The generator for conditioning tests. A naive difference of two nearby large values loses
/// every significant digit, and the whole point of a cancellation-safe form is to not — so the
/// property needs inputs where the naive version is actually bad. Random pairs essentially never
/// cancel, which is why this is built rather than hoped for.
#[must_use]
pub fn cancelling_pair() -> Gen<(f64, f64)> {
    super::pair(magnitude(-40, 40), int(1, 52)).map(|(x, shed)| {
        // y differs from x only in the low bits, so x - y is around x * 2^-shed.
        let delta = x * 2.0_f64.powi(-(shed as i32));
        (x, x - delta)
    })
}

#[cfg(test)]
mod tests;
