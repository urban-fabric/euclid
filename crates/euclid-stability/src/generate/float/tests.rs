//! That the float generators actually reach the values they exist to reach.
//!
//! Each of these would pass trivially against a generator that returned `0.0` forever, except
//! that each names a *specific* class of value and requires it to appear. That is the only way
//! to test a generator: assert coverage, not absence of crashes.

use super::{any, cancelling_pair, finite, in_range, magnitude, non_finite, notable, positive};
use crate::generate::Gen;
use crate::source::Source;

fn draw_many<T: 'static>(generator: &Gen<T>, count: usize) -> Vec<T> {
    let mut source = Source::generating(0xC0FFEE, 1 << 20);
    (0..count)
        .filter_map(|_| generator.draw(&mut source).ok())
        .collect()
}

#[test]
fn notable_reaches_every_named_value() {
    let drawn = draw_many(&notable(), 2_000);
    for wanted in super::NOTABLE {
        let found = drawn
            .iter()
            .any(|v| v.to_bits() == wanted.to_bits() || (v.is_nan() && wanted.is_nan()));
        assert!(found, "never generated {wanted:?}");
    }
}

/// The notable values **named**, independently of the list that produces them.
///
/// The test above is a tautology and mutation testing is what exposed it: it iterates over
/// `NOTABLE` to build its own expectations, so corrupting an entry moves the generator and the
/// assertion together and nothing fails. Six separate mutations of that list survived — `-0.0`
/// collapsing to `0.0`, `-f64::EPSILON` to `f64::EPSILON`, `1.0 + f64::EPSILON` to
/// `1.0 * f64::EPSILON`, and so on.
///
/// This spells out the values that must be reachable, so the list cannot silently lose one. Each
/// is here because the module header's argument applies to it: the probability of a random draw
/// landing on it is not small, it is nil, and these are the inputs where identities break —
/// signed zero, the ulp neighbours of one, the subnormal boundary, and the integer at which
/// consecutive integers stop being representable.
#[test]
fn the_named_values_are_reachable_without_consulting_the_list() {
    let drawn = draw_many(&notable(), 4_000);
    let reaches = |wanted: f64| drawn.iter().any(|v| v.to_bits() == wanted.to_bits());

    // Both signs of zero, which `==` cannot distinguish — hence the bitwise comparison above.
    assert!(reaches(0.0), "positive zero is missing");
    assert!(reaches(-0.0), "negative zero is missing");

    // Both signs of one and of epsilon.
    assert!(reaches(1.0) && reaches(-1.0), "a signed one is missing");
    assert!(
        reaches(f64::EPSILON) && reaches(-f64::EPSILON),
        "a signed epsilon is missing"
    );

    // The two neighbours of one, where a relative-error comparison changes behaviour.
    assert!(reaches(1.0 + f64::EPSILON), "the ulp above one is missing");
    assert!(
        reaches(1.0 - f64::EPSILON / 2.0),
        "the ulp below one is missing"
    );

    // The subnormal boundary, both signs of the smallest normal and the smallest subnormal.
    assert!(
        reaches(f64::MIN_POSITIVE) && reaches(-f64::MIN_POSITIVE),
        "a signed smallest-normal is missing"
    );
    assert!(reaches(5e-324), "the smallest subnormal is missing");

    // 2^53, and the integer above it that is not representable and rounds back onto it.
    assert!(reaches(9_007_199_254_740_992.0), "2^53 is missing");
    assert!(
        reaches(9_007_199_254_740_993.0),
        "the unrepresentable integer above 2^53 is missing"
    );

    assert!(
        reaches(f64::MAX) && reaches(f64::MIN),
        "an extreme is missing"
    );
}

#[test]
fn non_finite_reaches_both_infinities_and_nan() {
    let drawn = draw_many(&non_finite(), 500);
    assert!(drawn.contains(&f64::INFINITY));
    assert!(drawn.contains(&f64::NEG_INFINITY));
    assert!(drawn.iter().any(|v| v.is_nan()));
}

/// The point of drawing in log space: a uniform draw over the reals essentially never produces
/// a small number, and small numbers are where the interesting numerics live.
#[test]
fn magnitude_covers_its_exponent_range_in_log_space() {
    let drawn = draw_many(&magnitude(-200, 200), 2_000);
    assert!(!drawn.is_empty());

    // The bounds are binary exponents, so this range spans about 6e-61 to 1.6e60 -- a first
    // draft of this test asserted `< 1e-100`, which `magnitude(-200, 200)` cannot reach and
    // which was a bug in the test rather than in the generator.
    assert!(
        drawn.iter().any(|v| v.abs() < 1e-50),
        "never produced a very small magnitude"
    );
    assert!(
        drawn.iter().any(|v| v.abs() > 1e50),
        "never produced a very large magnitude"
    );
    assert!(
        drawn.iter().any(|v| (0.1..10.0).contains(&v.abs())),
        "never produced a magnitude near one"
    );
    assert!(drawn.iter().any(|v| *v < 0.0), "never produced a negative");
    assert!(drawn.iter().all(|v| v.is_finite()), "produced a non-finite");
}

/// Mantissas must not be round numbers, or alignment and rounding bugs hide.
#[test]
fn magnitude_produces_full_precision_mantissas() {
    let drawn = draw_many(&magnitude(0, 4), 500);
    let non_integral = drawn.iter().filter(|v| v.fract() != 0.0).count();
    assert!(
        non_integral > drawn.len() / 2,
        "only {non_integral}/{} values had a fractional part",
        drawn.len()
    );
}

#[test]
fn magnitude_respects_a_narrow_range() {
    let drawn = draw_many(&magnitude(3, 3), 200);
    assert!(!drawn.is_empty());
    for value in drawn {
        // One octave: |v| in [2^3, 2^4).
        assert!(
            (8.0..16.0).contains(&value.abs()),
            "{value} escaped the single-octave range"
        );
    }
}

#[test]
fn finite_is_always_finite_but_still_reaches_the_notables() {
    let drawn = draw_many(&finite(), 3_000);
    assert!(drawn.iter().all(|v| v.is_finite()), "produced a non-finite");
    assert!(drawn.contains(&0.0), "never produced zero");
    assert!(drawn.contains(&1.0), "never produced one");
    assert!(
        drawn
            .iter()
            .any(|v| *v != 0.0 && v.abs() < f64::MIN_POSITIVE),
        "never produced a subnormal"
    );
}

#[test]
fn any_reaches_the_non_finite_values_too() {
    let drawn = draw_many(&any(), 3_000);
    assert!(drawn.iter().any(|v| v.is_nan()), "never produced NaN");
    assert!(
        drawn.iter().any(|v| v.is_infinite()),
        "never produced an infinity"
    );
    assert!(
        drawn.iter().any(|v| v.is_finite()),
        "never produced a finite"
    );
}

#[test]
fn in_range_stays_in_range_and_hits_both_ends() {
    let drawn = draw_many(&in_range(-2.5, 7.5), 1_000);
    assert!(!drawn.is_empty());
    assert!(
        drawn.iter().all(|v| (-2.5..=7.5).contains(v)),
        "escaped the range"
    );
    assert!(drawn.contains(&-2.5), "never hit the lower bound");
    assert!(drawn.contains(&7.5), "never hit the upper bound");
}

#[test]
fn a_degenerate_range_is_constant() {
    let drawn = draw_many(&in_range(3.0, 3.0), 50);
    assert!(drawn.iter().all(|v| *v == 3.0));
}

#[test]
fn positive_is_strictly_positive() {
    let drawn = draw_many(&positive(), 1_000);
    assert!(!drawn.is_empty());
    assert!(
        drawn.iter().all(|v| *v > 0.0 && v.is_finite()),
        "produced a non-positive or non-finite value"
    );
}

/// The conditioning generator must actually produce ill-conditioned input, or the properties
/// built on it are vacuous. This is the test that keeps it honest.
#[test]
fn cancelling_pair_is_actually_ill_conditioned() {
    let drawn = draw_many(&cancelling_pair(), 500);
    assert!(!drawn.is_empty());

    let severe = drawn
        .iter()
        .filter(|(x, y)| {
            let difference = (x - y).abs();
            difference > 0.0 && x.abs() / difference > 1e6
        })
        .count();
    assert!(
        severe > 0,
        "no pair cancelled by even six digits, so the generator is not exercising cancellation"
    );

    // And the condition number agrees with that reading.
    let worst = drawn
        .iter()
        .map(|(x, y)| crate::tolerance::condition_number(&[*x, -*y]))
        .filter(|c| c.is_finite())
        .fold(0.0_f64, f64::max);
    assert!(worst > 1e6, "worst condition number was only {worst:e}");
}

/// The mantissa is drawn from `[1, 2)` — half-open, and the upper end matters.
///
/// `magnitude` builds its value as `fraction * 2^exp` with `fraction = 1 + bits/2^52`, so the
/// mantissa bound has to be `2^52 - 1`: one higher and `fraction` reaches exactly `2.0`, which
/// silently widens every range by an octave. The module doc already warns that the largest value
/// is `2^(max_exp + 1)` rather than `2^max_exp`, and this is what keeps that statement true.
///
/// Mutation testing found the bound unpinned in both directions.
#[test]
fn the_mantissa_stays_in_its_half_open_interval() {
    let drawn = draw_many(&magnitude(0, 0), 4_000);
    assert!(!drawn.is_empty());

    for value in &drawn {
        let mantissa = value.abs();
        assert!(
            (1.0..2.0).contains(&mantissa),
            "magnitude(0, 0) produced {mantissa:e}, outside the half-open [1, 2)"
        );
    }
    // And it must actually approach the top, or the bound is too tight rather than too loose.
    assert!(
        drawn.iter().any(|v| v.abs() > 1.99),
        "the mantissa never approached 2.0"
    );
}

/// `in_range` must produce the **interior**, not just the three named points.
///
/// The generator is a weighted choice between the two endpoints plus the midpoint, and a spread
/// across the range. Mutation testing turned the spread's `1 << 30` into `1 >> 30`, collapsing it
/// to zero so every "spread" draw returned the low endpoint — and the existing test still passed,
/// because it only asked that both ends be reached and the boundary branch still reached them.
#[test]
fn in_range_fills_the_interior_and_not_only_its_named_points() {
    let low = 0.0;
    let high = 1.0;
    let drawn = draw_many(&in_range(low, high), 2_000);
    let midpoint = low + (high - low) / 2.0;

    let interior = drawn
        .iter()
        .filter(|v| **v != low && **v != high && **v != midpoint)
        .count();
    assert!(
        interior > drawn.len() / 4,
        "only {interior} of {} draws were interior points; the spread has collapsed onto the \
         named boundaries",
        drawn.len()
    );
    // And those interior points must be spread, not clustered on one value.
    let distinct: std::collections::BTreeSet<u64> = drawn.iter().map(|v| v.to_bits()).collect();
    assert!(
        distinct.len() > 100,
        "only {} distinct values across {} draws",
        distinct.len(),
        drawn.len()
    );
}

/// A cancelling pair must cancel **at every scale**, not just at one.
///
/// [`cancelling_pair_is_actually_ill_conditioned`] already checks that the cancellation is
/// severe, and it is blind to the magnitude the cancellation happens at: narrowing the
/// generator's `magnitude(-40, 40)` to `magnitude(40, 40)` leaves every pair ill-conditioned
/// around `1e12` and that test green. Mutation testing found exactly that edit.
///
/// Scale is the whole reason the range is there. A cancellation-safe form can be correct around
/// one and wrong among subnormals, where the perturbation itself underflows, and a generator
/// pinned to a single octave would never ask.
#[test]
fn a_cancelling_pair_spans_the_magnitudes_it_declares() {
    let drawn = draw_many(&cancelling_pair(), 2_000);
    assert!(!drawn.is_empty());

    assert!(
        drawn.iter().any(|(x, _)| x.abs() < 1e-6),
        "never produced a small-magnitude pair, so cancellation near zero is untested"
    );
    assert!(
        drawn.iter().any(|(x, _)| x.abs() > 1e6),
        "never produced a large-magnitude pair"
    );
    assert!(
        drawn.iter().any(|(x, _)| (0.1..10.0).contains(&x.abs())),
        "never produced a pair near one"
    );
}

/// The mantissa bound, asserted as arithmetic rather than sampled.
///
/// See [`super::MANTISSA_MAX`] for why this cannot be a behavioural test: a bound one too high
/// lets `fraction` reach exactly `2.0`, which widens every `magnitude` range by an octave and is
/// reachable with probability `2^-52`. Four thousand draws asserting `[1, 2)` did not notice.
#[test]
fn the_mantissa_bound_keeps_the_interval_half_open() {
    assert_eq!(super::MANTISSA_BITS, 52, "an f64 mantissa is 52 bits");
    assert_eq!(
        super::MANTISSA_MAX,
        4_503_599_627_370_495,
        "the largest mantissa pattern must be 2^52 - 1, one short of the scale"
    );
    assert_eq!(super::MANTISSA_SCALE, 4_503_599_627_370_496.0);

    // The largest fraction the generator can build is strictly below two, and only just.
    #[allow(clippy::cast_precision_loss)]
    let largest = 1.0 + (super::MANTISSA_MAX as f64) / super::MANTISSA_SCALE;
    assert!(largest < 2.0, "the interval is not half-open: {largest}");
    assert_eq!(
        largest,
        f64::from_bits(2.0_f64.to_bits() - 1),
        "the largest fraction must be the ulp below two"
    );
}

/// `in_range` must reach its **midpoint**, not only its endpoints.
///
/// The generator names three points it will always hit — both bounds and the middle — and the
/// middle was unpinned. Mutation testing replaced `(high - low) / 2.0` with `(high - low) % 2.0`,
/// which for a range of width ten collapses the midpoint onto the lower bound: still in range,
/// still hitting both ends, and the existing tests had nothing to say about it.
///
/// The midpoint is deliberate because it is where a range's interesting value often sits — the
/// middle of `[0, pi]`, the crossover in a piecewise kernel — and losing it is a silent loss of
/// exactly the coverage this generator was built to guarantee.
#[test]
fn in_range_reaches_its_declared_midpoint() {
    let drawn = draw_many(&in_range(-2.5, 7.5), 2_000);
    assert!(
        drawn.contains(&2.5),
        "never produced the midpoint of [-2.5, 7.5]"
    );

    // And on an asymmetric range, so a midpoint that coincided with a bound would not pass.
    let drawn = draw_many(&in_range(0.0, 1.0), 2_000);
    assert!(
        drawn.contains(&0.5),
        "never produced the midpoint of [0, 1]"
    );
}
