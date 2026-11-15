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
