//! Replay, shrinking and the bit-level assembly.
//!
//! Part of the [`float`](super) generator's tests. These are about *determinism* rather than
//! coverage: a tape has to replay to the same value, shrink toward zero and assemble each
//! exponent exactly, or a recorded counterexample is not reproducible and the whole regression
//! mechanism is decoration.

use super::super::{binade_boundary, cancelling_pair, magnitude};
use crate::generate::Gen;
use crate::source::Source;

fn draw_many<T: 'static>(generator: &Gen<T>, count: usize) -> Vec<T> {
    let mut source = Source::generating(0xC0FFEE, 1 << 20);
    (0..count)
        .filter_map(|_| generator.draw(&mut source).ok())
        .collect()
}

#[test]
fn cancellation_replays_toward_zero_at_the_upper_magnitude_bound() {
    let generator = cancelling_pair();
    let ceiling = 2.0_f64.powi(41);
    for (sign_word, sign) in [(0, 1.0), (u64::MAX, -1.0)] {
        // Largest exponent and mantissa, then the sign and a one-bit perturbation.
        let mut tape = Source::replaying(vec![u64::MAX, u64::MAX, sign_word, 0]);
        let x = sign * ceiling.next_down();
        assert_eq!(generator.draw(&mut tape), Ok((x, x / 2.0)));
    }
}

#[test]
fn magnitude_replays_exact_powers_through_the_subnormal_range() {
    for exponent in -1074..=1023 {
        let generator = magnitude(exponent, exponent);
        let scale = if exponent < -1022 {
            f64::MIN_POSITIVE * 2.0_f64.powi(exponent + 1022)
        } else {
            2.0_f64.powi(exponent)
        };
        for (sign_word, sign) in [(0, 1.0), (u64::MAX, -1.0)] {
            for (mantissa_word, factor) in [(0, 1.0), (1 << 63, 1.5)] {
                let mut tape = Source::replaying(vec![0, mantissa_word, sign_word]);
                let value = generator.draw(&mut tape).expect("a complete tape");
                assert_eq!(value.to_bits(), (sign * (factor * scale)).to_bits());
            }
        }
    }
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

/// The mantissa bound, asserted as arithmetic rather than sampled.
///
/// See [`super::super::MANTISSA_MAX`] for why this cannot be a behavioural test: a bound one too high
/// lets `fraction` reach exactly `2.0`, which widens every `magnitude` range by an octave and is
/// reachable with probability `2^-52`. Four thousand draws asserting `[1, 2)` did not notice.
#[test]
fn the_mantissa_bound_keeps_the_interval_half_open() {
    assert_eq!(
        super::super::MANTISSA_BITS,
        52,
        "an f64 mantissa is 52 bits"
    );
    assert_eq!(
        super::super::MANTISSA_MAX,
        4_503_599_627_370_495,
        "the largest mantissa pattern must be 2^52 - 1, one short of the scale"
    );
    assert_eq!(super::super::MANTISSA_SCALE, 4_503_599_627_370_496.0);

    // The largest fraction the generator can build is strictly below two, and only just.
    #[allow(clippy::cast_precision_loss)]
    let largest = 1.0 + (super::super::MANTISSA_MAX as f64) / super::super::MANTISSA_SCALE;
    assert!(largest < 2.0, "the interval is not half-open: {largest}");
    assert_eq!(
        largest,
        f64::from_bits(2.0_f64.to_bits() - 1),
        "the largest fraction must be the ulp below two"
    );
}

/// `binade_boundary` must produce *only* exact powers of two, and reach both signs.
///
/// The generator exists because [`magnitude`] cannot reach a boundary, so the one thing it must
/// not do is drift off one. A mantissa bit anywhere in the output would make it a worse
/// `magnitude` rather than a different generator.
#[test]
fn binade_boundary_produces_only_exact_powers_of_two() {
    let drawn = draw_many(&binade_boundary(-8, 8), 2_000);
    for value in &drawn {
        let magnitude_bits = value.abs().to_bits();
        // A power of two has an all-zero mantissa field.
        assert_eq!(
            magnitude_bits & ((1u64 << 52) - 1),
            0,
            "{value:e} is not an exact power of two"
        );
        assert!(
            value.abs() >= 2.0_f64.powi(-8) && value.abs() <= 2.0_f64.powi(8),
            "{value:e} is outside the declared range"
        );
    }
    assert!(drawn.iter().any(|v| *v < 0.0), "never produced a negative");
    assert!(drawn.iter().any(|v| *v > 0.0), "never produced a positive");
    // The exact endpoints, which is the claim the docstring makes against `magnitude`'s
    // extra octave: nothing above `2^max_exp`.
    assert!(drawn.contains(&256.0), "never produced 2^8");
    assert!(drawn.contains(&-256.0), "never produced -2^8");
    assert!(
        drawn.iter().all(|v| v.abs() <= 256.0),
        "produced a value above 2^8"
    );
    // And the negative unit, which is the value that distinguishes a direction error.
    assert!(drawn.contains(&-1.0), "never produced -1.0");
}

/// The subnormal powers are reachable, since that is the other end where a scale goes wrong.
#[test]
fn binade_boundary_reaches_the_subnormals() {
    let drawn = draw_many(&binade_boundary(-1_074, -1_070), 500);
    assert!(
        drawn.iter().any(|v| v.abs() < f64::MIN_POSITIVE),
        "never produced a subnormal"
    );
    assert!(
        drawn.contains(&5e-324) || drawn.contains(&-5e-324),
        "never produced the smallest subnormal"
    );
    for value in &drawn {
        assert_ne!(*value, 0.0, "underflowed to zero: the power must survive");
    }
}

/// Every exponent in `f64`'s range assembles to the right value, checked against `log2`.
///
/// Exhaustive over all 2,098 exponents rather than sampled, because [`binade_boundary`] builds
/// its bit pattern by hand and a field assembled one bit off would still *look* like a power of
/// two — zero mantissa, plausible magnitude — while being the wrong one. `log2` of an exact
/// power of two is exact, so it recovers the exponent with no tolerance anywhere.
#[test]
fn every_power_of_two_assembles_to_its_exponent() {
    for exp in -1_074..=1_023 {
        let value = super::super::power_of_two(exp);
        assert!(value > 0.0, "2^{exp} assembled to {value:e}");
        assert!(value.is_finite(), "2^{exp} assembled to {value:e}");
        assert_eq!(
            value.log2(),
            f64::from(exp),
            "2^{exp} assembled to {value:e}, whose log2 is {}",
            value.log2()
        );
        // The subnormal powers are the ones `powi` loses; assert the boundary explicitly.
        if exp < -1_022 {
            assert!(value < f64::MIN_POSITIVE, "2^{exp} should be subnormal");
        } else {
            assert!(value >= f64::MIN_POSITIVE, "2^{exp} should be normal");
        }
    }
    // The named endpoints, so a reader can see the range is the whole format.
    assert_eq!(super::super::power_of_two(-1_074), 5e-324);
    assert_eq!(super::super::power_of_two(-1_022), f64::MIN_POSITIVE);
    assert_eq!(super::super::power_of_two(0), 1.0);
}
