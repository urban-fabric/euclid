//! Representation checks for the numeric module.

use super::*;

#[test]
fn next_up_steps_exactly_one_representable_value() {
    assert_eq!(next_up(0.0).to_bits(), 1);
    assert_eq!(next_up(1.0), f64::from_bits(1.0_f64.to_bits() + 1));
    // Downward through the negatives is still upward in value.
    assert!(next_up(-1.0) > -1.0);
    assert_eq!(next_up(f64::INFINITY), f64::INFINITY);
    assert!(next_up(f64::NAN).is_nan());
}

#[test]
fn next_down_is_the_inverse_of_next_up() {
    for x in [1.0_f64, -1.0, 1e-300, 1e300, 0.5] {
        assert_eq!(next_down(next_up(x)), x, "round trip failed at {x}");
    }
    assert_eq!(next_up(next_down(0.0)), 0.0);
}

#[test]
fn a_monotone_function_has_no_inversion() {
    assert!(scan_monotone(f64::exp, 1.0, 5_000).is_none());
    assert!(scan_monotone(f64::sqrt, 2.0, 5_000).is_none());
    // A plateau is still monotone: equality is not an inversion.
    assert!(scan_monotone(|_| 1.0, 1.0, 1_000).is_none());
}

/// The defect this detector exists for: a function accurate at every point whose error
/// oscillates by more than the output spacing, so it is not monotone even though no individual
/// value is wrong by much. A pointwise accuracy test cannot see this.
#[test]
fn ulp_level_wiggle_is_found() {
    // `exp` perturbed by an alternating one-ulp error. Every value is within one ulp of the
    // truth; the sequence is not monotone.
    let wiggly = |x: f64| {
        let base = x.exp();
        if x.to_bits().is_multiple_of(2) {
            base
        } else {
            f64::from_bits(base.to_bits().saturating_sub(2))
        }
    };

    let inversion =
        scan_monotone(wiggly, 1.0, 5_000).expect("alternating error must break monotonicity");
    assert!(
        inversion.values.1 < inversion.values.0,
        "an inversion must report a decrease: {:?}",
        inversion.values
    );
}

#[test]
fn a_decreasing_function_inverts_immediately() {
    let inversion = scan_monotone(|x: f64| -x, 1.0, 10).expect("negation decreases");
    assert!(inversion.values.1 < inversion.values.0);
}

/// `next_down` saturates at negative infinity rather than stepping off the end into NaN.
///
/// The guard is `x.is_nan() || x == f64::NEG_INFINITY`, and weakening that `||` to `&&` makes
/// `next_down(-inf)` decrement the bit pattern of negative infinity — which is a NaN. A scan
/// walking downward would then stop reporting values and start reporting NaN, silently.
#[test]
fn next_down_saturates_at_negative_infinity() {
    assert_eq!(next_down(f64::NEG_INFINITY), f64::NEG_INFINITY);
    assert!(next_down(f64::NAN).is_nan());
    // The upward counterpart, for symmetry.
    assert_eq!(next_up(f64::INFINITY), f64::INFINITY);
    assert!(next_up(f64::NAN).is_nan());
}
