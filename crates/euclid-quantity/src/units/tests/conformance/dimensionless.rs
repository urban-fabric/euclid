//! Ratio, reciprocal and clamping laws for factors and percentages.

use crate::{Factor, Percentage};

/// `unit`, `times`, `recip` and the two clamping helpers.
///
/// `recip` could return a default for every input, and `Factor::clamped` and
/// `Factor::non_negative` likewise — the helpers a caller reaches for precisely when arithmetic
/// has produced something out of range, so a broken one fails where it is most needed.
#[test]
fn the_dimensionless_tier_multiplies_and_inverts() {
    use euclid_units::Dimensionless;

    const TOL: f64 = 1e-12;

    assert_eq!(Factor::unit().as_f64(), 1.0, "the unit factor is one");
    assert_eq!(Percentage::unit().as_percent(), 100.0, "unit is 100%");

    for v in [0.5_f64, 1.0, 2.0, -4.0] {
        let a = Factor::new(v);
        assert!(
            (a.recip().as_f64() - 1.0 / v).abs() <= TOL,
            "recip must invert"
        );
        // An involution, which is the law a caller relies on.
        assert!(
            (a.recip().recip().as_f64() - v).abs() <= TOL,
            "recip twice returns"
        );
        assert!(
            (Dimensionless::times(a, Factor::new(3.0)).as_f64() - v * 3.0).abs() <= TOL,
            "times must multiply"
        );
    }

    // `Percentage` has its own `recip`, which is a separate implementation from `Factor`'s and
    // was separately unpinned. Its unit is 100, so the reciprocal is taken about that.
    for v in [25.0_f64, 50.0, 200.0] {
        let r = Percentage::new(v).recip().as_percent();
        assert!(
            (r - 100.0 * 100.0 / v).abs() <= 1e-9,
            "Percentage::recip({v}) = {r}, which is not the reciprocal about unit"
        );
    }

    // The clamping helpers, at and either side of their bounds.
    for (v, want) in [(-1.0, 0.0), (0.0, 0.0), (0.5, 0.5), (1.0, 1.0), (2.0, 1.0)] {
        assert_eq!(
            Factor::new(v).clamped(0.0, 1.0).as_f64(),
            want,
            "clamped({v}, 0, 1)"
        );
    }
    for (v, want) in [(-5.0, 0.0), (-0.0, 0.0), (0.0, 0.0), (3.0, 3.0)] {
        assert_eq!(
            Factor::new(v).non_negative().as_f64(),
            want,
            "non_negative({v})"
        );
    }
}
