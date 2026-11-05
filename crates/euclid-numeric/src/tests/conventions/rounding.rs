//! Rounding-mode conventions.

use crate::Real;

// --- rounding family ----------------------------------------------------

#[test]
fn round_is_half_away_from_zero_not_half_to_even() {
    // The genuinely ambiguous one. IEEE's default is half-to-even (`2.5 -> 2`); Rust's
    // `f64::round` is half-away-from-zero (`2.5 -> 3`). A carrier that lowers to a
    // banker's-rounding primitive would silently disagree.
    assert_eq!(Real::round(0.5f64), 1.0);
    assert_eq!(Real::round(1.5f64), 2.0);
    assert_eq!(
        Real::round(2.5f64),
        3.0,
        "half-away-from-zero, not half-to-even"
    );
    assert_eq!(Real::round(-0.5f64), -1.0);
    assert_eq!(Real::round(-2.5f64), -3.0);
    assert_eq!(Real::round(2.4f64), 2.0);
    assert_eq!(Real::round(2.6f64), 3.0);
}

#[test]
fn trunc_and_floor_differ_on_negatives() {
    // `trunc` rounds toward zero; `floor` toward −∞. They agree only for non-negatives.
    assert_eq!(Real::trunc(2.7f64), 2.0);
    assert_eq!(Real::floor(2.7f64), 2.0);
    assert_eq!(Real::trunc(-2.7f64), -2.0, "trunc goes toward zero");
    assert_eq!(Real::floor(-2.7f64), -3.0, "floor goes toward -inf");
    assert_eq!(Real::ceil(-2.7f64), -2.0);
    assert_eq!(Real::ceil(2.1f64), 3.0);
}

#[test]
fn fract_keeps_the_sign_of_its_argument() {
    // `fract` is `self − trunc(self)`, so it is negative for negative inputs. The other
    // common definition (`self − floor(self)`, always in `[0,1)`) would disagree on sign.
    assert!((Real::fract(2.75f64) - 0.75).abs() < 1e-15);
    assert!(
        (Real::fract(-2.75f64) + 0.75).abs() < 1e-15,
        "fract(-2.75) is -0.75, not +0.25"
    );
    // The defining identity, across signs.
    for i in -40..40 {
        let x = f64::from(i) * 0.37;
        let err = (Real::trunc(x) + Real::fract(x) - x).abs();
        assert!(err < 1e-14, "trunc + fract != x at {x}");
    }
}

#[test]
fn floor_and_ceil_bracket_every_value() {
    for i in -100..100 {
        let x = f64::from(i) * 0.41;
        let (f, c) = (Real::floor(x), Real::ceil(x));
        assert!(f <= x && x <= c, "floor <= x <= ceil failed at {x}");
        assert!(c - f <= 1.0, "floor and ceil differ by at most 1 at {x}");
        // Integral inputs are fixed points.
        assert_eq!(Real::floor(f), f);
        assert_eq!(Real::ceil(c), c);
    }
}

#[test]
fn rounding_is_idempotent_and_preserves_integers() {
    for i in -20..20 {
        let n = f64::from(i);
        assert_eq!(Real::round(n), n);
        assert_eq!(Real::floor(n), n);
        assert_eq!(Real::ceil(n), n);
        assert_eq!(Real::trunc(n), n);
        assert_eq!(Real::fract(n), 0.0);
    }
}

// --- polynomial evaluation: coefficient order, and the unfused multiply-add ---
//
// `polyval` deliberately does *not* fuse, which is unusual enough to pin: a
// fused multiply-add is more accurate, and that is the problem. The series
// expansions calling it are fitted against a reference implementation that does
// not fuse, so matching it bit-for-bit is what lets those fits be checked
// against published values.

#[test]
fn polyval_is_horner_with_the_highest_power_first() {
    // 2x² − 3x + 1
    assert_eq!(4.0_f64.polyval(&[2.0, -3.0, 1.0]), 21.0);
    assert_eq!(0.0_f64.polyval(&[2.0, -3.0, 1.0]), 1.0);
    // A single coefficient is a constant, and no coefficients is zero.
    assert_eq!(9.0_f64.polyval(&[7.0]), 7.0);
    assert_eq!(9.0_f64.polyval(&[]), 0.0);
    // Order matters, and getting it backwards is the likely mistake.
    assert_ne!(4.0_f64.polyval(&[1.0, -3.0, 2.0]), 21.0);
}

#[test]
fn polyval_does_not_fuse_its_multiply_add() {
    // The fused and unfused forms differ in the last ULP for a well-chosen
    // argument, and the callers depend on the unfused one. Construct a case
    // where `mul_add` and `a * x + b` disagree, then require the unfused answer.
    // (1 + 2⁻²⁷)² is 1 + 2⁻²⁶ + 2⁻⁵⁴: the last term falls below the ULP of 1, so
    // the rounded product is 1 + 2⁻²⁶ and the unfused sum with −(1 + 2⁻²⁶) is
    // exactly zero, while the fused form keeps the 2⁻⁵⁴.
    let x = 1.0 + 2.0_f64.powi(-27);
    let a = x;
    let b = -(1.0 + 2.0_f64.powi(-26));
    let unfused = a * x + b;
    let fused = a.mul_add(x, b);
    assert_ne!(
        unfused, fused,
        "the fixture must actually distinguish the two"
    );
    assert_eq!(x.polyval(&[a, b]), unfused);
}
