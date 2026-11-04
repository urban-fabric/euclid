//! The min/max NaN policy, and how it differs from mask-based selection.

use crate::{Compare, Real};

// --- min / max NaN policy -----------------------------------------------

#[test]
fn min_and_max_return_the_non_nan_operand() {
    // This is the sharpest divergence risk on the trait. `f64::min`/`max` implement IEEE
    // minNum/maxNum: a NaN operand is *ignored*. That is the opposite of NaN-propagating
    // arithmetic, and the opposite of what a mask-based `select(a < b, a, b)` would give
    // (a comparison with NaN is false, so the branch would pick `b` unconditionally).
    let n = f64::NAN;
    assert_eq!(Real::min(n, 1.0), 1.0, "NaN is ignored, not propagated");
    assert_eq!(Real::max(n, 1.0), 1.0);
    assert_eq!(Real::min(1.0, n), 1.0);
    assert_eq!(Real::max(1.0, n), 1.0);
    // Both NaN gives NaN, there being nothing else to return.
    assert!(Real::min(n, n).is_nan());

    // Ordinary operands behave as expected.
    assert_eq!(Real::min(2.0f64, 5.0), 2.0);
    assert_eq!(Real::max(2.0f64, 5.0), 5.0);
}

#[test]
fn min_and_max_differ_from_the_mask_based_selection_on_nan() {
    // Spelled out, because a carrier that implements `min` as `select(lt, a, b)` would *not*
    // match: with a NaN operand the comparison is false and the result differs.
    let n = f64::NAN;
    let mask_based = Compare::select(&Compare::lt(&n, &1.0), n, 1.0);
    assert_eq!(mask_based, 1.0);
    assert_eq!(Real::min(n, 1.0), 1.0);
    // Here they agree; with the operands swapped they do not.
    let swapped = Compare::select(&Compare::lt(&1.0, &n), 1.0, n);
    assert!(swapped.is_nan(), "mask-based selection propagates the NaN");
    assert_eq!(Real::min(1.0, n), 1.0, "min does not");
}

#[test]
fn clamp_composes_min_and_max() {
    for i in -50..50 {
        let x = f64::from(i) * 0.3;
        let (lo, hi) = (-2.0f64, 3.0f64);
        assert_eq!(Real::clamp(x, lo, hi), Real::min(Real::max(x, lo), hi));
    }
}
