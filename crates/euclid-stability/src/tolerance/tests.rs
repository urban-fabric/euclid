//! The comparisons, especially the cases where a naive tolerance gives the wrong answer.

use super::{
    compensated_sum, condition_number, ulps_between, within_absolute, within_relative, within_ulps,
};

#[test]
fn identical_values_are_zero_ulps_apart() {
    assert_eq!(ulps_between(1.0, 1.0), Some(0));
    assert_eq!(ulps_between(f64::MAX, f64::MAX), Some(0));
}

/// The two zeroes have different bit patterns but compare equal, and no numerical contract
/// distinguishes them as results.
#[test]
fn signed_zeroes_are_equal() {
    assert_eq!(ulps_between(0.0, -0.0), Some(0));
    assert!(within_ulps(0.0, -0.0, 0));
    assert!(within_relative(0.0, -0.0, 0.0));
}

#[test]
fn adjacent_floats_are_one_ulp_apart() {
    let next = f64::from_bits(1.0_f64.to_bits() + 1);
    assert_eq!(ulps_between(1.0, next), Some(1));
    assert!(within_ulps(1.0, next, 1));
    assert!(!within_ulps(1.0, next, 0));
}

/// The ordering must be monotone *across* zero, or the distance between a small negative and a
/// small positive would come out enormous.
#[test]
fn ulps_are_monotone_across_zero() {
    let smallest = f64::from_bits(1); // smallest positive subnormal
    assert_eq!(ulps_between(0.0, smallest), Some(1));
    assert_eq!(ulps_between(-smallest, 0.0), Some(1));
    assert_eq!(ulps_between(-smallest, smallest), Some(2));
}

/// Scale-freedom is the property that makes ULPs the right default: the same relative accuracy
/// is the same ULP distance at any magnitude.
#[test]
fn ulps_are_scale_free() {
    for exponent in [-300i32, -10, 0, 10, 300] {
        let value = 2.0_f64.powi(exponent);
        let next = f64::from_bits(value.to_bits() + 1);
        assert_eq!(ulps_between(value, next), Some(1), "failed at 2^{exponent}");
    }
}

#[test]
fn nan_is_close_to_nothing_including_itself() {
    assert_eq!(ulps_between(f64::NAN, f64::NAN), None);
    assert!(!within_ulps(f64::NAN, f64::NAN, u64::MAX));
    assert!(!within_relative(f64::NAN, f64::NAN, 1.0));
    assert!(!within_absolute(f64::NAN, f64::NAN, f64::INFINITY));
    assert!(!within_ulps(f64::NAN, 1.0, u64::MAX));
}

#[test]
fn equal_infinities_are_equal_but_unequal_ones_are_not_close() {
    assert!(within_ulps(f64::INFINITY, f64::INFINITY, 0));
    assert!(!within_ulps(f64::INFINITY, f64::NEG_INFINITY, u64::MAX));
    assert!(!within_ulps(f64::INFINITY, f64::MAX, u64::MAX));
}

/// The failure mode `within_absolute` has and `within_relative` does not: a fixed absolute
/// tolerance is meaningless once the values are small enough.
#[test]
fn an_absolute_tolerance_passes_vacuously_at_small_scale() {
    let a: f64 = 1e-300;
    let b = 3e-300; // three times larger -- 200% relative error
    assert!(
        within_absolute(a, b, 1e-9),
        "an absolute tolerance accepts a 200% error at this scale"
    );
    assert!(
        !within_relative(a, b, 1e-9),
        "a relative tolerance must reject it"
    );
}

/// And the mirror image: a fixed absolute tolerance is unreachable once the values are large.
#[test]
fn an_absolute_tolerance_fails_always_at_large_scale() {
    let a: f64 = 1e300;
    let b = f64::from_bits(a.to_bits() + 1); // one ulp apart -- as close as possible
    assert!(
        !within_absolute(a, b, 1e-9),
        "an absolute tolerance rejects adjacent floats at this scale"
    );
    assert!(within_ulps(a, b, 1), "they are one ulp apart");
    assert!(within_relative(a, b, 1e-15));
}

/// Relative comparison against zero must fail rather than silently pass, or an implementation
/// that returns zero for everything passes its own accuracy tests.
#[test]
fn relative_comparison_against_zero_fails_loudly() {
    assert!(!within_relative(0.0, 1e-300, 0.5));
    assert!(!within_relative(1e-300, 0.0, 0.5));
    // Absolute is the right tool when zero is a legitimate expected value.
    assert!(within_absolute(0.0, 1e-300, 1e-12));
}

#[test]
fn relative_comparison_handles_overflow_without_losing_small_gaps() {
    let max = f64::MAX;
    for (a, b) in [(max, -max), (-max, max)] {
        assert!(within_relative(a, b, 2.0));
        assert!(!within_relative(a, b, 2.0_f64.next_down()));
    }
    assert!(within_relative(max, -max / 2.0, 1.5));
    assert!(!within_relative(max, -max / 2.0, 1.5_f64.next_down()));
    assert!(!within_relative(
        3.0,
        3.0_f64.next_down(),
        0.625 * f64::EPSILON
    ));
    assert!(!within_relative(max, -f64::INFINITY, 2.0));
}

#[test]
fn the_condition_number_of_a_benign_sum_is_near_one() {
    assert!((condition_number(&[1.0, 2.0, 3.0]) - 1.0).abs() < 1e-15);
}

/// The case compensated summation exists for: terms that mostly cancel have a huge condition
/// number, and the condition number is what predicts how many digits a naive sum loses.
#[test]
fn a_cancelling_sum_is_ill_conditioned() {
    let terms = [1e16, 1.0, -1e16];
    let condition = condition_number(&terms);
    assert!(
        condition > 1e15,
        "expected an ill-conditioned sum, got {condition:e}"
    );

    // And the prediction holds: the naive left-to-right sum loses the 1.0 entirely.
    let naive = terms.iter().fold(0.0, |acc, t| acc + t);
    assert_eq!(naive, 0.0, "naive summation lost the small term");
}

#[test]
fn exact_cancellation_is_infinitely_ill_conditioned() {
    assert_eq!(condition_number(&[1.0, -1.0]), f64::INFINITY);
    assert_eq!(condition_number(&[]), f64::INFINITY);
}

/// The regression for the bug this function used to have: summing the denominator naively made
/// `condition_number` report `inf` for exactly the catastrophic-cancellation inputs it exists to
/// measure. The true sum of `[1e16, 1, -1e16]` is `1`, so the condition number is about `2e16`.
#[test]
fn the_condition_number_survives_the_cancellation_it_measures() {
    let terms = [1e16, 1.0, -1e16];

    // Precondition: the naive fold really does lose the small term, so this is the hard case.
    let naive = terms.iter().fold(0.0_f64, |acc, t| acc + t);
    assert_eq!(
        naive, 0.0,
        "precondition: naive summation must lose the term"
    );

    let condition = condition_number(&terms);
    assert!(
        condition.is_finite(),
        "condition_number reported {condition:e} rather than a finite value"
    );
    assert!(
        (1e16..1e17).contains(&condition),
        "expected a condition number near 2e16, got {condition:e}"
    );
}

/// The summation kernel itself, pinned to an **exact** value.
///
/// Every test above reaches `compensated_sum` through `condition_number`, and not one of them
/// pins it. They assert a *band* — `condition.is_finite()`, `(1e16..1e17)` — on a quantity
/// computed as `magnitude / total.abs()`, so a correction that comes out negated, doubled or
/// dropped still lands inside the band and the suite stays green. Mutation testing found four
/// separate ways to corrupt this function that the entire 1,169-test workspace suite accepted,
/// including flipping the sign of the correction term.
///
/// That is the gap line coverage cannot see: the line *executes* in three tests and is *pinned*
/// by none of them. The fix is to assert the number, not a range containing it.
///
/// Both orderings are checked because they take different branches of the
/// `total.abs() >= term.abs()` test, and that branch is the whole difference between
/// Kahan–Babuška–Neumaier and plain Kahan — the latter is wrong exactly when a later term is
/// larger than the running total, which is what `[1.0, 1e16, -1e16]` arranges.
#[test]
fn the_compensated_sum_is_exact_where_the_naive_fold_is_not() {
    for terms in [[1e16, 1.0, -1e16], [1.0, 1e16, -1e16]] {
        let naive = terms.iter().fold(0.0_f64, |acc, t| acc + t);
        assert_eq!(
            naive, 0.0,
            "precondition: the naive fold must lose the small term for {terms:?}"
        );
        assert_eq!(
            compensated_sum(&terms),
            1.0,
            "the compensated sum of {terms:?} is exactly 1.0"
        );
    }

    // A benign sum must come out exactly too, or the correction is leaking into cases that do
    // not need it.
    assert_eq!(compensated_sum(&[1.0, 2.0, 3.0]), 6.0);
    assert_eq!(compensated_sum(&[1.0, -1.0]), 0.0);
    assert_eq!(compensated_sum(&[]), 0.0);
}

/// `within_absolute` measures `|a - b|`, not `|a + b|`.
///
/// Two values straddling zero are as far apart as they get; a sum would call them identical.
/// Found by mutation testing, which replaced the subtraction with an addition and watched the
/// suite pass — because every existing case compared same-signed values, where the two
/// expressions rarely disagree enough to matter.
#[test]
fn within_absolute_measures_a_difference_not_a_sum() {
    assert!(
        !within_absolute(1.0, -1.0, 0.5),
        "values two apart must not compare within half"
    );
    assert!(!within_absolute(1e-3, -1e-3, 1e-9));
    // And the honest direction still works.
    assert!(within_absolute(1.0, 1.25, 0.5));
}

/// And the same at a magnitude where the naive fold loses the term entirely in both directions.
#[test]
fn the_condition_number_is_finite_across_the_cancellation_range() {
    for exponent in [16i32, 18, 20, 30] {
        let half = 10f64.powi(exponent);
        let condition = condition_number(&[half, 1.0, -half]);
        assert!(
            condition.is_finite(),
            "condition_number was {condition:e} at 1e{exponent}"
        );
    }
}
