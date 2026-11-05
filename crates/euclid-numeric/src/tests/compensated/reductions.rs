//! The reductions built on the primitives: compensated summation and an accurate
//! dot product.
//!
//! These are the "pay-off" tests this module's header refers to. Each pairs the
//! reduction against the naive fold it replaces and asserts *both* halves — that
//! the naive form loses the digits, and that this one keeps them. A test that
//! only checked the good answer would still pass if the naive form had been fine
//! all along, which would make the reduction pointless rather than correct.

use crate::compensated;

#[test]
fn compensated_sum_recovers_what_a_naive_fold_drops() {
    // One large term and many small ones, each below the running total's ULP.
    let mut xs = vec![1.0e16_f64];
    xs.extend(std::iter::repeat_n(1.0, 1000));
    let naive: f64 = xs.iter().sum();
    assert_eq!(naive, 1.0e16, "the naive fold must actually lose them");
    assert_eq!(compensated::sum(&xs), 1.0e16 + 1000.0);
    assert_eq!(compensated::sum_iter(xs.iter().copied()), 1.0e16 + 1000.0);
}

#[test]
fn compensated_sum_handles_a_cancelling_sequence() {
    // Kahan–Babuška–Neumaier's distinguishing case: a later term larger than the
    // running total, which plain Kahan gets wrong.
    let xs = [1.0_f64, 1.0e100, 1.0, -1.0e100];
    assert_eq!(compensated::sum(&xs), 2.0);
    assert_eq!(compensated::sum::<f64>(&[]), 0.0);
}

#[test]
fn compensated_dot_is_exact_where_products_cancel() {
    // Nearly orthogonal: the answer is the residue of two large products, so the
    // rounding inside each product is the whole answer.
    let a = [1.0e8_f64, 1.0];
    let b = [1.0e8_f64, -1.0e16];
    assert_eq!(compensated::dot(&a, &b), 0.0);
    // And it agrees with the obvious answer when nothing cancels.
    assert_eq!(compensated::dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]), 32.0);
    // Ragged input uses the shorter length rather than panicking.
    assert_eq!(compensated::dot(&[1.0, 2.0, 99.0], &[3.0, 4.0]), 11.0);
}

#[test]
fn compensated_reductions_saturate_rather_than_poisoning_the_accumulator() {
    // `two_sum` reports a zero remainder for a non-finite sum precisely so an
    // overflow does not leave a `NaN` in the correction term.
    assert!(compensated::sum(&[f64::MAX, f64::MAX]).is_infinite());
    assert!(compensated::sum(&[1.0, f64::NAN, 1.0]).is_nan());
}

#[test]
fn the_reductions_run_at_f32_too() {
    // `f32` is the other carrier this crate implements, and a compensated sum is
    // worth more there, not less: the ULP it recovers is 2^24 times coarser.
    assert_eq!(compensated::sum(&[1.0_f32, 2.0, 3.0]), 6.0);

    // The base has to be large enough that 1.0 falls below its ULP: at 1e7 the
    // `f32` spacing is still 1.0, so a naive fold there loses nothing and proves
    // nothing. At 1e8 the spacing is 8.
    let mut xs = vec![1.0e8_f32];
    xs.extend(std::iter::repeat_n(1.0, 100));
    let naive: f32 = xs.iter().sum();
    assert_eq!(naive, 1.0e8, "the naive f32 fold must actually lose them");

    // Compared in `f64` against the exact answer, because `1e8 + 100` is not
    // itself an `f32` — the claim is that this lands nearer, not that it is exact.
    let exact = 1.0e8_f64 + 100.0;
    let compensated_err = (f64::from(compensated::sum(&xs)) - exact).abs();
    let naive_err = (f64::from(naive) - exact).abs();
    assert!(
        compensated_err < naive_err,
        "compensated err {compensated_err} should beat naive {naive_err}"
    );
}
