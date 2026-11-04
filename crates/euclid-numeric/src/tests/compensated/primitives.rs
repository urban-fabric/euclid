//! The exact-arithmetic primitives, and the accurate reductions they buy.
//!
//! Part of the [`compensated`](super) suite.

use crate::Real;
use euclid_utils::{assert_close, assert_exact, rel, sweep};

// --- two_sum -------------------------------------------------------------

#[test]
fn two_sum_is_exact() {
    // The defining property: value + error reconstructs the exact sum. Checked on pairs
    // chosen so the rounded sum genuinely discards something.
    let cases = [
        (1.0f64, f64::EPSILON / 2.0),
        (1e300, 1.0),
        (1.0, -1.0 + f64::EPSILON),
        (0.1, 0.2),
        (1e16, 1.0),
        (f64::MIN_POSITIVE, f64::MIN_POSITIVE / 2.0),
    ];
    for (a, b) in cases {
        let (s, e) = Real::two_sum(a, b);
        assert_exact(
            s,
            a + b,
            &format!("two_sum({a},{b}) value is the rounded sum"),
        );
        // Reconstruct: (s, e) represents a + b exactly, so adding them back in the other
        // order recovers the larger operand.
        let reconstructed = s + e;
        assert_exact(
            reconstructed,
            a + b,
            &format!("two_sum({a},{b}) reconstructs"),
        );
        // And the error is genuinely the discarded part.
        if a.abs() >= b.abs() {
            assert_exact(e, b - (s - a), &format!("two_sum({a},{b}) error term"));
        }
    }
}

#[test]
fn two_sum_recovers_what_plain_addition_discards() {
    // 1 + eps/2 rounds back to 1, so the error term carries the whole second operand.
    let (s, e) = Real::two_sum(1.0f64, f64::EPSILON / 2.0);
    assert_exact(s, 1.0, "rounded away");
    assert_exact(e, f64::EPSILON / 2.0, "but recovered in the error term");
    // 1e16 + 1 likewise loses the 1.
    let (s, e) = Real::two_sum(1e16f64, 1.0);
    assert_exact(s, 1e16, "rounded away");
    assert_exact(e, 1.0, "recovered");
}

#[test]
fn two_sum_is_commutative_and_handles_zero() {
    for a in sweep(-3.0, 3.0, 13) {
        for b in sweep(-3.0, 3.0, 13) {
            let (s1, e1) = Real::two_sum(a, b);
            let (s2, e2) = Real::two_sum(b, a);
            assert_exact(s1, s2, "two_sum value is commutative");
            assert_exact(e1, e2, "two_sum error is commutative");
        }
        let (s, e) = Real::two_sum(a, 0.0);
        assert_exact(s, a, "a + 0");
        assert_exact(e, 0.0, "no error adding zero");
    }
}

// --- two_product ---------------------------------------------------------

#[test]
fn two_product_is_exact() {
    let cases = [
        (1.0f64 + f64::EPSILON, 1.0 + f64::EPSILON),
        (0.1, 0.1),
        (1e8 + 1.0, 1e8 + 1.0),
        (3.0, 1.0 / 3.0),
        (core::f64::consts::PI, core::f64::consts::PI),
    ];
    for (a, b) in cases {
        let (p, e) = Real::two_product(a, b);
        assert_exact(p, a * b, &format!("two_product({a},{b}) value"));
        // The error is the exact remainder, so it is representable and p + e is the best
        // double approximation of the true product.
        assert!(e.is_finite(), "error term must be finite");
        // Reconstructing in higher precision: (p, e) is a double-double for a*b.
        let (hi, lo) = (p, e);
        assert_exact(
            hi + lo,
            a * b,
            "double-double collapses to the rounded product",
        );
    }
}

#[test]
fn two_product_recovers_the_squaring_remainder() {
    // (1+eps)^2 = 1 + 2eps + eps^2; the eps^2 term is not representable in the product.
    let a = 1.0f64 + f64::EPSILON;
    let (p, e) = Real::two_product(a, a);
    assert_exact(p, a * a, "rounded square");
    assert!(e != 0.0, "the eps^2 remainder must be captured, got {e}");
    assert_close(e, f64::EPSILON * f64::EPSILON, 1e-12, "remainder is eps^2");
}

// --- the pay-off: accurate reductions ------------------------------------

/// Compensated (Kahan–Babuška–Neumaier) summation, written only through `two_sum`.
fn compensated_sum(xs: &[f64]) -> f64 {
    let mut total = 0.0f64;
    let mut correction = 0.0f64;
    for &x in xs {
        let (s, e) = Real::two_sum(total, x);
        total = s;
        correction += e;
    }
    total + correction
}

#[test]
fn compensated_summation_beats_naive_accumulation() {
    // The classic case: one large value followed by many small ones that each round away
    // individually but sum to something significant.
    let mut xs = vec![1e16f64];
    xs.extend(core::iter::repeat_n(1.0f64, 10_000));
    let exact = 1e16 + 10_000.0;

    let naive: f64 = xs.iter().sum();
    let compensated = compensated_sum(&xs);

    assert_exact(naive, 1e16, "naive summation loses every small term");
    assert_exact(
        compensated,
        exact,
        "compensated summation recovers all of them",
    );
}

#[test]
fn compensated_summation_handles_a_cancelling_sequence() {
    // Terms that cancel to a small residue: the naive sum's error can swamp the answer.
    let xs = [1.0f64, 1e17, -1e17, -1.0, 1e-9];
    let naive: f64 = xs.iter().sum();
    let compensated = compensated_sum(&xs);
    // The true sum is 1e-9. The naive accumulation loses the leading 1 when it meets 1e17
    // and never recovers it, ending up at -1 — wrong by a factor of 1e9.
    assert!(
        rel(naive, 1e-9) > 0.5,
        "the naive sum is expected to be badly wrong here, got {naive}"
    );
    // Compensation recovers about eight digits rather than all sixteen, because the *final*
    // `total + correction` is itself a cancelling addition: both operands are order 1 and the
    // result is 1e-9, so it gives up ~9 digits at the last step. Reaching full precision
    // needs a second compensated pass over the error terms. Recovering 8 digits where the
    // naive form recovers none is still the point.
    assert!(
        rel(compensated, 1e-9) < 1e-6,
        "compensated sum = {compensated}, want 1e-9"
    );
}

#[test]
fn an_accurate_dot_product_beats_the_naive_one() {
    // Both primitives together: exact products accumulated with exact sums.
    let a = [1e8f64 + 1.0, 1.0, -1e8, 1.0];
    let b = [1e8f64 + 1.0, 1.0, 1e8 + 2.0, 1.0];
    // (1e8+1)^2 + 1 - 1e8(1e8+2) + 1 = (1e16+2e8+1) + 1 - (1e16+2e8) + 1 = 3
    let exact = 3.0f64;

    let naive: f64 = a.iter().zip(&b).map(|(&x, &y)| x * y).sum();

    let mut total = 0.0f64;
    let mut correction = 0.0f64;
    for (&x, &y) in a.iter().zip(&b) {
        let (p, pe) = Real::two_product(x, y);
        let (s, se) = Real::two_sum(total, p);
        total = s;
        correction += se + pe;
    }
    let accurate = total + correction;

    assert_exact(accurate, exact, "compensated dot product is exact here");
    assert!(
        rel(naive, exact) > 1e-12,
        "the naive dot product is expected to drift, got {naive}"
    );
}
