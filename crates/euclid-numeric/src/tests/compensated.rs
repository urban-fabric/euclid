//! The compensated-arithmetic primitives and the cancellation-prone elementary forms added
//! alongside them.
//!
//! [`two_sum`](Real::two_sum) and [`two_product`](Real::two_product) are exactness claims, not
//! approximation claims, so most of these tests assert *bit-exact* identities rather than a
//! tolerance. The pay-off tests show compensated summation and an accurate dot product beating
//! naive accumulation by orders of magnitude — which is the whole reason the primitives exist.

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

// --- cos_m1 --------------------------------------------------------------

#[test]
fn cos_m1_keeps_digits_where_the_naive_form_loses_them() {
    // cos(1e-8) rounds to exactly 1.0, so `cos(x) - 1.0` is exactly 0 — total loss.
    let x = 1e-8f64;
    assert_exact(x.cos() - 1.0, 0.0, "the naive form collapses");
    // cos x - 1 ~ -x^2/2
    assert_close(Real::cos_m1(x), -0.5 * x * x, 1e-12, "cos_m1 at 1e-8");

    // -x^2/2 is the leading term only; the x^4/24 correction is 4e-6 at x = 0.1, so compare
    // against it only where it is genuinely negligible, and against the two-term series above
    // that.
    for i in 4..40 {
        let x = 10f64.powi(-i);
        assert_close(
            Real::cos_m1(x),
            -0.5 * x * x,
            1e-10,
            &format!("cos_m1({x})"),
        );
    }
    // Four terms over [0.01, 0.2]: the first dropped term is x^10/10!, which is 3e-14 at the
    // top of that range. (Three terms over [0.01, 0.3] is *not* enough — x^8/8! reaches
    // 1.1e-12 by x = 0.12, which is the series truncating, not `cos_m1` drifting.)
    for x in sweep(0.01, 0.2, 30) {
        let series = -0.5 * x * x + x.powi(4) / 24.0 - x.powi(6) / 720.0 + x.powi(8) / 40320.0;
        assert_close(
            Real::cos_m1(x),
            series,
            1e-12,
            &format!("cos_m1({x}) vs series"),
        );
    }
}

#[test]
fn cos_m1_agrees_with_the_naive_form_away_from_zero() {
    // Where cancellation is not a problem, the two must match.
    for x in sweep(0.5, 6.0, 40) {
        assert_close(
            Real::cos_m1(x),
            x.cos() - 1.0,
            1e-14,
            &format!("cos_m1({x})"),
        );
    }
}

#[test]
fn cos_m1_is_even_and_bounded() {
    for x in sweep(0.0, 10.0, 50) {
        assert_close(Real::cos_m1(-x), Real::cos_m1(x), 1e-15, "cos_m1 is even");
        let v = Real::cos_m1(x);
        assert!(
            (-2.0..=0.0).contains(&v),
            "cos_m1({x}) = {v} escaped [-2, 0]"
        );
    }
    assert_exact(Real::cos_m1(0.0f64), 0.0, "cos_m1(0) = 0");
}

#[test]
fn haversine_stays_accurate_at_short_distances() {
    // The motivating use. hav(θ) = (1 − cos θ)/2 = −cos_m1(θ)/2. Built the naive way, the
    // central term collapses to zero for nearby points and the distance comes out as 0.
    let earth_radius_m = 6_371_000.0f64;
    let haversine = |lat1: f64, lon1: f64, lat2: f64, lon2: f64| {
        let (dlat, dlon) = (lat2 - lat1, lon2 - lon1);
        let h = -Real::cos_m1(dlat) / 2.0 + lat1.cos() * lat2.cos() * (-Real::cos_m1(dlon) / 2.0);
        2.0 * earth_radius_m * h.sqrt().asin()
    };
    let naive = |lat1: f64, lon1: f64, lat2: f64, lon2: f64| {
        let (dlat, dlon) = (lat2 - lat1, lon2 - lon1);
        let h = (1.0 - dlat.cos()) / 2.0 + lat1.cos() * lat2.cos() * ((1.0 - dlon.cos()) / 2.0);
        2.0 * earth_radius_m * h.sqrt().asin()
    };

    // Two points about 1 metre apart: dlat ~ 1.57e-7 rad.
    let lat = 0.9f64;
    let dlat = 1.0 / earth_radius_m;
    let stable = haversine(lat, 0.0, lat + dlat, 0.0);
    assert_close(stable, 1.0, 1e-9, "1 m separation recovered");

    // At 1 mm the naive form has nothing left.
    let dlat_mm = 1e-3 / earth_radius_m;
    let stable_mm = haversine(lat, 0.0, lat + dlat_mm, 0.0);
    assert_close(stable_mm, 1e-3, 1e-6, "1 mm separation recovered");
    let naive_mm = naive(lat, 0.0, lat + dlat_mm, 0.0);
    assert!(
        rel(naive_mm, 1e-3) > 1e-4,
        "the naive haversine is expected to be inaccurate at 1 mm, got {naive_mm}"
    );
}

// --- sinc ----------------------------------------------------------------

#[test]
fn sinc_fills_the_removable_singularity() {
    assert_exact(Real::sinc(0.0f64), 1.0, "sinc(0) = 1");
    assert_exact(Real::sinc(-0.0f64), 1.0, "sinc(-0) = 1");
    // Continuous across it: sinc -> 1 as x -> 0. Only meaningful once x^2/6 is below the
    // tolerance, which it is from about 1e-8 down.
    for i in 8..40 {
        let x = 10f64.powi(-i);
        assert_close(Real::sinc(x), 1.0, 1e-15, &format!("sinc({x}) -> 1"));
    }
}

#[test]
fn sinc_agrees_with_sin_over_x_away_from_zero() {
    for x in sweep(0.1, 20.0, 80) {
        assert_close(Real::sinc(x), x.sin() / x, 1e-14, &format!("sinc({x})"));
    }
}

#[test]
fn sinc_is_even_and_vanishes_at_multiples_of_pi() {
    for x in sweep(0.5, 15.0, 40) {
        assert_close(Real::sinc(-x), Real::sinc(x), 1e-15, "sinc is even");
    }
    for k in 1..8 {
        let x = f64::from(k) * <f64 as Real>::pi();
        assert!(
            Real::sinc(x).abs() < 1e-15,
            "sinc({k}pi) = {} should vanish",
            Real::sinc(x)
        );
    }
}

#[test]
fn sinc_matches_its_series_near_the_crossover() {
    // The implementation switches between series and sin(x)/x; the join must be seamless.
    for x in sweep(0.009, 0.011, 41) {
        let series = 1.0 - x * x / 6.0 + x.powi(4) / 120.0 - x.powi(6) / 5040.0;
        assert_close(
            Real::sinc(x),
            series,
            1e-14,
            &format!("sinc({x}) at crossover"),
        );
    }
}

// --- hypot3 --------------------------------------------------------------

#[test]
fn hypot3_matches_the_direct_formula_in_range() {
    for x in sweep(-5.0, 5.0, 11) {
        for y in sweep(-5.0, 5.0, 11) {
            for z in sweep(-5.0, 5.0, 11) {
                let want = (x * x + y * y + z * z).sqrt();
                assert_close(Real::hypot3(x, y, z), want, 1e-14, "hypot3 in range");
            }
        }
    }
}

#[test]
fn hypot3_does_not_overflow_where_the_naive_form_does() {
    // Squaring 1e200 overflows f64 even though the norm is perfectly representable.
    let big = 1e200f64;
    assert!(
        (big * big + big * big + big * big).sqrt().is_infinite(),
        "the naive form is expected to overflow"
    );
    let h = Real::hypot3(big, big, big);
    assert!(h.is_finite(), "hypot3 overflowed: {h}");
    assert_close(
        h,
        big * 3.0f64.sqrt(),
        1e-14,
        "hypot3 of three equal components",
    );
}

#[test]
fn hypot3_does_not_underflow_where_the_naive_form_does() {
    let tiny = 1e-200f64;
    assert_exact(
        (tiny * tiny + tiny * tiny + tiny * tiny).sqrt(),
        0.0,
        "the naive form is expected to underflow to zero",
    );
    assert_close(
        Real::hypot3(tiny, tiny, tiny),
        tiny * 3.0f64.sqrt(),
        1e-14,
        "hypot3 of three tiny components",
    );
}

#[test]
fn hypot3_agrees_with_nested_hypot_and_handles_edges() {
    // It must equal hypot(hypot(x,y), z) — nesting the two-argument form.
    for x in sweep(0.5, 4.0, 8) {
        for y in sweep(0.5, 4.0, 8) {
            for z in sweep(0.5, 4.0, 8) {
                assert_close(
                    Real::hypot3(x, y, z),
                    Real::hypot(Real::hypot(x, y), z),
                    1e-14,
                    "hypot3 == nested hypot",
                );
            }
        }
    }
    assert_exact(Real::hypot3(0.0f64, 0.0, 0.0), 0.0, "all zero");
    assert_exact(Real::hypot3(3.0f64, 4.0, 0.0), 5.0, "degenerates to hypot");
    assert!(Real::hypot3(f64::INFINITY, 1.0, 1.0).is_infinite());
    assert!(Real::hypot3(f64::NAN, 1.0, 1.0).is_nan());
}

// --- ln_rising_factorial -------------------------------------------------

#[test]
fn ln_rising_factorial_matches_the_product_for_integer_n() {
    // (a)_n = a(a+1)...(a+n-1), so the log is the sum of logs.
    for a in [0.5f64, 1.0, 2.5, 10.0, 1e3] {
        for n in 1..12u32 {
            let mut want = 0.0f64;
            for k in 0..n {
                want += (a + f64::from(k)).ln();
            }
            assert_close(
                Real::ln_rising_factorial(a, f64::from(n)),
                want,
                1e-13,
                &format!("ln (a={a})_{n}"),
            );
        }
    }
}

#[test]
fn ln_rising_factorial_matches_the_gamma_difference_where_that_is_accurate() {
    // Small `a`, where the two ln_gamma terms are not nearly equal.
    for a in sweep(0.25, 8.0, 32) {
        for n in [0.5f64, 1.5, 3.25, 7.75] {
            assert_close(
                Real::ln_rising_factorial(a, n),
                Real::ln_gamma(a + n) - Real::ln_gamma(a),
                1e-12,
                &format!("ln ({a})_{n}"),
            );
        }
    }
}

#[test]
fn ln_rising_factorial_survives_a_huge_base_where_the_difference_cancels() {
    // The motivating case: a enormous, n small. ln_gamma(a+n) and ln_gamma(a) agree to many
    // digits, so their difference throws away precisely the answer.
    for a in [1e8f64, 1e12, 1e15] {
        for n in [1.0f64, 2.0, 0.5] {
            let got = Real::ln_rising_factorial(a, n);
            // For n << a the rising factorial is ~a^n, so the log is ~n ln a.
            let approx = n * a.ln();
            assert!(
                rel(got, approx) < 1e-7,
                "ln ({a})_{n} = {got}, expected about {approx}"
            );
            assert!(got.is_finite());
        }
    }
    // Integer n is exact via the product form, so compare against it directly.
    let a = 1e12f64;
    assert_close(
        Real::ln_rising_factorial(a, 2.0),
        a.ln() + (a + 1.0).ln(),
        1e-14,
        "ln (1e12)_2",
    );
}

#[test]
fn ln_rising_factorial_is_zero_at_n_zero_and_additive_in_n() {
    for a in sweep(0.5, 20.0, 20) {
        assert_exact(Real::ln_rising_factorial(a, 0.0), 0.0, "(a)_0 = 1");
        // (a)_{m+n} = (a)_m (a+m)_n  =>  logs add.
        for (m, n) in [(2.0f64, 3.0f64), (1.5, 2.5), (4.0, 1.0)] {
            let lhs = Real::ln_rising_factorial(a, m + n);
            let rhs = Real::ln_rising_factorial(a, m) + Real::ln_rising_factorial(a + m, n);
            assert_close(
                lhs,
                rhs,
                1e-11,
                &format!("additivity at a={a}, m={m}, n={n}"),
            );
        }
    }
}

// --- the f32 carrier -----------------------------------------------------

#[test]
fn the_new_surface_is_available_and_consistent_on_f32() {
    assert_eq!(Real::sinc(0.0f32), 1.0);
    assert_eq!(Real::cos_m1(0.0f32), 0.0);
    assert_eq!(Real::hypot3(3.0f32, 4.0, 0.0), 5.0);

    // two_sum is exact in f32 too.
    let (s, e) = Real::two_sum(1.0f32, f32::EPSILON / 2.0);
    assert_eq!(s, 1.0);
    assert_eq!(e, f32::EPSILON / 2.0);

    // cos_m1 keeps digits where the naive form collapses much sooner than in f64.
    let x = 1e-4f32;
    assert_eq!(x.cos() - 1.0, 0.0, "naive form collapses in f32");
    assert!((f64::from(Real::cos_m1(x)) - (-0.5 * f64::from(x) * f64::from(x))).abs() < 1e-12);

    // hypot3 avoids the much earlier f32 overflow.
    let big = 1e30f32;
    assert!((big * big).is_infinite(), "f32 squares overflow here");
    assert!(Real::hypot3(big, big, big).is_finite());

    // And the rising factorial agrees with the product form.
    let a = 10.0f32;
    let want = a.ln() + (a + 1.0).ln() + (a + 2.0).ln();
    assert!((Real::ln_rising_factorial(a, 3.0) - want).abs() < 1e-5);
}
