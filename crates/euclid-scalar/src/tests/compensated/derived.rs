//! The derived stable forms: hypot3 and the log rising factorial.
//!
//! Part of the [`compensated`](super) suite.

use crate::Real;
use euclid_utils::{assert_close, assert_exact, rel, sweep};

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
