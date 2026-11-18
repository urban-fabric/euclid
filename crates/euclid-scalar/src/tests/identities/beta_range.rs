//! `B(a, b)` across its *whole* parameter range, against the shapes where it is elementary.
//!
//! The companion to [`super::gamma_range`], and it exists for the same reason: the existing
//! identity tests exercise `ln_beta` over ordinary parameters, and the defect lived past them.
//!
//! `ln B(a, b)` was computed as `lnΓ(a) + lnΓ(b) − lnΓ(a + b)`, which cancels whenever
//! `a + b ≫ min(a, b)` — the result is `O(ln)` while the terms are `O(a ln a)`. Measured
//! against the exact `ln B(a, 1) = −ln a`:
//!
//! ```text
//! a = 1e2   1.7e-14 relative       a = 1e15   4e-2 relative
//! a = 1e8   1.4e-9  relative       a = 1e16   ln B gave 0, and B gave 1
//! ```
//!
//! Past `a = 1/ε` the sum `a + b` rounds back to `a`, so the two log-gammas are bit-identical
//! and the answer cancels to nothing: `B(1e300, 1)` returned `1` where the truth is `1e-300`.
//! Large shape parameters are not exotic here — they are what a Student-t or F distribution
//! with many degrees of freedom asks for.
//!
//! # The elementary shapes
//!
//! `B` collapses at small integer second argument, which gives exact references needing no
//! reference table and no quadrature:
//!
//! ```text
//! B(a, 1) = 1/a        B(a, 2) = 1/(a(a+1))        B(a, 3) = 2/(a(a+1)(a+2))
//! ```
//!
//! and `B(½, ½) = π`. In the log domain the first of these is `−ln a` exactly, which is what
//! makes it usable at `a = 1e300` where `B` itself has long underflowed.

use crate::Real;
use euclid_utils::assert_close;

#[test]
fn the_log_beta_matches_its_exact_closed_forms_over_the_whole_range() {
    // ln B(a, 1) = -ln a, for every representable a. This is the test the old form failed.
    for &a in &[
        1e-300f64,
        1e-20,
        0.5,
        1.0,
        2.0,
        1e2,
        1e8,
        1e15,
        1e16,
        1e17,
        1e50,
        1e200,
        1e300,
        f64::MAX,
    ] {
        let expected = -a.ln();
        for (label, got) in [
            ("ln_beta(a, 1)", Real::ln_beta(a, 1.0)),
            ("ln_beta(1, a)", Real::ln_beta(1.0, a)),
        ] {
            // Mixed absolute/relative, scaled by the magnitude of the answer: at a = 1 the
            // expected value is exactly zero, and the bound is otherwise `ln_gamma`'s own —
            // the two Lanczos evaluations each carry an ulp, so their difference carries a
            // handful.
            let bound = 16.0 * f64::EPSILON * expected.abs().max(1.0);
            assert!(
                (got - expected).abs() <= bound,
                "{label} at a = {a:e}: got {got:e}, expected {expected:e}"
            );
        }
    }
    // ln B(a, 2) = -ln(a(a+1)) and ln B(a, 3) = ln 2 - ln(a(a+1)(a+2)), formed in the log
    // domain so the references themselves survive to the top of the range.
    for &a in &[1e-10f64, 1.0, 1e2, 1e8, 1e15, 1e100, 1e300] {
        let two = -(a.ln() + (a + 1.0).ln());
        assert!(
            ((Real::ln_beta(a, 2.0) - two) / two).abs() <= 8.0 * f64::EPSILON,
            "ln_beta({a:e}, 2) = {:e}, expected {two:e}",
            Real::ln_beta(a, 2.0)
        );
        let three = core::f64::consts::LN_2 - (a.ln() + (a + 1.0).ln() + (a + 2.0).ln());
        assert!(
            ((Real::ln_beta(a, 3.0) - three) / three).abs() <= 8.0 * f64::EPSILON,
            "ln_beta({a:e}, 3) = {:e}, expected {three:e}",
            Real::ln_beta(a, 3.0)
        );
    }
    // B(1/2, 1/2) = pi, the one case where the function itself is a famous constant.
    assert_close(
        Real::beta(0.5, 0.5),
        core::f64::consts::PI,
        1e-15,
        "B(1/2, 1/2) = pi",
    );
}

#[test]
fn the_beta_function_itself_is_right_where_it_is_representable() {
    // B(a, 1) = 1/a. Checked on the unlogged function too, because `beta` returned exactly
    // `1.0` for every large `a` — a value that looks entirely plausible.
    for &a in &[1e-300f64, 1e-8, 1.0, 1e2, 1e8, 1e15, 1e100, 1e300] {
        let expected = 1.0 / a;
        if expected == 0.0 || !expected.is_finite() {
            continue;
        }
        let got = Real::beta(a, 1.0);
        // `beta` is `exp(ln_beta)`, so it necessarily carries `eps·|ln B|` of relative error —
        // 7.7e-14 at a = 1e-300, where `ln B` is 690. That gap is not a defect; it is the
        // reason `ln_beta` is on the surface at all, and the bound says so.
        let bound = 4.0 * f64::EPSILON * Real::ln_beta(a, 1.0).abs().max(1.0);
        assert!(
            ((got - expected) / expected).abs() <= bound,
            "beta({a:e}, 1) = {got:e}, expected {expected:e}"
        );
    }
}

#[test]
fn log_beta_is_symmetric_and_satisfies_its_recurrence() {
    // B(a, b) = B(b, a) exactly — the implementation orders its arguments internally, so this
    // is a real constraint on that choice rather than a tautology.
    for &a in &[1e-8f64, 0.5, 1.0, 7.0, 1e5, 1e200] {
        for &b in &[1e-8f64, 0.5, 1.0, 7.0, 1e5, 1e200] {
            assert_eq!(
                Real::ln_beta(a, b),
                Real::ln_beta(b, a),
                "ln_beta must be symmetric at ({a:e}, {b:e})"
            );
        }
    }
    // B(a, b+1) = B(a, b)·b/(a+b), the recurrence in the second argument. In the log domain
    // this is a difference of two O(ln) quantities, so it stays sharp at every magnitude.
    for &a in &[0.5f64, 1.0, 3.0, 1e4, 1e100] {
        for &b in &[0.5f64, 1.0, 3.0, 1e4] {
            // `ln b − ln(a+b)` is `−ln1p(a/b)`, and spelling it that way matters: at
            // (a, b) = (0.5, 1e4) the two logs are both 9.2 and differ by 5e-5, so the
            // literal form loses five digits and the *reference* becomes the inaccurate side.
            let expected = Real::ln_beta(a, b) - (a / b).ln_1p();
            let got = Real::ln_beta(a, b + 1.0);
            assert!(
                ((got - expected) / expected).abs() < 1e-14,
                "ln_beta recurrence at ({a:e}, {b:e}): {got:e} vs {expected:e}"
            );
        }
    }
}

#[test]
fn the_incomplete_beta_inherits_the_repair() {
    // `beta_reg` divides by `B(a, b)` through the same `ln_beta`, so the collapse reached the
    // CDF too. `I_x(a, 1) = x^a` is exact, which pins it at parameters where `B` itself is far
    // outside the format.
    for &a in &[1.0f64, 1e2, 1e8, 1e15, 1e17, 1e100] {
        for &x in &[0.25f64, 0.5, 0.9] {
            // x^a underflows for large a, and zero is then the correct answer.
            let expected = x.powf(a);
            let got = Real::beta_reg(x, a, 1.0);
            assert!(
                (got - expected).abs() <= 1e-13,
                "beta_reg({x}, {a:e}, 1) = {got:e}, expected x^a = {expected:e}"
            );
        }
        // And at the ends, regardless of parameters.
        assert_eq!(Real::beta_reg(0.0, a, 1.0), 0.0);
        assert_eq!(Real::beta_reg(1.0, a, 1.0), 1.0);
    }
    // I_x(1, b) = 1 - (1-x)^b, the mirror statement.
    for &b in &[1.0f64, 1e2, 1e8, 1e15] {
        for &x in &[1e-12f64, 0.1, 0.5] {
            // `1 − (1−x)^b` as `−expm1(b·ln1p(−x))`. Written literally the reference is the
            // inaccurate side: at x = 1e-12, b = 1e8 the base is one ulp from 1 and `powf`
            // cannot resolve it, giving 9.9993e-5 where the truth is 9.9995e-5.
            let expected = -(b * (-x).ln_1p()).exp_m1();
            let got = Real::beta_reg(x, 1.0, b);
            assert!(
                ((got - expected) / expected).abs() <= 1e-12,
                "beta_reg({x:e}, 1, {b:e}) = {got:e}, expected {expected:e}"
            );
        }
    }
}

#[test]
fn the_f32_carrier_has_the_same_repair() {
    // f32's 1/eps is 8.4e6, so the collapse began at a far more ordinary parameter there —
    // `ln B(1e8, 1)` had no digits left at all.
    for &a in &[1.0f32, 1e2, 1e6, 1e7, 1e8, 1e15, 1e30, f32::MAX] {
        let expected = -a.ln();
        let got = Real::ln_beta(a, 1.0);
        let bound = 16.0 * f32::EPSILON * expected.abs().max(1.0);
        assert!(
            (got - expected).abs() <= bound,
            "f32 ln_beta({a:e}, 1) = {got:e}, expected {expected:e}"
        );
        assert_eq!(
            Real::ln_beta(a, 1.0f32),
            Real::ln_beta(1.0f32, a),
            "f32 ln_beta must be symmetric"
        );
    }
}

#[test]
fn the_incomplete_beta_is_exact_at_the_symmetric_median() {
    // `I_{1/2}(a, a) = 1/2` for every `a`, by symmetry alone — the two tails of a symmetric
    // density are equal. No reference value, no truncation, nothing to transcribe, and it holds
    // at every magnitude, which makes it the sharpest available probe of the large-parameter
    // regime.
    //
    // It is also the case that was broken. `beta_reg`'s continued fraction converges slowest at
    // exactly `x = a/(a+b)`, and its iteration cap used to be 300 — enough only to about
    // `a = 1e4`. Past that it returned a truncated partial sum with no indication, giving:
    //
    // | `a` | 1e5 | 1e6 | 1e7 | 1e8 |
    // |-----|-----|-----|-----|-----|
    // | old error | `8.4e-11` | `4.1e-7` | `9.5e-3` | `0.187` |
    //
    // At `a = 1e8` the first digit was wrong. The suite did not catch it because its largest
    // *equal* pair was `(500, 500)` — a factor of 20 short of where the defect began. Hence
    // this test, which sweeps until the exponent range runs out.
    for &a in &[
        0.5f64, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e10, 1e12, 1e15, 1e30, 1e100,
    ] {
        let got = Real::beta_reg(0.5f64, a, a);
        // Below the Edgeworth handover the continued fraction's `exp` prefactor cancels three
        // terms of magnitude `1.386(a+b)`, so the floor there is `≈ 2.8aε` — 4.7e-10 at a = 1e6,
        // which is the worst point of the whole range. Above the handover the skewness term
        // vanishes identically and `Φ(0)` is exactly 1/2.
        let floor = 4.0 * a * f64::EPSILON + 1e-15;
        assert!(
            (got - 0.5).abs() <= floor,
            "I_1/2({a:e}, {a:e}) = {got:.17e}, must be 1/2 (bound {floor:e})"
        );
    }
    // Above the handover it is not merely close but bit-exact, which is worth pinning
    // separately: it is what tells us the symmetric skewness really is identically zero rather
    // than merely small.
    for &a in &[1e7f64, 1e8, 1e10, 1e12, 1e15, 1e100] {
        assert_eq!(
            Real::beta_reg(0.5f64, a, a),
            0.5,
            "I_1/2({a:e}, {a:e}) must be exactly 1/2 above the Edgeworth handover"
        );
    }
}

#[test]
fn the_incomplete_beta_complements_itself_at_every_magnitude() {
    // `I_x(a,b) + I_{1−x}(b,a) = 1`. Both sides run through the same code with the parameters
    // swapped, so this catches a branch that is accurate on one side of the median and not the
    // other — including the Edgeworth handover, which must agree with the fraction across it.
    //
    // The pairs deliberately mix symmetric large (where the fraction is slowest), asymmetric
    // large (where the normal approximation is *invalid* however large `a + b` is, because the
    // skewness goes like `2/√min(a,b)`), and tiny.
    for &(a, b) in &[
        (0.01f64, 0.01f64),
        (0.5, 0.5),
        (2.0, 5.0),
        (500.0, 500.0),
        (1e4, 1e4),
        (1e6, 1e6),
        (1e7, 1e7),
        (1e8, 1e8),
        (1e10, 1e10),
        (1e4, 1.0),
        (1.0, 1e4),
        (1e8, 1e6),
        (1e6, 1e8),
        (1e-8, 1e8),
        (1e9, 1e3),
    ] {
        for i in 1..20 {
            let x = f64::from(i) * 0.05;
            let sum = Real::beta_reg(x, a, b) + Real::beta_reg(1.0 - x, b, a);
            // Measured worst across these pairs is 9.5e-10, at (1e6, 1e6) and x = 1/2 — the
            // prefactor floor described above, doubled because both terms carry it.
            assert!(
                (sum - 1.0).abs() < 2e-9,
                "I_{x}({a:e},{b:e}) + I_{{1-x}}({b:e},{a:e}) = {sum:.17e}, must be 1"
            );
        }
    }
}
