//! **How accurate is it?** Reference-value and dense-sweep accuracy checks for the special
//! functions on the public [`Real`] surface.
//!
//! Every tolerance here is set from *measured* error, and annotated with what was measured.
//! A tolerance is a claim about this implementation, so loosening one without explaining why
//! hides exactly the regression it exists to catch.
//!
//! References come from two independent sources, never from euclid itself:
//!
//! - **Exact closed forms**, computed here in the test (factorials, harmonic sums, the
//!   binomial sum for an integer-parameter incomplete beta, power series for `I₀`/`I₁`).
//! - **60-digit `Decimal` values**, generated out-of-band and pasted in. Where the natural
//!   closed form would cancel catastrophically, the *tail* form was used instead — see
//!   [`gamma_lr_deep_tail_is_accurate`], which is the case that proves it matters.

// Reference literals are written out in full. Clippy would shorten them to the shortest
// round-tripping form, but the extra digits are deliberate provenance: they are what the
// 60-digit `Decimal` reference produced, so a reader can see the source value rather than a
// re-rounding of it. The shortened and full forms denote the same `f64` either way.
#![allow(clippy::excessive_precision)]

use crate::Real;

/// Relative error, falling back to absolute error when the reference is exactly zero.
fn rel(got: f64, want: f64) -> f64 {
    if want == 0.0 {
        got.abs()
    } else {
        ((got - want) / want).abs()
    }
}

#[track_caller]
fn assert_rel(got: f64, want: f64, tol: f64, what: &str) {
    let e = rel(got, want);
    assert!(
        e <= tol,
        "{what}: got {got:.17e}, want {want:.17e}, rel err {e:.3e} > tol {tol:.1e}"
    );
}

// --- erf / erfc ---------------------------------------------------------

/// Measured max relative error 8.6e-16 over this table (≈ 4 ulp).
const ERF_TOL: f64 = 5e-15;
/// Measured max relative error 2.5e-12, notably weaker than `erf` — `erfc` is formed by
/// complementation in the mid range, which costs digits the direct series keeps.
const ERFC_TOL: f64 = 1e-11;

#[test]
fn erf_matches_reference_values() {
    // Reference: CPython `math.erf`, an independent C implementation.
    let table = [
        (0.1, 1.12462916018284897e-01),
        (0.5, 5.20499877813046519e-01),
        (1.0, 8.42700792949714894e-01),
        (1.5, 9.66105146475310761e-01),
        (2.0, 9.95322265018952712e-01),
        (3.0, 9.99977909503001361e-01),
        (4.0, 9.99999984582742085e-01),
        (5.0, 9.99999999998462563e-01),
    ];
    for (x, want) in table {
        assert_rel(Real::erf(x), want, ERF_TOL, &format!("erf({x})"));
    }
}

#[test]
fn erfc_matches_reference_values_into_the_far_tail() {
    // Reference: CPython `math.erfc`. The last entries are far below where a naive
    // `1 - erf(x)` has any digits left at all.
    let table = [
        (0.1, 8.87537083981715047e-01),
        (0.5, 4.79500122186953481e-01),
        (1.0, 1.57299207050285134e-01),
        (1.5, 3.38948535246892738e-02),
        (2.0, 4.67773498104726623e-03),
        (3.0, 2.20904969985854412e-05),
        (4.0, 1.54172579002800200e-08),
        (5.0, 1.53745979442803494e-12),
        (6.0, 2.15197367124989128e-17),
        (8.0, 1.12242971729829264e-29),
    ];
    for (x, want) in table {
        assert_rel(Real::erfc(x), want, ERFC_TOL, &format!("erfc({x})"));
    }
}

// --- log-gamma family ---------------------------------------------------

/// Measured max relative error 1.5e-15 against exact `ln((n−1)!)` for n = 1..170.
const LN_GAMMA_TOL: f64 = 1e-14;
/// Measured max relative error 1.8e-12 against the exact harmonic-sum form. About three
/// orders of magnitude weaker than `ln_gamma`; the recurrence-shifted asymptotic series
/// used here does not reach full double precision.
const DIGAMMA_TOL: f64 = 1e-11;
/// Measured max relative error 7.0e-12 against the exact `π²/6 − Σ1/k²` form.
const TRIGAMMA_TOL: f64 = 5e-11;

#[test]
fn ln_gamma_matches_exact_log_factorials() {
    // ln Γ(n) = ln((n−1)!), accumulated exactly here rather than taken on faith.
    for n in 1..=170u32 {
        let mut want = 0.0f64;
        for k in 1..n {
            want += f64::from(k).ln();
        }
        assert_rel(
            Real::ln_gamma(f64::from(n)),
            want,
            LN_GAMMA_TOL,
            &format!("ln_gamma({n})"),
        );
    }
}

#[test]
fn digamma_matches_the_exact_harmonic_form() {
    // ψ(n) = −γ + Σ_{k<n} 1/k
    const EULER: f64 = 0.577_215_664_901_532_86;
    for n in 1..=200u32 {
        let mut h = 0.0f64;
        for k in 1..n {
            h += 1.0 / f64::from(k);
        }
        assert_rel(
            Real::digamma(f64::from(n)),
            -EULER + h,
            DIGAMMA_TOL,
            &format!("digamma({n})"),
        );
    }
}

#[test]
fn trigamma_matches_the_exact_zeta_form() {
    // ψ'(n) = π²/6 − Σ_{k<n} 1/k²
    for n in 1..=200u32 {
        let mut s = 0.0f64;
        for k in 1..n {
            s += 1.0 / (f64::from(k) * f64::from(k));
        }
        let want = core::f64::consts::PI * core::f64::consts::PI / 6.0 - s;
        assert_rel(
            Real::trigamma(f64::from(n)),
            want,
            TRIGAMMA_TOL,
            &format!("trigamma({n})"),
        );
    }
}

#[test]
fn digamma_and_trigamma_hit_their_closed_forms_at_half_integers() {
    const EULER: f64 = 0.577_215_664_901_532_86;
    let pi2 = core::f64::consts::PI * core::f64::consts::PI;
    // ψ(1/2) = −γ − 2 ln 2 ; ψ'(1/2) = π²/2
    assert_rel(
        Real::digamma(0.5),
        -EULER - 2.0 * core::f64::consts::LN_2,
        DIGAMMA_TOL,
        "digamma(1/2)",
    );
    assert_rel(
        Real::trigamma(0.5),
        pi2 / 2.0,
        TRIGAMMA_TOL,
        "trigamma(1/2)",
    );
    // ψ'(1) = π²/6
    assert_rel(Real::trigamma(1.0), pi2 / 6.0, TRIGAMMA_TOL, "trigamma(1)");
}

// --- incomplete gamma ---------------------------------------------------

/// Measured max relative error 1.1e-14 against the cancellation-free tail reference, over
/// shapes 1..20 and `x ∈ (0, 40]`. Worst case at large shape (a=20, x=18.5).
const GAMMA_LR_TOL: f64 = 5e-14;

/// Exact `P(a, x)` for integer `a`, cancellation-free.
///
/// Uses the *tail* identity `P(a,x) = e^{−x} Σ_{k≥a} x^k/k!` rather than the textbook
/// `1 − e^{−x} Σ_{k<a} x^k/k!`. Both are algebraically exact, but the second subtracts two
/// nearly-equal numbers, so it loses every digit once `P` is small: at `a=3, x=0.1` it is
/// already wrong by 4.6e-13, which is 400x euclid's own error there. Summing the tail never
/// subtracts, so it stays accurate into the deep underflow regime.
fn gamma_lr_tail(a: u32, x: f64) -> f64 {
    let mut term = 1.0f64;
    for k in 1..=a {
        term *= x / f64::from(k); // x^a / a!
    }
    let (mut sum, mut t, mut k) = (0.0f64, term, a);
    loop {
        sum += t;
        k += 1;
        t *= x / f64::from(k);
        if t < sum * 1e-19 || k > 5000 {
            break;
        }
    }
    (-x).exp() * sum
}

#[test]
fn gamma_lr_matches_the_exact_tail_reference() {
    for a in [1u32, 2, 3, 5, 10, 20] {
        for i in 1..400 {
            let x = f64::from(i) * 0.1;
            assert_rel(
                Real::gamma_lr(x, f64::from(a)),
                gamma_lr_tail(a, x),
                GAMMA_LR_TOL,
                &format!("gamma_lr(x={x}, a={a})"),
            );
        }
    }
}

#[test]
fn gamma_lr_deep_tail_is_accurate() {
    // `P(10, 0.1)` is ~2.5e-17 — far below where the textbook complement form has any digits
    // left. References from 60-digit `Decimal` arithmetic via the tail series.
    let table = [
        (0.1f64, 10.0f64, 2.51634780677031630e-17),
        (0.5, 10.0, 1.70967002934890327e-10),
        (1.5, 20.0, 3.28343419661364400e-16),
        (3.0, 20.0, 8.31442358819169941e-11),
        (5.0, 30.0, 2.81751761557792253e-14),
    ];
    for (x, a, want) in table {
        assert_rel(
            Real::gamma_lr(x, a),
            want,
            GAMMA_LR_TOL,
            &format!("gamma_lr(x={x}, a={a})"),
        );
    }
}

// --- incomplete beta ----------------------------------------------------

/// Measured max relative error 1.4e-14 against the exact binomial sum.
const BETA_REG_TOL: f64 = 1e-12;

/// Exact `I_x(a, b)` for integer `a`, `b`: `Σ_{j=a}^{a+b-1} C(a+b-1, j) x^j (1−x)^{n−j}`.
fn beta_reg_closed_form(x: f64, a: u32, b: u32) -> f64 {
    let n = a + b - 1;
    let mut total = 0.0f64;
    for j in a..=n {
        let mut c = 1.0f64;
        for i in 0..j {
            c = c * f64::from(n - i) / f64::from(i + 1);
        }
        total += c * x.powi(j as i32) * (1.0 - x).powi((n - j) as i32);
    }
    total
}

#[test]
fn beta_reg_matches_the_integer_parameter_closed_form() {
    for (a, b) in [(1u32, 1u32), (2, 3), (5, 2), (4, 7), (10, 10)] {
        for i in 1..100 {
            let x = f64::from(i) * 0.01;
            assert_rel(
                Real::beta_reg(x, f64::from(a), f64::from(b)),
                beta_reg_closed_form(x, a, b),
                BETA_REG_TOL,
                &format!("beta_reg(x={x}, a={a}, b={b})"),
            );
        }
    }
}

// --- modified Bessel ----------------------------------------------------

/// **The weakest functions on the real surface.** `bessel_i0e`/`bessel_i1e` rest on the
/// Abramowitz & Stegun polynomial approximations, which top out near 1e-7 *by construction* —
/// roughly `f32` accuracy from an `f64` function, and about eight orders of magnitude weaker
/// than the gamma/beta family beside them.
///
/// Measured relative error grows with `x` and then plateaus: 8.4e-8 over `x ∈ (0, 25]`,
/// 3.0e-7 by `x = 50`, and 4.73e-7 from `x ≈ 100` onward (flat out to `x = 1000`). The
/// tolerance sits just above that plateau, so *improving* the implementation fails this test
/// and prompts tightening it.
const BESSEL_TOL: f64 = 1e-6;

/// `I₀` by its defining power series `Σ (x²/4)^k / (k!)²`, which converges quickly and is
/// accurate to full double precision for the moderate `x` used here.
fn i0_series(x: f64) -> f64 {
    let t = (x / 2.0) * (x / 2.0);
    let mut s = 1.0f64;
    let mut term = 1.0f64;
    for k in 1..400u32 {
        term *= t / f64::from(k * k);
        s += term;
        if term < s * 1e-18 {
            break;
        }
    }
    s
}

/// `I₁` by its power series `(x/2) Σ (x²/4)^k / (k!(k+1)!)`.
fn i1_series(x: f64) -> f64 {
    let t = (x / 2.0) * (x / 2.0);
    let mut s = 1.0f64;
    let mut term = 1.0f64;
    for k in 1..400u32 {
        term *= t / f64::from(k * (k + 1));
        s += term;
        if term < s * 1e-18 {
            break;
        }
    }
    s * x / 2.0
}

#[test]
fn bessel_scaled_forms_hold_to_their_documented_accuracy() {
    for i in 0..250 {
        let x = f64::from(i) * 0.1;
        let scale = (-x.abs()).exp();
        assert_rel(
            Real::bessel_i0e(x),
            scale * i0_series(x),
            BESSEL_TOL,
            &format!("bessel_i0e({x})"),
        );
        if i > 0 {
            assert_rel(
                Real::bessel_i1e(x),
                scale * i1_series(x),
                BESSEL_TOL,
                &format!("bessel_i1e({x})"),
            );
        }
    }
}

#[test]
fn bessel_accuracy_is_no_better_than_single_precision() {
    // Pins the *known limitation* so it cannot be forgotten: somewhere in the supported range
    // the error genuinely exceeds what `f64` could deliver. If an implementation change makes
    // this fail, the A&S polynomials were replaced and `BESSEL_TOL` should be tightened and
    // the public docs updated.
    let worst = (0..250)
        .map(|i| {
            let x = f64::from(i) * 0.1;
            rel(Real::bessel_i0e(x), (-x.abs()).exp() * i0_series(x))
        })
        .fold(0.0f64, f64::max);
    assert!(
        worst > 1e-9,
        "bessel_i0e is now accurate to {worst:.2e}; tighten BESSEL_TOL and update the \
         precision note on Real::bessel_i0e"
    );
}

#[test]
fn bessel_is_accurate_across_the_branch_switch() {
    // The implementation changes formula at |x| = 3.75; a mismatched join there would show up
    // as a jump rather than as drift.
    for (x, i0e, i1e) in [
        (
            3.74f64,
            2.14772648612818584e-01f64,
            1.83141622320588088e-01f64,
        ),
        (3.76, 2.14142872424759739e-01, 1.82795563491229912e-01),
    ] {
        assert_rel(Real::bessel_i0e(x), i0e, BESSEL_TOL, "i0e at branch");
        assert_rel(Real::bessel_i1e(x), i1e, BESSEL_TOL, "i1e at branch");
    }
    // Continuity: the two sides agree to much better than the window between them.
    let jump = (Real::bessel_i0e(3.7499) - Real::bessel_i0e(3.7501)).abs();
    assert!(
        jump < 1e-5,
        "discontinuity at the branch switch: {jump:.3e}"
    );
}

#[test]
fn bessel_large_argument_stays_finite_and_decaying() {
    // Reference values at large x, where the unscaled I₀ would have overflowed long ago.
    for (x, i0e) in [
        (50.0f64, 5.65616266474541912e-02f64),
        (100.0, 3.99443792990966801e-02),
    ] {
        assert_rel(Real::bessel_i0e(x), i0e, BESSEL_TOL, &format!("i0e({x})"));
    }
    // I₀ᵉ decays like 1/√(2πx) — monotonically, with no overflow.
    let mut prev = f64::INFINITY;
    for i in 1..200 {
        let v = Real::bessel_i0e(f64::from(i) * 5.0);
        assert!(v.is_finite() && v > 0.0 && v < prev);
        prev = v;
    }
}
