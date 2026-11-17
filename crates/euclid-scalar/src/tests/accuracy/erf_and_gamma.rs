//! Measured accuracy of the error-function and log-gamma families.
//!
//! Part of the [`accuracy`](super) suite.

#![allow(clippy::excessive_precision)]

// Reference literals are written out in full. Clippy would shorten them to the shortest
// round-tripping form, but the extra digits are deliberate provenance: they are what the
// 60-digit `Decimal` reference produced, so a reader can see the source value rather than a
// re-rounding of it. The shortened and full forms denote the same `f64` either way.

use crate::Real;
use euclid_utils::{EULER_GAMMA as EULER, assert_close};

// --- erf / erfc ---------------------------------------------------------

/// Measured max relative error 8.6e-16 over this table (≈ 4 ulp).
const ERF_TOL: f64 = 5e-15;
/// Measured max relative error 3.7e-15 over this table (≈ 17 ulp), the same bar as `erf`.
///
/// This was `1e-11`, annotated "notably weaker than `erf`, formed by complementation in the mid
/// range". That was true but it was a *defect*, not a property: the continued-fraction handover
/// sat at `x = 4` when the fraction is exact from `x = 1.8`, so `1 − erf` was being used across
/// a band where it had already lost nine digits — `1.2e-9` at `x = 3.9`. Moving the join to
/// `2.0` (see the kernel's `ERFCX_CF_MIN`) removed it.
///
/// The loose tolerance is why it survived: the table below originally skipped `2.5`, `3.5` and
/// `3.9`, and at the one mid-range point it did sample, `x = 3`, the error was `2.4e-12` —
/// comfortably inside `1e-11`. Both the gap and the slack are closed here.
const ERFC_TOL: f64 = 5e-15;

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
        assert_close(Real::erf(x), want, ERF_TOL, &format!("erf({x})"));
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
        // 2.5 / 3.5 / 3.9 bracket the old handover at 4.0; 3.9 is where the former
        // `1 - erf` branch was at its worst.
        (2.5, 4.06952017444958919e-04),
        (3.0, 2.20904969985854412e-05),
        (3.5, 7.43098372341412777e-07),
        (3.9, 3.47922485972317454e-08),
        (4.0, 1.54172579002800200e-08),
        (5.0, 1.53745979442803494e-12),
        (6.0, 2.15197367124989128e-17),
        (8.0, 1.12242971729829264e-29),
    ];
    for (x, want) in table {
        assert_close(Real::erfc(x), want, ERFC_TOL, &format!("erfc({x})"));
    }
}

#[test]
fn erfcx_matches_reference_values_across_its_handover() {
    // `erfcx(x) = e^{x²}erfc(x)`, sampled either side of `ERFCX_CF_MIN` so a mis-sited
    // handover shows up here and not only in `erfc`. The two share the branch, and `erfcx` is
    // the one the normal family's log-CDF actually reaches for.
    //
    // Reference: a 40-digit `Decimal` evaluation, cross-checked two independent ways — the
    // Maclaurin series for `erf` scaled by `e^{x²}`, and the continued fraction — which agree
    // exactly in their overlap. Not CPython, which has no `erfcx`.
    let table = [
        (0.5, 6.15690344192925898e-01),
        (1.0, 4.27583576155806999e-01),
        (1.5, 3.21585416454317485e-01),
        (1.9, 2.66509373661672655e-01),
        (2.0, 2.55395676310505748e-01),
        (2.1, 2.45119123345172357e-01),
        (2.5, 2.10806364061143586e-01),
        (3.0, 1.79001151181389956e-01),
        (4.0, 1.36999457625061383e-01),
        (10.0, 5.61409927438225875e-02),
        (100.0, 5.64161378298943302e-03),
        // Far enough out that `erfcx ~ 1/(x√π)` to the last bit.
        (1e8, 5.64189583547756227e-09),
    ];
    for (x, want) in table {
        assert_close(Real::erfcx(x), want, ERFC_TOL, &format!("erfcx({x})"));
    }
}

// --- log-gamma family ---------------------------------------------------

/// Measured max relative error 1.5e-15 against exact `ln((n−1)!)` for n = 1..170.
const LN_GAMMA_TOL: f64 = 1e-14;
/// Measured max relative error 8.8e-16 against the exact harmonic-sum form (≈ 4 ulp).
///
/// This was `1e-11`, annotated "about three orders of magnitude weaker than `ln_gamma`; the
/// recurrence-shifted asymptotic series does not reach full double precision". The second half
/// of that was wrong — the series was fine, it was simply being evaluated too close in. The
/// recurrence walked the argument only to `x = 10`, where the first omitted term of the
/// expansion is `1/(132·10^10) = 7.6e-13`; walking to `20` instead puts it at `7.4e-16`. See
/// the kernel's `PSI_ASYMPTOTIC_MIN`.
const DIGAMMA_TOL: f64 = 5e-15;
/// Measured max relative error 8.5e-15 against the exact `π²/6 − Σ1/k²` form; was `7.0e-12`
/// before the shift described on [`DIGAMMA_TOL`], which both functions share.
const TRIGAMMA_TOL: f64 = 5e-14;

#[test]
fn ln_gamma_matches_exact_log_factorials() {
    // ln Γ(n) = ln((n−1)!), accumulated exactly here rather than taken on faith.
    for n in 1..=170u32 {
        let mut want = 0.0f64;
        for k in 1..n {
            want += f64::from(k).ln();
        }
        assert_close(
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
    for n in 1..=200u32 {
        let mut h = 0.0f64;
        for k in 1..n {
            h += 1.0 / f64::from(k);
        }
        assert_close(
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
        assert_close(
            Real::trigamma(f64::from(n)),
            want,
            TRIGAMMA_TOL,
            &format!("trigamma({n})"),
        );
    }
}

#[test]
fn digamma_and_trigamma_hit_their_closed_forms_at_half_integers() {
    let pi2 = core::f64::consts::PI * core::f64::consts::PI;
    // ψ(1/2) = −γ − 2 ln 2 ; ψ'(1/2) = π²/2
    assert_close(
        Real::digamma(0.5),
        -EULER - 2.0 * core::f64::consts::LN_2,
        DIGAMMA_TOL,
        "digamma(1/2)",
    );
    assert_close(
        Real::trigamma(0.5),
        pi2 / 2.0,
        TRIGAMMA_TOL,
        "trigamma(1/2)",
    );
    // ψ'(1) = π²/6
    assert_close(Real::trigamma(1.0), pi2 / 6.0, TRIGAMMA_TOL, "trigamma(1)");
}
