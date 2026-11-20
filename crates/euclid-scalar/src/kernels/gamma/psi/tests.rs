//! Unit tests for `ψ` and `ψ₁`.
//!
//! Beside [`super`] rather than with `Γ`'s own tests, because what they reach is this module's:
//! the [`PsiReduction`] table, the Taylor series about `ψ`'s real zero and the recurrence shift.
//! The window edge is pinned explicitly — either side of it a different series runs, and both
//! are accurate there, so the boundary is only visible as a decision.

use super::*;

fn close(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
}

#[test]
fn digamma_known_values() {
    // Tightened to 1e-11 after adding the 1/(240·x⁸) asymptotic term: the prior
    // truncation reached only ~1e-10 (vs the ~15-digit intent of the coefficients).
    close(digamma(1.0), -crate::constants::f64::EULER_GAMMA, 1e-11);
    close(
        digamma(2.0),
        1.0 - crate::constants::f64::EULER_GAMMA,
        1e-11,
    );
    // ψ(3) = 3/2 − γ.
    close(
        digamma(3.0),
        1.5 - crate::constants::f64::EULER_GAMMA,
        1e-11,
    );
}

#[test]
fn trigamma_known_values() {
    let pi2 = core::f64::consts::PI * core::f64::consts::PI;
    close(trigamma(1.0), pi2 / 6.0, 1e-10);
    close(trigamma(2.0), pi2 / 6.0 - 1.0, 1e-10);
    close(trigamma(0.5), pi2 / 2.0, 1e-10);
    close(trigamma(3.3) - trigamma(4.3), 1.0 / (3.3 * 3.3), 1e-10);
}

#[test]
fn positive_trigamma_matches_the_series_on_both_sides_of_the_shift() {
    // At integer and half-integer arguments the series is a finite sum
    // subtracted from π²/6 or π²/2. References use 70-digit Decimal sums.
    for (x, expected) in [
        (0.5, 4.934_802_200_544_679),
        (1.0, 1.644_934_066_848_226_4),
        (19.5, 0.052_619_441_213_655_93),
        (20.0, 0.051_270_822_935_203_12),
        (20.5, 0.049_989_592_429_960_99),
        (64.0, 0.015_747_706_064_338_93),
    ] {
        let actual = trigamma_positive(x);
        assert!(
            (actual - expected).abs() < 2e-14 * expected,
            "{x}: {actual} vs {expected}"
        );
    }
}

/// The Taylor series about `ψ`'s zero, at the edge of the window it is used on.
///
/// # Why the edge, and why against pasted digits
///
/// Five of the twenty-one coefficients' signs survived the suite. They are the deep ones, and
/// their contribution falls off as `d^k`: at the window's half-width of `2e-1` flipping `a₁₂`
/// moves the result by `2.1e6` ulp and flipping `a₁₈` by `14`, while anywhere nearer the zero
/// they vanish — at `d = 1e-1` the `a₁₄` flip is already down to `4` ulp and `a₁₆` to nothing.
/// So the edge is the only argument that pins the tail of the series.
///
/// The references are `ψ` at 40 digits from Stirling plus a sixty-step upward recurrence,
/// computed outside the crate; the series itself is accurate to **0 to 2 ulp** against them,
/// which is what lets the bound be four. `a₂₀` is *not* reachable: its flip moves the value by
/// one ulp at the edge and zero everywhere else, so it sits at the format's resolution and no
/// assertion on this function can separate it.
#[test]
fn the_series_about_the_psi_zero_is_pinned_at_its_window_edge() {
    for (d, want) in [
        (0.2_f64, 0.177_660_044_996_198_25_f64),
        (-0.2, -0.213_615_129_720_246_74),
        (0.1, 0.092_582_708_956_254_55),
        (-0.05, -0.049_523_892_169_472),
        (0.15, 0.135_985_547_380_247_57),
    ] {
        let got = psi_near_zero(d);
        let ulp = want.abs() * f64::EPSILON;
        assert!(
            (got - want).abs() <= 4.0 * ulp,
            "psi_near_zero({d}) = {got:e}, want {want:e} ({:.1} ulp)",
            (got - want).abs() / ulp
        );
    }
    // The constant term is the offset from the true zero, not zero itself: `ψ` vanishes
    // `9.55e-17` above the representable `PSI_ZERO`, and the high-precision reference agrees
    // with `a₀` to every digit it carries.
    assert!((psi_near_zero(0.0) - -9.241_265_521_729_427e-17).abs() <= 1e-31);
    // Odd about the zero to leading order, since `a₁` dominates across the window.
    assert!(psi_near_zero(0.2) > 0.0 && psi_near_zero(-0.2) < 0.0);
}

/// The digamma reductions, which divide the line by how `ψ` is *reached* rather than by value.
///
/// `Undefined` and `Pole` both answer `NaN`, so only the decision separates them — and the
/// distinction is real: `−∞` has no answer because `ψ` oscillates there, while a negative integer
/// has none because `ψ` takes opposite signs on the two sides of it.
#[test]
fn the_digamma_reductions_name_every_region() {
    use super::PsiReduction::{Asymptotic, NearZero, Pole, Reflected, Undefined};

    assert_eq!(PsiReduction::for_digamma(f64::NAN), Undefined);
    assert_eq!(PsiReduction::for_digamma(f64::NEG_INFINITY), Undefined);
    // `+∞` is not undefined: the series gives `ψ(+∞) = +∞`.
    assert_eq!(PsiReduction::for_digamma(f64::INFINITY), Asymptotic);
    assert_eq!(digamma(f64::INFINITY), f64::INFINITY);

    // Negative integers are poles; everything else left of zero reflects.
    for x in [-1.0, -2.0, -39.0, -1e15] {
        assert_eq!(PsiReduction::for_digamma(x), Pole, "at {x:e}");
        assert!(digamma(x).is_nan(), "at {x:e}");
    }
    for x in [
        -0.5,
        -1.5,
        -0.001,
        (-1.0_f64).next_up(),
        (-1.0_f64).next_down(),
    ] {
        assert_eq!(PsiReduction::for_digamma(x), Reflected, "at {x:e}");
        assert!(digamma(x).is_finite(), "at {x:e}");
    }

    // The Taylor window about `ψ`'s real zero, at and either side of both edges.
    let (lo, hi) = (PSI_ZERO - PSI_ZERO_WINDOW, PSI_ZERO + PSI_ZERO_WINDOW);
    assert_eq!(PsiReduction::for_digamma(PSI_ZERO), NearZero);
    assert_eq!(PsiReduction::for_digamma(lo), NearZero);
    assert_eq!(PsiReduction::for_digamma(hi), NearZero);
    assert_eq!(PsiReduction::for_digamma(lo.next_down()), Asymptotic);
    assert_eq!(PsiReduction::for_digamma(hi.next_up()), Asymptotic);
    // `ψ` really does vanish there, which is what the window exists to resolve.
    assert!(digamma(PSI_ZERO).abs() < 1e-15, "{}", digamma(PSI_ZERO));

    // Zero itself, and the rest of the positive axis.
    for x in [0.0, 1.0, 0.5, 19.0, f64::from(PSI_ASYMPTOTIC_MIN), 1e300] {
        assert_eq!(PsiReduction::for_digamma(x), Asymptotic, "at {x:e}");
    }
    // The two arms of the walk-up agree across the threshold: both are the same series after the
    // recurrence, so the only thing the comparison changes is how many terms are shed.
    let threshold = f64::from(PSI_ASYMPTOTIC_MIN);
    close(digamma(threshold.next_down()), digamma(threshold), 1e-14);
}
