//! `Γ` across its *whole* representable range, and at its poles.
//!
//! The existing identity tests in [`super::gamma_family`] check `Γ` where it is comfortable:
//! factorials to `20!`, and `exp(ln_gamma)` agreement out to `x = 14.9`. That left two large
//! regions untested, and both held bugs:
//!
//! - **Between the last tested point and the true overflow boundary.** The Lanczos form
//!   evaluates `t^(x+0.5)·e^{−t}`, whose first factor overflows long before the product does,
//!   since the second is a compensating underflow. So `Γ` returned `+∞` for arguments whose
//!   values are perfectly representable: `Γ(26)` through `Γ(35)` in `f32`, and `Γ(144)`
//!   through `Γ(171)` in `f64`. Further out still, `e^{−t}` underflowed to zero as well and
//!   `inf · 0` gave `NaN`.
//! - **At the poles.** `sin(πx)` does not evaluate to zero at an integer `x`; it evaluates to
//!   the rounding residue of `πx`, with an arbitrary sign. So the reflection formula returned
//!   finite nonsense — `Γ(−20) = 5.27e-4`, which is the dangerous kind of wrong because it
//!   looks like an answer — and `ln_gamma` returned `NaN` at some negative integers and finite
//!   nonsense at others, depending on that sign.
//!
//! The lesson these tests encode: a function with a limited range has to be tested *up to*
//! that range, and a function with poles has to be tested *at* them.

use crate::Real;
use euclid_utils::assert_close;

/// The largest integer with `Γ(n)` representable, per carrier: `Γ(171) = 7.26e306` fits an
/// `f64` and `Γ(172)` does not; `Γ(35) = 2.95e38` fits an `f32` and `Γ(36)` does not.
const LAST_FINITE_F64: u32 = 171;
const LAST_FINITE_F32: u32 = 35;

#[test]
fn gamma_matches_the_factorials_all_the_way_to_the_overflow_boundary() {
    // Γ(n) = (n−1)!, with the reference built by an independent incremental product rather
    // than from the crate's own `factorial` — which routes through `ln_gamma` past 20! and so
    // would not be independent.
    let mut factorial = 1.0f64; // (n-1)! at n = 1
    for n in 1..=LAST_FINITE_F64 {
        if n > 1 {
            factorial *= f64::from(n - 1);
        }
        let g = Real::gamma(f64::from(n));
        assert!(
            g.is_finite(),
            "gamma({n}) = {g} but {}! = {factorial:e} is representable",
            n - 1
        );
        // The bound is the *reference's*, not `gamma`'s: an incremental product over 170
        // multiplications drifts by about 1e-13 relative by the top of the range (measured
        // 1.0e-13 at n = 165). Tight accuracy for `gamma` is established by the recurrence
        // and duplication tests below, which need no external reference at all; what this
        // test is for is the range, and the defect it guards against is an infinity — which
        // no tolerance forgives.
        assert_close(g, factorial, 1e-12, &format!("gamma({n}) = {}!", n - 1));
    }
    // One step past the boundary it must saturate, not wrap or go NaN.
    for n in [172u32, 200, 1000] {
        assert_eq!(
            Real::gamma(f64::from(n)),
            f64::INFINITY,
            "gamma({n}) should be +inf"
        );
    }
}

#[test]
fn the_f32_carrier_reaches_its_own_boundary_too() {
    // The same statement one carrier down, where the gap between "last tested" and "actually
    // overflows" was 26..35 — ten integers of wrong answers.
    let mut factorial = 1.0f32;
    for n in 1..=LAST_FINITE_F32 {
        if n > 1 {
            factorial *= n as f32 - 1.0;
        }
        let g = Real::gamma(n as f32);
        assert!(
            g.is_finite(),
            "f32 gamma({n}) = {g} but {}! = {factorial:e} is representable",
            n - 1
        );
        assert!(
            ((g - factorial) / factorial).abs() < 1e-5,
            "f32 gamma({n}) = {g:e} vs {}! = {factorial:e}",
            n - 1
        );
    }
    for n in [36u32, 50, 98, 1000] {
        assert_eq!(
            Real::gamma(n as f32),
            f32::INFINITY,
            "f32 gamma({n}) should be +inf"
        );
    }
}

#[test]
fn gamma_has_poles_rather_than_values_at_the_non_positive_integers() {
    // `Γ` has opposite signs on the two sides of each pole, so no limit exists and `NaN` is
    // the answer. At the origin the sign of the zero picks a side, so `±∞` is.
    assert_eq!(Real::gamma(0.0f64), f64::INFINITY);
    assert_eq!(Real::gamma(-0.0f64), f64::NEG_INFINITY);
    assert_eq!(Real::gamma(0.0f32), f32::INFINITY);
    assert_eq!(Real::gamma(-0.0f32), f32::NEG_INFINITY);

    for n in 1..60 {
        let x = -f64::from(n);
        assert!(Real::gamma(x).is_nan(), "gamma({x}) must be NaN at a pole");
        // `ln|Γ|` *does* have a limit at a pole — `+∞` from both sides — so it is returned.
        assert_eq!(
            Real::ln_gamma(x),
            f64::INFINITY,
            "ln_gamma({x}) must be +inf at a pole"
        );
        // `rgamma` is entire and vanishes there; it always did, which is what showed `gamma`
        // was the one in the wrong.
        assert_eq!(Real::rgamma(x), 0.0, "rgamma({x}) must be exactly zero");
        assert!(Real::gamma(-f64::from(n) as f32).is_nan());
    }
    // Including the far-out integers, where every representable value *is* an integer.
    for &x in &[-1e5f64, -1e17, -1e300, f64::NEG_INFINITY] {
        assert!(Real::gamma(x).is_nan(), "gamma({x:e}) must be NaN");
    }
    // Immediately beside a pole the sign must flip, which is what makes `NaN` the right
    // answer *at* it. Γ is negative just above −1 and positive just below.
    for n in 1..8 {
        let x = -f64::from(n);
        let (lo, hi) = (Real::gamma(x - 1e-6), Real::gamma(x + 1e-6));
        assert!(
            lo * hi < 0.0,
            "gamma should change sign across the pole at {x}: {lo:e} then {hi:e}"
        );
        // Both sides blow up like 1/(n!·δ), so the magnitude shrinks with n as n! grows —
        // the sign flip above is the part that is scale-free and worth asserting.
        assert!(
            lo.is_finite() && hi.is_finite() && lo != 0.0 && hi != 0.0,
            "gamma should be large and finite beside the pole at {x}"
        );
    }
}

#[test]
fn gamma_satisfies_its_recurrence_across_the_whole_range() {
    // Γ(x+1) = x·Γ(x). Swept right up to the boundary, so it also constrains the last decade
    // before overflow — the region the factorial test covers only at integers.
    for i in 1..1_700 {
        let x = f64::from(i) * 0.1;
        let (g, g_next) = (Real::gamma(x), Real::gamma(x + 1.0));
        if !g.is_finite() || !g_next.is_finite() {
            continue; // past the boundary, where the identity is inf = inf
        }
        assert_close(
            g_next,
            x * g,
            1e-13,
            &format!("gamma({}) = {x} gamma({x})", x + 1.0),
        );
    }
    // And on the negative side, between the poles, where the reflection formula is in play.
    for i in 1..200 {
        let x = -0.05 - f64::from(i) * 0.1;
        if x.fract() == 0.0 {
            continue;
        }
        let (g, g_next) = (Real::gamma(x), Real::gamma(x + 1.0));
        if !g.is_finite() || !g_next.is_finite() || g == 0.0 {
            continue;
        }
        assert_close(
            g_next / (x * g),
            1.0,
            1e-11,
            &format!("gamma recurrence at {x}"),
        );
    }
}

#[test]
fn gamma_satisfies_the_legendre_duplication_formula() {
    // Γ(z)Γ(z+½) = 2^{1−2z}√π·Γ(2z).
    //
    // An independent constraint of a different shape from the recurrence: it relates a point
    // to its *double*, so it ties the small-argument region to the large one in a single step
    // and would catch a branch that was self-consistent but globally mis-scaled.
    for i in 1..170 {
        let z = f64::from(i) * 0.05;
        let (lhs, rhs) = (
            Real::gamma(z) * Real::gamma(z + 0.5),
            2f64.powf(1.0 - 2.0 * z) * core::f64::consts::PI.sqrt() * Real::gamma(2.0 * z),
        );
        if !lhs.is_finite() || !rhs.is_finite() {
            continue;
        }
        assert_close(lhs / rhs, 1.0, 1e-12, &format!("duplication at z = {z}"));
    }
}

#[test]
fn gamma_and_its_reciprocal_and_its_log_agree_over_the_whole_range() {
    // Three ways of saying the same thing, each with a different range:
    //
    //   Γ(x)·rgamma(x) = 1   where Γ is finite and nonzero
    //   ln Γ(x) = ln Γ(x)    compared in the log domain, which stays finite everywhere
    //
    // The log-domain comparison is the one that reaches the top of the range, and it is how
    // the premature overflow shows up as a disagreement rather than as a lost test.
    for i in 1..1_720 {
        let x = f64::from(i) * 0.1;
        let g = Real::gamma(x);
        if g.is_finite() && g > 0.0 {
            assert_close(
                g * Real::rgamma(x),
                1.0,
                1e-12,
                &format!("gamma({x}) rgamma({x}) = 1"),
            );
            // `ln(Γ(x))` against `ln_gamma(x)`: equal where Γ is representable, and the
            // comparison is well conditioned because both sides are O(ln Γ).
            assert_close(
                g.ln(),
                Real::ln_gamma(x),
                1e-13,
                &format!("ln(gamma({x})) = ln_gamma({x})"),
            );
        } else {
            // Past the boundary `Γ` saturates, but `ln_gamma` must carry on — that asymmetry
            // is the entire reason `ln_gamma` exists.
            assert!(
                Real::ln_gamma(x).is_finite(),
                "ln_gamma({x}) should outlive gamma"
            );
            // `rgamma` outlives `gamma` too, and by a wide margin: `1/Γ(171.7)` is about
            // `1e-309`, a subnormal but a perfectly real number. It reaches zero only once
            // `Γ` passes `2e323`, near x = 180.
            let reciprocal = Real::rgamma(x);
            assert!(
                (0.0..1e-300).contains(&reciprocal),
                "rgamma({x}) = {reciprocal:e} should be tiny but need not be zero"
            );
        }
    }
}
