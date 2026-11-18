//! The error-function family on both carriers: its defining round trip, and its behaviour at
//! the values that are not numbers.
//!
//! See the [parent module](super) for why a cross-precision budget is not enough here.
//!
//! # Why a cross-precision budget cannot pin `erf_inv`
//!
//! `kernels/erf/f32.rs` kept **34** survivors after the cross-precision pass, nearly all of them
//! in `erf_inv` — every Horner coefficient in both of its tables, and every operator in the
//! Newton correction that used to follow them.
//!
//! The budget cannot reach any of that, and not because the sweep missed a branch. The two
//! carriers use **different approximations**: Giles' single-precision coefficient set at `f32`
//! and a separate one at `f64`. They legitimately disagree by up to 15 ulp, so the budget has to
//! be 15 — and the early coefficients contribute about `7e-7` relative, well *inside* a 15-ulp
//! bound at this magnitude. Right instrument for a kernel that computes in f64 and rounds once;
//! wrong instrument for a kernel with its own fit.
//!
//! # And why the round trip is not enough either
//!
//! `erf(erf_inv(p)) = p` is the definition, so it looks like the answer. It is a useful test and
//! it is in this file — but it compares the function against *its own inverse*, so it is blind to
//! any error the pair share. Measured at `f32` it reports a residual of `5.0e-7` for a function
//! whose true error was `1.9e-6`, nearly four times larger.
//!
//! In [`erf_inv`](crate::kernels::erf::f32), the Newton step was making
//! the f32 kernel **eight times less accurate** than its bare seed, because it divided `erf`'s own
//! f32 error by a vanishing derivative. The thirty-four survivors were not reporting untestable
//! coefficients behind a correct correction; they were reporting that the correction had destroyed
//! the precision they would have perturbed.
//!
//! With the step removed, the coefficients are directly observable and are pinned against the
//! **f64 kernel as truth** — four orders of magnitude finer than f32 can represent — by
//! `the_f32_inverse_is_as_accurate_as_the_format_allows`, beside the implementation where the
//! private coefficient tables are visible. The round trip below stays as a cheap independent
//! check of the pair; it is no longer asked to carry the coefficients.

use super::{F32_TOL, agrees};
use crate::tests::*;
use core::fmt::Debug;

/// `erf(erf_inv(p)) = p` — the definition, checked on each carrier.
///
/// The bound is the measured residual with roughly a factor of two of headroom, and it is
/// dominated by `erf`'s own error rather than the inverse's, which is why it does not tighten
/// towards the format's resolution.
///
/// **Read this as a consistency check on the pair, not as an accuracy claim about either.** A
/// round trip cannot see an error the two functions share — see the module header for the
/// factor-of-four it concealed. The accuracy claims live beside each implementation.
fn erf_inv_round_trip<T>(bound: f64, what: &str)
where
    T: Real<Mask = bool> + Debug,
{
    // Sampled across the open interval, including the small-`p` region where `w − 2.5` reaches
    // the end of its range and the high-order coefficients carry the most weight — which is
    // exactly where the mutations showed up.
    for i in 1..400 {
        let p = -0.999 + 2.0 * 0.999 * f64::from(i) / 400.0;
        if p == 0.0 {
            continue;
        }
        let back = T::scalar(p).erf_inv().erf();
        let residual = (back - T::scalar(p)).abs();
        assert!(
            residual.le(&(T::scalar(bound) * T::scalar(p).abs())),
            "{what}: erf(erf_inv({p})) must return {p}, residual {residual:?} exceeds \
             {bound} relative",
        );
    }
    // The three exactly-specified points.
    assert!(T::zero().erf_inv().eq(&T::zero()), "{what}: erf_inv(0) = 0");
    assert!(
        T::one().erf_inv().eq(&T::infinity()),
        "{what}: erf_inv(1) = ∞"
    );
    assert!(
        (-T::one()).erf_inv().eq(&T::neg_infinity()),
        "{what}: erf_inv(−1) = −∞",
    );
    // Outside the support the inverse saturates rather than returning NaN.
    assert!(
        T::scalar(2.0).erf_inv().eq(&T::infinity()),
        "{what}: erf_inv above 1 saturates to ∞",
    );
    assert!(
        T::scalar(-2.0).erf_inv().eq(&T::neg_infinity()),
        "{what}: erf_inv below −1 saturates to −∞",
    );
}

/// The error-function family at `±∞`, `NaN` and zero.
///
/// Each of these is a *sign test* inside an `is_infinite` branch — `if x > 0.0 { 1.0 } else
/// { -1.0 }` and its three relatives. A mutation that swaps the comparison or deletes the
/// negation is invisible to every finite argument, and all four functions had one surviving.
///
/// The limits themselves are not conventions to be looked up: `erf` saturates at `±1` because
/// it is a normalised integral of a positive function, `erfc = 1 − erf` therefore runs between
/// `0` and `2`, and `erfcx(x) = e^{x²} erfc(x)` diverges on the left because the exponential
/// wins while `erfc` tends to two.
fn erf_family_special_values<T>(tol: f64, what: &str)
where
    T: Real<Mask = bool> + Debug,
{
    let s = T::scalar;

    // `erf`: an odd function saturating at ±1.
    assert!(T::infinity().erf().eq(&T::one()), "{what}: erf(∞) = 1");
    assert!(
        T::neg_infinity().erf().eq(&-T::one()),
        "{what}: erf(−∞) = −1",
    );
    assert!(T::nan().erf().is_nan(), "{what}: erf(NaN) = NaN");
    assert!(T::zero().erf().eq(&T::zero()), "{what}: erf(0) = 0");

    // `erfc = 1 − erf`, so it runs from 2 down to 0.
    assert!(T::infinity().erfc().eq(&T::zero()), "{what}: erfc(∞) = 0");
    assert!(T::neg_infinity().erfc().eq(&s(2.0)), "{what}: erfc(−∞) = 2",);
    assert!(T::nan().erfc().is_nan(), "{what}: erfc(NaN) = NaN");
    agrees(&T::zero().erfc(), 1.0, tol, what, "erfc(0) = 1");

    // `erfcx(x) = e^{x²} erfc(x)`: zero on the right, divergent on the left.
    assert!(T::infinity().erfcx().eq(&T::zero()), "{what}: erfcx(∞) = 0");
    assert!(
        T::neg_infinity().erfcx().eq(&T::infinity()),
        "{what}: erfcx(−∞) = ∞",
    );
    assert!(T::nan().erfcx().is_nan(), "{what}: erfcx(NaN) = NaN");
    agrees(&T::zero().erfcx(), 1.0, tol, what, "erfcx(0) = 1");

    // The oddness of `erf` and the complement relation, across the sign boundary the
    // `x > 0.0` branches sit on.
    for &x in &[0.25_f64, 1.0, 2.0, 3.5] {
        let positive = s(x).erf();
        let negative = s(-x).erf();
        assert!(
            (positive + negative).abs().le(&s(tol)),
            "{what}: erf is odd at {x}",
        );
        // `erfc(x) + erfc(−x) = 2`.
        let total = s(x).erfc() + s(-x).erfc();
        agrees(&total, 2.0, tol, what, "erfc(x) + erfc(−x) = 2");
    }

    // `erf` against its Maclaurin series, below the argument where that series stops being a
    // usable reference.
    //
    // Two independent limits coincide at about `x = 2`, which is why the sample stops there.
    // The series alternates with terms peaking at `x^{2n+1}/n!`, so by `x = 3.5` it cancels
    // some twenty-five orders of magnitude down to a result near one and carries no useful
    // digits — it disagreed with the kernel by `2.5e-7` relative, a hundred thousand times this
    // tolerance, and the series was the wrong one. Coincidentally `ERFCX_CF_MIN` is also `2`,
    // above which `erfc` and `erfcx` both route through the same continued fraction, so an
    // identity between *them* would be circular up there as well. The accuracy of the large-`x`
    // tail is `super::super::accuracy`'s question, against out-of-band reference values.
    for &x in &[0.25_f64, 0.5, 1.0, 1.5, 2.0, -0.75] {
        agrees(
            &s(x).erf(),
            erf_series(x),
            tol,
            what,
            "erf matches its Maclaurin series",
        );
        agrees(
            &s(x).erfc(),
            1.0 - erf_series(x),
            tol,
            what,
            "erfc = 1 − erf against the series",
        );
        // `erfcx(x) = e^{x²} erfc(x)`. Below `ERFCX_CF_MIN` this is how `erfcx` is computed, so
        // the independent part is the series rather than the relation.
        agrees(
            &s(x).erfcx(),
            (x * x).exp() * (1.0 - erf_series(x)),
            tol,
            what,
            "erfcx(x) = e^{x²}·erfc(x)",
        );
    }
}

/// `erf` from its Maclaurin series, as an independent reference.
///
/// `erf(x) = 2/√π · Σ (−1)ⁿ x^{2n+1} / (n!(2n+1))`. Shares no code with either kernel, and is
/// trustworthy only for `|x| <= 2` — see the caller for why.
fn erf_series(x: f64) -> f64 {
    let mut term = x;
    let mut sum = x;
    for n in 1..60 {
        term *= -x * x / f64::from(n);
        sum += term / (2.0 * f64::from(n) + 1.0);
    }
    2.0 / core::f64::consts::PI.sqrt() * sum
}

#[test]
fn the_f64_carrier_computes_the_erf_family() {
    // Measured residual `5.8e-15`; pinned with a factor of three of headroom.
    erf_inv_round_trip::<f64>(2e-14, "f64");
    erf_family_special_values::<f64>(1e-12, "f64");
}

#[test]
fn the_f32_carrier_computes_the_erf_family() {
    // Measured residual `8.6e-7`, against mutated coefficients at `4.1e-5` and worse — so this
    // bound sits a factor of twenty below the nearest thing it has to reject.
    erf_inv_round_trip::<f32>(2e-6, "f32");
    erf_family_special_values::<f32>(F32_TOL * 10.0, "f32");
}
