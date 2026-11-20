//! Unit tests for the single-precision exponential and trigonometric integrals.
//!
//! A child module so implementation plus tests stay inside the crate's 400-line budget. What is
//! added beyond the reference comparisons is the **accumulator** and the **routes**: fourteen
//! mutants survived in `compensated_series`, the complex Lentz helpers, `ei_route` and
//! `ei_asymptotic_sum`, none
//! of which a whole-function comparison can reach — a compensation term is invisible unless the
//! naive sum would have lost it, and a convergence break changes cost rather than value.

use super::*;
use crate::kernels::integrals as reference;
use euclid_utils::assert_close_f32;

/// The native values against the `f64` kernel, at budgets a native path earns.
#[test]
fn the_native_values_track_the_f64_reference() {
    for &x in &[0.25f32, 0.5, 1.0, 2.0, 3.0, 4.0, 7.5, 20.0, 45.0, 100.0] {
        let wide = f64::from(x);
        assert_close_f32(sin_int(x), reference::sin_int(wide), 16.0, "Si");
        assert_close_f32(cos_int(x), reference::cos_int(wide), 16.0, "Ci");
        assert_close_f32(exp_int_e1(x), reference::exp_int_e1(wide), 16.0, "E1");
    }
    for &x in &[0.5f32, 1.0, 5.0, 20.0, 60.0] {
        assert_close_f32(
            exp_int_ei(x),
            reference::exp_int_ei(f64::from(x)),
            16.0,
            "Ei",
        );
    }
}

#[test]
fn the_f32_surface_keeps_the_f64_conventions() {
    // The same edge behaviour, which is what a carrier swap must not change.
    assert_eq!(sin_int(0.0), 0.0);
    assert_eq!(sin_int(f32::INFINITY), core::f32::consts::FRAC_PI_2);
    assert_eq!(sin_int(f32::NEG_INFINITY), -core::f32::consts::FRAC_PI_2);
    assert!(sin_int(f32::NAN).is_nan());
    assert!(cos_int(-1.0).is_nan());
    assert_eq!(cos_int(0.0), f32::NEG_INFINITY);
    assert_eq!(cos_int(f32::INFINITY), 0.0);
    assert!(exp_int_e1(-1.0).is_nan());
    assert_eq!(exp_int_e1(0.0), f32::INFINITY);
    assert_eq!(exp_int_e1(f32::INFINITY), 0.0);
    assert_eq!(exp_int_ei(0.0), f32::NEG_INFINITY);
    // Ei overflows f32 far earlier than f64; saturating to infinity is correct.
    assert_eq!(exp_int_ei(100.0), f32::INFINITY);
    assert!(exp_int_ei(f32::NAN).is_nan());
}

#[test]
fn si_is_odd_and_ci_matches_its_known_values() {
    for &x in &[0.5f32, 1.0, 3.0, 12.0] {
        assert_eq!(sin_int(-x), -sin_int(x));
    }
    // Spot values, to f32 resolution.
    assert_close_f32(sin_int(1.0), 9.460_830_703_671_83e-1, 4.0, "Si(1)");
    assert_close_f32(cos_int(1.0), 3.374_039_229_009_682e-1, 4.0, "Ci(1)");
    assert_close_f32(sin_int(45.0), 1.558_715_000_896_413e0, 16.0, "Si(45)");
}

/// Both trig routes must be reached, and must agree across the join.
///
/// The join is the one place a two-route function can be wrong while both routes are right,
/// so it is checked directly rather than left to a sweep that might straddle it by luck.
#[test]
fn the_trig_routes_agree_across_their_join() {
    assert_eq!(
        TrigIntegralRoute::for_argument(TRIG_SERIES_MAX - 0.1),
        TrigIntegralRoute::Series
    );
    assert_eq!(
        TrigIntegralRoute::for_argument(TRIG_SERIES_MAX),
        TrigIntegralRoute::Fraction
    );
    // Either side of the boundary, against the f64 kernel.
    for x in [
        TRIG_SERIES_MAX - 1e-3,
        TRIG_SERIES_MAX,
        TRIG_SERIES_MAX + 1e-3,
    ] {
        let wide = f64::from(x);
        assert_close_f32(sin_int(x), reference::sin_int(wide), 16.0, "Si at the join");
        assert_close_f32(cos_int(x), reference::cos_int(wide), 16.0, "Ci at the join");
    }
}

/// The compensated accumulator, where the compensation is the whole point.
///
/// Six mutants lived here. `correction` only matters when the naive running total has *already*
/// discarded a term, which no ordinary series in this file arranges — so the test arranges it:
/// `1e8 + 1 − 1e8` is `0` summed naively in `f32`, because `1e8 + 1` rounds back to `1e8`, and
/// `1` once the discarded low part is carried.
#[test]
fn the_accumulator_carries_what_the_running_total_discards() {
    let catastrophic = [1e8_f32, 1.0, -1e8];
    let naive: f32 = catastrophic.iter().sum();
    assert_eq!(naive, 0.0, "the naive sum must lose the middle term");
    let got = compensated_series(|k| catastrophic.get(k as usize).copied());
    assert_eq!(got, 1.0);

    // Exact on a short exact series, so the compensation never *adds* error.
    assert_eq!(compensated_series(|k| (k < 4).then_some(0.25)), 1.0);
    assert_eq!(compensated_series(|_| None), 0.0);

    // The break is `negligible && k > 2`, so the first three terms are always taken and the
    // fourth is where it can first fire. A term of exactly zero is negligible against any
    // magnitude, which pins the index: four calls, not three and not the cap.
    let mut calls = 0;
    let value = compensated_series(|k| {
        calls += 1;
        Some(if k == 0 { 1.0 } else { 0.0 })
    });
    assert_eq!(value, 1.0);
    assert_eq!(calls, 4, "the guard admits a break only past k = 2");

    // And a series that never goes negligible runs to the cap rather than forever.
    let mut calls = 0;
    let value = compensated_series(|_| {
        calls += 1;
        Some(1.0)
    });
    assert_eq!(calls, MAX_TERMS, "the cap is the backstop");
    assert_eq!(value, MAX_TERMS as f32);
}

/// The Lentz nudge at this format's floor.
///
/// The one-norm it tests and the complex arithmetic around it now live in
/// [`crate::math::complex`] and [`crate::math::continued_fraction`], where they are generic and
/// tested once. What stays here is the floor: `LENTZ_TINY_F32` is per format, and this is the
/// only place that pairs it with the nudge.
#[test]
fn the_nudge_fires_only_on_a_genuine_zero() {
    assert_eq!(nudge_at((0.0, 0.0)), (LENTZ_TINY_F32, 0.0));
    assert_eq!(nudge_at((-0.0, -0.0)), (LENTZ_TINY_F32, 0.0));
    // Equal-magnitude opposite parts are not a zero, which is why the test is a 1-norm.
    assert_eq!(nudge_at((2.0, -2.0)), (2.0, -2.0));
    assert_eq!(
        nudge_at((f32::MIN_POSITIVE, -f32::MIN_POSITIVE)),
        (f32::MIN_POSITIVE, -f32::MIN_POSITIVE)
    );
}

/// `Ei`'s route table, over the whole line.
#[test]
fn the_ei_routes_cover_the_line() {
    use EiRoute::{Asymptotic, Reflection, Series, Value};
    // Compared by variant: a `NaN` payload is not equal to itself, so `assert_eq` would fail on
    // a route that is in fact correct.
    assert!(matches!(ei_route(f32::NAN), Value(v) if v.is_nan()));
    assert_eq!(ei_route(0.0), Value(f32::NEG_INFINITY));
    assert_eq!(ei_route(-0.0), Value(f32::NEG_INFINITY));
    assert_eq!(ei_route(f32::INFINITY), Value(f32::INFINITY));
    assert_eq!(ei_route(-1.0), Reflection);
    assert_eq!(ei_route(f32::NEG_INFINITY), Reflection);
    assert_eq!(ei_route(1.0), Series);
    assert_eq!(ei_route(EI_ASYMPTOTIC_MIN.next_down()), Series);
    assert_eq!(ei_route(EI_ASYMPTOTIC_MIN), Asymptotic);
    assert_eq!(ei_route(1e30), Asymptotic);
}

/// `E₁`'s three routes, including the series crossover and the whole negative half-line.
///
/// The negative arm is the one no value assertion reaches, which is what the replaced guard's
/// own comment recorded: a negative argument falling through to the series evaluates `ln x` and
/// returns `NaN` regardless, so the domain statement and the accident agree on the answer. Only
/// the decision separates them.
#[test]
fn the_e1_routes_cover_the_line_and_its_series_crossover() {
    use E1Route::{ContinuedFraction, Series, Value};

    assert_eq!(e1_route(E1_SERIES_MAX.next_down()), Series);
    assert_eq!(e1_route(E1_SERIES_MAX), Series);
    assert_eq!(e1_route(E1_SERIES_MAX.next_up()), ContinuedFraction);
    assert_eq!(e1_route(f32::MIN_POSITIVE), Series);
    assert_eq!(e1_route(1e30), ContinuedFraction);
    assert_eq!(e1_route(f32::MAX), ContinuedFraction);
    // The pole at the origin, from both signed zeros.
    assert_eq!(e1_route(0.0), Value(f32::INFINITY));
    assert_eq!(e1_route(-0.0), Value(f32::INFINITY));
    assert_eq!(e1_route(f32::INFINITY), Value(0.0));
    // The negative half-line and `NaN` are off the domain.
    for x in [
        -f32::MIN_POSITIVE,
        -0.5,
        -1.0,
        -1e30,
        f32::NEG_INFINITY,
        f32::NAN,
    ] {
        assert!(matches!(e1_route(x), Value(v) if v.is_nan()), "at {x:e}");
        assert!(exp_int_e1(x).is_nan(), "at {x:e}");
    }
}

/// The asymptotic sum, which stops at its least term because the expansion diverges.
#[test]
fn the_ei_asymptotic_sum_tracks_the_wide_kernel() {
    // Non-integer arguments are deliberately included. At an *integer* `x` the ratio `k/x`
    // reaches exactly one, so `term == prev` fires a narrowed `==` break by luck; off the
    // integers it never fires at all and the loop runs on through the expansion's divergent
    // tail to the term cap, which is what this catches.
    for x in [EI_ASYMPTOTIC_MIN, 22.5, 25.0, 37.3, 50.0, 100.0] {
        // `Ei(x) ≈ e^x/x · Σ k!/x^k`, so the sum is the wide kernel's `Ei` stripped of its
        // prefactor — which is how this function is used and the only reference it has.
        let want = (reference::exp_int_ei(f64::from(x)) * f64::from(x) / f64::from(x).exp()) as f32;
        let got = ei_asymptotic_sum(x);
        assert!(
            (got - want).abs() <= 2e-4 * want,
            "ei_asymptotic_sum({x}): {got} vs {want}"
        );
        // Every term is positive, so the sum exceeds one and grows as `x` falls.
        assert!(got > 1.0, "at {x}: {got}");
    }
    assert!(ei_asymptotic_sum(25.0) > ei_asymptotic_sum(100.0));
    // The sum is bounded: the series is truncated at its least term, so it cannot run away.
    for x in [20.3_f32, 22.5, 37.3, 99.1] {
        let got = ei_asymptotic_sum(x);
        assert!((1.0..2.0).contains(&got), "at {x}: {got}");
    }

    // `>=` against `>` is **not** reachable, and the reason is worth recording: the two differ
    // only at an integer `x`, where it drops the single least term — `2.3e-8` relative at
    // `x = 20` and smaller above it, which is below this format's resolution. An equivalent
    // mutant by the expansion's own error floor.
}

/// The trig integrals' route tables, which differ because the two functions differ.
///
/// `Si` is odd and entire — every arm is about the format — while `Ci` carries a `ln x` term and
/// so has a real singularity at the origin and no value at all to its left.
#[test]
fn the_trig_integral_routes_distinguish_an_odd_function_from_a_cut_one() {
    assert_eq!(SinIntRoute::for_argument(f32::NAN), SinIntRoute::Undefined);
    assert_eq!(SinIntRoute::for_argument(0.0), SinIntRoute::AtOrigin);
    assert_eq!(SinIntRoute::for_argument(-0.0), SinIntRoute::AtOrigin);
    for x in [f32::INFINITY, f32::NEG_INFINITY] {
        assert_eq!(SinIntRoute::for_argument(x), SinIntRoute::AtInfinity);
    }
    for x in [
        -1e30,
        -1.0,
        -f32::MIN_POSITIVE,
        f32::MIN_POSITIVE,
        1.0,
        1e30,
    ] {
        assert_eq!(
            SinIntRoute::for_argument(x),
            SinIntRoute::Quadrature,
            "at {x:e}"
        );
    }
    // The odd symmetry the sign-bit test delivers, including at the origin.
    assert_eq!(sin_int(-0.0).to_bits(), (-0.0_f32).to_bits());
    for x in [0.5_f32, 1.0, 5.0, 50.0] {
        assert_eq!(sin_int(-x), -sin_int(x), "at {x}");
    }

    assert_eq!(CosIntRoute::for_argument(f32::NAN), CosIntRoute::Undefined);
    for x in [
        -f32::MIN_POSITIVE,
        -1.0,
        -1e30,
        f32::NEG_INFINITY,
        (-0.0_f32).next_down(),
    ] {
        assert_eq!(
            CosIntRoute::for_argument(x),
            CosIntRoute::Undefined,
            "at {x:e}"
        );
        assert!(cos_int(x).is_nan(), "at {x:e}");
    }
    // Both signed zeros are the singularity: `-0.0 < 0.0` is false, so it falls to this arm.
    assert_eq!(CosIntRoute::for_argument(0.0), CosIntRoute::AtOrigin);
    assert_eq!(CosIntRoute::for_argument(-0.0), CosIntRoute::AtOrigin);
    assert_eq!(cos_int(-0.0), f32::NEG_INFINITY);
    assert_eq!(
        CosIntRoute::for_argument(f32::INFINITY),
        CosIntRoute::AtInfinity
    );
    for x in [f32::MIN_POSITIVE, 1.0, 1e30] {
        assert_eq!(
            CosIntRoute::for_argument(x),
            CosIntRoute::Quadrature,
            "at {x:e}"
        );
    }
}
