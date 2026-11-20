//! Unit tests for the single-precision `J₀`/`J₁` and `Y₀`/`Y₁`.
//!
//! A child module so implementation plus tests stay inside the crate's 400-line budget. What is
//! added beyond the reference comparisons is the **node budget** and the **routes**: twenty-eight
//! mutants survived here, eleven of them in `alias_nodes` alone, which decides how many backward
//! recurrence steps the aliasing needs. Its output is an integer and its effect is a cost, so a
//! value comparison on `J` cannot see it at all — the recurrence converges from any sufficient
//! node count, and from a few too many just as well.

use super::*;
use crate::kernels::bessel::{ordinary_first, ordinary_second};
use euclid_utils::assert_close_f32;

/// Both kinds against the `f64` kernel, at budgets a native path earns.
#[test]
fn the_values_track_the_f64_reference() {
    for i in 1..80 {
        let x = (i as f32) * 0.5;
        let wide = f64::from(x);
        assert_close_f32(j0(x), ordinary_first::j0(wide), 16.0, "J0");
        assert_close_f32(j1(x), ordinary_first::j1(wide), 16.0, "J1");
        // `Y` is compared against its envelope rather than relatively: it has infinitely
        // many zeros, so a relative comparison is meaningless at each one.
        let envelope = (2.0 / (core::f64::consts::PI * wide)).sqrt();
        for (name, got, want) in [
            ("Y0", y0(x), ordinary_second::y0(wide)),
            ("Y1", y1(x), ordinary_second::y1(wide)),
        ] {
            let error = (f64::from(got) - want).abs() / envelope;
            assert!(
                error <= 1e-6,
                "{name}({x}) is {error:.3e} of the envelope from the f64 value"
            );
        }
    }
}

/// `J` is even, `J₁` odd, and both exact at the origin.
#[test]
fn the_symmetries_hold_exactly() {
    for i in 1..40 {
        let x = (i as f32) * 0.5;
        assert_eq!(j0(-x), j0(x), "J0 is even");
        assert_eq!(j1(-x), -j1(x), "J1 is odd");
    }
    assert_eq!(j0(0.0), 1.0);
    assert_eq!(j1(0.0), 0.0);
    assert_eq!(j1(-0.0), 0.0, "J1 preserves the sign of zero");
    assert!(j1(-0.0).is_sign_negative());
}

#[test]
fn the_f32_surface_keeps_the_f64_conventions() {
    assert!(j0(f32::NAN).is_nan());
    assert!(j1(f32::NAN).is_nan());
    assert_eq!(j0(f32::INFINITY), 0.0);
    assert_eq!(j1(f32::INFINITY), 0.0);
    assert_eq!(j0(f32::NEG_INFINITY), 0.0);
}

/// All three `J` routes must be reached, and must agree across both joins.
#[test]
fn the_j_routes_agree_across_their_joins() {
    assert_eq!(JRoute::for_magnitude(1.0), JRoute::Series);
    assert_eq!(JRoute::for_magnitude(J_SERIES_MAX), JRoute::Series);
    assert_eq!(JRoute::for_magnitude(J_SERIES_MAX + 0.1), JRoute::Aliased);
    assert_eq!(JRoute::for_magnitude(HANKEL_MIN - 0.1), JRoute::Aliased);
    assert_eq!(JRoute::for_magnitude(HANKEL_MIN), JRoute::Hankel);
    for x in [
        J_SERIES_MAX - 1e-3,
        J_SERIES_MAX,
        J_SERIES_MAX + 1e-3,
        HANKEL_MIN - 1e-3,
        HANKEL_MIN,
        HANKEL_MIN + 1e-3,
    ] {
        let wide = f64::from(x);
        assert_close_f32(j0(x), ordinary_first::j0(wide), 16.0, "J0 at a join");
        assert_close_f32(j1(x), ordinary_first::j1(wide), 16.0, "J1 at a join");
    }
}

/// The Wronskian `J₀Y₁ − J₁Y₀ = −2/(πx)`, which ties the two native kinds together.
///
/// Worth more than either value check: it holds only if both families are right *and* wired
/// to each other, and it is the only test here that would catch the `J` and `Y` parts of the
/// shared Hankel expansion being swapped.
#[test]
fn the_wronskian_ties_the_two_native_kinds_together() {
    for i in 1..60 {
        let x = (i as f32) * 0.5;
        let w = j0(x) * y1(x) - j1(x) * y0(x) + 2.0 / (PI * x);
        let scale = 2.0 / (PI * x);
        assert!(
            w.abs() <= 1e-5 * scale.max(1e-2),
            "J/Y Wronskian broken at x = {x}: {w:e}"
        );
    }
}

/// The aliasing node budget, as exact integers.
///
/// Eleven mutants lived here. The count has to be *enough* and even; anything above enough gives
/// the same `J` to the bit, so every operator in it is invisible to a value assertion. Measured
/// values, rounded up to even and offset by the order:
#[test]
fn the_alias_node_budget_is_enough_and_even() {
    for (n, x, want) in [
        (0_u32, 0.5_f32, 10_u32),
        (0, 1.0, 12),
        (0, 2.0, 14),
        (0, 2.5, 16),
        (0, 5.0, 20),
        (0, 6.9, 24),
        (1, 0.5, 12),
        (1, 1.0, 12),
        (1, 2.0, 16),
        (1, 2.5, 16),
        (1, 5.0, 22),
        (1, 6.9, 24),
    ] {
        assert_eq!(alias_nodes(n, x), want, "at n={n}, x={x}");
    }
    // Always even, which the backward recurrence's parity depends on, and always above the
    // argument — the series needs to have passed its peak at `m ≈ x/2` before it can be truncated.
    for n in 0..2 {
        for i in 1..70 {
            let x = i as f32 * 0.1;
            let nodes = alias_nodes(n, x);
            assert_eq!(nodes % 2, 0, "at n={n}, x={x}: {nodes} is odd");
            assert!(nodes as f32 > x * 0.5, "at n={n}, x={x}: {nodes}");
        }
    }
    // Monotone in both arguments, and a higher order needs at least as many.
    assert!(alias_nodes(0, 6.9) > alias_nodes(0, 0.5));
    assert!(alias_nodes(1, 5.0) >= alias_nodes(0, 5.0));
    // The cap binds above the aliasing range: `m` stops at `ALIAS_MAX_NODES`, so the budget
    // saturates at `64 + n + 2` rounded to even rather than growing without bound.
    assert_eq!(alias_nodes(0, 100.0), ALIAS_MAX_NODES + 2);
    assert_eq!(alias_nodes(1, 100.0), ALIAS_MAX_NODES + 4);
}

/// `Y`'s route table, over the whole line.
#[test]
fn the_y_routes_cover_the_line() {
    use YRoute::{Hankel, March, Series, Value};
    assert!(matches!(YRoute::for_argument(f32::NAN), Value(v) if v.is_nan()));
    assert!(matches!(YRoute::for_argument(-1.0), Value(v) if v.is_nan()));
    assert_eq!(YRoute::for_argument(0.0), Value(f32::NEG_INFINITY));
    assert_eq!(YRoute::for_argument(f32::INFINITY), Value(0.0));
    assert_eq!(YRoute::for_argument(1.0), Series);
    assert_eq!(YRoute::for_argument(Y_SERIES_MAX), Series);
    assert_eq!(YRoute::for_argument(Y_SERIES_MAX.next_up()), March);
    assert_eq!(YRoute::for_argument(4.0), March);
    assert_eq!(YRoute::for_argument(HANKEL_MIN.next_down()), March);
    assert_eq!(YRoute::for_argument(HANKEL_MIN), Hankel);
    assert_eq!(YRoute::for_argument(1e30), Hankel);
}

/// Each private route against the `f64` kernel, on the band it is selected for.
///
/// A sweep over the public surface checks whichever arm its arguments happen to land on; these
/// call the arms directly, so each one is checked on its own band rather than by luck.
#[test]
fn every_route_tracks_the_wide_kernel_on_its_own_band() {
    // The `J` series, below `J_SERIES_MAX`, for both orders.
    for x in [1e-4_f32, 0.1, 0.5, 1.0, J_SERIES_MAX] {
        let wide = f64::from(x);
        let got = j_series(0, x);
        let want = ordinary_first::j0(wide) as f32;
        assert!(
            (got - want).abs() <= 2e-6,
            "j_series(0,{x}): {got} vs {want}"
        );
        let got = j_series(1, x);
        let want = ordinary_first::j1(wide) as f32;
        assert!(
            (got - want).abs() <= 2e-6,
            "j_series(1,{x}): {got} vs {want}"
        );
    }
    // The aliased backward recurrence, between the series and Hankel.
    for x in [J_SERIES_MAX.next_up(), 3.0, 5.0, HANKEL_MIN.next_down()] {
        let wide = f64::from(x);
        let got = j_aliased(0, x);
        let want = ordinary_first::j0(wide) as f32;
        assert!(
            (got - want).abs() <= 2e-6,
            "j_aliased(0,{x}): {got} vs {want}"
        );
        let got = j_aliased(1, x);
        let want = ordinary_first::j1(wide) as f32;
        assert!(
            (got - want).abs() <= 2e-6,
            "j_aliased(1,{x}): {got} vs {want}"
        );
    }
    // The `Y` series, below `Y_SERIES_MAX`. `Y` diverges logarithmically at the origin, so the
    // comparison is relative there.
    for x in [0.01_f32, 0.1, 0.5, 1.0, Y_SERIES_MAX] {
        let wide = f64::from(x);
        let got = y0_series(x);
        let want = ordinary_second::y0(wide) as f32;
        assert!(
            (got - want).abs() <= 2e-5 * want.abs().max(1.0),
            "y0_series({x}): {got} vs {want}"
        );
        let got = y1_series(x);
        let want = ordinary_second::y1(wide) as f32;
        assert!(
            (got - want).abs() <= 2e-5 * want.abs().max(1.0),
            "y1_series({x}): {got} vs {want}"
        );
    }
    // And the public surface at and above the Hankel handover, which is the shared expansion.
    for x in [HANKEL_MIN, 10.0, 50.0, 500.0] {
        let wide = f64::from(x);
        assert_close_f32(j0(x), ordinary_first::j0(wide), 64.0, "j0 Hankel");
        assert_close_f32(j1(x), ordinary_first::j1(wide), 64.0, "j1 Hankel");
        assert_close_f32(y0(x), ordinary_second::y0(wide), 64.0, "y0 Hankel");
        assert_close_f32(y1(x), ordinary_second::y1(wide), 64.0, "y1 Hankel");
    }
}

/// One Taylor step of Bessel's equation, checked against the equation.
#[test]
fn the_taylor_step_transports_bessels_equation() {
    // A zero step is the identity, exactly.
    assert_eq!(taylor_step(3.0, 2.0, -1.0, 0.0, 0.0), (2.0, -1.0));
    // `x²y″ + xy′ + (x² − ν²)y = 0` fixes the second coefficient: `a₂ = −(x₀y′ + (x₀² − ν²)y)
    // / (2x₀²)`, so a small step is `y + y′δ + a₂δ²` and the rest is negligible.
    let (x0, y, dy, order_sq) = (4.0_f32, 0.5_f32, -0.25_f32, 1.0_f32);
    let delta = 0.01_f32;
    let (value, _) = taylor_step(x0, y, dy, delta, order_sq);
    let a2 = -(x0 * dy + (x0 * x0 - order_sq) * y) / (2.0 * x0 * x0);
    let taylor = y + dy * delta + a2 * delta * delta;
    assert!((value - taylor).abs() <= 2e-6, "{value} vs {taylor}");
    // And it reproduces `J₀` when seeded from it, which is the march's whole purpose.
    let x0 = 3.0_f32;
    let (value, derivative) = taylor_step(x0, j0(x0), -j1(x0), 0.25, 0.0);
    let wide = f64::from(x0) + 0.25;
    assert!(
        (value - ordinary_first::j0(wide) as f32).abs() <= 1e-5,
        "{value}"
    );
    assert!(
        (derivative + ordinary_first::j1(wide) as f32).abs() <= 1e-5,
        "{derivative}"
    );
}

/// The **cost** of both `Y` series, which is all their convergence breaks change.
///
/// Seven mutants lived on those two lines. Narrowing `<` to `==`, or `k > 1` to `==` or `<`,
/// makes the break unreachable and runs every call to [`Y_MAX_TERMS`] — thirty-two terms where
/// fewer than ten suffice — without moving the value by a bit. The count is the observable.
/// `<=` against `<`, and `>=` against `>`, differ only at an exact tolerance hit or one index
/// sooner than convergence ever arrives: equivalent mutants, recorded rather than chased.
#[test]
fn both_y_series_converge_well_inside_the_term_cap() {
    for (x, y0_want, y1_want) in [
        (0.01_f32, 3, 3),
        (0.1, 4, 4),
        (0.5, 5, 6),
        (1.0, 6, 7),
        (Y_SERIES_MAX, 8, 9),
    ] {
        let (value, terms) = y0_series_terms(x);
        assert_eq!(terms, y0_want, "y0 at x = {x}");
        assert!(value.is_finite());
        assert!(terms < Y_MAX_TERMS, "y0 at x = {x} reached the cap");
        let (value, terms) = y1_series_terms(x);
        assert_eq!(terms, y1_want, "y1 at x = {x}");
        assert!(value.is_finite());
        assert!(terms < Y_MAX_TERMS, "y1 at x = {x} reached the cap");
    }
    // Monotone in the argument: more terms are needed further from the origin, which is what
    // makes the break a convergence test rather than a fixed schedule.
    assert!(y0_series_terms(Y_SERIES_MAX).1 > y0_series_terms(0.01).1);
    assert!(y1_series_terms(Y_SERIES_MAX).1 > y1_series_terms(0.01).1);
}

/// `J₁` is **odd**, including at zero, which only the sign bit records.
///
/// The comparison that applies the reflection survived because `-0.0 == 0.0`: both arms return a
/// zero, and every value assertion in the crate is satisfied by either. The sign is the whole
/// content of oddness at the origin, so it is asserted on the bits — the same distinction
/// `erf(-0.0)` and `exprel(-0.0)` carry.
#[test]
fn the_first_order_keeps_the_sign_of_its_zero() {
    assert_eq!(j1(-0.0_f32).to_bits(), (-0.0_f32).to_bits());
    assert_eq!(j1(0.0_f32).to_bits(), 0.0_f32.to_bits());
    assert!(j1(-0.0_f32).is_sign_negative());
    assert!(!j1(0.0_f32).is_sign_negative());
    // And odd away from the origin too, which is what the reflection is for.
    for x in [0.5_f32, 1.0, 3.0, 10.0] {
        assert_eq!(j1(-x), -j1(x), "at {x}");
    }
    // `J₀` is even by contrast, so its reflection must *not* carry a sign.
    assert_eq!(j0(-0.0_f32).to_bits(), j0(0.0_f32).to_bits());
    for x in [0.5_f32, 3.0, 10.0] {
        assert_eq!(j0(-x), j0(x), "at {x}");
    }
}
