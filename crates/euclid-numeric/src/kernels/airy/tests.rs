//! Unit tests for the Airy family, reaching the private thresholds and the individual routes.
//!
//! These live beside the implementation because every one of them needs something the
//! `crate::tests` tree cannot see: a crossover constant, or one route evaluated in isolation so
//! it can be compared against the route it hands over to. Tests that only need the public
//! surface — the Wronskian, the equation, the values at the origin — live in
//! [`crate::tests::identities::airy_family`] instead.
//!
//! # What is actually being checked here
//!
//! A piecewise implementation has two failure modes that reference values do not catch. One is
//! a **step at a join**, where both branches are individually accurate but disagree about the
//! value at the boundary. The other is a **branch that is accurate only where it is tested**,
//! which a sweep landing on one side of a threshold will miss entirely. The tests below attack
//! both: first by probing every crossover at ulp resolution, then by running each pair of
//! neighbouring routes against *each other* across the overlap where both should be good.

use super::march;
use super::series::{AI_0, NEG_AI_PRIME_0, SQRT_3};
use super::{
    AI_SERIES_MAX, BI_SERIES_MAX, LEADING_MIN, MARCH_MAX, SERIES_MAX, ai, ai_pair_scaled, ai_prime,
    bi, bi_pair_scaled, bi_prime, bie, leading_scaled, negative_asymptotic, positive_scaled,
    series, zeta,
};
use crate::Real;
use euclid_utils::{assert_close, no_step_across, rel, sweep};

#[test]
fn the_origin_constants_agree_with_the_crate_gamma() {
    // `c₁ = 3^{−2/3}/Γ(⅔)` and `c₂ = 3^{−1/3}/Γ(⅓)`, carried as literals because the series
    // needs them to full precision and `Γ` is only good to ~1e-16. Checking them against the
    // crate's own `Γ` is what keeps the literals honest: a mistyped digit shows up here, and
    // the check is non-circular because nothing in `Γ` consults the Airy module.
    let three = 3.0f64;
    assert_close(
        AI_0,
        three.powf(-2.0 / 3.0) / Real::gamma(2.0 / 3.0),
        1e-15,
        "Ai(0) = 3^(-2/3)/gamma(2/3)",
    );
    assert_close(
        NEG_AI_PRIME_0,
        three.powf(-1.0 / 3.0) / Real::gamma(1.0 / 3.0),
        1e-15,
        "-Ai'(0) = 3^(-1/3)/gamma(1/3)",
    );
    assert_close(SQRT_3, three.sqrt(), 0.0, "SQRT_3");

    // And the four values at the origin follow from them, with the √3 ratios exact.
    assert_close(ai(0.0), AI_0, 0.0, "Ai(0)");
    assert_close(ai_prime(0.0), -NEG_AI_PRIME_0, 0.0, "Ai'(0)");
    assert_close(bi(0.0) / ai(0.0), SQRT_3, 1e-15, "Bi(0)/Ai(0) = sqrt(3)");
    assert_close(
        bi_prime(0.0) / ai_prime(0.0),
        -SQRT_3,
        1e-15,
        "Bi'(0)/Ai'(0) = -sqrt(3)",
    );
}

#[test]
fn every_crossover_leaves_no_step_at_ulp_resolution() {
    // Each threshold probed at the two representable values that straddle it. The derivative
    // term matters most at `LEADING_MIN`, where `Aiᵉ′ ≈ −x^{1/4}/(2√π)` is about 2800 and one
    // ulp of 1e10 is 2e-6, so the function is entitled to move by ~5e-3 in absolute terms.
    //
    // `ai`/`bi` rather than the scaled forms at the two series joins, because those are the
    // routes that actually change there; the scaled forms are checked at the same points by
    // the overlap tests below.
    no_step_across(
        "Ai at AI_SERIES_MAX",
        ai,
        AI_SERIES_MAX,
        ai_prime(AI_SERIES_MAX),
        1e-15,
    );
    no_step_across(
        "Ai' at AI_SERIES_MAX",
        ai_prime,
        AI_SERIES_MAX,
        // `Ai″ = x·Ai` by the equation itself, so the derivative of `Ai′` is free.
        AI_SERIES_MAX * ai(AI_SERIES_MAX),
        1e-15,
    );
    no_step_across(
        "Bi at BI_SERIES_MAX",
        bi,
        BI_SERIES_MAX,
        bi_prime(BI_SERIES_MAX),
        1e-14,
    );
    no_step_across(
        "Bi' at BI_SERIES_MAX",
        bi_prime,
        BI_SERIES_MAX,
        BI_SERIES_MAX * bi(BI_SERIES_MAX),
        1e-14,
    );
    no_step_across(
        "Ai at -SERIES_MAX",
        ai,
        -SERIES_MAX,
        ai_prime(-SERIES_MAX),
        1e-15,
    );
    no_step_across(
        "Bi at -SERIES_MAX",
        bi,
        -SERIES_MAX,
        bi_prime(-SERIES_MAX),
        1e-15,
    );
    no_step_across(
        "Ai at -MARCH_MAX",
        ai,
        -MARCH_MAX,
        ai_prime(-MARCH_MAX),
        1e-13,
    );
    no_step_across(
        "Bi at -MARCH_MAX",
        bi,
        -MARCH_MAX,
        bi_prime(-MARCH_MAX),
        1e-13,
    );
    // The scaled forms are the only ones still finite at `LEADING_MIN`.
    no_step_across(
        "Bie at LEADING_MIN",
        bie,
        LEADING_MIN,
        // `d/dx[1/(√π x^{1/4})]`, the leading form's own slope.
        -0.25 / (core::f64::consts::PI.sqrt() * LEADING_MIN.powf(1.25)),
        1e-15,
    );
}

#[test]
fn the_series_and_the_k_quadrature_agree_across_their_overlap() {
    // Both routes are good on `[0.3, 1]` — the series has not yet lost anything to the
    // `c₁f − c₂g` cancellation and the quadrature's node count is still modest. Disagreement
    // here would mean one of them is wrong in a way no single-route test could localize.
    for x in sweep(0.3, AI_SERIES_MAX, 40) {
        let z = zeta(x);
        let (series_ai, series_ai_prime, _, _) = series::airy(x);
        let (quad_ai, quad_ai_prime) = ai_pair_scaled(x, z);
        let decay = (-z).exp();
        assert_close(
            quad_ai * decay,
            series_ai,
            2e-15,
            "Ai: quadrature vs series",
        );
        assert_close(
            quad_ai_prime * decay,
            series_ai_prime,
            2e-15,
            "Ai': quadrature vs series",
        );
    }
}

#[test]
fn the_series_and_the_i_asymptotic_agree_across_their_overlap() {
    // `Bi`'s overlap is far wider than `Ai`'s, because the series is a sum of positive terms
    // here and loses nothing: it is still exact at x = 20, while the asymptotic is at machine
    // precision from about x = 9. Both are checked over the whole band.
    for x in sweep(BI_SERIES_MAX, 20.0, 40) {
        let z = zeta(x);
        let (_, _, series_bi, series_bi_prime) = series::airy(x);
        let (asym_bi, asym_bi_prime) = bi_pair_scaled(x, z);
        let decay = (-z).exp();
        // The bound carries `ζ·ε`, because *this comparison* is what introduces it: the series
        // produces the unscaled value, so reaching the scaled one costs a multiplication by
        // `e^{−ζ}`, whose relative error is `ζ·ε` from the rounding of `ζ` alone. At x = 20
        // that is 2.6e-14 and the measured disagreement is 1.5e-14; at x = 9 the bound tightens
        // to 1e-14. Writing it as a function of `ζ` rather than as a flat worst case keeps the
        // small-x end of the sweep as strict as it can be.
        let bound = 2e-15 + 2.0 * z * f64::EPSILON;
        assert_close(
            asym_bi,
            series_bi * decay,
            bound,
            "Bi: asymptotic vs series",
        );
        assert_close(
            asym_bi_prime,
            series_bi_prime * decay,
            bound,
            "Bi': asymptotic vs series",
        );
    }
}

#[test]
fn the_march_agrees_with_both_routes_it_joins() {
    // The march is the one route with no independent closed form behind it, so it is pinned
    // from both ends: against the series where the series is still exact, and against the
    // Hankel expansion where that has become so. Nothing else constrains the band between.
    for x in sweep(-SERIES_MAX, -0.5, 25) {
        let marched = march::airy(x);
        let (s_ai, s_ai_prime, s_bi, s_bi_prime) = series::airy(x);
        for (name, got, want) in [
            ("Ai", marched.0, s_ai),
            ("Ai'", marched.1, s_ai_prime),
            ("Bi", marched.2, s_bi),
            ("Bi'", marched.3, s_bi_prime),
        ] {
            assert_close(got, want, 1e-14, &format!("{name}: march vs series"));
        }
    }
    // At the far end the comparison has to be absolute: both routes are accurate to ~1e-15
    // *absolute* there, and the functions pass through zero inside this range.
    for x in sweep(-14.0, -MARCH_MAX, 25) {
        let marched = march::airy(x);
        let hankel = negative_asymptotic(-x);
        for (name, got, want) in [
            ("Ai", marched.0, hankel.0),
            ("Ai'", marched.1, hankel.1),
            ("Bi", marched.2, hankel.2),
            ("Bi'", marched.3, hankel.3),
        ] {
            assert!(
                (got - want).abs() < 1e-12,
                "{name}: march {got:e} vs hankel {want:e} at x = {x}"
            );
        }
    }
}

#[test]
fn the_leading_order_branch_matches_the_route_it_replaces() {
    // `LEADING_MIN` is where the first asymptotic correction `u₁/ζ = (5/72)/ζ` drops below
    // rounding. Just *under* the threshold the full Bessel route still runs, so the two can be
    // compared directly: if the threshold were set too low, this would catch it.
    for x in [LEADING_MIN * 0.5, LEADING_MIN * 0.9, LEADING_MIN] {
        let full = positive_scaled(x, zeta(x));
        let leading = leading_scaled(x);
        for (name, got, want) in [
            ("Aie", leading.0, full.0),
            ("Aie'", leading.1, full.1),
            ("Bie", leading.2, full.2),
            ("Bie'", leading.3, full.3),
        ] {
            assert!(
                rel(got, want) < 1e-15,
                "{name}: leading {got:e} vs full {want:e} at x = {x:e}"
            );
        }
    }
}

#[test]
fn the_quadrature_is_reached_with_a_usable_zeta_everywhere_it_is_used() {
    // `ζ = ⅔x^{3/2}` is what every positive-axis route is parameterized by, and two things can
    // go wrong with it: it underflows to zero for tiny `x`, which would leave the quadrature
    // integrating nothing, and it overflows for `x > 3e205`. Both are avoided by thresholds
    // rather than by luck, so both are asserted.
    for x in [AI_SERIES_MAX, 1.5, 10.0, 1e3, 1e6, LEADING_MIN * 0.99] {
        let z = zeta(x);
        assert!(
            z.is_finite() && z > 0.0,
            "zeta({x:e}) = {z:e} is unusable, yet a Bessel route runs here"
        );
    }
    // Below the series threshold `ζ` may be anything at all, including zero — nothing uses it.
    assert_eq!(zeta(0.0), 0.0);
    assert!(zeta(1e-300) == 0.0 || zeta(1e-300) > 0.0);
}
