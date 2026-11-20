//! Where two routes meet, checked from both sides.
//!
//! Split from the accuracy tests because a join is the one place a multi-route function can be
//! wrong while every route is right. Each of these evaluates the *same argument* through both
//! arms and compares them to each other rather than to a reference — the only instrument that
//! sees a moved or mis-ordered crossover.

use super::super::march;
use super::super::regimes::{negative_asymptotic, positive_scaled};
use super::super::{
    AI_SERIES_MAX, BI_SERIES_MAX, LEADING_MIN, MARCH_MAX, SERIES_MAX, ai_pair_scaled,
    bi_pair_scaled, series, zeta,
};
use euclid_utils::{assert_close, rel, sweep};

#[test]
fn the_series_and_the_k_quadrature_agree_across_their_overlap() {
    // Both routes are good on `[0.3, 1]` — the series has not yet lost anything to the
    // `c₁f − c₂g` cancellation and the quadrature's node count is still modest. Disagreement
    // here would mean one of them is wrong in a way no single-route test could localize.
    for x in sweep(0.3, AI_SERIES_MAX, 40) {
        let z = zeta(x);
        let (series_ai, series_ai_prime, _, _) = series::airy(x);
        let (quad_ai, quad_ai_prime) = ai_pair_scaled(x, z).into_parts();
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
        let (asym_bi, asym_bi_prime) = bi_pair_scaled(x, z).into_parts();
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
        let leading = crate::math::airy::leading_scaled(x);
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

/// `positive_scaled`'s crossover at `BI_SERIES_MAX`, where both routes are correct.
///
/// Relaxing `x > BI_SERIES_MAX` to `>=` changes which route evaluates `x = 9` exactly. Both are
/// accurate there — that is what makes a crossover a free choice — so the mutation is equivalent
/// to within the agreement of the two branches, which this measures rather than assumes.
#[test]
fn the_positive_crossover_agrees_from_both_sides() {
    let x = super::BI_SERIES_MAX;
    let z = super::zeta(x);
    let (ai_a, aip_a, bi_a, bip_a) = positive_scaled(x, z);
    // The other side of the comparison: what the mutant would take at exactly this argument.
    let (ai_b, aip_b) = super::ai_pair_scaled(x, z).into_parts();
    let (bi_b, bip_b) = super::bi_pair_scaled(x, z).into_parts();

    for (name, from_series, from_asymptotic) in [
        ("Ai^e", ai_a, ai_b),
        ("Ai^e'", aip_a, aip_b),
        ("Bi^e", bi_a, bi_b),
        ("Bi^e'", bip_a, bip_b),
    ] {
        let relative = ((from_series - from_asymptotic) / from_asymptotic).abs();
        assert!(
            relative < 1e-13,
            "{name} must agree across the crossover at x = {x}: {from_series:e} against \
             {from_asymptotic:e}, relative {relative:e}"
        );
    }
}

/// The march's Taylor order carries more terms than `f64` can use, **over the domain it is
/// actually reached on**.
///
/// Three survivors live in `march::step`: the coefficient array's length `ORDER + 2` mutated to
/// `* 2`, and the Horner loop's bound `ORDER + 1` mutated to `* 1` and to `- 1`. The first is
/// slack — the array needs 26 entries and the mutation gives it 48, which are never written or
/// read. The other two drop the top one and two Taylor coefficients.
///
/// The previous pass filed those two as *probably* dead precision and declined to exclude them,
/// for a specific reason worth keeping: the truncation bound in `ORDER`'s own docstring,
/// `|h|^{N+1}/(N+1)!`, **ignores the coefficient growth** in
/// `c[n+2] = (x₀c[n] + c[n−1])/((n+2)(n+1))`, which rises with `|x₀|`. A bound that ignores the
/// thing that could break it is not a bound.
///
/// The missing step was the reachability one. `march::airy` marches from `−SERIES_MAX = −2`
/// outward to at most `MARCH_MAX = 9`, in steps of at most `MAX_STEP = 0.5`. So `|x₀| ≤ 9` and
/// `|h| ≤ 0.5` — never anything larger — and at order 24 the term is about
/// `9¹²/24! · 0.5²⁴ ≈ 3e-20`, which is four orders below an ulp of an `O(1)` value. The growth
/// is real and the factorial still wins by a wide margin at the *reachable* arguments.
///
/// Measured here rather than estimated, by transcribing `step` with the mutations under a
/// selector and sweeping the domain the march uses.
#[test]
fn the_marchs_top_taylor_terms_are_below_resolution_where_it_runs() {
    /// `0` is the real body; `1` shortens the Horner loop by one term, `2` by two.
    fn step_variant(x0: f64, y: f64, y_prime: f64, h: f64, mutation: u8) -> (f64, f64) {
        let order = super::march::ORDER;
        let mut c = [0.0f64; 64];
        c[0] = y;
        c[1] = y_prime;
        for n in 0..order {
            let previous = if n >= 1 { c[n - 1] } else { 0.0 };
            #[allow(clippy::cast_precision_loss)]
            let divisor = ((n + 2) * (n + 1)) as f64;
            c[n + 2] = (x0 * c[n] + previous) / divisor;
        }
        let top = match mutation {
            1 => order,
            2 => order - 1,
            _ => order + 1,
        };
        let mut value = 0.0f64;
        let mut derivative = 0.0f64;
        for n in (0..=top).rev() {
            value = value * h + c[n];
            if n >= 1 {
                #[allow(clippy::cast_precision_loss)]
                let coefficient = n as f64;
                derivative = derivative * h + coefficient * c[n];
            }
        }
        (value, derivative)
    }

    // The reachable domain, stated explicitly: the march never sees an `x0` outside this, nor a
    // step longer than `MAX_STEP`.
    // Checked at compile time, because the domain below is *derived* from these two
    // constants: if either moves, the reasoning in the docstring no longer applies and this
    // should stop building rather than silently measure the wrong range.
    const _: () = assert!(super::MARCH_MAX == 9.0, "the march's outer bound moved");
    const _: () = assert!(
        super::march::MAX_STEP == 0.5,
        "the march's step bound moved"
    );

    let mut worst_value = 0.0f64;
    let mut worst_derivative = 0.0f64;
    for i in 0..=120 {
        let x0 = -super::MARCH_MAX + 2.0 * super::MARCH_MAX * f64::from(i) / 120.0;
        for step_index in 0..=20 {
            let h = -super::march::MAX_STEP
                + 2.0 * super::march::MAX_STEP * f64::from(step_index) / 20.0;
            // Starting values of the magnitudes the march actually transports: the series' own
            // `f`/`g` pair at the join, and a few scalings of them.
            for &(y, y_prime) in &[(1.0f64, 0.0f64), (0.0, 1.0), (0.535, -0.73), (-2.1, 3.4)] {
                let real = step_variant(x0, y, y_prime, h, 0);
                for mutation in [1u8, 2] {
                    let mutated = step_variant(x0, y, y_prime, h, mutation);
                    let scale = real.0.abs().max(1e-300);
                    let scale_prime = real.1.abs().max(1e-300);
                    worst_value = worst_value.max((real.0 - mutated.0).abs() / scale);
                    worst_derivative =
                        worst_derivative.max((real.1 - mutated.1).abs() / scale_prime);
                }
            }
        }
    }
    // Below half an ulp means the dropped terms cannot change the rounded result.
    assert!(
        worst_value <= f64::EPSILON / 2.0 && worst_derivative <= f64::EPSILON / 2.0,
        "the top Taylor terms must be below half an ulp over the reachable domain: \
         value {worst_value:e}, derivative {worst_derivative:e}"
    );

    // And the array-length slack: 26 entries are written, so a longer array changes nothing.
    const _: () = assert!(
        super::march::ORDER + 2 == 26,
        "the coefficient array needs ORDER + 2 = 26 entries"
    );
}
