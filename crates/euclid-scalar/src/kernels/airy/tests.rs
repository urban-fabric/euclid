//! Airy accuracy against independent references, and the origin constants.
//!
//! Split from [`joins`] so each file stays inside the crate's 400-line budget. What stays here
//! compares the functions to something outside this module: the series constants against the
//! crate's own `Γ`, the `Bi` asymptotic against its sum, and the whole axis against a sweep.

mod joins;
mod routes;

use super::march;
use super::series::SQRT_3;
use super::{
    AI_SERIES_MAX, BI_SERIES_MAX, LEADING_MIN, MARCH_MAX, SERIES_MAX, ai, ai_pair_scaled, ai_prime,
    bi, bi_pair_scaled, bi_prime, bie, zeta,
};
use crate::Real;
use crate::math::airy::{AI_0, NEG_AI_PRIME_0};
use euclid_utils::{assert_close, no_step_across};

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

/// The written connection formula and its simplified asymptotic agree bit for bit.
#[test]
fn bi_asymptotic_matches_the_sum_at_opposite_orders() {
    use crate::kernels::bessel::asymptotic::i_scaled;

    for x in [9.0_f64, 9.5, 12.0, 20.0, 50.0, 1e3, 1e8] {
        let z = super::zeta(x);
        let bi = (x / 3.0).sqrt() * (i_scaled(-1.0 / 3.0, z) + i_scaled(1.0 / 3.0, z));
        let bi_prime = (x / super::SQRT_3) * (i_scaled(-2.0 / 3.0, z) + i_scaled(2.0 / 3.0, z));
        let (actual, actual_prime) = super::bi_pair_scaled(x, z).into_parts();
        assert_eq!(actual.to_bits(), bi.to_bits());
        assert_eq!(actual_prime.to_bits(), bi_prime.to_bits());
    }
}

#[test]
fn airy_routes_choose_the_series_at_exact_cutoffs() {
    use super::regimes::PositiveRegime::{AiQuadrature, BothBessel, BothSeries};

    let ai = super::AI_SERIES_MAX;
    let bi = super::BI_SERIES_MAX;
    for (x, regime) in [
        (0.0, BothSeries),
        (ai.next_down(), BothSeries),
        (ai, BothSeries),
        (ai.next_up(), AiQuadrature),
        (bi.next_down(), AiQuadrature),
        (bi, AiQuadrature),
        (bi.next_up(), BothBessel),
    ] {
        assert_eq!(super::regimes::PositiveRegime::for_argument(x), regime);
    }

    assert!(super::negative_axis(-f64::from_bits(1)));
    assert!(!super::negative_axis(f64::from_bits(1)));
    for x in [0.0_f64, -0.0] {
        assert!(!super::negative_axis(x));
        assert_eq!(super::quad(x), super::series::airy(x));
        assert_eq!(super::quad_scaled(x), super::series::airy(x));
    }
}
