//! Unit tests for the uniform asymptotic expansion.
//!
//! A child module so the parent stays inside the crate's 400-line budget. What these reach is
//! private to [`super`] — the band predicate, the tail enum and the coefficient table — and the
//! expansion's *accuracy* is guarded from outside by
//! `the_incomplete_gamma_matches_the_poisson_sum_at_integer_shape`, whose shapes and arguments
//! now land in this band by construction.

use super::*;

/// The band, at and either side of both of its edges.
#[test]
fn the_band_covers_large_shapes_near_their_mean_and_nothing_else() {
    use TemmeTail::{Lower, Upper};

    // Shape: the expansion needs `μ(a)`, which is only valid from `STIRLING_MIN`.
    assert!(use_temme(TEMME_MIN_SHAPE, TEMME_MIN_SHAPE));
    assert!(!use_temme(
        TEMME_MIN_SHAPE.next_down(),
        TEMME_MIN_SHAPE.next_down()
    ));

    // Offset: the edge belongs to the expansion, and both signs route alike.
    for a in [20.0_f64, 100.0, 1e6] {
        assert!(use_temme(a, a));
        assert!(use_temme(a, a * (1.0 + TEMME_MAX_OFFSET)));
        assert!(use_temme(a, a * (1.0 - TEMME_MAX_OFFSET)));
        assert!(!use_temme(a, a * (1.0 + TEMME_MAX_OFFSET) + a * 1e-9));
        assert!(!use_temme(a, a * (1.0 - TEMME_MAX_OFFSET) - a * 1e-9));
    }

    // The tail split is `x` against `a`, and it is what keeps the small side from being formed
    // by a subtraction.
    assert_eq!(TemmeTail::for_arguments(100.0, 100.0), Upper);
    assert_eq!(TemmeTail::for_arguments(100.0, 110.0), Upper);
    assert_eq!(TemmeTail::for_arguments(100.0, 99.0), Lower);
}

/// The expansion agrees with the series it replaced, on the edge where both are sound.
///
/// Reference-free, and the only place the comparison means anything: at `|σ| = 0.4` the series
/// is short — its term ratio is `x/(a+k)`, so it converges geometrically away from `x = a` — and
/// the expansion is at the widest `η` its table was sized for. Further in, the series is the one
/// that is wrong, by the `√a` accumulation this route exists to remove, so agreement there would
/// prove nothing. It is the same argument `erf_inv`'s two Newton numerators are compared under.
///
/// # Both routes must be asked about the *same* `x`
///
/// The first version nudged the argument across the band edge so the router would pick the
/// other arm, and reported `5.6e7` ulp at `a = 20`. That was the nudge, not a disagreement:
/// `P(20, ·)` has a logarithmic derivative of order ten there, so moving `x` by `1e-9` relative
/// moves `P` by `1e-8`. Calling each route directly at one `x` is the only comparison that says
/// anything — the same trap as feeding `erf_inv`'s two routes their own complements.
///
/// Measured worst over the shapes below: `4` ulp.
#[test]
fn the_expansion_agrees_with_the_series_at_the_edge_of_the_band() {
    let mut worst = (0.0_f64, 0.0_f64);
    for &a in &[20.0_f64, 35.0, 60.0, 100.0, 250.0, 1000.0] {
        // The lower edge, where the series route is the lower series and `x < a + 1` holds.
        let x = a * (1.0 - TEMME_MAX_OFFSET);
        let series = crate::math::gamma::gamma_lower_series(a, x, f64::EPSILON, 100_000);
        let by_series = crate::math::gamma::log_scaled_gamma(a, x, series.ln()).exp();
        let by_expansion = temme_lower(a, x);

        let ulps = (by_expansion - by_series).abs() / crate::Real::ulp(by_expansion);
        if ulps > worst.0 {
            worst = (ulps, a);
        }
        assert!(
            ulps <= 24.0,
            "at a={a}, x={x} the expansion gives {by_expansion} and the series {by_series}, \
             {ulps} ulp apart"
        );
    }
    assert!(
        worst.0 > 0.0,
        "the two routes were bit-identical everywhere, so this compared nothing"
    );
}

/// The coefficient table is the one the derivation produced.
///
/// Pins the two rows that have an independent closed form. `d_0 = 1/σ − 1/η`, expanded by hand,
/// is `−1/3 + η/12 − 2η²/135 + η³/864`; a mis-transcribed table is otherwise a smooth plausible
/// function, which is the failure mode a value assertion on `gamma_lr` would not localise.
#[test]
fn the_coefficient_table_matches_the_derivation() {
    let expected = [
        -1.0 / 3.0,
        1.0 / 12.0,
        -2.0 / 135.0,
        1.0 / 864.0,
        1.0 / 2835.0,
    ];
    for (index, &want) in expected.iter().enumerate() {
        let got = TEMME_D[0][index];
        assert!(
            (got - want).abs() <= 4.0 * crate::Real::ulp(want),
            "d_0[{index}] is {got}, derived value {want}"
        );
    }
    // Every row is non-empty and starts at the order its weight requires: `d_k/a^k` with
    // `a ≥ 20`, so the last row needs far fewer terms than the first.
    assert_eq!(TEMME_D.len(), 10);
    assert!(TEMME_D[0].len() > TEMME_D[9].len());
    assert!(TEMME_D.iter().all(|row| !row.is_empty()));
}
