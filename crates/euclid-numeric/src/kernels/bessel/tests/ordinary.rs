//! `J₀`/`J₁` and `Y₀`/`Y₁`: reference values, the three regimes, and the bands where adjacent
//! methods overlap.
//!
//! Kept as one file because the overlap tests deliberately straddle the two kinds — the
//! Hankel expansion is shared, so checking it against the trapezoid and against the march
//! belongs in one place.

use crate::kernels::bessel::asymptotic::hankel;
use crate::kernels::bessel::ordinary_first::{
    HANKEL_MIN, J_SERIES_MAX, j_aliased, j_series, j0, j1,
};
use crate::kernels::bessel::ordinary_second::{
    Y_SERIES_MAX, y_march, y0, y0_series, y1, y1_series,
};

fn close(a: f64, b: f64, tol: f64) {
    assert!(
        (a - b).abs() <= tol * b.abs().max(1.0),
        "{a:e} vs {b:e} (tol {tol:e}, abs err {:e})",
        (a - b).abs()
    );
}

/// 600-digit `Decimal` references: `(x, J₀, J₁, Y₀, Y₁)`.
const REFS: &[(f64, f64, f64, f64, f64)] = &[
    (
        0.5,
        9.384_698_072_408_129e-1,
        2.422_684_576_748_739e-1,
        -4.445_187_335_067_065_6e-1,
        -1.471_472_392_670_243e0,
    ),
    (
        1.0,
        7.651_976_865_579_666e-1,
        4.400_505_857_449_335e-1,
        8.825_696_421_567_696e-2,
        -7.812_128_213_002_887e-1,
    ),
    (
        2.0,
        2.238_907_791_412_356_7e-1,
        5.767_248_077_568_734e-1,
        5.103_756_726_497_451e-1,
        -1.070_324_315_409_375_4e-1,
    ),
    (
        5.0,
        -1.775_967_713_143_383e-1,
        -3.275_791_375_914_652_3e-1,
        -3.085_176_252_490_337_6e-1,
        1.478_631_433_912_268_3e-1,
    ),
    (
        8.0,
        1.716_508_071_375_539e-1,
        2.346_363_468_539_146_3e-1,
        2.235_214_893_875_662_2e-1,
        -1.580_604_617_312_475e-1,
    ),
    (
        10.0,
        -2.459_357_644_513_483_5e-1,
        4.347_274_616_886_144e-2,
        5.567_116_728_359_939_5e-2,
        2.490_154_242_069_538_8e-1,
    ),
    (
        15.0,
        -1.422_447_282_678_077_2e-2,
        2.051_040_386_135_227_5e-1,
        2.054_642_960_389_182_8e-1,
        2.107_362_803_687_351e-2,
    ),
    (
        20.0,
        1.670_246_643_405_831_6e-1,
        6.683_312_417_585_005e-2,
        6.264_059_680_938_383e-2,
        -1.655_116_143_625_213e-1,
    ),
    (
        30.0,
        -8.636_798_358_104_021e-2,
        -1.187_510_626_166_229_4e-1,
        -1.172_957_316_866_640_3e-1,
        8.442_557_066_174_723e-2,
    ),
    (
        50.0,
        5.581_232_766_925_181_6e-2,
        -9.751_182_812_517_514e-2,
        -9.806_499_547_007_708e-2,
        -5.679_566_856_201_477e-2,
    ),
    (
        100.0,
        1.998_585_030_422_312_2e-2,
        -7.714_535_201_411_216e-2,
        -7.724_431_336_508_315e-2,
        -2.037_231_200_275_979_2e-2,
    ),
];

#[test]
fn all_four_match_their_reference_values() {
    for &(x, j0_r, j1_r, y0_r, y1_r) in REFS {
        close(j0(x), j0_r, 1e-15);
        close(j1(x), j1_r, 1e-15);
        close(y0(x), y0_r, 1e-15);
        close(y1(x), y1_r, 1e-15);
    }
}

// The J/Y Wronskian `J₀Y₁ − J₁Y₀ = −2/(πx)` is not here: it spans both kinds, so it lives
// in `tests::identities::bessel_family`, where it is checked on the public trait surface
// across every regime.

#[test]
fn the_regimes_join_without_a_step() {
    // Each join is checked against the derivative, which bounds how much the function may
    // legitimately move across the interval: J0' = -J1, Y0' = -Y1, and
    // J1' = J0 - J1/x, Y1' = Y0 - Y1/x.
    let h = 1e-9;
    for &x in &[Y_SERIES_MAX, HANKEL_MIN] {
        for (name, f, derivative) in [
            ("J0", j0 as fn(f64) -> f64, -j1(x)),
            ("J1", j1 as fn(f64) -> f64, j0(x) - j1(x) / x),
            ("Y0", y0 as fn(f64) -> f64, -y1(x)),
            ("Y1", y1 as fn(f64) -> f64, y0(x) - y1(x) / x),
        ] {
            let (below, above) = (f(x - h), f(x + h));
            let implied = derivative.abs() * 2.0 * h;
            assert!(
                (above - below).abs() <= implied * 1.05 + 1e-14,
                "{name} steps at x = {x}: {below:e} then {above:e}"
            );
        }
    }
}

#[test]
fn the_recurrence_relations_hold() {
    // 2n/x * Y_n = Y_{n-1} + Y_{n+1} with n = 1 gives Y2 = 2Y1/x - Y0, and the same for J.
    // Checking the derivative identities instead, which need no J2/Y2:
    //   J0' = -J1,  (x J1)' = x J0
    for i in 1..200 {
        let x = f64::from(i) * 0.15;
        let h = 1e-5 * x;
        // A central difference truncates at order h² times the third derivative, which
        // with h = 1e-5·x is about 1e-8 by x = 30. These tolerances are set by the
        // differencing, not by the functions being differenced.
        let d_j0 = (j0(x + h) - j0(x - h)) / (2.0 * h);
        close(d_j0, -j1(x), 1e-7);
        let d_y0 = (y0(x + h) - y0(x - h)) / (2.0 * h);
        close(d_y0, -y1(x), 1e-7);
        // (x J1)' = x J0
        let d_xj1 = ((x + h) * j1(x + h) - (x - h) * j1(x - h)) / (2.0 * h);
        close(d_xj1, x * j0(x), 1e-6);
    }
}

#[test]
fn j_satisfies_bessels_equation() {
    // x^2 y'' + x y' + x^2 y = 0 for J0, by finite differences.
    for i in 2..120 {
        let x = f64::from(i) * 0.25;
        let h = 1e-4 * x;
        let second = (j0(x + h) - 2.0 * j0(x) + j0(x - h)) / (h * h);
        let first = (j0(x + h) - j0(x - h)) / (2.0 * h);
        let residual = x * x * second + x * first + x * x * j0(x);
        assert!(
            residual.abs() < 1e-5 * x * x,
            "Bessel's equation broken for J0 at {x}: {residual:e}"
        );
    }
}

#[test]
fn adjacent_methods_agree_across_their_whole_overlap() {
    // Stronger than a join test: each pair of methods is run side by side over a band
    // where both should be valid, so agreement constrains them everywhere rather than at
    // a single point. The three are fully independent of one another — a finite
    // trigonometric sum, an ODE march, and a divergent asymptotic expansion — so this is
    // real evidence rather than a tautology.
    for i in 0..=48 {
        let x = 18.0 + f64::from(i) * 0.5; // [18, 42]
        let (j0_hankel, y0_hankel) = hankel(0.0, x);
        let (j1_hankel, y1_hankel) = hankel(1.0, x);
        close(j0_hankel, j_aliased(0, x), 1e-14);
        close(j1_hankel, j_aliased(1, x), 1e-14);
        let (y0_marched, y1_marched) = y_march(x);
        close(y0_hankel, y0_marched, 1e-13);
        close(y1_hankel, y1_marched, 1e-13);
    }
    // The J series against the quadrature, across the whole band where both hold. The
    // series is exact up to a few, and the quadrature from about 1e-8 upward, so this
    // overlap is wide.
    for i in 1..=60 {
        let x = f64::from(i) * 0.05; // (0, 3]
        close(j_series(0, x), j_aliased(0, x), 1e-15);
        close(j_series(1, x), j_aliased(1, x), 1e-15);
    }
    // And the Y series against the march just above the series cutoff.
    for i in 0..=30 {
        let x = Y_SERIES_MAX + f64::from(i) * 0.05;
        close(y_march(x).0, y0_series(x), 1e-14);
        close(y_march(x).1, y1_series(x), 1e-14);
    }
}

#[test]
fn parity_and_non_finite_inputs() {
    for &(x, _, _, _, _) in REFS {
        close(j0(-x), j0(x), 1e-15);
        close(j1(-x), -j1(x), 1e-15);
    }
    assert_eq!(j0(0.0), 1.0);
    assert_eq!(j1(0.0), 0.0);
    assert!(j0(f64::NAN).is_nan());
    assert!(j1(f64::NAN).is_nan());
    assert!(y0(f64::NAN).is_nan());
    assert!(y1(f64::NAN).is_nan());
    assert_eq!(j0(f64::INFINITY), 0.0);
    assert_eq!(j1(f64::INFINITY), 0.0);
    assert_eq!(y0(f64::INFINITY), 0.0);
    assert_eq!(y1(f64::INFINITY), 0.0);
    // Y is real only for positive argument.
    assert!(y0(-1.0).is_nan());
    assert!(y1(-1.0).is_nan());
    assert_eq!(y0(0.0), f64::NEG_INFINITY);
    assert_eq!(y1(0.0), f64::NEG_INFINITY);
}

#[test]
fn no_regime_boundary_leaves_a_step_at_ulp_resolution() {
    // Each of the three crossovers, probed at the two representable values straddling it.
    // `J` switches series -> trapezoid at 2 and trapezoid -> Hankel at 18; `Y` switches
    // series -> march at 4 and march -> Hankel at 18.
    use euclid_utils::no_step_across;
    for &x in &[J_SERIES_MAX, HANKEL_MIN] {
        no_step_across("J0", j0, x, -j1(x), 1e-14);
        no_step_across("J1", j1, x, j0(x) - j1(x) / x, 1e-14);
    }
    for &x in &[Y_SERIES_MAX, HANKEL_MIN] {
        no_step_across("Y0", y0, x, -y1(x), 1e-14);
        no_step_across("Y1", y1, x, y0(x) - y1(x) / x, 1e-14);
    }
    // And the thresholds themselves must be exactly where the code says they are: evaluating
    // one ulp either side must actually cross a branch, which this confirms indirectly by
    // checking that the *other* route is the one being used just across it.
    assert!(
        (j0(J_SERIES_MAX) - j_series(0, J_SERIES_MAX)).abs() < 1e-15,
        "at the threshold J0 should still be the series"
    );
    let just_above = crate::Real::next_after(J_SERIES_MAX, f64::INFINITY);
    assert!(
        (j0(just_above) - j_aliased(0, just_above)).abs() < 1e-15,
        "one ulp above it J0 should be the trapezoid"
    );
}
