//! `J₀`/`J₁` and `Y₀`/`Y₁`: the reference values, and the domain's edges.
//!
//! Split into sub-topics because the family's tests passed this crate's 400-line budget. What
//! stays here is what compares a *value* against an outside number — the 600-digit reference
//! table, and the parity and non-finite laws that need no reference at all. The rest divides by
//! the question it asks:
//!
//! | module | question |
//! |--------|----------|
//! | [`joins`] | do two adjacent methods agree where both are valid? |
//! | [`routes`] | does each argument reach the method the table says? |
//! | [`equations`] | does the output satisfy the differential equation it solves? |
//! | [`stopping`] | how many terms did it take, and does the cap hold? |
//!
//! That last one is a whole file because a stopping rule's *cost* is invisible to every
//! value assertion here: the tolerances sit below `f64`'s resolution, so the only observable is
//! the term count. See [`crate::math::pair::Converged`].
//!
//! `close` lives here and the children reach it through `super`, since three of them compare
//! values at a relative tolerance and a second copy would be a second tolerance.

mod equations;
mod joins;
mod routes;
mod stopping;

use crate::kernels::bessel::ordinary_first::{j0, j1};
use crate::kernels::bessel::ordinary_second::{y0, y1};

/// Relative comparison, falling back to absolute below one — the Bessel functions pass through
/// zero, where a purely relative tolerance is meaningless.
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
