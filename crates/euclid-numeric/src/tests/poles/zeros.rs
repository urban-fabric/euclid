//! Zeros: the arguments where the functions vanish, and the sign structure between them.
//!
//! Part of the [poles suite](super); see there for what these tests are for and why
//! accuracy at a zero is asserted in absolute rather than relative terms.

use crate::Real;
use euclid_utils::assert_close;

/// The first several zeros of `J₀`, `J₁`, `Y₀` and `Y₁`, each verified to 25 digits by an
/// independent high-precision evaluation of the defining series.
const J0_ZEROS: &[f64] = &[
    2.404_825_557_695_773e0,
    5.520_078_110_286_311e0,
    8.653_727_912_911_013e0,
    1.179_153_443_901_428_1e1,
    1.493_091_770_848_778_5e1,
];
const J1_ZEROS: &[f64] = &[
    3.831_705_970_207_512_5e0,
    7.015_586_669_815_619e0,
    1.017_346_813_506_272_2e1,
    1.332_369_193_631_422_3e1,
];
const Y0_ZEROS: &[f64] = &[
    8.935_769_662_791_675e-1,
    3.957_678_419_314_858e0,
    7.086_051_060_301_773e0,
    1.022_234_504_349_641_8e1,
];
const Y1_ZEROS: &[f64] = &[
    2.197_141_326_031_017e0,
    5.429_681_040_794_135e0,
    8.596_005_868_331_169e0,
    1.174_915_483_083_988e1,
];

// --- the zeros ----------------------------------------------------------

#[test]
fn the_oscillatory_functions_vanish_at_their_known_zeros() {
    // Absolute smallness, for the reason given in the module header: at a zero the input
    // rounding dominates. The derivative at a zero is O(0.3), and the argument carries an
    // uncertainty of eps*x ~ 3e-15 at x = 15, so ~1e-15 is the floor here however the
    // function is computed.
    for (name, zeros, f) in [
        ("J0", J0_ZEROS, Real::bessel_j0 as fn(f64) -> f64),
        ("J1", J1_ZEROS, Real::bessel_j1 as fn(f64) -> f64),
        ("Y0", Y0_ZEROS, Real::bessel_y0 as fn(f64) -> f64),
        ("Y1", Y1_ZEROS, Real::bessel_y1 as fn(f64) -> f64),
    ] {
        for &zero in zeros {
            let value = f(zero);
            assert!(
                value.abs() < 1e-14,
                "{name} should vanish at {zero}, got {value:e}"
            );
            // The sign must change across it, which no amount of smallness guarantees.
            let step = 1e-6;
            assert!(
                f(zero - step) * f(zero + step) < 0.0,
                "{name} does not cross zero at {zero}"
            );
        }
    }
}

#[test]
fn consecutive_zeros_bracket_exactly_one_crossing() {
    // A structural check that survives at any precision: between two consecutive zeros the
    // function keeps one sign, so a spurious oscillation -- the signature of a bad asymptotic
    // expansion or a mis-joined branch -- is caught even where values are tiny.
    for (name, zeros, f) in [
        ("J0", J0_ZEROS, Real::bessel_j0 as fn(f64) -> f64),
        ("J1", J1_ZEROS, Real::bessel_j1 as fn(f64) -> f64),
        ("Y0", Y0_ZEROS, Real::bessel_y0 as fn(f64) -> f64),
        ("Y1", Y1_ZEROS, Real::bessel_y1 as fn(f64) -> f64),
    ] {
        for pair in zeros.windows(2) {
            let (lo, hi) = (pair[0], pair[1]);
            let mid_sign = f((lo + hi) * 0.5).signum();
            for i in 1..40 {
                let x = lo + (hi - lo) * f64::from(i) / 40.0;
                assert_eq!(
                    f(x).signum(),
                    mid_sign,
                    "{name} changes sign inside ({lo}, {hi}) at {x}"
                );
            }
        }
    }
}

#[test]
fn ci_and_ei_vanish_at_their_known_roots() {
    // Ci's first zero and Ei's only positive zero, both verified to 25 digits independently.
    let ci_zero = 6.165_054_856_207_162e-1;
    assert!(
        Real::cos_int(ci_zero).abs() < 1e-15,
        "Ci should vanish at {ci_zero}, got {:e}",
        Real::cos_int(ci_zero)
    );
    assert!(Real::cos_int(ci_zero - 1e-6) < 0.0 && Real::cos_int(ci_zero + 1e-6) > 0.0);

    let ei_zero = 3.725_074_107_813_668e-1;
    assert!(
        Real::exp_int_ei(ei_zero).abs() < 1e-15,
        "Ei should vanish at {ei_zero}, got {:e}",
        Real::exp_int_ei(ei_zero)
    );
    assert!(Real::exp_int_ei(ei_zero - 1e-6) < 0.0 && Real::exp_int_ei(ei_zero + 1e-6) > 0.0);
}

#[test]
fn zeta_is_exactly_zero_at_the_trivial_zeros() {
    // Returned as exact zeros rather than left to the sin(pi s/2) factor to cancel, which it
    // would not do exactly.
    for k in 1..40 {
        let s = -2.0 * f64::from(k);
        assert_eq!(Real::zeta(s), 0.0, "zeta({s}) must be exactly zero");
    }
    // But *not* zero at the negative odd integers, where the Bernoulli values live.
    assert_close(Real::zeta(-1.0), -1.0 / 12.0, 1e-12, "zeta(-1) = -1/12");
    assert_close(Real::zeta(-3.0), 1.0 / 120.0, 1e-11, "zeta(-3) = 1/120");
    assert_close(Real::zeta(-5.0), -1.0 / 252.0, 1e-11, "zeta(-5) = -1/252");
}
