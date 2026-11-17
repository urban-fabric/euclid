//! Domain edges: out-of-domain arguments, NaN propagation, and the infinite limits.
//!
//! Part of the [poles suite](super); see there for what these tests are for and why
//! accuracy at a zero is asserted in absolute rather than relative terms.

use crate::Real;
use euclid_utils::assert_close;

// --- domain edges -------------------------------------------------------

#[test]
fn out_of_domain_arguments_give_nan_rather_than_a_plausible_number() {
    // Y, K, Ci and E1 are real only for positive argument. Returning NaN is the contract;
    // returning the value of some other branch would be worse than failing.
    for &x in &[
        -1e-300f64,
        -1e-8,
        -0.5,
        -1.0,
        -7.0,
        -1e10,
        f64::NEG_INFINITY,
    ] {
        assert!(Real::bessel_y0(x).is_nan(), "Y0({x}) must be NaN");
        assert!(Real::bessel_y1(x).is_nan(), "Y1({x}) must be NaN");
        assert!(Real::bessel_k0(x).is_nan(), "K0({x}) must be NaN");
        assert!(Real::bessel_k1(x).is_nan(), "K1({x}) must be NaN");
        assert!(Real::bessel_k0e(x).is_nan(), "K0e({x}) must be NaN");
        assert!(Real::bessel_k1e(x).is_nan(), "K1e({x}) must be NaN");
        assert!(Real::cos_int(x).is_nan(), "Ci({x}) must be NaN");
        assert!(Real::exp_int_e1(x).is_nan(), "E1({x}) must be NaN");
    }
    // The Lambert W's real branches end at -1/e; just below it there is no real value.
    let branch_point = -(1.0 / core::f64::consts::E);
    for &x in &[-0.4f64, -0.5, -1.0, -1e10] {
        assert!(Real::lambert_w(x).is_nan(), "W({x}) must be NaN");
    }
    assert!(
        Real::lambert_w(branch_point - 1e-15).is_nan()
            || Real::lambert_w(branch_point - 1e-15) >= -1.0,
        "just below the branch point is either NaN or still >= -1"
    );
}

#[test]
fn nan_propagates_everywhere() {
    let n = f64::NAN;
    assert!(Real::bessel_j0(n).is_nan());
    assert!(Real::bessel_j1(n).is_nan());
    assert!(Real::bessel_y0(n).is_nan());
    assert!(Real::bessel_y1(n).is_nan());
    assert!(Real::bessel_k0(n).is_nan());
    assert!(Real::bessel_k1(n).is_nan());
    assert!(Real::bessel_k0e(n).is_nan());
    assert!(Real::bessel_k1e(n).is_nan());
    assert!(Real::sin_int(n).is_nan());
    assert!(Real::cos_int(n).is_nan());
    assert!(Real::exp_int_e1(n).is_nan());
    assert!(Real::exp_int_ei(n).is_nan());
    assert!(Real::lambert_w(n).is_nan());
    assert!(Real::zeta(n).is_nan());
}

#[test]
fn the_infinite_limits_are_the_analytic_ones() {
    // Each of these is a limit, not a saturation artefact.
    assert_eq!(Real::bessel_j0(f64::INFINITY), 0.0);
    assert_eq!(Real::bessel_j1(f64::INFINITY), 0.0);
    assert_eq!(Real::bessel_j0(f64::NEG_INFINITY), 0.0);
    assert_eq!(Real::bessel_y0(f64::INFINITY), 0.0);
    assert_eq!(Real::bessel_y1(f64::INFINITY), 0.0);
    assert_eq!(Real::bessel_k0(f64::INFINITY), 0.0);
    assert_eq!(Real::bessel_k1(f64::INFINITY), 0.0);
    assert_eq!(Real::bessel_k0e(f64::INFINITY), 0.0);
    assert_eq!(Real::exp_int_e1(f64::INFINITY), 0.0);
    assert_eq!(Real::exp_int_ei(f64::INFINITY), f64::INFINITY);
    assert_eq!(Real::cos_int(f64::INFINITY), 0.0);
    // Si has a finite limit, which is the interesting one.
    assert_eq!(Real::sin_int(f64::INFINITY), core::f64::consts::FRAC_PI_2);
    assert_eq!(
        Real::sin_int(f64::NEG_INFINITY),
        -core::f64::consts::FRAC_PI_2
    );
    assert_eq!(Real::lambert_w(f64::INFINITY), f64::INFINITY);
    // zeta(s) -> 1 as s -> +inf, and has no limit as s -> -inf.
    assert_close(Real::zeta(f64::INFINITY), 1.0, 1e-15, "zeta(inf) = 1");
    assert_close(Real::zeta(300.0), 1.0, 1e-15, "zeta(300) = 1");
}
