//! Identities of the error-function family.

use crate::Real;
use euclid_utils::assert_close;

// --- error function -----------------------------------------------------

#[test]
fn erf_plus_erfc_is_one() {
    for i in -60..60 {
        let x = f64::from(i) * 0.1;
        assert_close(
            Real::erf(x) + Real::erfc(x),
            1.0,
            1e-15,
            &format!("erf({x}) + erfc({x}) = 1"),
        );
    }
}

#[test]
fn erf_is_odd_and_erfc_reflects() {
    for i in 1..60 {
        let x = f64::from(i) * 0.1;
        // erf(−x) = −erf(x)
        assert_close(Real::erf(-x), -Real::erf(x), 1e-15, "erf is odd");
        // erfc(−x) = 2 − erfc(x)
        assert_close(
            Real::erfc(-x),
            2.0 - Real::erfc(x),
            1e-14,
            "erfc(-x) = 2 - erfc(x)",
        );
    }
}

#[test]
fn erfcx_is_the_scaled_erfc() {
    // erfcx(x) = e^{x²} erfc(x), checked where both sides are representable.
    for i in 0..27 {
        let x = f64::from(i) * 0.1;
        assert_close(
            Real::erfcx(x) * (-x * x).exp(),
            Real::erfc(x),
            1e-14,
            &format!("erfcx({x}) e^-x² = erfc({x})"),
        );
    }
}

#[test]
fn erf_inv_round_trips_through_erf() {
    for i in 1..50 {
        let x = f64::from(i) * 0.05;
        assert_close(
            Real::erf_inv(Real::erf(x)),
            x,
            1e-12,
            &format!("erf_inv(erf({x})) = {x}"),
        );
    }
    // And the other direction, on the probability scale.
    for i in 1..20 {
        let p = f64::from(i) * 0.05;
        assert_close(Real::erf(Real::erf_inv(p)), p, 1e-12, "erf(erf_inv(p)) = p");
    }
}

#[test]
fn erf_is_monotone_increasing() {
    let mut prev = f64::NEG_INFINITY;
    for i in -400..400 {
        let v = Real::erf(f64::from(i) * 0.05);
        assert!(v >= prev, "erf must be non-decreasing");
        prev = v;
    }
}
