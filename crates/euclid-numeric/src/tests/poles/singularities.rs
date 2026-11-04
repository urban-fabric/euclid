//! Poles: where the functions diverge, and how they diverge.
//!
//! Part of the [poles suite](super); see there for what these tests are for and why
//! accuracy at a zero is asserted in absolute rather than relative terms.

use crate::Real;
use euclid_utils::assert_close;

// --- the logarithmic and simple poles -----------------------------------

#[test]
fn the_second_kind_bessels_have_a_logarithmic_pole_at_the_origin() {
    assert_eq!(Real::bessel_y0(0.0), f64::NEG_INFINITY);
    assert_eq!(Real::bessel_y1(0.0), f64::NEG_INFINITY);
    assert_eq!(Real::bessel_k0(0.0), f64::INFINITY);
    assert_eq!(Real::bessel_k1(0.0), f64::INFINITY);

    // Approaching it: Y0 ~ (2/pi)(ln(x/2) + gamma), K0 ~ -ln(x/2) - gamma, and both Y1 and K1
    // have a *simple* pole rather than a logarithmic one -- Y1 ~ -2/(pi x), K1 ~ 1/x.
    //
    // Only arguments where the leading term is itself accurate to 1e-12 are used: the next
    // correction is O(x^2 ln x), so this starts at 1e-7 rather than at 0.1.
    for &x in &[1e-7f64, 1e-12, 1e-30, 1e-100, 1e-200, 1e-300] {
        let log_term = (x * 0.5).ln() + euclid_utils::EULER_GAMMA;
        assert_close(
            Real::bessel_y0(x) / (2.0 / core::f64::consts::PI * log_term),
            1.0,
            1e-12,
            &format!("Y0({x}) ~ (2/pi)(ln(x/2) + gamma)"),
        );
        assert_close(
            Real::bessel_k0(x) / -log_term,
            1.0,
            1e-12,
            &format!("K0({x}) ~ -ln(x/2) - gamma"),
        );
        assert_close(
            Real::bessel_y1(x) / (-2.0 / (core::f64::consts::PI * x)),
            1.0,
            1e-12,
            &format!("Y1({x}) ~ -2/(pi x)"),
        );
        assert_close(
            Real::bessel_k1(x) * x,
            1.0,
            1e-12,
            &format!("K1({x}) ~ 1/x"),
        );
    }
}

#[test]
fn the_integrals_have_the_poles_their_definitions_imply() {
    // Ci and Ei both diverge to -inf at the origin, E1 to +inf, and Si is simply zero there.
    assert_eq!(Real::cos_int(0.0), f64::NEG_INFINITY);
    assert_eq!(Real::exp_int_ei(0.0), f64::NEG_INFINITY);
    assert_eq!(Real::exp_int_e1(0.0), f64::INFINITY);
    assert_eq!(Real::sin_int(0.0), 0.0);
    // Si is odd, so it must carry the sign of a signed zero through.
    assert!(
        Real::sin_int(-0.0f64).is_sign_negative(),
        "Si(-0) must be -0"
    );

    // Approaching: Ci ~ gamma + ln x, Ei ~ gamma + ln x, E1 ~ -gamma - ln x, Si ~ x.
    //
    // Starting at 1e-12 rather than higher because the next term of both Ei and E1 is `±x`,
    // which is a *relative* correction of x/|gamma + ln x|: at 1e-7 that is 6.4e-9, so a
    // tighter sweep would be measuring the truncated reference rather than the implementation.
    // At 1e-12 it is 3.7e-14, comfortably under the tolerance.
    for &x in &[1e-12f64, 1e-50, 1e-150, 1e-300] {
        let log_term = euclid_utils::EULER_GAMMA + x.ln();
        assert_close(
            Real::cos_int(x) / log_term,
            1.0,
            1e-12,
            &format!("Ci({x}) ~ gamma + ln x"),
        );
        assert_close(
            Real::exp_int_ei(x) / log_term,
            1.0,
            1e-12,
            &format!("Ei({x}) ~ gamma + ln x"),
        );
        assert_close(
            Real::exp_int_e1(x) / -log_term,
            1.0,
            1e-12,
            &format!("E1({x}) ~ -gamma - ln x"),
        );
        assert_close(Real::sin_int(x) / x, 1.0, 1e-12, &format!("Si({x}) ~ x"));
    }
}

#[test]
fn zeta_has_a_simple_pole_at_one_and_approaches_it_from_both_sides() {
    assert_eq!(Real::zeta(1.0), f64::INFINITY, "the pole itself");
    // zeta(s) ~ 1/(s-1) + gamma, which is what distinguishes a correct implementation from one
    // that merely returns something large. Checked from *both* sides, which also means the
    // critical-strip route and the Euler-Maclaurin route are both exercised against the same
    // local form.
    for &delta in &[1e-3f64, 1e-5, 1e-7, 1e-10] {
        for signed in [delta, -delta] {
            let s = 1.0 + signed;
            let expected = 1.0 / signed + euclid_utils::EULER_GAMMA;
            assert_close(
                Real::zeta(s) / expected,
                1.0,
                1e-4,
                &format!("zeta({s}) ~ 1/(s-1) + gamma"),
            );
        }
    }
    // The divergence has the right sign on each side.
    assert!(Real::zeta(1.0 + 1e-9) > 1e8);
    assert!(Real::zeta(1.0 - 1e-9) < -1e8);
}
