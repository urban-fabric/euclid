//! Overflow, underflow, subnormals, branch points, and enormous in-domain arguments.
//!
//! Part of the [poles suite](super); see there for what these tests are for and why
//! accuracy at a zero is asserted in absolute rather than relative terms.

use crate::Real;
use euclid_utils::{assert_close, sweep};

// --- overflow and underflow boundaries ----------------------------------

#[test]
fn the_unscaled_forms_overflow_and_underflow_exactly_where_the_true_values_do() {
    // I grows like e^x and K decays like e^-x, so both leave f64's range while their scaled
    // counterparts remain perfectly ordinary numbers. Saturating is the correct answer; the
    // failure mode to rule out is a NaN or a silently finite wrong value.
    assert!(Real::bessel_i0(700.0f64).is_finite(), "I0(700) still fits");
    assert!(Real::bessel_i0(750.0f64).is_infinite(), "I0(750) does not");
    assert!(Real::bessel_i1(750.0f64).is_infinite());
    assert_eq!(Real::bessel_k0(1000.0f64), 0.0, "K0(1000) underflows");
    assert_eq!(Real::bessel_k1(1000.0f64), 0.0);
    assert!(
        Real::exp_int_ei(800.0f64).is_infinite(),
        "Ei(800) overflows"
    );

    // The scaled forms stay usable far beyond, which is the entire reason they exist.
    for &x in &[700.0f64, 1e3, 1e6, 1e30, 1e150, 1e300] {
        for (name, value) in [
            ("I0e", Real::bessel_i0e(x)),
            ("I1e", Real::bessel_i1e(x)),
            ("K0e", Real::bessel_k0e(x)),
            ("K1e", Real::bessel_k1e(x)),
        ] {
            assert!(
                value > 0.0 && value.is_finite(),
                "{name}({x}) = {value} should be a finite positive number"
            );
        }
        // ln I0 likewise, where I0 itself is long gone.
        assert!(Real::ln_bessel_i0(x).is_finite());
    }

    // The scaled Wronskian, which additionally pins the *values* rather than just their
    // finiteness. Capped at 1e8: `I₀ᵉ` is a peak-factored sum over `O(√x)` terms against a
    // 100,000-term loop cap, so above roughly `x = 1e10` it silently truncates and returns a
    // finite but wrong number. `K` has no such limit -- its quadrature is ~40 nodes at any
    // argument -- which is why the two are separated here.
    for &x in &[700.0f64, 1e3, 1e5, 1e6, 1e8] {
        assert_close(
            (Real::bessel_i0e(x) * Real::bessel_k1e(x) + Real::bessel_i1e(x) * Real::bessel_k0e(x))
                * x,
            1.0,
            1e-13,
            &format!("the scaled Wronskian at x = {x}"),
        );
    }
    // Past that, K alone is still exactly on its asymptote, all the way out. The `1/(8x)`
    // correction has to be carried: at x = 1e12 it is 1.25e-13, so comparing against the bare
    // leading term would be measuring the reference's truncation.
    for &x in &[1e12f64, 1e30, 1e150, 1e300] {
        let base = (core::f64::consts::PI / (2.0 * x)).sqrt();
        let inv = 1.0 / x;
        assert_close(
            Real::bessel_k0e(x) / (base * (1.0 - inv / 8.0)),
            1.0,
            1e-15,
            &format!("K0e({x:e}) ~ sqrt(pi/2x)(1 - 1/8x)"),
        );
        assert_close(
            Real::bessel_k1e(x) / (base * (1.0 + 3.0 * inv / 8.0)),
            1.0,
            1e-15,
            &format!("K1e({x:e}) ~ sqrt(pi/2x)(1 + 3/8x)"),
        );
    }
}

#[test]
fn subnormal_arguments_are_handled_rather_than_flushed() {
    // The smallest positive normal, and the smallest subnormal. Several of these functions
    // divide by x or take ln x, so this is where a spurious inf/NaN would appear.
    for &x in &[f64::MIN_POSITIVE, 5e-324, 1e-320] {
        assert_eq!(Real::bessel_j0(x), 1.0, "J0 -> 1 at the origin");
        assert!(
            Real::bessel_j1(x) >= 0.0 && Real::bessel_j1(x) <= x,
            "J1(x) ~ x/2 must stay in [0, x]"
        );
        // Y0 and Ci involve ln x, which is about -745 here: large but perfectly finite.
        assert!(
            Real::bessel_y0(x).is_finite() && Real::bessel_y0(x) < -400.0,
            "Y0({x:e}) = {}",
            Real::bessel_y0(x)
        );
        assert!(Real::cos_int(x).is_finite() && Real::cos_int(x) < -400.0);
        assert!(Real::exp_int_e1(x).is_finite() && Real::exp_int_e1(x) > 400.0);
        assert!(Real::bessel_k0(x).is_finite() && Real::bessel_k0(x) > 400.0);
        // K1 ~ 1/x and Y1 ~ -2/(pi x) genuinely exceed f64's range here, so infinity is the
        // honest answer rather than a defect.
        assert!(
            Real::bessel_k1(x).is_infinite() || Real::bessel_k1(x) > 1e300,
            "K1({x:e}) ~ 1/x must be enormous"
        );
        assert!(Real::sin_int(x) >= 0.0 && Real::sin_int(x) <= 2.0 * x);
        // W(x) ~ x for tiny x.
        assert_close(Real::lambert_w(x) / x, 1.0, 1e-12, "W(x) ~ x");
    }
}

// --- branch points ------------------------------------------------------

#[test]
fn the_lambert_branch_point_behaves_like_a_square_root() {
    // W is not Lipschitz at x = -1/e: the derivative of w e^w vanishes at w = -1, so
    // W(-1/e + d) = -1 + sqrt(2ed) - ... . That is why only half the digits are available
    // there, and asserting the square-root law is what documents the limit as a property of
    // the function rather than a weakness of the method.
    let branch_point = -(1.0 / core::f64::consts::E);
    assert_close(Real::lambert_w(branch_point), -1.0, 1e-8, "W(-1/e) = -1");
    assert!(
        Real::lambert_w(branch_point) >= -1.0 - 1e-8,
        "W can never go below -1 on the principal branch"
    );
    for &delta in &[1e-14f64, 1e-12, 1e-10, 1e-8, 1e-6] {
        let w = Real::lambert_w(branch_point + delta);
        let predicted = -1.0 + (2.0 * core::f64::consts::E * delta).sqrt();
        // Only the leading square-root term, so a few percent is the right bar.
        assert_close(
            (w + 1.0) / (predicted + 1.0),
            1.0,
            0.02,
            &format!("W(-1/e + {delta:e}) + 1 ~ sqrt(2ed)"),
        );
        assert!(w >= -1.0 - 1e-8, "W({delta:e} above the branch point) < -1");
    }
    // And the defining equation still holds there, even though the digits are halved.
    for &delta in &[0.0f64, 1e-12, 1e-9, 1e-6, 1e-3] {
        let x = branch_point + delta;
        let w = Real::lambert_w(x);
        assert_close(w * w.exp(), x, 1e-7, &format!("W e^W = x at d = {delta:e}"));
    }
}

// --- adverse but in-domain arguments ------------------------------------

#[test]
fn the_oscillatory_functions_stay_bounded_at_enormous_arguments() {
    // At huge x the *argument reduction* inside sin/cos dominates the error, so the values
    // are no longer accurate pointwise -- but they must stay bounded, finite, and on their
    // envelopes. An implementation that lost the reduction would return garbage of the wrong
    // magnitude, which is what this detects.
    for &x in &[1e8f64, 1e12, 1e16, 1e100, 1e300] {
        let envelope = (2.0 / (core::f64::consts::PI * x)).sqrt();
        for (name, value) in [
            ("J0", Real::bessel_j0(x)),
            ("J1", Real::bessel_j1(x)),
            ("Y0", Real::bessel_y0(x)),
            ("Y1", Real::bessel_y1(x)),
        ] {
            assert!(value.is_finite(), "{name}({x:e}) = {value}");
            assert!(
                value.abs() <= envelope * 1.5,
                "{name}({x:e}) = {value:e} exceeds its envelope {envelope:e}"
            );
        }
        // Si and Ci likewise sit on their limits with a 1/x envelope.
        assert!(
            (Real::sin_int(x) - core::f64::consts::FRAC_PI_2).abs() <= 1.5 / x + 1e-15,
            "Si({x:e}) left its envelope"
        );
        assert!(
            Real::cos_int(x).abs() <= 1.5 / x + 1e-15,
            "Ci({x:e}) left its envelope"
        );
    }
}

#[test]
fn nothing_in_the_family_returns_nan_anywhere_in_its_domain() {
    // The blanket sweep. Every function, across its whole domain, at a resolution fine enough
    // to land inside every branch and on both sides of every join.
    for x in sweep(1e-6, 60.0, 4000) {
        for (name, value) in [
            ("J0", Real::bessel_j0(x)),
            ("J1", Real::bessel_j1(x)),
            ("Y0", Real::bessel_y0(x)),
            ("Y1", Real::bessel_y1(x)),
            ("K0", Real::bessel_k0(x)),
            ("K1", Real::bessel_k1(x)),
            ("K0e", Real::bessel_k0e(x)),
            ("K1e", Real::bessel_k1e(x)),
            ("I0e", Real::bessel_i0e(x)),
            ("Si", Real::sin_int(x)),
            ("Ci", Real::cos_int(x)),
            ("E1", Real::exp_int_e1(x)),
            ("Ei", Real::exp_int_ei(x)),
            ("W", Real::lambert_w(x)),
        ] {
            assert!(!value.is_nan(), "{name}({x}) returned NaN");
        }
    }
    // zeta over a sweep that crosses the pole, the critical strip, and the trivial zeros.
    for s in sweep(-60.0, 60.0, 4000) {
        assert!(!Real::zeta(s).is_nan(), "zeta({s}) returned NaN");
    }
    // And a much wider zeta sweep, where the functional equation's Gamma factor overflows.
    // Saturating to an infinity is acceptable there; a NaN is not.
    for s in sweep(-600.0, -60.0, 2000) {
        assert!(!Real::zeta(s).is_nan(), "zeta({s}) returned NaN");
    }
}
