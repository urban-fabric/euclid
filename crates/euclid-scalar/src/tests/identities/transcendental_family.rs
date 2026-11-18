//! Identities of the non-elementary integrals (`Si`, `Ci`, `E₁`, `Ei`), the Lambert `W`, and
//! the Riemann zeta.
//!
//! What ties this group together is that each member is *defined* by a relation the
//! implementation does not use: `Si` by an integral, `W` by an inverse, `ζ` by a series and a
//! functional equation. So the identity is genuinely independent of the method under test —
//! `W` is computed by Halley iteration but checked by substituting back into `w·e^w = x`, and
//! `Si` is computed from a continued fraction but checked against quadrature of `sin t/t`.

use crate::Real;
use euclid_utils::{assert_close, integrate, sweep};

// --- the trigonometric and exponential integrals ------------------------

#[test]
fn the_integrals_match_quadrature_of_their_own_integrands() {
    // Si(x) = int_0^x sin(t)/t dt, with the removable singularity at the origin filled in.
    for x in sweep(0.5, 12.0, 24) {
        let quadrature = integrate(|t| if t == 0.0 { 1.0 } else { t.sin() / t }, 0.0, x, 20_000);
        assert_close(
            Real::sin_int(x),
            quadrature,
            1e-10,
            &format!("Si({x}) by quadrature"),
        );
    }
    // Ci(x) = gamma + ln x + int_0^x (cos t - 1)/t dt. The integrand is -t/2 + O(t^3) at the
    // origin, so it is smooth there and Simpson behaves.
    for x in sweep(0.5, 12.0, 24) {
        let quadrature = integrate(
            |t| if t == 0.0 { 0.0 } else { (t.cos() - 1.0) / t },
            0.0,
            x,
            20_000,
        );
        assert_close(
            Real::cos_int(x),
            euclid_utils::EULER_GAMMA + x.ln() + quadrature,
            1e-10,
            &format!("Ci({x}) by quadrature"),
        );
    }
}

#[test]
fn e1_matches_quadrature_of_its_own_integrand() {
    // E1(x) = int_x^inf e^{-t}/t dt, truncated at x + 60 where the integrand is ~e^{-60}.
    for x in sweep(0.25, 8.0, 16) {
        let quadrature = integrate(|t| (-t).exp() / t, x, x + 60.0, 40_000);
        assert_close(
            Real::exp_int_e1(x) / quadrature,
            1.0,
            1e-9,
            &format!("E1({x}) by quadrature"),
        );
    }
}

#[test]
fn the_integrals_differentiate_to_their_integrands() {
    // Si' = sin x / x, Ci' = cos x / x, E1' = -e^-x/x, Ei' = e^x/x. A derivative identity is
    // the sharpest test of a *join*, because a discontinuity there appears as an unbounded
    // derivative rather than as a small value error.
    for x in sweep(0.5, 30.0, 150) {
        let h = 1e-5 * x;
        assert_close(
            (Real::sin_int(x + h) - Real::sin_int(x - h)) / (2.0 * h),
            x.sin() / x,
            1e-7,
            "Si' = sin x / x",
        );
        assert_close(
            (Real::cos_int(x + h) - Real::cos_int(x - h)) / (2.0 * h),
            x.cos() / x,
            1e-7,
            "Ci' = cos x / x",
        );
    }
    for x in sweep(0.25, 15.0, 60) {
        let h = 1e-5 * x;
        assert_close(
            (Real::exp_int_e1(x + h) - Real::exp_int_e1(x - h)) / (2.0 * h),
            -(-x).exp() / x,
            1e-7 * (-x).exp() / x,
            "E1' = -e^-x/x",
        );
    }
}

#[test]
fn ei_is_the_reflection_of_e1() {
    // Ei(-x) = -E1(x), the relation by which negative arguments are evaluated.
    for x in sweep(0.05, 30.0, 200) {
        assert_close(
            Real::exp_int_ei(-x),
            -Real::exp_int_e1(x),
            1e-13,
            &format!("Ei(-{x}) = -E1({x})"),
        );
    }
}

#[test]
fn si_and_ci_lie_on_their_shared_modulus_envelope() {
    // Si and Ci are the imaginary and real parts of one exponential integral:
    //     E1(ix) = -Ci(x) + i(Si(x) - pi/2)
    // and |E1(ix)| -> 1/x. So Ci^2 + (Si - pi/2)^2 -> 1/x^2 even though both oscillate. This
    // is the Si/Ci counterpart of the Bessel Nicholson envelope, and it catches a phase error
    // that pointwise reference values would not.
    for x in sweep(20.0, 400.0, 200) {
        let modulus =
            Real::cos_int(x).powi(2) + (Real::sin_int(x) - core::f64::consts::FRAC_PI_2).powi(2);
        assert_close(
            modulus * x * x,
            1.0,
            2e-2,
            &format!("Si/Ci modulus envelope at x = {x}"),
        );
    }
}

#[test]
fn si_is_odd_and_bounded_by_its_limit() {
    for x in sweep(0.1, 50.0, 200) {
        assert_close(Real::sin_int(-x), -Real::sin_int(x), 1e-15, "Si is odd");
        // Si overshoots pi/2 but never leaves the Gibbs band around it.
        assert!(
            Real::sin_int(x) > 0.0 && Real::sin_int(x) < 1.9,
            "Si({x}) left its band"
        );
    }
    // The global maximum of Si is at the first zero of sin(x)/x, i.e. x = pi.
    let peak = Real::sin_int(core::f64::consts::PI);
    for x in sweep(0.1, 60.0, 500) {
        assert!(
            Real::sin_int(x) <= peak + 1e-15,
            "Si({x}) exceeds its maximum at pi"
        );
    }
}

// --- the Lambert W ------------------------------------------------------

#[test]
fn lambert_w_inverts_its_defining_map() {
    // w e^w = x across the whole real domain, which is the definition rather than a
    // consequence of it.
    for x in sweep(-0.36, 200.0, 400) {
        let w = Real::lambert_w(x);
        assert!(w.is_finite(), "W({x}) = {w}");
        assert_close(w * w.exp(), x, 1e-13, &format!("W({x}) e^W = x"));
    }
    // And far out, where the starting guess is ln x - ln ln x.
    for &x in &[1e3f64, 1e10, 1e100, 1e300] {
        let w = Real::lambert_w(x);
        // Checked as w + ln w = ln x, since w e^w would overflow.
        assert_close(w + w.ln(), x.ln(), 1e-13, &format!("W({x}) + ln W = ln x"));
    }
}

#[test]
fn lambert_w_satisfies_its_log_and_derivative_identities() {
    for x in sweep(0.1, 100.0, 200) {
        let w = Real::lambert_w(x);
        // ln W(x) = ln x - W(x), for x > 0.
        assert_close(w.ln(), x.ln() - w, 1e-12, "ln W = ln x - W");
        // W'(x) = W / (x (1 + W)).
        let h = 1e-6 * x;
        let derivative = (Real::lambert_w(x + h) - Real::lambert_w(x - h)) / (2.0 * h);
        assert_close(derivative, w / (x * (1.0 + w)), 1e-7, "W' = W/(x(1+W))");
    }
    // W(x ln x) = ln x for x >= 1/e, an identity that holds for no wrong implementation.
    for x in sweep(0.5, 40.0, 100) {
        let l = x.ln();
        assert_close(Real::lambert_w(l * x), l, 1e-12, "W(x ln x) = ln x");
    }
    // W(1) is the omega constant, which the crate also carries independently.
    assert_close(Real::lambert_w(1.0), Real::omega(), 1e-15, "W(1) = omega");
}

// --- the Riemann zeta ---------------------------------------------------

/// Compensated sum, built on the crate's own [`Real::two_sum`].
///
/// Needed because these references are 200,000 terms long: summed naively, `ζ(4)`'s series
/// drifts by `2.8e-13`, which is larger than the tolerance being asserted — the test would
/// then be measuring the accuracy of its own reference rather than of `zeta`. Compensation
/// removes the `n`-fold growth, and since every term here is positive there is no cancellation
/// left to worry about.
fn compensated_sum(terms: impl Iterator<Item = f64>) -> f64 {
    let mut total = 0.0f64;
    let mut correction = 0.0f64;
    for term in terms {
        let (sum, error) = Real::two_sum(total, term);
        total = sum;
        correction += error;
    }
    total + correction
}

#[test]
fn zeta_matches_its_defining_series_where_that_converges() {
    // zeta(s) = sum n^{-s}, summed directly. Only usable for s comfortably above 1, which is
    // exactly why the implementation does something else -- so this is an independent check.
    for &s in &[4.0f64, 6.0, 8.0, 12.0, 20.0] {
        let total = compensated_sum((1..200_000u32).map(|n| f64::from(n).powf(-s)));
        assert_close(Real::zeta(s), total, 1e-13, &format!("zeta({s}) by series"));
    }
}

#[test]
fn zeta_matches_its_closed_forms_at_the_even_integers() {
    let pi = core::f64::consts::PI;
    assert_close(Real::zeta(2.0), pi * pi / 6.0, 1e-14, "zeta(2) = pi^2/6");
    assert_close(
        Real::zeta(4.0),
        pi.powi(4) / 90.0,
        1e-14,
        "zeta(4) = pi^4/90",
    );
    assert_close(
        Real::zeta(6.0),
        pi.powi(6) / 945.0,
        1e-13,
        "zeta(6) = pi^6/945",
    );
    assert_close(
        Real::zeta(8.0),
        pi.powi(8) / 9450.0,
        1e-13,
        "zeta(8) = pi^8/9450",
    );
    // And the catalogue constants, which were verified independently of zeta. `apery` is on
    // the trait surface; ZETA_2 and ZETA_4 are not lifted there, so they come from the
    // constants module.
    assert_close(
        Real::zeta(2.0),
        crate::constants::f64::ZETA_2,
        1e-14,
        "zeta(2) catalogue",
    );
    assert_close(Real::zeta(3.0), Real::apery(), 1e-14, "zeta(3) = Apery");
    assert_close(
        Real::zeta(4.0),
        crate::constants::f64::ZETA_4,
        1e-14,
        "zeta(4) catalogue",
    );
}

#[test]
fn zeta_satisfies_the_functional_equation_across_all_three_regimes() {
    // zeta(s) = 2^s pi^{s-1} sin(pi s/2) Gamma(1-s) zeta(1-s).
    //
    // Swept over non-integer s only: at an integer s >= 2 the identity degenerates into
    // 0 * inf, since sin(pi s/2) vanishes exactly where Gamma(1-s) has a pole, and carries no
    // information there.
    //
    // The sweep deliberately straddles s = 1, so it exercises the Euler-Maclaurin route, the
    // critical-strip route and the reflection route, and pins them against each other.
    for i in 1..60 {
        let s = -4.43 + f64::from(i) * 0.17;
        if s.fract() == 0.0 || (s - 1.0).abs() < 0.05 {
            continue; // skip the integers and the neighbourhood of the pole
        }
        let rhs = 2f64.powf(s)
            * core::f64::consts::PI.powf(s - 1.0)
            * (core::f64::consts::FRAC_PI_2 * s).sin()
            * Real::gamma(1.0 - s)
            * Real::zeta(1.0 - s);
        assert_close(
            Real::zeta(s),
            rhs,
            1e-9 * rhs.abs().max(1.0),
            &format!("functional equation at s = {s}"),
        );
    }
}

#[test]
fn zeta_relates_to_the_eta_function_and_the_polygamma_ladder() {
    // eta(s) = (1 - 2^{1-s}) zeta(s) = sum (-1)^{n-1} n^{-s}, summed directly for large s.
    for &s in &[3.0f64, 5.0, 9.0] {
        let total = compensated_sum((1..200_000u32).map(|n| {
            let term = f64::from(n).powf(-s);
            if n % 2 == 1 { term } else { -term }
        }));
        assert_close(
            (1.0 - 2f64.powf(1.0 - s)) * Real::zeta(s),
            total,
            1e-13,
            &format!("eta({s})"),
        );
    }
    // eta(1) = ln 2, so (1 - 2^{1-s}) zeta(s) -> ln 2 as s -> 1 from either side. This is the
    // one statement about the pole that stays finite, and it crosses the pole.
    for &s in &[0.999f64, 0.9999, 1.0001, 1.001] {
        assert_close(
            (1.0 - 2f64.powf(1.0 - s)) * Real::zeta(s),
            2f64.ln(),
            1e-3,
            &format!("eta({s}) -> ln 2"),
        );
    }
    // zeta(n+1) = (-1)^{n+1} psi^(n)(1) / n!, tying zeta to the polygamma ladder.
    for n in 1..6u32 {
        let factorial = (1..=n).map(f64::from).product::<f64>();
        let sign = if n % 2 == 0 { -1.0 } else { 1.0 };
        assert_close(
            Real::zeta(f64::from(n) + 1.0),
            sign * Real::polygamma(1.0, n) / factorial,
            1e-11,
            &format!("zeta({}) from psi^({n})(1)", n + 1),
        );
    }
}
