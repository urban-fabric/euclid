//! Unit tests for the Lambert `W` and the Riemann zeta.
//!
//! A child module in its own file, so `use super::*` still reaches the private critical-strip
//! routine and the iteration constants while keeping each file inside the 400-line budget.

use super::*;
use crate::constants::f64::{APERY, OMEGA, ZETA_2, ZETA_4};

fn close(a: f64, b: f64, tol: f64) {
    assert!(
        (a - b).abs() <= tol * b.abs().max(1.0),
        "{a:e} vs {b:e} (tol {tol:e})"
    );
}

#[test]
fn lambert_w_satisfies_its_defining_equation() {
    // w e^w = x, checked across the whole real domain.
    for i in 0..200 {
        let x = NEG_INV_E + f64::from(i) * 0.5;
        let w = lambert_w(x);
        assert!(w.is_finite(), "W({x}) = {w}");
        close(w * w.exp(), x, 1e-11);
    }
}

#[test]
fn lambert_w_hits_its_known_values() {
    assert_eq!(lambert_w(0.0), 0.0);
    close(lambert_w(1.0), OMEGA, 1e-14);
    close(lambert_w(core::f64::consts::E), 1.0, 1e-14);
    close(lambert_w(NEG_INV_E), -1.0, 1e-7);
    // W(x ln x) = ln x for x >= 1/e.
    for i in 1..40 {
        let x = f64::from(i) * 0.5 + 0.5;
        let l = x.ln();
        close(lambert_w(l * x), l, 1e-10);
    }
}

#[test]
fn lambert_w_is_accurate_near_the_branch_point_and_far_out() {
    // The branch point is where Newton-type iteration stalls, so it gets its own series.
    for &d in &[0.0f64, 1e-12, 1e-9, 1e-6, 1e-3] {
        let x = NEG_INV_E + d;
        let w = lambert_w(x);
        assert!(w >= -1.0 - 1e-6, "W({x}) = {w} must be >= -1");
        close(w * w.exp(), x, 1e-6);
    }
    // And for very large arguments, where the start is ln x - ln ln x.
    for &x in &[1e3f64, 1e10, 1e100, 1e300] {
        let w = lambert_w(x);
        assert!(w.is_finite() && w > 0.0, "W({x}) = {w}");
        // w + ln w = ln x
        close(w + w.ln(), x.ln(), 1e-12);
    }
}

#[test]
fn lambert_w_is_monotone_and_rejects_its_domain_edge() {
    let mut prev = f64::NEG_INFINITY;
    for i in 0..500 {
        let x = NEG_INV_E + f64::from(i) * 0.1;
        let w = lambert_w(x);
        assert!(w >= prev - 1e-9, "W must increase: {prev} then {w}");
        prev = w;
    }
    assert!(
        lambert_w(NEG_INV_E - 1e-6).is_nan(),
        "below the branch point"
    );
    assert!(lambert_w(-1.0).is_nan());
    assert!(lambert_w(f64::NAN).is_nan());
    assert!(lambert_w(f64::INFINITY).is_infinite());
}

#[test]
fn zeta_matches_the_constants_catalogue() {
    // These tie zeta to values verified independently in `constants`.
    close(zeta(2.0), ZETA_2, 1e-13);
    close(zeta(3.0), APERY, 1e-13);
    close(zeta(4.0), ZETA_4, 1e-13);
}

#[test]
fn zeta_matches_its_closed_forms() {
    let pi = core::f64::consts::PI;
    close(zeta(2.0), pi * pi / 6.0, 1e-13);
    close(zeta(4.0), pi.powi(4) / 90.0, 1e-13);
    close(zeta(6.0), pi.powi(6) / 945.0, 1e-12);
    close(zeta(8.0), pi.powi(8) / 9450.0, 1e-12);
    // zeta(0) = -1/2 and zeta(-1) = -1/12 come through the functional equation.
    close(zeta(0.0), -0.5, 1e-13);
    close(zeta(-1.0), -1.0 / 12.0, 1e-11);
    close(zeta(-3.0), 1.0 / 120.0, 1e-10);
}

#[test]
fn zeta_is_correct_inside_the_critical_strip() {
    // The regime the other two routes cannot reach. Before `zeta_critical_strip` existed
    // this called the Euler-Maclaurin sum with s in (0, 1), outside its domain of
    // convergence — it tripped a debug assertion, and in release returned nonsense.
    //
    // References from a 60-digit `Decimal` evaluation of eta under repeated Euler
    // transformation, a method independent of the one under test; it reproduces
    // zeta(2) = pi^2/6 and zeta(4) = pi^4/90 to 59 digits.
    close(zeta(0.25), -8.132_784_052_618_917e-1, 1e-13);
    close(zeta(0.5), -1.460_354_508_809_586_8e0, 1e-13);
    close(zeta(0.75), -3.441_285_386_945_222_7e0, 1e-13);
    close(zeta(0.001), -5.009_199_427_132_187e-1, 1e-12);
    // Just below the pole, where zeta ~ 1/(s-1) + gamma.
    close(zeta(0.999), -9.994_228_571_557_887e2, 1e-10);
    close(
        zeta(0.999),
        1.0 / (0.999 - 1.0) + crate::constants::f64::EULER_GAMMA,
        1e-3,
    );

    // The functional equation must hold across the strip, tying it to the s > 1 route.
    // At s in (0, 1) the reflected point 1 - s is also in the strip, so this additionally
    // checks the new branch against *itself* under reflection — a real constraint, since
    // the two sides are evaluated at different arguments.
    for i in 1..20 {
        let s = f64::from(i) * 0.05;
        let rhs = 2f64.powf(s)
            * core::f64::consts::PI.powf(s - 1.0)
            * (core::f64::consts::FRAC_PI_2 * s).sin()
            * super::super::gamma::gamma(1.0 - s)
            * zeta(1.0 - s);
        close(zeta(s), rhs, 1e-11);
    }

    // zeta is negative throughout [0, 1) and decreasing toward the pole.
    let mut previous = f64::INFINITY;
    for i in 0..99 {
        let s = f64::from(i) * 0.01;
        let v = zeta(s);
        assert!(v < 0.0, "zeta({s}) = {v} must be negative on [0, 1)");
        assert!(v < previous, "zeta must decrease toward the pole at {s}");
        previous = v;
    }
}

#[test]
fn zeta_vanishes_at_the_trivial_zeros() {
    for k in 1..12 {
        let s = -2.0 * f64::from(k);
        assert_eq!(zeta(s), 0.0, "zeta({s}) must be exactly zero");
    }
}

#[test]
fn zeta_approaches_one_and_poles_correctly() {
    // zeta(s) -> 1 as s -> +inf.
    for &s in &[20.0f64, 50.0, 200.0] {
        close(zeta(s), 1.0, 1e-5);
    }
    assert!(zeta(1.0).is_infinite(), "s = 1 is a pole");
    // Just above the pole it diverges: zeta(1+e) ~ 1/e.
    assert!(zeta(1.0 + 1e-6) > 1e5);
    assert!(zeta(f64::NAN).is_nan());
    close(zeta(f64::INFINITY), 1.0, 1e-15);
}

#[test]
fn zeta_satisfies_its_own_functional_equation() {
    // The relation used for s < 1 must also hold where the series is used directly, which
    // checks the two regimes agree rather than just that each runs.
    //
    // Evaluated only at *non-integer* s. At an integer s >= 2 the identity degenerates:
    // sin(pi s / 2) is zero while Gamma(1 - s) has a pole, so the product is 0 * inf and
    // carries no information. The implementation never hits that case, because it only
    // applies the equation for s < 1, where 1 - s > 0 and Gamma is finite.
    for i in 1..30 {
        let s = 1.37 + f64::from(i) * 0.31;
        debug_assert!(s.fract() != 0.0);
        let lhs = zeta(s);
        let one_minus = 1.0 - s;
        let rhs = 2f64.powf(s)
            * core::f64::consts::PI.powf(s - 1.0)
            * (core::f64::consts::FRAC_PI_2 * s).sin()
            * super::super::gamma::gamma(one_minus)
            * zeta(one_minus);
        close(lhs, rhs, 1e-8);
    }
}

#[test]
fn zeta_relates_to_the_polygamma_ladder() {
    // zeta(n+1) = (-1)^{n+1} psi^(n)(1) / n!
    for n in 1..6u32 {
        let factorial = (1..=n).map(f64::from).product::<f64>();
        let sign = if n.is_multiple_of(2) { -1.0 } else { 1.0 };
        let via_polygamma = sign * super::super::polygamma::polygamma(n, 1.0) / factorial;
        close(zeta(f64::from(n) + 1.0), via_polygamma, 1e-10);
    }
}

#[test]
fn cvz_steps_match_their_exact_coefficients() {
    assert_eq!(super::cvz_normalizer(0), 1);
    assert_eq!(super::cvz_normalizer(1), 3);
    assert_eq!(super::cvz_normalizer(2), 17);
    assert_eq!(super::cvz_normalizer(3), 99);
    assert_eq!(super::cvz_normalizer(24), 1_180_872_205_318_713_601);
    assert_eq!(super::cvz_normalizer(25), 6_882_627_592_338_442_563);
    assert_eq!(super::next_cvz_coefficient(0.0, 3.0, -1.0), 18.0);
    assert_eq!(super::next_cvz_coefficient(1.0, 3.0, 18.0), -48.0);
    assert_eq!(super::next_cvz_coefficient(2.0, 3.0, -48.0), 32.0);

    let below = 1.0_f64.next_down();
    let linear = -(1.0 - below) * core::f64::consts::LN_2;
    assert_eq!(super::eta_divisor(below), linear);
    assert_eq!(super::eta_divisor(0.0), -1.0);
}

#[test]
fn zeta_routes_include_the_pole_and_the_reflection_boundary() {
    use super::ZetaRoute::{Eta, Hurwitz, Reflection, Value};
    let one = 1.0_f64;
    for (s, route) in [
        (one.next_down(), Eta),
        (one, Value(f64::INFINITY)),
        (one.next_up(), Hurwitz),
        ((-one).next_up(), Eta),
        (-one, Reflection),
        ((-one).next_down(), Reflection),
        (-2.0, Value(0.0)),
        (-4.0, Value(0.0)),
        (-6.0, Value(0.0)),
        ((-6.0_f64).next_down(), Reflection),
        (-f64::MAX, Value(0.0)),
        (-0.0, Value(-0.5)),
        (f64::INFINITY, Value(1.0)),
    ] {
        assert_eq!(super::zeta_route(s), route);
    }
    for s in [f64::NEG_INFINITY, f64::NAN] {
        assert!(matches!(super::zeta_route(s), Value(x) if x.is_nan()));
    }
}

#[test]
fn zeta_resolves_signed_neighbours_of_trivial_zeros() {
    // MPFR at 256 bits, evaluated at the exact floating-point inputs.
    for (s, expected) in [
        ((-2.0_f64).next_down(), 1.352_183_123_621_542_3e-17),
        ((-2.0_f64).next_up(), -6.760_915_618_107_716e-18),
        ((-20.0_f64).next_down(), -4.699_565_092_785_927e-13),
        ((-20.0_f64).next_up(), 4.699_565_092_785_888e-13),
        ((-168.0_f64).next_down(), -2.889_672_893_862_134_3e154),
        ((-168.0_f64).next_up(), 2.889_672_893_861_593_7e154),
    ] {
        let got = super::zeta(s);
        assert!((got / expected - 1.0).abs() < 4e-13, "s={s}, got={got}");
    }
}
