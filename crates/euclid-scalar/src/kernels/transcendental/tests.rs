//! Unit tests for the Riemann zeta.
//!
//! A child module in its own file, so `use super::*` still reaches the private critical-strip
//! routine and the term counts while keeping each file inside the 400-line budget. The Lambert
//! `W`'s own tests sit beside it in [`super::lambert`], which is where its routing enums are.

use super::*;
use crate::constants::f64::{APERY, ZETA_2, ZETA_4};

fn close(a: f64, b: f64, tol: f64) {
    assert!(
        (a - b).abs() <= tol * b.abs().max(1.0),
        "{a:e} vs {b:e} (tol {tol:e})"
    );
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
    assert_eq!(
        crate::math::zeta::next_cvz_coefficient(0.0, 3.0, -1.0),
        18.0
    );
    assert_eq!(
        crate::math::zeta::next_cvz_coefficient(1.0, 3.0, 18.0),
        -48.0
    );
    assert_eq!(
        crate::math::zeta::next_cvz_coefficient(2.0, 3.0, -48.0),
        32.0
    );

    let below = 1.0_f64.next_down();
    let linear = -(1.0 - below) * core::f64::consts::LN_2;
    assert_eq!(crate::math::zeta::eta_divisor(below), linear);
    assert_eq!(crate::math::zeta::eta_divisor(0.0), -1.0);
}

#[test]
fn zeta_routes_include_the_pole_and_the_reflection_boundary() {
    use super::ZetaRoute::{Eta, Hurwitz, Reflection, Value};
    let one = 1.0_f64;
    for (s, route) in [
        (one.next_down(), Eta),
        (one, Value(f64::INFINITY)),
        (one.next_up(), Hurwitz),
        // The eta route no longer reaches `−1`: it stops at `ZETA_ETA_MIN = −1/8`, because the
        // reflection is eight to twenty times more accurate from there down. Both sides of the
        // new boundary, and the old one, which is now well inside the reflection.
        (super::ZETA_ETA_MIN, Reflection),
        (super::ZETA_ETA_MIN.next_up(), Eta),
        ((-one).next_up(), Reflection),
        (-one, Reflection),
        ((-one).next_down(), Reflection),
        (-0.5, Reflection),
        ((-0.01_f64), Eta),
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

/// The two reflection forms agree where both are valid.
///
/// Reference-free, and the comparison that makes the boundary meaningful: the forms are
/// algebraically the same product, so where both are sound their disagreement bounds the error of
/// each. Asserted at the *same* `s` through both branches rather than across the threshold —
/// stepping over it would compare two different arguments, and `ζ` moves fast enough on the
/// negative axis that the step alone would dominate.
///
/// # The range is bounded by the log form, not by the product
///
/// This comparison is only as tight as the *worse* route, and the log form's error grows with
/// `|lnΓ(1−s)|` — `49` ulp by `s = −11`, `959` by `s = −92`. Run to the bottom of the product's
/// range it would be measuring the log form's known degradation and calling it a disagreement.
/// So it stops at `−13`, while both are within tens of ulp of each other and a real defect in
/// either still shows.
///
/// Measured worst over the arguments below: `58` ulp. A bar of `120` catches a form that has
/// stopped being the same product — a mis-signed factor, a dropped `2^s` — without flagging the
/// difference that is the whole reason the boundary exists.
#[test]
fn the_two_reflection_forms_agree_where_both_are_valid() {
    fn log_domain(s: f64) -> f64 {
        let one_minus_s = 1.0 - s;
        let sin_term = crate::math::gamma::reflection_sine(0.5 * s);
        let ln_magnitude = s * core::f64::consts::LN_2
            + (s - 1.0) * core::f64::consts::PI.ln()
            + sin_term.abs().ln()
            + super::ln_gamma(one_minus_s);
        (ln_magnitude.exp() * super::hurwitz_zeta(one_minus_s, 1.0)).copysign(sin_term)
    }

    let mut worst = (0.0_f64, 0.0_f64);
    let mut checked = 0usize;
    for step in 0..=60 {
        // Odd half-integers would land on the trivial zeros, where a relative comparison is
        // meaningless; the offset keeps every sample away from them.
        let s = -4.3 - f64::from(step) * 0.15;

        let product = super::zeta(s);
        let logged = log_domain(s);
        let ulps = (product - logged).abs() / crate::Real::ulp(product);
        if ulps > worst.0 {
            worst = (ulps, s);
        }
        assert!(
            ulps <= 120.0,
            "at s = {s} the product gives {product} and the log form {logged}, {ulps} ulp apart"
        );
        checked += 1;
    }
    assert!(
        checked > 50,
        "only {checked} arguments reached the assertion"
    );
    assert!(
        worst.0 > 0.0,
        "the two forms were bit-identical everywhere, so this compared nothing"
    );
}

/// The reflection's form boundary, and the overflow boundary beyond it.
#[test]
fn the_reflection_picks_its_form_at_the_measured_boundary() {
    use super::ReflectionForm::{LogDomain, Product};
    use super::ZETA_PRODUCT_MAX;
    assert_eq!(super::ReflectionForm::for_argument(-1.0), LogDomain);
    assert_eq!(super::ReflectionForm::for_argument(-3.0), LogDomain);
    assert_eq!(
        super::ReflectionForm::for_argument(ZETA_PRODUCT_MAX.next_up()),
        LogDomain
    );
    assert_eq!(
        super::ReflectionForm::for_argument(super::ZETA_PRODUCT_MAX),
        Product
    );
    assert_eq!(super::ReflectionForm::for_argument(-50.0), Product);
    // `Γ(1−s)` leaves the format past 171, so the product cannot be formed at all.
    assert_eq!(
        super::ReflectionForm::for_argument(1.0 - super::GAMMA_PRODUCT_MAX),
        Product
    );
    assert_eq!(
        super::ReflectionForm::for_argument((1.0 - super::GAMMA_PRODUCT_MAX).next_down()),
        LogDomain
    );
    // The pinned closed forms are both on the log side, which is why the boundary is not at the
    // measured crossover of −5/2. A `const` block, so moving it past `ζ(−3)` fails to compile
    // rather than failing the identity suite in another crate.
    const {
        assert!(
            ZETA_PRODUCT_MAX < -3.0,
            "ZETA_PRODUCT_MAX must stay below ζ(−3), whose closed form is pinned to zero ulp"
        );
    }
}
