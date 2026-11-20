//! Unit tests for the arbitrary-order polygamma and the reciprocal gamma.
//!
//! A child module so implementation plus tests stay inside the crate 400-line budget, while
//! `use super::*` still reaches the private `PolygammaRoute` and `RgammaRoute` tables and the
//! Euler–Maclaurin split. Both route tables need asserting as *decisions*: three of
//! `PolygammaRoute` arms return `NaN` and two of `RgammaRoute` return `0.0`, so within each
//! group only the decision distinguishes them.

use super::*;

fn close(a: f64, b: f64, tol: f64) {
    assert!(
        (a - b).abs() <= tol * b.abs().max(1.0),
        "{a:e} vs {b:e} (tol {tol:e})"
    );
}

#[test]
fn polygamma_reduces_to_digamma_and_trigamma() {
    for i in 1..40 {
        let x = f64::from(i) * 0.25;
        close(polygamma(0, x), digamma(x), 1e-14);
        close(polygamma(1, x), trigamma(x), 1e-14);
    }
}

#[test]
fn polygamma_matches_its_known_values_at_one() {
    // psi^(n)(1) = (-1)^{n+1} n! zeta(n+1)
    // zeta(3) = 1.2020569031595942854, zeta(4) = pi^4/90, zeta(5) = 1.0369277551433699263
    let pi4_90 = core::f64::consts::PI.powi(4) / 90.0;
    close(polygamma(2, 1.0), -2.0 * 1.202_056_903_159_594_3, 1e-12);
    close(polygamma(3, 1.0), 6.0 * pi4_90, 1e-12);
    // zeta(5), written to the digits f64 can hold.
    close(polygamma(4, 1.0), -24.0 * 1.036_927_755_143_37, 1e-12);
}

#[test]
fn polygamma_satisfies_its_recurrence() {
    // psi^(n)(x+1) = psi^(n)(x) + (-1)^n n! / x^{n+1}
    for order in 2..7u32 {
        let n = f64::from(order);
        let factorial = (1..=order).map(f64::from).product::<f64>();
        let sign = if order.is_multiple_of(2) { 1.0 } else { -1.0 };
        for i in 1..30 {
            let x = f64::from(i) * 0.5;
            let lhs = polygamma(order, x + 1.0);
            let rhs = polygamma(order, x) + sign * factorial / x.powf(n + 1.0);
            close(lhs, rhs, 1e-10);
        }
    }
}

#[test]
fn polygamma_is_the_derivative_of_the_previous_order() {
    for order in 1..5u32 {
        for i in 2..20 {
            let x = f64::from(i) * 0.5;
            let h = 1e-5 * x;
            let d = (polygamma(order, x + h) - polygamma(order, x - h)) / (2.0 * h);
            close(d, polygamma(order + 1, x), 1e-7);
        }
    }
}

#[test]
fn rgamma_is_zero_at_the_poles_and_inverts_gamma_elsewhere() {
    for n in 0..10 {
        assert_eq!(rgamma(-f64::from(n)), 0.0, "1/Gamma(-{n}) = 0");
    }
    for i in 1..60 {
        let x = f64::from(i) * 0.25;
        close(rgamma(x), (-ln_gamma(x)).exp(), 1e-14);
    }
    // Negative, non-integer arguments through the reflection formula.
    for &x in &[-0.5f64, -1.5, -2.5, -3.5, -0.25, -4.75] {
        let g = super::super::gamma::gamma(x);
        close(rgamma(x), 1.0 / g, 1e-11);
    }
}

#[test]
fn rgamma_is_defined_where_gamma_overflows() {
    // Gamma(175) overflows; 1/Gamma(175) is a representable subnormal-range value.
    assert!(super::super::gamma::gamma(175.0).is_infinite());
    let r = rgamma(175.0);
    assert!(r > 0.0 && r.is_finite(), "rgamma(175) = {r}");
    // 1/Gamma(175) is about 1e-316, which is *subnormal* — below f64::MIN_POSITIVE at
    // 2.2e-308 — so it carries well under 53 bits of mantissa. The value is recoverable,
    // but a log round trip through it cannot be checked at full precision.
    assert!(
        r < f64::MIN_POSITIVE,
        "expected a subnormal result, got {r}"
    );
    close(r.ln(), -ln_gamma(175.0), 1e-10);

    // Where the result is normal, the round trip is tight.
    let r160 = rgamma(160.0);
    assert!(r160 > f64::MIN_POSITIVE);
    close(r160.ln(), -ln_gamma(160.0), 1e-14);
}

/// The five routes, asserted as decisions.
///
/// [`RgammaRoute::Pole`] and [`RgammaRoute::AtPositiveInfinity`] return the same `0.0` for
/// different reasons, so no value assertion can tell them apart — nor tell either from a
/// reflection that happened to round to zero, which is exactly what the exact-zero arm is
/// there to prevent.
#[test]
fn the_reciprocal_gamma_routes_split_the_line_at_the_integers() {
    use RgammaRoute::{AtPositiveInfinity, Exponential, Pole, Reflection, Undefined};
    for (x, want) in [
        (f64::NAN, Undefined),
        (f64::NEG_INFINITY, Undefined),
        (f64::INFINITY, AtPositiveInfinity),
        // The non-positive integers, and the signed zeros among them.
        (0.0, Pole),
        (-0.0, Pole),
        (-1.0, Pole),
        (-39.0, Pole),
        (-f64::MAX, Pole),
        // Either side of a pole is reflection, however close.
        ((-1.0_f64).next_up(), Reflection),
        ((-1.0_f64).next_down(), Reflection),
        (-0.5, Reflection),
        (-39.000_01, Reflection),
        // The positive axis, including the subnormal floor and the top of the format.
        (f64::MIN_POSITIVE, Exponential),
        (0.5, Exponential),
        (1.0, Exponential),
        (175.0, Exponential),
        (f64::MAX, Exponential),
    ] {
        assert_eq!(RgammaRoute::for_argument(x), want, "at {x:e}");
    }
    // The positive integers are *not* poles: `1/Γ(n)` is `1/(n−1)!`, finite and nonzero.
    for n in 1..6 {
        assert_eq!(RgammaRoute::for_argument(f64::from(n)), Exponential);
        assert!(rgamma(f64::from(n)) > 0.0);
    }
}

/// The six polygamma routes, including the three that all return `NaN`.
///
/// The `Pole`/`Unsupported` split is invisible to any value assertion — both answer `NaN` —
/// but it is the difference between "this function has no value here" and "this
/// implementation does not reach a value that exists". Order is tested before argument, so
/// `ψ` and `ψ₁` keep their own edge behaviour: `polygamma(0, -0.5)` is a finite reflection
/// while `polygamma(2, -0.5)` is not reached at all.
#[test]
fn the_polygamma_routes_separate_a_pole_from_an_unimplemented_region() {
    use PolygammaRoute::{Digamma, Hurwitz, Pole, Trigamma, Undefined, Unsupported};

    // The two low orders take their own routes at every argument, `NaN` included.
    for x in [f64::NAN, -3.0, -0.5, 0.0, 1.0, 1e10] {
        assert_eq!(PolygammaRoute::for_arguments(0, x), Digamma, "at {x:e}");
        assert_eq!(PolygammaRoute::for_arguments(1, x), Trigamma, "at {x:e}");
    }
    // And order 0 at a negative non-integer really does return a value, which is what makes
    // the ordering load-bearing rather than cosmetic.
    assert!(polygamma(0, -0.5).is_finite());
    assert!(polygamma(2, -0.5).is_nan());

    for order in [2, 3, 7] {
        assert_eq!(PolygammaRoute::for_arguments(order, f64::NAN), Undefined);
        // Poles at the non-positive integers, both signed zeros among them.
        for x in [0.0, -0.0, -1.0, -2.0, -39.0] {
            assert_eq!(PolygammaRoute::for_arguments(order, x), Pole, "at {x:e}");
        }
        // Real and finite there, but not reachable without a reflection formula.
        for x in [-0.5, -1.5, -10.5, (-1.0_f64).next_up(), -1e10 - 0.5] {
            assert_eq!(
                PolygammaRoute::for_arguments(order, x),
                Unsupported,
                "at {x:e}"
            );
        }
        for x in [f64::MIN_POSITIVE, 0.5, 1.0, 1e10, f64::INFINITY] {
            assert_eq!(PolygammaRoute::for_arguments(order, x), Hurwitz, "at {x:e}");
        }
        // Every NaN-returning arm agrees on the value, which is why only the route separates
        // them.
        for x in [f64::NAN, 0.0, -2.0, -0.5] {
            assert!(polygamma(order, x).is_nan(), "at {x:e}");
        }
    }
}

/// The two power schemes, asserted as decisions, plus the guard that makes the fast one safe.
///
/// The arms agree to about eleven ulp, so no value assertion can reliably tell them apart — but
/// the *guard* is a different matter. Without it the squaring arm is not merely less accurate but
/// wrong: `b^{32}` at `a = 1e10` is `1e320`, which overflows to infinity and makes the reciprocal
/// zero where the true term is a subnormal. That case is pinned by value as well as by route.
#[test]
fn the_power_scheme_is_chosen_by_integrality_and_by_range() {
    use ZetaPower::{Squaring, Transcendental};

    // `polygamma` only ever arrives with an integer `s`, which is the case worth being fast.
    assert_eq!(ZetaPower::for_scheme(3.0, 1.0, 17.0), Squaring(3));
    assert_eq!(ZetaPower::for_scheme(12.0, 0.5, 16.5), Squaring(12));
    // `zeta` passes arbitrary reals, which have to go the general way.
    assert_eq!(ZetaPower::for_scheme(2.5, 1.0, 17.0), Transcendental);
    assert_eq!(
        ZetaPower::for_scheme(0.5_f64.next_up() + 2.0, 1.0, 17.0),
        Transcendental
    );
    // An integer too large for the cast, and one whose product would leave the format.
    assert_eq!(ZetaPower::for_scheme(1.0e28, 1.0, 17.0), Transcendental);
    assert_eq!(
        ZetaPower::for_scheme(SQUARING_ORDER_MAX, 1.0, 17.0),
        Transcendental
    );
    // The range guard, at the case that motivated it: `b = 1e10 + 16`, so `s·ln b` is `737`.
    assert_eq!(
        ZetaPower::for_scheme(32.0, 1.0e10, 1.0e10 + 16.0),
        Transcendental
    );
    // Either side of the reach boundary, with the base held fixed.
    let b = 17.0_f64;
    let edge = (SQUARING_LOG_MAX / b.ln()).floor();
    assert_eq!(ZetaPower::for_scheme(edge, 1.0, b), Squaring(edge as i32));
    assert!(matches!(
        ZetaPower::for_scheme(edge + 1.0, 1.0, b),
        Transcendental
    ));

    // And the guarded result is the subnormal rather than the zero a product would give.
    let term = hurwitz_zeta(32.0, 1.0e10);
    assert!(term > 0.0 && term.is_finite(), "ζ(32, 1e10) = {term:e}");
    assert!(
        term < f64::MIN_POSITIVE,
        "expected a subnormal, got {term:e}"
    );

    // Both arms land on `ζ(2k)`'s exact closed forms, which is the only place the truth is known
    // exactly — and there they carry the same error. `ζ(2) = π²/6`, `ζ(4) = π⁴/90`.
    let pi = core::f64::consts::PI;
    close(hurwitz_zeta(2.0, 1.0), pi * pi / 6.0, 1e-15);
    close(hurwitz_zeta(4.0, 1.0), pi.powi(4) / 90.0, 1e-15);
    close(hurwitz_zeta(6.0, 1.0), pi.powi(6) / 945.0, 1e-15);
}
