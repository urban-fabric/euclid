//! Unit tests for the single-precision polygamma, reciprocal gamma and `exprel`.
//!
//! A child module so implementation plus tests stay inside the crate 400-line budget, while
//! `use super::*` still reaches the private route tables and the shortened Euler–Maclaurin
//! head. What is added beyond the `f64` peer comparisons is this format own edges: `Γ`
//! overflows between 35 and 36 here while `1/Γ` is still representable, and `exprel` two
//! infinities give different answers where the `f64` peer never had to say so.

use super::*;

fn close(a: f32, b: f32, tol: f32) {
    assert!(
        (a - b).abs() <= tol * b.abs().max(1.0),
        "{a:e} vs {b:e} (tol {tol:e})"
    );
}

#[test]
fn polygamma_reduces_and_recurs_in_f32() {
    for i in 1..20 {
        let x = (i as f32) * 0.5;
        close(polygamma(0, x), digamma(x), 1e-5);
        close(polygamma(1, x), trigamma(x), 1e-5);
    }
    // psi^(2)(x+1) = psi^(2)(x) + 2/x^3
    for i in 1..20 {
        let x = (i as f32) * 0.5;
        close(
            polygamma(2, x + 1.0),
            polygamma(2, x) + 2.0 / x.powi(3),
            1e-4,
        );
    }
}

#[test]
fn rgamma_and_exprel_in_f32() {
    assert_eq!(rgamma(0.0f32), 0.0);
    assert_eq!(rgamma(-3.0f32), 0.0);
    for i in 1..20 {
        let x = (i as f32) * 0.5;
        close(rgamma(x), (-ln_gamma(x)).exp(), 1e-5);
    }
    assert_eq!(exprel(0.0f32), 1.0);
    for i in 1..20 {
        let x = (i as f32) * 0.25;
        close(exprel(x), x.exp_m1() / x, 1e-5);
    }
}

/// The five reciprocal-gamma routes, asserted as decisions.
///
/// [`RgammaRoute::Pole`] and [`RgammaRoute::AtPositiveInfinity`] return the same `0.0` for
/// different reasons, so no value assertion can separate them — nor either from a reflection
/// that happened to round to zero, which is what the exact-zero arm prevents.
#[test]
fn the_reciprocal_gamma_routes_split_the_line_at_the_integers() {
    use RgammaRoute::{AtPositiveInfinity, Exponential, Pole, Reflection, Undefined};
    for (x, want) in [
        (f32::NAN, Undefined),
        (f32::NEG_INFINITY, Undefined),
        (f32::INFINITY, AtPositiveInfinity),
        (0.0, Pole),
        (-0.0, Pole),
        (-1.0, Pole),
        (-39.0, Pole),
        (-f32::MAX, Pole),
        ((-1.0_f32).next_up(), Reflection),
        ((-1.0_f32).next_down(), Reflection),
        (-0.5, Reflection),
        (f32::MIN_POSITIVE, Exponential),
        (0.5, Exponential),
        (1.0, Exponential),
        (f32::MAX, Exponential),
    ] {
        assert_eq!(RgammaRoute::for_argument(x), want, "at {x:e}");
    }
    // The positive integers are *not* poles: `1/Γ(n)` is `1/(n−1)!`, finite and nonzero.
    for n in 1..6 {
        assert_eq!(RgammaRoute::for_argument(n as f32), Exponential);
        assert!(rgamma(n as f32) > 0.0);
    }
    // `Γ` overflows in this format between 35 and 36 — `Γ(35) = 2.95e38` against a ceiling of
    // `3.4e38` — while `1/Γ(36)` is `9.7e-41`, subnormal but representable. That window is
    // the whole reason the positive axis goes through the log domain rather than `recip()`.
    assert!(super::super::gamma::gamma(35.0_f32).is_finite());
    assert!(super::super::gamma::gamma(36.0_f32).is_infinite());
    let r = rgamma(36.0_f32);
    assert!(r > 0.0 && r.is_finite(), "rgamma(36) = {r}");
    assert!(r < f32::MIN_POSITIVE, "expected a subnormal, got {r}");
}

/// The non-finite arms, which nothing reached — three mutants lived in this comparison.
///
/// `(eˣ − 1)/x` tends to `+∞` as `x → +∞` and to `0` as `x → −∞`, so the two infinities give
/// different answers and the sign test between them is load-bearing. Swapping it is a defect
/// a caller sees as a saturating value where it expected a vanishing one.
#[test]
fn exprel_separates_the_two_infinities() {
    assert_eq!(exprel(f32::INFINITY), f32::INFINITY);
    assert_eq!(exprel(f32::NEG_INFINITY), 0.0);
    assert!(exprel(f32::NAN).is_nan());
}

/// Below the crossover the series must be **correctly rounded**, which `exp_m1(x)/x` is not.
///
/// This is what makes the branch worth having, and the bound is set from the gap rather than
/// from taste: measured over 400,000 arguments in `(0, 1e-5]`, `1 + x/2` is accurate to
/// `5.96e-8` and `exp_m1(x)/x` to `1.07e-7`. A bound of `7e-8` admits the first and refuses
/// the second, so it fails if the crossover is narrowed until the series stops being used —
/// which is otherwise invisible, since both paths are accurate to the `1e-5` that the
/// round-trip check above allows.
///
/// The reference is `f64`, four orders finer than f32 can represent, so the comparison is
/// against the truth rather than against euclid's other path.
#[test]
fn exprel_is_correctly_rounded_below_its_crossover() {
    let mut worst = 0.0f64;
    for i in 1..=4_000u32 {
        for sign in [1.0f32, -1.0] {
            #[allow(clippy::cast_possible_truncation)]
            let x = sign * 1e-5 * (f64::from(i) / 4_000.0) as f32;
            let truth = f64::from(x).exp_m1() / f64::from(x);
            let relative = ((f64::from(exprel(x)) - truth) / truth).abs();
            worst = worst.max(relative);
        }
    }
    assert!(
        worst <= 7e-8,
        "exprel below its crossover is off by {worst:.3e}, which is the accuracy of \
         exp_m1(x)/x rather than of the series"
    );
}

/// The five `exprel` routes, at and either side of each boundary.
#[test]
fn the_exprel_forms_partition_the_line() {
    use ExprelForm::{AtOrigin, Infinite, Quotient, Series, Undefined};
    for (x, want) in [
        (0.0_f32, AtOrigin),
        (-0.0, AtOrigin),
        (f32::NAN, Undefined),
        (f32::INFINITY, Infinite),
        (f32::NEG_INFINITY, Infinite),
        // The magnitude decides, so both signs route alike, and the bound is open.
        (SERIES_MAX.next_down(), Series),
        (-SERIES_MAX.next_down(), Series),
        (1e-30, Series),
        (SERIES_MAX, Quotient),
        (-SERIES_MAX, Quotient),
        (1.0, Quotient),
        (90.0, Quotient),
    ] {
        assert_eq!(ExprelForm::for_argument(x), want, "at {x:e}");
    }
    // Two terms, not the `f64` peer's three: at the cutoff `x²/6` is `1.7e-11` against a
    // result near one, four orders under this format's `1.19e-7` resolution.
    assert_eq!(exprel(0.0), 1.0);
    assert!(exprel(f32::NAN).is_nan());
    assert_eq!(exprel(f32::INFINITY), f32::INFINITY);
    assert_eq!(exprel(f32::NEG_INFINITY), 0.0);
    assert_eq!(exprel(1e-6), 1.0 + 1e-6 / 2.0);

    use ExprelQuotient::{Direct, Shifted};
    assert_eq!(ExprelQuotient::for_numerator(1.0), Direct);
    assert_eq!(ExprelQuotient::for_numerator(f32::INFINITY), Shifted);
    // The band `(88.72, 93.26)`, where `exp_m1` is gone and the quotient is not: it used to
    // return `+∞` for every argument in it.
    assert!(!90.0_f32.exp_m1().is_finite());
    let value = exprel(90.0);
    assert!(value.is_finite() && value > 0.0, "exprel(90) = {value}");
    assert_eq!(value, (90.0_f32 - SHIFT).exp() * (SHIFT.exp() / 90.0));
    // Past the top of the band the quotient overflows too, and `+∞` is the right answer.
    assert_eq!(exprel(94.0), f32::INFINITY);
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
    for x in [f32::NAN, -3.0, -0.5, 0.0, 1.0, 1e10] {
        assert_eq!(PolygammaRoute::for_arguments(0, x), Digamma, "at {x:e}");
        assert_eq!(PolygammaRoute::for_arguments(1, x), Trigamma, "at {x:e}");
    }
    // And order 0 at a negative non-integer really does return a value, which is what makes
    // the ordering load-bearing rather than cosmetic.
    assert!(polygamma(0, -0.5).is_finite());
    assert!(polygamma(2, -0.5).is_nan());

    for order in [2, 3, 7] {
        assert_eq!(PolygammaRoute::for_arguments(order, f32::NAN), Undefined);
        // Poles at the non-positive integers, both signed zeros among them.
        for x in [0.0, -0.0, -1.0, -2.0, -39.0] {
            assert_eq!(PolygammaRoute::for_arguments(order, x), Pole, "at {x:e}");
        }
        // Real and finite there, but not reachable without a reflection formula.
        for x in [-0.5, -1.5, -10.5, (-1.0_f32).next_up()] {
            assert_eq!(
                PolygammaRoute::for_arguments(order, x),
                Unsupported,
                "at {x:e}"
            );
        }
        for x in [f32::MIN_POSITIVE, 0.5, 1.0, 1e10, f32::INFINITY] {
            assert_eq!(PolygammaRoute::for_arguments(order, x), Hurwitz, "at {x:e}");
        }
        // Every NaN-returning arm agrees on the value, which is why only the route separates
        // them.
        for x in [f32::NAN, 0.0, -2.0, -0.5] {
            assert!(polygamma(order, x).is_nan(), "at {x:e}");
        }
    }
}
