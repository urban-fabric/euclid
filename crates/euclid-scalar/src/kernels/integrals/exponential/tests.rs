//! Unit tests for the exponential integrals `E₁` and `Ei`.
//!
//! Beside [`super`] rather than with the shared machinery, because what they reach is this
//! module's: both route tables, both series and the divergent asymptotic sum. The reference
//! values are 600-digit `Decimal` evaluations of the defining series, which at that precision
//! are immune to the cancellation that defeats the same series in `f64`.

use super::*;

fn close(a: f64, b: f64, tol: f64) {
    assert!(
        (a - b).abs() <= tol * b.abs().max(1.0),
        "{a:e} vs {b:e} (tol {tol:e})"
    );
}

// --- reference values ------------------------------------------------
// 600-digit `Decimal` evaluations of the defining power series, which at that precision
// are immune to the cancellation that defeats the same series in `f64`.

#[test]
fn e1_matches_reference_values() {
    close(exp_int_e1(0.1), 1.822_923_958_419_390_6e0, 1e-15);
    close(exp_int_e1(0.5), 5.597_735_947_761_608e-1, 1e-15);
    close(exp_int_e1(1.0), 2.193_839_343_955_202_9e-1, 1e-15);
    close(exp_int_e1(1.5), 1.000_195_824_066_326_5e-1, 1e-15);
    close(exp_int_e1(2.0), 4.890_051_070_806_112e-2, 1e-15);
    close(exp_int_e1(5.0), 1.148_295_591_275_325_7e-3, 1e-15);
    close(exp_int_e1(10.0), 4.156_968_929_685_325e-6, 1e-15);
    close(exp_int_e1(20.0), 9.835_525_290_649_882e-11, 1e-15);
    close(exp_int_e1(50.0), 3.783_264_029_550_459e-24, 1e-15);
}

#[test]
fn ei_matches_reference_values() {
    close(exp_int_ei(1.0), 1.895_117_816_355_936_8e0, 1e-15);
    close(exp_int_ei(2.0), 4.954_234_356_001_89e0, 1e-15);
    close(exp_int_ei(5.0), 4.018_527_535_580_318e1, 1e-15);
    close(exp_int_ei(10.0), 2.492_228_976_241_877_7e3, 1e-15);
    close(exp_int_ei(20.0), 2.561_565_266_405_658_8e7, 1e-15);
    close(exp_int_ei(40.0), 6.039_718_263_611_242e15, 1e-15);
    close(exp_int_ei(50.0), 1.058_563_689_713_169e20, 1e-15);
}

#[test]
fn e1_joins_its_two_regimes_smoothly() {
    // E1' = -e^-x/x, so over 2h the value may change by at most about that much.
    let h = 1e-8;
    let (below, above) = (exp_int_e1(E1_SERIES_MAX - h), exp_int_e1(E1_SERIES_MAX + h));
    let implied = (-E1_SERIES_MAX).exp() / E1_SERIES_MAX * 2.0 * h;
    assert!(
        (above - below).abs() <= implied * 1.01 + 1e-15,
        "step at the E1 join: {below:e} then {above:e}"
    );
}

#[test]
fn ei_crosses_its_asymptotic_switch_smoothly() {
    // Comparing Ei(x-h) to Ei(x+h) directly would be wrong: Ei grows like e^x, so it
    // changes by about 2e-6 relative over any interval that straddles x = 40. The
    // derivative e^x/x bounds how much it may change, which is what makes a jump
    // detectable without a reference value — the same technique used for the incomplete
    // gamma's branch switch.
    let x = EI_ASYMPTOTIC_MIN;
    let h = 1e-7;
    let below = exp_int_ei(x - h);
    let above = exp_int_ei(x + h);
    let implied = x.exp() / x * 2.0 * h;
    assert!(
        (above - below).abs() <= implied * 1.01 + 1e-9 * above.abs(),
        "step at the Ei switch: {below:e} then {above:e}, change {:e} exceeds the \
         derivative-implied {implied:e}",
        above - below
    );
}

#[test]
fn e1_is_the_negated_ei_of_the_negated_argument() {
    for i in 1..60 {
        let x = f64::from(i) * 0.25;
        close(exp_int_ei(-x), -exp_int_e1(x), 1e-14);
    }
}

#[test]
fn e1_is_positive_decreasing_and_handles_its_edges() {
    let mut prev = f64::INFINITY;
    for i in 1..400 {
        let x = f64::from(i) * 0.25;
        let v = exp_int_e1(x);
        assert!(v > 0.0, "E1({x}) = {v} must be positive");
        assert!(v < prev, "E1 must decrease");
        prev = v;
    }
    assert!(exp_int_e1(0.0).is_infinite());
    assert_eq!(exp_int_e1(f64::INFINITY), 0.0);
    assert!(exp_int_e1(-1.0).is_nan());
    assert!(exp_int_e1(f64::NAN).is_nan());
}

/// `E₁`'s three routes, including the series crossover and the whole negative half-line.
///
/// The negative arm is the one no value assertion reaches: a negative argument that fell through
/// to the series would evaluate `ln x` and return `NaN` regardless, so the domain statement and
/// the accident agree on the answer. Only the decision separates them.
#[test]
fn the_e1_routes_cover_the_line_and_its_series_crossover() {
    use super::E1Route::{ContinuedFraction, Series, Value};

    let threshold = super::E1_SERIES_MAX;
    assert_eq!(super::e1_route(threshold.next_down()), Series);
    assert_eq!(super::e1_route(threshold), Series);
    assert_eq!(super::e1_route(threshold.next_up()), ContinuedFraction);
    assert_eq!(super::e1_route(f64::MIN_POSITIVE), Series);
    assert_eq!(super::e1_route(1e30), ContinuedFraction);
    assert_eq!(super::e1_route(f64::MAX), ContinuedFraction);
    // The pole at the origin, from both signed zeros.
    assert_eq!(super::e1_route(0.0), Value(f64::INFINITY));
    assert_eq!(super::e1_route(-0.0), Value(f64::INFINITY));
    assert_eq!(super::e1_route(f64::INFINITY), Value(0.0));
    // The negative half-line and `NaN` are off the domain.
    for x in [
        -f64::MIN_POSITIVE,
        -0.5,
        -1.0,
        -1e30,
        f64::NEG_INFINITY,
        f64::NAN,
    ] {
        assert!(
            matches!(super::e1_route(x), Value(v) if v.is_nan()),
            "at {x:e}"
        );
        assert!(super::exp_int_e1(x).is_nan(), "at {x:e}");
    }
}

#[test]
fn ei_asymptotic_sum_includes_equal_minimal_terms() {
    // Check the recurrence on exact dyadic terms. The caller uses the expansion at x >= 40.
    assert_eq!(super::ei_asymptotic_sum(2.0), 2.0);
    assert_eq!(super::ei_asymptotic_sum(4.0), 1.5625);
}

#[test]
fn ei_remains_finite_until_the_result_overflows() {
    // MPFR at 256 bits, rounded to binary64. The final two inputs are adjacent.
    let last_finite = 716.355_490_542_451_7_f64;
    for (x, reference) in [
        (700.0, 1.450_978_736_052_560_8e301),
        (710.0, 3.150_915_688_206_201_4e305),
        (716.0, 1.260_502_910_604_089_3e308),
        (last_finite, 1.797_693_134_862_150_3e308),
    ] {
        let value = super::exp_int_ei(x);
        assert!((value / reference - 1.0).abs() <= 8.0 * f64::EPSILON);
    }
    assert_eq!(super::exp_int_ei(last_finite.next_up()), f64::INFINITY);
    assert_eq!(super::exp_int_ei(f64::MAX), f64::INFINITY);
}

#[test]
fn integral_routes_distinguish_switches_and_signed_limits() {
    use super::EiRoute::{Asymptotic, Reflection, Series, Value};

    let threshold = super::EI_ASYMPTOTIC_MIN;
    assert_eq!(super::ei_route(threshold.next_down()), Series);
    assert_eq!(super::ei_route(threshold), Asymptotic);
    assert_eq!(super::ei_route(threshold.next_up()), Asymptotic);
    assert_eq!(super::ei_route(f64::INFINITY), Value(f64::INFINITY));
    assert_eq!(super::ei_route(f64::NEG_INFINITY), Reflection);
    assert!(matches!(super::ei_route(f64::NAN), Value(x) if x.is_nan()));
    // Qualified, because `Series` is also an `EiRoute` arm and the two are different decisions.
    let threshold = super::TRIG_SERIES_MAX;
    use crate::kernels::integrals::trigonometric::TrigIntegralRoute;
    assert_eq!(
        TrigIntegralRoute::for_argument(threshold.next_down()),
        TrigIntegralRoute::Series
    );
    assert_eq!(
        TrigIntegralRoute::for_argument(threshold),
        TrigIntegralRoute::Fraction
    );
    assert_eq!(
        TrigIntegralRoute::for_argument(threshold.next_up()),
        TrigIntegralRoute::Fraction
    );
    for x in [0.0_f64, -0.0] {
        assert_eq!(super::ei_route(x), Value(f64::NEG_INFINITY));
        assert_eq!(super::sin_int(x).to_bits(), x.to_bits());
    }
    let half_pi = core::f64::consts::FRAC_PI_2;
    assert_eq!(super::sin_int(f64::INFINITY), half_pi);
    assert_eq!(super::sin_int(f64::NEG_INFINITY), -half_pi);
}
