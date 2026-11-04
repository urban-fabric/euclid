//! Unit tests for the non-elementary integrals.
//!
//! Separated from the implementation because the two together exceed this crate's
//! 400-line file budget. A child module rather than a sibling, so `use super::*`
//! still reaches the private series, continued fraction and crossover constants.

use super::*;
use euclid_utils::no_step_across;

fn close(a: f64, b: f64, tol: f64) {
    assert!(
        (a - b).abs() <= tol * b.abs().max(1.0),
        "{a:e} vs {b:e} (tol {tol:e})"
    );
}

/// The asymptotic auxiliaries `f ~ Σ(−1)ᵏ(2k)!/x^{2k+1}` and `g ~ Σ(−1)ᵏ(2k+1)!/x^{2k+2}`,
/// from which `Si = π/2 − cos x·f − sin x·g` and `Ci = sin x·f − cos x·g`.
///
/// Test-only, and deliberately so: the implementation reaches large `x` through the
/// continued fraction, so this is a genuinely *independent* route to the same values. Both
/// are divergent asymptotic series, truncated at their minimum term — continuing past it
/// makes them worse, which is what the growth check in each loop watches for.
fn trig_asymptotic(x: f64) -> (f64, f64) {
    let inv = 1.0 / x;
    let inv2 = inv * inv;

    let mut f = 0.0f64;
    let mut term = inv;
    for k in 0..200u32 {
        f += if k.is_multiple_of(2) { term } else { -term };
        let kf = f64::from(k);
        let next = term * (2.0 * kf + 1.0) * (2.0 * kf + 2.0) * inv2;
        if next.abs() >= term.abs() {
            break;
        }
        term = next;
    }

    let mut g = 0.0f64;
    let mut term = inv2;
    for k in 0..200u32 {
        g += if k.is_multiple_of(2) { term } else { -term };
        let kf = f64::from(k);
        let next = term * (2.0 * kf + 2.0) * (2.0 * kf + 3.0) * inv2;
        if next.abs() >= term.abs() {
            break;
        }
        term = next;
    }
    (f, g)
}

/// `(Si, Ci)` from the asymptotic expansion.
fn si_ci_asymptotic(x: f64) -> (f64, f64) {
    let (f, g) = trig_asymptotic(x);
    (
        core::f64::consts::FRAC_PI_2 - x.cos() * f - x.sin() * g,
        x.sin() * f - x.cos() * g,
    )
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
fn si_matches_reference_values() {
    close(sin_int(0.5), 4.931_074_180_430_667e-1, 1e-15);
    close(sin_int(1.0), 9.460_830_703_671_83e-1, 1e-15);
    close(sin_int(2.0), 1.605_412_976_802_694_8e0, 1e-15);
    close(sin_int(3.0), 1.848_652_527_999_468_3e0, 1e-15);
    close(sin_int(5.0), 1.549_931_244_944_674e0, 1e-15);
    close(sin_int(10.0), 1.658_347_594_218_874_1e0, 1e-15);
    close(sin_int(20.0), 1.548_241_701_043_44e0, 1e-15);
    // Where the old series-only implementation returned 10.46 instead of 1.56.
    close(sin_int(45.0), 1.558_715_000_896_412_8e0, 1e-15);
    close(sin_int(50.0), 1.551_617_072_485_936e0, 1e-15);
    close(sin_int(100.0), 1.562_225_466_889_056_3e0, 1e-15);
}

#[test]
fn ci_matches_reference_values() {
    close(cos_int(0.5), -1.777_840_788_066_129e-1, 1e-15);
    close(cos_int(1.0), 3.374_039_229_009_681_6e-1, 1e-15);
    close(cos_int(2.0), 4.229_808_287_748_65e-1, 1e-15);
    close(cos_int(3.0), 1.196_297_860_080_003_2e-1, 1e-14);
    close(cos_int(5.0), -1.900_297_496_566_438_8e-1, 1e-15);
    close(cos_int(10.0), -4.545_643_300_445_537e-2, 1e-14);
    close(cos_int(20.0), 4.441_982_084_535_331_4e-2, 1e-14);
    close(cos_int(45.0), 1.863_174_370_355_653e-2, 1e-14);
    close(cos_int(50.0), -5.628_386_324_116_306e-3, 1e-13);
    close(cos_int(100.0), -5.148_825_142_610_492e-3, 1e-13);
}

// --- the join, and agreement with an independent expansion -----------

#[test]
fn the_trigonometric_integrals_join_their_two_regimes_smoothly() {
    // A direct equality across the switch would be wrong even though Si and Ci are
    // bounded: their derivatives sin(x)/x and cos(x)/x are O(1/4) here, so over 2h the
    // value legitimately moves by ~2h/4, which dwarfs any rounding tolerance. The
    // derivative is what bounds a *legitimate* change, so comparing the observed step
    // against it is what detects a join discontinuity.
    let x = TRIG_SERIES_MAX;
    let h = 1e-8;
    for (name, f, derivative) in [
        ("Si", sin_int as fn(f64) -> f64, x.sin() / x),
        ("Ci", cos_int as fn(f64) -> f64, x.cos() / x),
    ] {
        let (below, above) = (f(x - h), f(x + h));
        let implied = derivative.abs() * 2.0 * h;
        assert!(
            (above - below).abs() <= implied * 1.01 + 1e-15,
            "{name} steps at the join: {below:e} then {above:e}, change {:e} exceeds the \
             derivative-implied {implied:e}",
            above - below
        );
    }
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
fn the_continued_fraction_agrees_with_the_independent_asymptotic_expansion() {
    // This is the check that the old series-based implementation could not pass. The
    // asymptotic expansion reaches machine precision past about x = 40 and shares no code
    // with the continued fraction, so agreement here pins both.
    for &x in &[40.0f64, 45.0, 50.0, 80.0, 150.0, 400.0] {
        let (si_a, ci_a) = si_ci_asymptotic(x);
        close(sin_int(x), si_a, 1e-14);
        close(cos_int(x), ci_a, 1e-13);
    }
}

#[test]
fn the_series_and_the_continued_fraction_agree_across_the_whole_overlap() {
    // Between x = 3 and x = 9 both routes are good, which is the strongest statement
    // available about the join: it is not just continuous, the two agree throughout.
    for i in 0..60 {
        let x = 3.0 + f64::from(i) * 0.1;
        let (ci_cf, si_cf) = {
            let (re, im) = e1_continued_fraction((0.0, x), (x.cos(), -x.sin()));
            (-re, core::f64::consts::FRAC_PI_2 + im)
        };
        // Re-run the series past its normal cutoff for the comparison.
        let mut pf = 1.0f64;
        let ci_series = EULER_GAMMA
            + x.ln()
            + compensated_series(|k| {
                let m = 2.0 * (f64::from(k) + 1.0);
                pf *= x * x / ((m - 1.0) * m);
                let t = pf / m;
                Some(if k.is_multiple_of(2) { -t } else { t })
            });
        let mut pf = x;
        let si_series = compensated_series(|k| {
            if k > 0 {
                let m = 2.0 * f64::from(k);
                pf *= x * x / (m * (m + 1.0));
            }
            let t = pf / (2.0 * f64::from(k) + 1.0);
            Some(if k.is_multiple_of(2) { t } else { -t })
        });
        close(si_cf, si_series, 1e-13);
        close(ci_cf, ci_series, 1e-13);
    }
}

// --- identities ------------------------------------------------------

#[test]
fn e1_is_the_negated_ei_of_the_negated_argument() {
    for i in 1..60 {
        let x = f64::from(i) * 0.25;
        close(exp_int_ei(-x), -exp_int_e1(x), 1e-14);
    }
}

#[test]
fn the_integrals_differentiate_to_their_integrands() {
    // E1' = -e^-x/x, Ei' = e^x/x, Si' = sin(x)/x, Ci' = cos(x)/x.
    for i in 1..60 {
        let x = f64::from(i) * 0.5;
        let h = 1e-6 * x;
        let d_e1 = (exp_int_e1(x + h) - exp_int_e1(x - h)) / (2.0 * h);
        close(d_e1, -(-x).exp() / x, 1e-6);
        let d_si = (sin_int(x + h) - sin_int(x - h)) / (2.0 * h);
        close(d_si, x.sin() / x, 1e-6);
        let d_ci = (cos_int(x + h) - cos_int(x - h)) / (2.0 * h);
        close(d_ci, x.cos() / x, 1e-6);
    }
    for i in 1..40 {
        let x = f64::from(i) * 0.25;
        let h = 1e-6 * x;
        let d = (exp_int_ei(x + h) - exp_int_ei(x - h)) / (2.0 * h);
        close(d, x.exp() / x, 1e-6);
    }
}

#[test]
fn si_is_odd_and_approaches_half_pi() {
    for i in 1..40 {
        let x = f64::from(i) * 0.5;
        close(sin_int(-x), -sin_int(x), 1e-15);
    }
    assert_eq!(sin_int(0.0), 0.0);
    assert!(
        sin_int(-0.0).is_sign_negative(),
        "the sign of zero survives"
    );
    assert_eq!(sin_int(f64::INFINITY), core::f64::consts::FRAC_PI_2);
    assert_eq!(sin_int(f64::NEG_INFINITY), -core::f64::consts::FRAC_PI_2);
    // Si oscillates toward pi/2 with envelope 1/x, so by 1e6 it is within ~1e-6.
    close(sin_int(1e6), core::f64::consts::FRAC_PI_2, 2e-6);
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

#[test]
fn ci_handles_its_edges() {
    assert!(cos_int(0.0).is_infinite() && cos_int(0.0) < 0.0);
    assert_eq!(cos_int(f64::INFINITY), 0.0);
    assert!(cos_int(-1.0).is_nan());
    assert!(cos_int(f64::NAN).is_nan());
    assert!(sin_int(f64::NAN).is_nan());
}

#[test]
fn every_crossover_leaves_no_step_at_ulp_resolution() {
    // All three switches in this module, each probed at the two representable values that
    // straddle it -- the smallest interval the format allows, so the function's own slope
    // contributes almost nothing and what is left is the agreement between the two methods.
    //
    // `Ei` is the exception that proves the derivative term is needed: it grows like e^x, so
    // across one ulp of x = 40 it legitimately moves by Ei'(40)*ulp(40) = 42.
    no_step_across(
        "Si",
        sin_int,
        TRIG_SERIES_MAX,
        TRIG_SERIES_MAX.sin() / TRIG_SERIES_MAX,
        1e-14,
    );
    no_step_across(
        "Ci",
        cos_int,
        TRIG_SERIES_MAX,
        TRIG_SERIES_MAX.cos() / TRIG_SERIES_MAX,
        1e-14,
    );
    no_step_across(
        "E1",
        exp_int_e1,
        E1_SERIES_MAX,
        -(-E1_SERIES_MAX).exp() / E1_SERIES_MAX,
        1e-14,
    );
    no_step_across(
        "Ei",
        exp_int_ei,
        EI_ASYMPTOTIC_MIN,
        EI_ASYMPTOTIC_MIN.exp() / EI_ASYMPTOTIC_MIN,
        1e-14,
    );
}
