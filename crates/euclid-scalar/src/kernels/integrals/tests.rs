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

/// Knuth's two-sum, pinned by its defining identity rather than through a caller.
///
/// This is the shape that made `.cargo/mutants.toml` exist, found again. All three of
/// `two_sum`'s mutations survived the suite — `a + b` to `a - b`, to `a * b`, and the residual's
/// `+` to `*` — because the function is **self-correcting in its total**: whatever the primary
/// sum becomes, the residual absorbs the difference and `s + e` still reconstructs `a + b`. At
/// `(0.1, 0.2)` the mutated pair is `s = -0.1, e = 0.4` against the real `s = 0.30000000000000004,
/// e = -2.78e-17`, and every downstream caller adds the two straight back together, so a test
/// that checks only a sum cannot see it.
///
/// What is destroyed is the *decomposition*, and that is the whole product: `total` and
/// `correction` are accumulated separately across terms, so a residual carrying `0.4` instead of
/// `-2.78e-17` makes the compensation meaningless and the error `O(n·ε·Σ|tₖ|)` again. The
/// identity has to be asserted on the two halves, not on their sum.
///
/// Each case is exact, so the assertions are equalities and none of the values is transcribed:
/// where `a + b` is representable the residual is zero, and where it needs more than 53 bits the
/// residual is exactly the part that did not fit.
#[test]
fn two_sum_splits_a_sum_into_a_rounded_total_and_its_exact_residual() {
    let two_pow_60 = 2f64.powi(60);
    for (a, b, residual) in [
        // An exactly representable sum leaves nothing over.
        (1.0f64, 2.0f64, 0.0f64),
        (0.5, 0.25, 0.0),
        // A sum needing a 61st bit: the trailing one does not fit, and is the residual.
        (two_pow_60, 1.0, 1.0),
        (two_pow_60, -1.0, -1.0),
        // The same below the ulp of 1, in both directions.
        (1.0, 1e-20, 1e-20),
        (1.0, -1e-20, -1e-20),
    ] {
        let (s, e) = two_sum(a, b);
        assert_eq!(
            s,
            a + b,
            "the primary sum is the rounded sum, at ({a:e}, {b:e})"
        );
        assert_eq!(e, residual, "and the residual is exact, at ({a:e}, {b:e})");
        // Which is the identity the compensation rests on.
        assert_eq!(s + e, a + b, "s + e reconstructs at ({a:e}, {b:e})");
    }
    // A residual that is not representable alongside the total is what the accumulator is for:
    // 0.1 + 0.2 is inexact, so the residual is non-zero and carries the lost bits.
    let (s, e) = two_sum(0.1_f64, 0.2);
    assert_eq!(s, 0.1 + 0.2, "the rounded sum");
    assert!(
        e != 0.0,
        "an inexact sum has a non-zero residual, got {e:e}"
    );
    assert!(
        e.abs() < f64::EPSILON * s.abs(),
        "and it is below one ulp of the total: {e:e}"
    );
    // Commutative in the total, as a sum should be.
    assert_eq!(two_sum(0.1, 0.2).0, two_sum(0.2, 0.1).0);
}

/// `compensated_series` must actually compensate, which only a series that defeats naive
/// summation can require.
///
/// `correction += e` was replaceable with `correction *= e` — zeroing the compensation outright,
/// since `correction` starts at zero — and the suite did not notice, because every existing
/// caller is checked against a tolerance that plain summation already meets. That is the
/// `compensated_sum` finding this repository's mutation config was written for, in a second
/// function.
///
/// The series below is the standard witness: one unit term followed by a thousand at `1e-17`,
/// each individually below half an ulp of the running total. Added naively every one of them is
/// lost and the answer is exactly `1.0`; compensated, they sum in the correction accumulator and
/// the answer is `1 + 1e-14`. No reference value is needed — the exact sum is known by
/// construction.
#[test]
fn compensated_series_recovers_terms_naive_summation_drops_entirely() {
    let terms = |k: u32| {
        if k == 0 {
            Some(1.0)
        } else if k <= 1_000 {
            Some(1e-17)
        } else {
            None
        }
    };

    // Naive summation loses all thousand: 1e-17 is below half an ulp of 1.0.
    let mut naive = 0.0f64;
    for k in 0..=1_001 {
        match terms(k) {
            Some(t) => naive += t,
            None => break,
        }
    }
    assert_eq!(
        naive, 1.0,
        "the premise: added naively the small terms vanish"
    );

    let compensated = compensated_series(terms);
    let exact = 1.0 + 1e-14; // 1000 × 1e-17, which is representable alongside 1.0
    assert!(
        compensated > 1.0,
        "the compensation must recover something, got {compensated:e}"
    );
    close(compensated, exact, 1e-16);
    // And the recovered amount is the whole of what naive summation dropped — to the resolution
    // the format has there, which is the only claim available: `1 + 1e-14` sits 45 ulps above
    // `1.0`, so the difference can be read back only to within one ulp of it (9.992e-15 for an
    // exact 1e-14). That is still a hundred million times the signal needed to separate a
    // working compensation from none at all.
    let recovered = compensated - naive;
    assert!(
        (recovered - 1e-14).abs() <= f64::EPSILON,
        "recovered {recovered:e}, expected 1e-14 to within one ulp of 1.0"
    );
}

#[test]
fn compensated_series_checks_convergence_after_four_terms() {
    let terms = [1.0, 0.0, 0.0, 0.0, 4.0];
    let mut evaluated = Vec::new();
    let sum = super::compensated_series(|k| {
        evaluated.push(k);
        terms.get(k as usize).copied()
    });
    assert_eq!(sum, 1.0);
    assert_eq!(evaluated, [0, 1, 2, 3]);
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
    let threshold = super::TRIG_SERIES_MAX;
    assert!(super::use_trig_series(threshold.next_down()));
    assert!(!super::use_trig_series(threshold));
    assert!(!super::use_trig_series(threshold.next_up()));
    for x in [0.0_f64, -0.0] {
        assert_eq!(super::ei_route(x), Value(f64::NEG_INFINITY));
        assert_eq!(super::sin_int(x).to_bits(), x.to_bits());
    }
    let half_pi = core::f64::consts::FRAC_PI_2;
    assert_eq!(super::sin_int(f64::INFINITY), half_pi);
    assert_eq!(super::sin_int(f64::NEG_INFINITY), -half_pi);
}

/// The two regime crossovers.
///
/// `exp_int_ei`'s `x < EI_ASYMPTOTIC_MIN` and `trig_integrals`' `x < TRIG_SERIES_MAX` decide
/// which route evaluates the threshold *exactly*. Both routes are accurate there — that is what
/// makes a crossover a free choice — and the existing join tests measure the step against the
/// derivative. This measures the thing the mutation actually changes: the two routes' agreement
/// at the single argument that moves between them.
#[test]
fn the_integral_crossovers_agree_at_their_thresholds() {
    // `Ei` at its asymptotic switch: both routes, one argument.
    let x = super::EI_ASYMPTOTIC_MIN;
    let from_below = super::exp_int_ei(x - f64::EPSILON * x);
    let at_threshold = super::exp_int_ei(x);
    let from_above = super::exp_int_ei(x + f64::EPSILON * x);
    let spread = ((from_above - from_below) / at_threshold).abs();
    assert!(
        spread < 1e-13,
        "the two Ei routes must agree across the switch at x = {x}: spread {spread:e}"
    );

    // `Si`/`Ci` at theirs.
    let t = super::TRIG_SERIES_MAX;
    for (name, f) in [
        ("Si", super::sin_int as fn(f64) -> f64),
        ("Ci", super::cos_int as fn(f64) -> f64),
    ] {
        let below = f(t - f64::EPSILON * t);
        let above = f(t + f64::EPSILON * t);
        let scale = f(t).abs().max(1.0);
        assert!(
            ((above - below) / scale).abs() < 1e-13,
            "{name}'s two routes must agree at the join: {below:e} against {above:e}"
        );
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
