//! Unit tests for the trigonometric integrals `Si` and `Ci`.
//!
//! Beside [`super`] rather than with the shared machinery, because what they reach is this
//! module's: the route tables and the paired `TrigIntegrals` evaluation. The independent
//! asymptotic expansion in the parent's test preamble is the referee at large argument — it
//! shares no code with the continued fraction, which is what makes agreement meaningful.

use super::*;

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
fn ci_handles_its_edges() {
    assert!(cos_int(0.0).is_infinite() && cos_int(0.0) < 0.0);
    assert_eq!(cos_int(f64::INFINITY), 0.0);
    assert!(cos_int(-1.0).is_nan());
    assert!(cos_int(f64::NAN).is_nan());
    assert!(sin_int(f64::NAN).is_nan());
}

/// The trig integrals' route tables, which differ because the two functions differ.
///
/// `Si` is odd and entire — every arm is about the format — while `Ci` carries a `ln x` term and
/// so has a real singularity at the origin and no value at all to its left. The negative arm is
/// the one no value assertion separates: a negative argument reaching the quadrature would
/// produce a `NaN` from `ln x` anyway.
#[test]
fn the_trig_integral_routes_distinguish_an_odd_function_from_a_cut_one() {
    use super::CosIntRoute;
    use super::SinIntRoute;

    // `Si` is defined on the whole line, with `±π/2` at the two ends.
    assert_eq!(SinIntRoute::for_argument(f64::NAN), SinIntRoute::Undefined);
    assert_eq!(SinIntRoute::for_argument(0.0), SinIntRoute::AtOrigin);
    assert_eq!(SinIntRoute::for_argument(-0.0), SinIntRoute::AtOrigin);
    for x in [f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(SinIntRoute::for_argument(x), SinIntRoute::AtInfinity);
    }
    for x in [
        -1e300,
        -1.0,
        -f64::MIN_POSITIVE,
        f64::MIN_POSITIVE,
        1.0,
        1e300,
    ] {
        assert_eq!(
            SinIntRoute::for_argument(x),
            SinIntRoute::Quadrature,
            "at {x:e}"
        );
    }
    // The odd symmetry the sign-bit test delivers, including at the origin.
    assert_eq!(super::sin_int(-0.0).to_bits(), (-0.0_f64).to_bits());
    for x in [0.5_f64, 1.0, 5.0, 50.0] {
        assert_eq!(super::sin_int(-x), -super::sin_int(x), "at {x}");
    }

    // `Ci` has no real value left of the origin, and the negative half-line is one arm with
    // `NaN` rather than a reflection.
    assert_eq!(CosIntRoute::for_argument(f64::NAN), CosIntRoute::Undefined);
    for x in [
        -f64::MIN_POSITIVE,
        -1.0,
        -1e300,
        f64::NEG_INFINITY,
        (-0.0_f64).next_down(),
    ] {
        assert_eq!(
            CosIntRoute::for_argument(x),
            CosIntRoute::Undefined,
            "at {x:e}"
        );
        assert!(super::cos_int(x).is_nan(), "at {x:e}");
    }
    // Both signed zeros are the singularity: `-0.0 < 0.0` is false, so it falls to this arm.
    assert_eq!(CosIntRoute::for_argument(0.0), CosIntRoute::AtOrigin);
    assert_eq!(CosIntRoute::for_argument(-0.0), CosIntRoute::AtOrigin);
    assert_eq!(super::cos_int(-0.0), f64::NEG_INFINITY);
    assert_eq!(
        CosIntRoute::for_argument(f64::INFINITY),
        CosIntRoute::AtInfinity
    );
    for x in [f64::MIN_POSITIVE, 1.0, 1e300] {
        assert_eq!(
            CosIntRoute::for_argument(x),
            CosIntRoute::Quadrature,
            "at {x:e}"
        );
    }
}
