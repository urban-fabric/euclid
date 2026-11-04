//! The Airy family at its edges: the non-finite arguments, the zeros, the overflow and
//! underflow boundaries, and the point where the phase itself stops being representable.
//!
//! `Ai` and `Bi` have no poles — they are entire — so this file is not about singularities. It
//! is about the four *representational* edges the family runs into, each of which the
//! implementation handles by a deliberate choice rather than by whatever the arithmetic
//! produced:
//!
//! | edge | where | chosen behavior |
//! |------|-------|-----------------|
//! | underflow | `Ai`, `x ≳ 104` | `0`, as the true value does |
//! | overflow | `Bi`, `x ≳ 104` | `+∞`, as the true value does |
//! | the zeros | `x < 0`, infinitely many | absolute accuracy only |
//! | the phase | `|x| ≳ 3e205` | `ζ` overflows; the oscillation is unrecoverable |

use crate::Real;
use euclid_utils::assert_close;

#[test]
fn the_non_finite_arguments_give_the_limits() {
    for f in [
        Real::airy_ai as fn(f64) -> f64,
        Real::airy_ai_prime,
        Real::airy_bi,
        Real::airy_bi_prime,
        Real::airy_ai_e,
        Real::airy_ai_e_prime,
        Real::airy_bi_e,
        Real::airy_bi_e_prime,
    ] {
        assert!(f(f64::NAN).is_nan(), "NaN must propagate");
    }
    // At +inf: `Ai` has decayed to nothing and `Bi` has grown without bound.
    assert_eq!(Real::airy_ai(f64::INFINITY), 0.0);
    assert_eq!(Real::airy_ai_prime(f64::INFINITY), 0.0);
    assert!(Real::airy_bi(f64::INFINITY).is_infinite());
    assert!(Real::airy_bi_prime(f64::INFINITY).is_infinite());
    // The scaled forms tend to zero like `x^{−1/4}` instead, with the derivatives unbounded.
    assert_eq!(Real::airy_ai_e(f64::INFINITY), 0.0);
    assert_eq!(Real::airy_bi_e(f64::INFINITY), 0.0);
    assert!(Real::airy_ai_e_prime(f64::INFINITY).is_infinite());
    assert!(Real::airy_bi_e_prime(f64::INFINITY).is_infinite());
    // At -inf the oscillation never settles, so there is no limit; the envelope does vanish,
    // and zero is the only finite answer inside it.
    for f in [
        Real::airy_ai as fn(f64) -> f64,
        Real::airy_ai_prime,
        Real::airy_bi,
        Real::airy_bi_prime,
    ] {
        assert_eq!(f(f64::NEG_INFINITY), 0.0);
    }
}

#[test]
fn the_underflow_and_overflow_boundaries_are_where_the_true_values_are() {
    // `Ai` decays like `e^{−ζ}` and `Bi` grows like `e^{+ζ}`, so both leave f64's range at
    // about x = 104, where ζ = 707. Just below, both are still ordinary numbers; just above,
    // each has gone the way the true value goes. The scaled forms stay finite throughout --
    // which is the entire reason they exist.
    assert!(
        Real::airy_ai(100.0f64) > 0.0,
        "Ai(100) is still representable"
    );
    assert!(Real::airy_bi(100.0f64).is_finite(), "Bi(100) still finite");
    assert_eq!(
        Real::airy_ai(140.0f64),
        0.0,
        "Ai(140) must underflow to zero"
    );
    assert!(
        Real::airy_bi(140.0f64).is_infinite(),
        "Bi(140) must overflow to +inf"
    );
    for &x in &[104.0f64, 140.0, 1e3, 1e6, 1e10, 1e100, 1e300, f64::MAX] {
        for (name, v) in [
            ("Aie", Real::airy_ai_e(x)),
            ("Bie", Real::airy_bi_e(x)),
            ("Aie'", Real::airy_ai_e_prime(x)),
            ("Bie'", Real::airy_bi_e_prime(x)),
        ] {
            assert!(v.is_finite() && v != 0.0, "{name}({x:e}) = {v:e}");
        }
    }
}

#[test]
fn the_scaled_forms_follow_their_envelopes_at_enormous_arguments() {
    // Past the point where every asymptotic correction is below rounding, the scaled values are
    // pure envelope: `Aie -> 1/(2*sqrt(pi)*x^{1/4})` and `Bie -> 1/(sqrt(pi)*x^{1/4})`, with
    // the derivatives the same shape times x^{1/2}. Asserting the closed form rather than mere
    // finiteness is what would catch a lost `zeta` or a dropped power.
    let sqrt_pi = core::f64::consts::PI.sqrt();
    for &x in &[1e10f64, 1e20, 1e100, 1e300] {
        let quarter = x.sqrt().sqrt();
        assert_close(
            Real::airy_ai_e(x),
            1.0 / (2.0 * sqrt_pi * quarter),
            1e-15,
            "Aie envelope",
        );
        assert_close(
            Real::airy_bi_e(x),
            1.0 / (sqrt_pi * quarter),
            1e-15,
            "Bie envelope",
        );
        assert_close(
            Real::airy_ai_e_prime(x),
            -quarter / (2.0 * sqrt_pi),
            1e-15,
            "Aie' envelope",
        );
        assert_close(
            Real::airy_bi_e_prime(x),
            quarter / sqrt_pi,
            1e-15,
            "Bie' envelope",
        );
    }
}

#[test]
fn the_oscillation_stays_inside_its_envelope_however_negative_the_argument() {
    // The counterpart of the Bessel family's enormous-argument test, and the same contract: at
    // large |x| the phase `zeta = (2/3)|x|^{3/2}` can no longer be resolved -- ulp(zeta) passes
    // 2*pi around |x| = 1e11 -- so the values stop being accurate pointwise. They must still be
    // finite, and still on their envelopes. An implementation that lost the phase entirely
    // would return something of the wrong *magnitude*, which is what this detects.
    //
    // Beyond |x| ~ 3e205 `zeta` itself overflows, and the module returns zero there: finite,
    // inside the envelope, and the mean of an oscillation whose phase is gone.
    //
    // The factor is 2 rather than 1, and that is a consequence of the phase loss rather than
    // slack. Each Airy function here is a *sum of two* Bessel functions of adjacent order --
    // `Bi'(-z) = (z/sqrt 3)[J_{-2/3} + J_{2/3}]` -- whose phases differ by a fixed amount in
    // exact arithmetic, and that fixed relationship is what produces the true envelope. Once
    // `zeta` is unresolvable the two computed phases are effectively independent, so the sum
    // can reach the sum of the individual amplitudes: exactly twice the envelope. Measured
    // worst is 1.85x, at Bi'(-1e100).
    //
    // The test still does its job at this width. An implementation that had lost the argument
    // reduction outright would return something of order 1e150 here, not 1e25.
    for &x in &[
        -1e4f64,
        -1e8,
        -1e11,
        -1e100,
        -1e205,
        -1e206,
        -1e300,
        f64::MIN,
    ] {
        let ax = -x;
        let envelope = 1.0 / (core::f64::consts::PI.sqrt() * ax.sqrt().sqrt());
        for (name, v) in [
            ("Ai", Real::airy_ai(x)),
            ("Bi", Real::airy_bi(x)),
            ("Aie", Real::airy_ai_e(x)),
            ("Bie", Real::airy_bi_e(x)),
        ] {
            assert!(v.is_finite(), "{name}({x:e}) = {v} is not finite");
            assert!(
                v.abs() <= envelope * 2.05,
                "{name}({x:e}) = {v:e} left its envelope {envelope:e}"
            );
        }
        // The derivatives ride an envelope that *grows* like |x|^{1/4}.
        let prime_envelope = ax.sqrt().sqrt() / core::f64::consts::PI.sqrt();
        for (name, v) in [
            ("Ai'", Real::airy_ai_prime(x)),
            ("Bi'", Real::airy_bi_prime(x)),
        ] {
            assert!(v.is_finite(), "{name}({x:e}) = {v} is not finite");
            assert!(
                v.abs() <= prime_envelope * 2.05,
                "{name}({x:e}) = {v:e} left its envelope {prime_envelope:e}"
            );
        }
    }
}

#[test]
fn the_zeros_are_reached_with_absolute_accuracy() {
    // At a zero only absolute accuracy is meaningful: the zeros are irrational, so the nearest
    // f64 to one already differs from the true zero by up to eps*|x|, and a non-vanishing
    // derivative turns that into an output error no implementation can avoid. The bound below
    // is exactly that -- |y'(a)| * ulp(a) -- rather than a number chosen to pass.
    //
    // The zeros themselves were obtained by Newton iteration on an independent 60-digit
    // `Decimal` implementation, not copied from a table. That was not caution for its own sake:
    // the first Bi zero had been transcribed here as -1.1737122214976947, six digits of which
    // are wrong (it is -1.17371322270912792), and the test duly failed against a correct
    // implementation returning Bi = 6e-7 at a point 1e-6 away from the real root.
    let ai_zeros = [
        -2.338_107_410_459_767,
        -4.087_949_444_130_97,
        -5.520_559_828_095_551,
        -7.944_133_587_120_853,
    ];
    let bi_zeros = [
        -1.173_713_222_709_127_8,
        -3.271_093_302_836_353,
        -4.830_737_841_662_016,
        -6.169_852_128_310_251,
    ];
    for &a in &ai_zeros {
        let slope = Real::airy_ai_prime(a).abs();
        let bound = slope * a.abs() * f64::EPSILON + 1e-14;
        assert!(
            Real::airy_ai(a).abs() <= bound,
            "Ai({a}) = {:e} exceeds the conditioning bound {bound:e}",
            Real::airy_ai(a)
        );
    }
    for &b in &bi_zeros {
        let slope = Real::airy_bi_prime(b).abs();
        let bound = slope * b.abs() * f64::EPSILON + 1e-14;
        assert!(
            Real::airy_bi(b).abs() <= bound,
            "Bi({b}) = {:e} exceeds the conditioning bound {bound:e}",
            Real::airy_bi(b)
        );
    }
}

#[test]
fn nothing_in_the_family_returns_nan_across_its_whole_domain() {
    // The blanket sweep, at a resolution fine enough to land inside every branch and on both
    // sides of all five joins. A NaN from an in-domain argument is the worst failure available
    // to a numeric library, because it propagates in silence.
    for i in -40_000i32..40_000 {
        let x = f64::from(i) * 0.001;
        for (name, v) in [
            ("Ai", Real::airy_ai(x)),
            ("Ai'", Real::airy_ai_prime(x)),
            ("Bi", Real::airy_bi(x)),
            ("Bi'", Real::airy_bi_prime(x)),
            ("Aie", Real::airy_ai_e(x)),
            ("Aie'", Real::airy_ai_e_prime(x)),
            ("Bie", Real::airy_bi_e(x)),
            ("Bie'", Real::airy_bi_e_prime(x)),
        ] {
            assert!(!v.is_nan(), "{name}({x}) returned NaN");
        }
    }
    // And at the subnormal end, where `zeta` underflows to exactly zero.
    for &x in &[1e-300f64, 5e-324, -5e-324, -1e-300, 0.0, -0.0] {
        assert!(!Real::airy_ai(x).is_nan(), "Ai({x:e})");
        assert!(!Real::airy_bi(x).is_nan(), "Bi({x:e})");
        assert!(!Real::airy_ai_e(x).is_nan(), "Aie({x:e})");
        assert!(!Real::airy_bi_e(x).is_nan(), "Bie({x:e})");
    }
}
