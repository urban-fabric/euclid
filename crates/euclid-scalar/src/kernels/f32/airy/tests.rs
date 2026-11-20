//! Unit tests for the single-precision Airy routes, reaching what the trait surface cannot.
//!
//! The split exponential that unscales `Bi` moved to [`crate::math::scaling`], where it is
//! generic and tested once for both formats — its arm is selected only for `x ∈ (26.0666,
//! 26.3380)` here, and the sweep covering that band checks `airy_bi` without an `airy_bi_prime`
//! row, so the derivative had crossed it unchecked in both precisions.

use super::*;
// The regime tables and the four-at-once evaluations moved into `regimes` when this module
// outgrew one file.
use super::regimes::{AiryRegime, PositiveRegime, positive_unscaled};
use crate::math::pair::Solutions;

/// The whole axis as a table.
///
/// Five mutants lived in the two comparison chains this replaced. Each moved an argument between
/// routes that *agree* about its value — the handovers are sited where they agree — so the table
/// is the only thing that can separate them.
#[test]
fn the_airy_regimes_partition_the_axis() {
    use AiryRegime::{
        AtNegativeInfinity, AtPositiveInfinity, Leading, NegativeAsymptotic, NegativeMarch,
        NegativeSeries, Positive, Undefined,
    };
    for (x, want) in [
        (f32::NAN, Undefined),
        (f32::INFINITY, AtPositiveInfinity),
        (f32::NEG_INFINITY, AtNegativeInfinity),
        // The negative ladder, closed on the right at each step.
        (-1e-30, NegativeSeries),
        (-1.0, NegativeSeries),
        (-SERIES_MAX, NegativeSeries),
        (-SERIES_MAX.next_up(), NegativeMarch),
        (-4.0, NegativeMarch),
        (-MARCH_MAX, NegativeMarch),
        (-MARCH_MAX.next_up(), NegativeAsymptotic),
        (-100.0, NegativeAsymptotic),
        (-1e30, NegativeAsymptotic),
        // Both zeros are on the positive side: `-0.0 < 0.0` is false, which is what keeps
        // `Ai(-0.0)` on the series rather than the oscillatory routes.
        (0.0, Positive),
        (-0.0, Positive),
        (1.0, Positive),
        (LEADING_MIN.next_down(), Positive),
        (LEADING_MIN, Leading),
        (1e30, Leading),
    ] {
        assert_eq!(AiryRegime::for_argument(x), want, "at {x:e}");
    }

    // The positive band's own subdivision, which the table above defers to.
    use PositiveRegime::{AiQuadrature, BothBessel, BothSeries};
    for (x, want) in [
        (0.0_f32, BothSeries),
        (0.5, BothSeries),
        (AI_SERIES_MAX, BothSeries),
        (AI_SERIES_MAX.next_up(), AiQuadrature),
        (3.0, AiQuadrature),
        (BI_SERIES_MAX, AiQuadrature),
        (BI_SERIES_MAX.next_up(), BothBessel),
        (100.0, BothBessel),
    ] {
        assert_eq!(PositiveRegime::for_argument(x), want, "at {x}");
    }
}

/// One Taylor step of the equation, checked against the equation rather than against itself.
#[test]
fn the_march_step_transports_the_airy_equation() {
    // A zero step is the identity, exactly — which pins the Horner loops' seeding.
    let at = Slope::new(2.0, -1.0);
    assert_eq!(march_step(3.0, at, 0.0), at);
    assert_eq!(
        march_step(0.0, Slope::uniform(0.0), 0.5),
        Slope::uniform(0.0)
    );

    // At `x₀ = 0` the recurrence collapses to `cₙ₊₂ = cₙ₋₁/((n+2)(n+1))`, whose two solutions are
    // the **closed-form Airy series** `f = 1 + x³/6 + x⁶/180 + ⋯` and `g = x + x⁴/12 + x⁷/504 +
    // ⋯`. Independent mathematics, so this checks the denominators against something other than
    // a transcription of them.
    let h = 0.5_f32;
    let (value, derivative) = march_step(0.0, Slope::new(1.0, 0.0), h).into_parts();
    let f = 1.0 + h.powi(3) / 6.0 + h.powi(6) / 180.0 + h.powi(9) / 12960.0;
    let f_prime = h.powi(2) / 2.0 + h.powi(5) / 30.0 + h.powi(8) / 1440.0;
    assert!((value - f).abs() <= 1e-6, "{value} vs {f}");
    assert!(
        (derivative - f_prime).abs() <= 1e-6,
        "{derivative} vs {f_prime}"
    );

    let (value, derivative) = march_step(0.0, Slope::new(0.0, 1.0), h).into_parts();
    let g = h + h.powi(4) / 12.0 + h.powi(7) / 504.0 + h.powi(10) / 45360.0;
    let g_prime = 1.0 + h.powi(3) / 3.0 + h.powi(6) / 72.0 + h.powi(9) / 4536.0;
    assert!((value - g).abs() <= 1e-6, "{value} vs {g}");
    assert!(
        (derivative - g_prime).abs() <= 1e-6,
        "{derivative} vs {g_prime}"
    );

    // With `x₀ ≠ 0` the `x₀·cₙ` factor enters, and the equation fixes the first three terms:
    // `y + y′h + (x₀y/2)h² + ((x₀y′ + y)/6)h³`. A small step leaves the rest negligible.
    let (x0, y, y_prime, h) = (3.0_f32, 2.0_f32, -1.0_f32, 0.01_f32);
    let value = march_step(x0, Slope::new(y, y_prime), h).value;
    let taylor = y + y_prime * h + (x0 * y / 2.0) * h * h + ((x0 * y_prime + y) / 6.0) * h * h * h;
    assert!((value - taylor).abs() <= 1e-6, "{value} vs {taylor}");
}

/// The saturated limit's **signed** zero, and the unscaling multiply that reaches it.
#[test]
fn the_saturated_limit_keeps_the_sign_of_its_zero() {
    // Above `LEADING_MIN` the unscaled pair saturates, and `Ai′` approaches zero *from below* —
    // so the limit is `−0.0`, not `0.0`. The two compare equal, which is exactly why dropping
    // the sign survived every value assertion; only the bit pattern separates them. This is the
    // same distinction `erf(−0.0)` carries, and an odd function should not lose it.
    let Solutions {
        first_kind: ai,
        second_kind: bi,
    } = quad(LEADING_MIN);
    assert_eq!(ai.value.to_bits(), 0.0_f32.to_bits(), "Ai saturates to +0");
    assert_eq!(
        ai.derivative.to_bits(),
        (-0.0_f32).to_bits(),
        "Ai' saturates to -0"
    );
    assert!(ai.derivative.is_sign_negative());
    assert!(!ai.value.is_sign_negative());
    assert_eq!(bi.value, f32::INFINITY);
    assert_eq!(bi.derivative, f32::INFINITY);
    assert_eq!(
        crate::kernels::f32::airy::ai_prime(1e6).to_bits(),
        (-0.0_f32).to_bits()
    );
}

/// `Ai` and `Ai′` across the positive band, where the scaled pair is multiplied back down.
///
/// The `Aiᵉ′·e^{−ζ}` multiply survived: the unscaled surface's own sweeps carry a budget wide
/// enough to absorb it, and `Ai′` is small there. Against the `f64` kernel at a relative
/// tolerance it is unmissable — at `x = 3` the mutated form gives `−0.35` for a true `−0.0119`.
#[test]
fn the_positive_band_unscales_both_of_the_first_pair() {
    for x in [
        AI_SERIES_MAX.next_up(),
        1.5_f32,
        3.0,
        BI_SERIES_MAX,
        BI_SERIES_MAX.next_up(),
        9.0,
        20.0,
    ] {
        let Solutions {
            first_kind: ai,
            second_kind: bi,
        } = positive_unscaled(x);
        let wide = f64::from(x);
        let want_ai = crate::kernels::airy::ai(wide) as f32;
        let want_ai_prime = crate::kernels::airy::ai_prime(wide) as f32;
        assert!(
            (ai.value - want_ai).abs() <= 1e-4 * want_ai.abs(),
            "Ai({x}): {:e} vs {want_ai:e}",
            ai.value
        );
        assert!(
            (ai.derivative - want_ai_prime).abs() <= 1e-4 * want_ai_prime.abs(),
            "Ai'({x}): {:e} vs {want_ai_prime:e}",
            ai.derivative
        );
        // Both decay and both keep their signs: `Ai > 0`, `Ai′ < 0` on the whole positive axis.
        assert!(ai.value > 0.0 && ai.derivative < 0.0, "at {x}");
        assert!(bi.value > 0.0 && bi.derivative > 0.0, "at {x}");
    }
    // And they decay monotonically, which the exponential factor is what delivers.
    assert!(positive_unscaled(1.5).first_kind.value > positive_unscaled(9.0).first_kind.value);
    assert!(
        positive_unscaled(1.5).first_kind.derivative < positive_unscaled(9.0).first_kind.derivative
    );

    // The sweep stops at `20` because that is where the format does. `Ai ~ e^{−ζ}/(2√π x^{1/4})`
    // with `ζ = ⅔x^{3/2}`, so by `x = 28.4` the true value is under `f32::MIN_POSITIVE` — at
    // `x = 30` it is `3.2e-49`. Returning zero there is the format's answer, not a lost one.
    assert_eq!(positive_unscaled(30.0).first_kind.value, 0.0);
    assert!(positive_unscaled(28.0).first_kind.value > 0.0);
}
