//! Gamma-family properties selected from surviving mutation sites.
//!
//! Generate the domain that distinguishes a guard from the arithmetic behind it:
//! negative non-integer trigamma arguments, invalid shapes paired with infinities,
//! and rising-factorial shifts that cross the Stirling endpoint in either direction.

use euclid_scalar::Real;
use euclid_stability::{Config, check, check_relative, classify, generate, property};

#[test]
fn rising_factorial_uses_the_gamma_difference_for_infinite_increments() {
    // Γ(-3/2) and Γ(3/2) are both positive.
    for a in [-1.5_f64, 1.5] {
        assert_eq!(a.ln_rising_factorial(f64::INFINITY), f64::INFINITY);
        assert_eq!((a as f32).ln_rising_factorial(f32::INFINITY), f32::INFINITY);
    }
}

#[test]
fn single_precision_rising_factorial_checks_both_stirling_arguments() {
    // When one endpoint is 1/2, that endpoint needs the direct gamma kernel.
    // References are MPFR differences at 256 bits.
    for (a, n, expected) in [
        (0.5_f32, 127.25, 489.769_298_805_583_8),
        (128.25, -127.75, -492.193_357_949_042_54),
    ] {
        let actual = f64::from(a.ln_rising_factorial(n));
        assert!(
            (actual - expected).abs() <= 0.000_15,
            "{actual} vs {expected}"
        );
    }
}

fn settings() -> Config {
    Config::default()
        .cases(400)
        .regressions("tests/regressions/gamma_properties.txt")
}

property! {
    #![config(settings().require("invalid shape", 10.0).require("invalid argument", 10.0))]

    /// Invalid arguments remain invalid even when an infinity shortcut follows the guard.
    fn incomplete_gamma_rejects_invalid_inputs_before_infinity_shortcuts(
        kind in generate::int(0, 3),
        positive in generate::float::in_range(0.125, 100.0),
    ) {
        let (x, shape) = match kind {
            0 => (f64::INFINITY, f64::NAN),
            1 => (f64::INFINITY, -positive),
            2 => (f64::NEG_INFINITY, positive),
            _ => (f64::NEG_INFINITY, f64::NAN),
        };
        classify!("invalid shape", kind != 2);
        classify!("invalid argument", kind >= 2);
        check!(Real::gamma_lr(x, shape).is_nan(), "P({shape}, {x}) must reject the domain");
        check!(Real::gamma_ur(x, shape).is_nan(), "Q({shape}, {x}) must reject the domain");
    }
}

property! {
    #![config(settings())]

    /// psi_1(x) = psi_1(x+1) + 1/x^2 holds between negative poles as well as above zero.
    fn trigamma_recurrence_crosses_negative_noninteger_arguments(
        pole in generate::int(-8, -1),
        fraction in generate::float::in_range(0.125, 0.875),
    ) {
        let x = pole as f64 + fraction;
        let got = Real::trigamma(x);
        let expected = Real::trigamma(x + 1.0) + 1.0 / (x * x);
        check!(got.is_finite() && got > 0.0, "psi_1({x}) = {got}");
        // Measured recurrence residual below 4e-14 relative. Reflection's sine
        // reduction has its own rounding; allow 8e-14 across these eight intervals.
        check_relative!(got, expected, 8e-14, "trigamma recurrence at {x}");
    }

    /// Gamma(1/2) = sqrt(pi) anchors both ends of the Stirling-domain decision.
    fn rising_factorial_handles_shifts_across_the_stirling_domain(
        large in generate::float::in_range(301.0, 10_000.0),
    ) {
        let half_log_pi = 0.5 * core::f64::consts::PI.ln();
        let expected = Real::ln_gamma(large) - half_log_pi;
        // One endpoint is exactly 1/2, so the large-a asymptotic is invalid
        // in either direction. Allow the two log-gamma evaluations to round.
        check_relative!(
            Real::ln_rising_factorial(0.5, large - 0.5), expected, 4e-14,
            "rising factorial from 1/2 to {large}"
        );
        check_relative!(
            Real::ln_rising_factorial(large, 0.5 - large), -expected, 4e-14,
            "rising factorial from {large} to 1/2"
        );
    }
}

property! {
    // One label per regime of `ln_rising_factorial`, and all three required.
    //
    // The old pair was `n <= 64` against `n > 64`, which names the *first* cutoff and hides the
    // second: above 64 the kernel still chooses between the integrated-digamma asymptotic and a
    // direct difference of two log-gammas, on `a > STIRLING_RATIO_MIN`. With `a` uniform over
    // `(0.125, 1000)` and `n` over `1..80` both arms do come up, but nothing *required* either,
    // so a generator narrowed later — to smaller `a`, or to `n <= 64` — would have silently
    // stopped testing the asymptotic while every assertion still passed.
    //
    // That is the failure this repository just paid for once: a sweep that reached a branch and
    // then stopped reaching it is indistinguishable from one that never did, unless coverage is
    // asserted.
    //
    // Measured over 400 cases: **81.00% exact, 14.75% asymptotic, 4.25% direct.** The bounds
    // below sit at roughly half of each, which is far enough under to absorb seed variation and
    // still fail loudly if an arm goes quiet.
    //
    // The direct arm is thin on purpose rather than by oversight: it needs `n > 64` (20% of
    // `1..=80`) *and* `a <= 300` (30% of `(0.125, 1000)`), so 6% is the most the current
    // generators can offer and 4.25% is what they deliver. Widening it would mean changing a
    // generator's range, which changes the tape layout and invalidates the persisted
    // regression beside this file — a worse trade than a 2% floor on a branch that is also
    // covered by the fixed suites.
    #![config(settings()
        .require("exact integer product", 40.0)
        .require("stirling asymptotic", 7.0)
        .require("direct log-gamma difference", 2.0))]

    /// The integer rising factorial is a finite product, including beyond the direct-sum cutoff.
    fn log_rising_factorial_agrees_with_an_independent_finite_product(
        a in generate::float::in_range(0.125, 1_000.0),
        n in generate::int(1, 80),
    ) {
        // Mirrors the kernel's three regimes exactly: an exact product for small integer `n`,
        // the asymptotic when *both* endpoints clear the Stirling threshold, the direct
        // difference otherwise.
        let shift = n as f64;
        let exact_product = n <= 64;
        let stirling = !exact_product && a > 300.0 && a + shift > 300.0;
        classify!("exact integer product", exact_product);
        classify!("stirling asymptotic", stirling);
        classify!("direct log-gamma difference", !exact_product && !stirling);
        let product = (0..n).map(|k| a + k as f64).product::<f64>();
        check_relative!(Real::ln_rising_factorial(a, shift), product.ln(), 8e-14);
    }
}
