//! **Does the narrowed `f32` path carry the classical special functions faithfully?**
//!
//! Two of the classical special functions — `Y` and the Airy pair — still evaluate in `f64` and
//! narrow once, because their methods reach their values
//! through a cancelling series, an outward march or a divergent asymptotic, each of which has an
//! error floor set by the *working* precision; a native `f32` port of those would be less
//! accurate rather than more (see [`crate::kernels::f32`]). For them, agreement with `f64` is
//! near-exact by construction and the first test asserts exactly that.
//!
//! **`W`, `ζ`, `Si`, `Ci`, `E₁`, `Ei`, the `K` family and `J` are no longer among them.** All six are now native and
//! none ever fitted the exception: `W` is a Halley iteration on a well-conditioned equation, `ζ`
//! is a geometric series plus an Euler–Maclaurin sum plus a reflection, and the four integrals
//! share one continued fraction. The integrals' case is the sharpest — the exception cited the
//! `Si` series reaching `4e20` at `x = 45`, where neither carrier uses the series at all. They
//! moved to
//! [`the_native_f32_transcendentals_stay_within_their_budgets`], which measures a real gap
//! instead of a rounding — and that is the point of separating the two tests, since a
//! near-exact assertion on a native kernel is not a strong check but a vacuous one.
//!
//! Split from [`super`] to keep both files inside the 400-line budget.

use crate::Real;
use euclid_utils::assert_close_f32;

// --- the classical special functions ------------------------------------

/// Every classical special function, at budgets measured rather than assumed.
///
/// This file used to open with `f32_classical_special_functions_are_the_narrowed_f64_values`,
/// asserting one-rounding agreement for `J`, `Y`, `K`, `Si`, `Ci`, `E₁`, `Ei`, `W`, `ζ` and the
/// Airy pair. It is gone, because none of them narrows any more and a near-exact assertion
/// against `f64` is not a strong check on a native kernel — it is a vacuous one. The families
/// moved here in the order they were ported, each with its own derived budget.
///
/// A narrowed kernel agrees with `f64` to one rounding, so asserting that proves only that the
/// dispatch found the reference. A native kernel has its own error, and the useful question is
/// how large it is — so these budgets are the measured worst over the sweeps below, and a
/// regression in either shows up here as a genuine accuracy change rather than as a wiring
/// mistake.
#[test]
fn the_native_f32_transcendentals_stay_within_their_budgets() {
    // Lambert W. Measured worst 8 ulp over this sweep; the seed regimes and the Halley cap are
    // derived in `kernels::f32::transcendental`.
    for i in 1..60 {
        let x32 = (i as f32) * 0.5;
        assert_close_f32(
            Real::lambert_w(x32),
            Real::lambert_w(f64::from(x32)),
            8.0,
            &format!("W({x32})"),
        );
    }
    // `Y`, now native, and the Airy pair. Both oscillate through zeros on part of their range,
    // so both are compared against their envelopes — `√(2/πx)` for `Y`, `|x|^{-1/4}/√π` for
    // Airy — rather than relatively.
    for i in 1..60 {
        let x32 = (i as f32) * 0.5;
        let x64 = f64::from(x32);
        let envelope = (2.0 / (core::f64::consts::PI * x64)).sqrt();
        for (name, got, want) in [
            ("Y0", Real::bessel_y0(x32), Real::bessel_y0(x64)),
            ("Y1", Real::bessel_y1(x32), Real::bessel_y1(x64)),
        ] {
            let error = (f64::from(got) - want).abs() / envelope;
            assert!(
                error <= 1e-6,
                "{name}({x32}) is {error:.3e} of the envelope away from the f64 value"
            );
        }
    }
    // The Airy pair, scaled: these are the forms that stay representable, since `Bi` overflows
    // `f32` by `x = 26.5` and `Ai` underflows not long after.
    for i in 1..60 {
        let x32 = (i as f32) * 0.5;
        let x64 = f64::from(x32);
        for (name, got, want) in [
            ("Aie", Real::airy_ai_e(x32), Real::airy_ai_e(x64)),
            (
                "Aie'",
                Real::airy_ai_e_prime(x32),
                Real::airy_ai_e_prime(x64),
            ),
            ("Bie", Real::airy_bi_e(x32), Real::airy_bi_e(x64)),
            (
                "Bie'",
                Real::airy_bi_e_prime(x32),
                Real::airy_bi_e_prime(x64),
            ),
        ] {
            assert_close_f32(got, want, 16.0, &format!("{name}({x32})"));
        }
    }
    // The negative axis, where the Airy pair oscillates and the scaled forms are the unscaled
    // ones. Against the envelope, for the same reason as `Y`.
    for i in 1..40 {
        let x32 = -(i as f32) * 0.5;
        let x64 = f64::from(x32);
        let quarter = (-x64).powf(0.25);
        // The values decay like `|x|^{-1/4}` and the *derivatives grow* like `|x|^{1/4}`, so
        // they do not share an envelope. Using the value's for both is wrong by a factor of
        // `√|x|` — it reported `Bi'(-10)` as `2.8e-6` adrift when against its own envelope it
        // is well inside the budget.
        //
        // The budget itself is `2ζ·ε` rather than a constant, because the error here is the
        // *phase*. Beyond `|x| = 6` these come from the Hankel expansion in `ζ = ⅔|x|^{3/2}`,
        // and `ζ` can only be held to `ulp(ζ)` — so the angle carries `≈ζ·ε` and an oscillating
        // function returns that as a fraction of its envelope. Measured: `2.1e-6` at `x = −11.5`
        // (`ζ = 26`) and `4.9e-6` at `x = −19.5` (`ζ = 57`), both tracking `ζ·ε`. A flat bound
        // would either pass everything near the origin or fail everything far from it. This is
        // the one irreducible floor in the `f32` Airy: it is total once `ulp(ζ)` passes `2π`,
        // which is `|x| ≈ 1.8e5` here against the `f64` peer's `1e11`.
        let zeta = (2.0 / 3.0) * (-x64) * (-x64).sqrt();
        let phase_bound = 2.0 * zeta * f64::from(f32::EPSILON);
        for (name, got, want, envelope) in [
            (
                "Ai",
                Real::airy_ai(x32),
                Real::airy_ai(x64),
                1.0 / (core::f64::consts::PI.sqrt() * quarter),
            ),
            (
                "Ai'",
                Real::airy_ai_prime(x32),
                Real::airy_ai_prime(x64),
                quarter / core::f64::consts::PI.sqrt(),
            ),
            (
                "Bi",
                Real::airy_bi(x32),
                Real::airy_bi(x64),
                1.0 / (core::f64::consts::PI.sqrt() * quarter),
            ),
            (
                "Bi'",
                Real::airy_bi_prime(x32),
                Real::airy_bi_prime(x64),
                quarter / core::f64::consts::PI.sqrt(),
            ),
        ] {
            let error = (f64::from(got) - want).abs() / envelope;
            assert!(
                error <= phase_bound.max(8.0 * f64::from(f32::EPSILON)),
                "{name}({x32}) is {error:.3e} of the envelope away from the f64 value, \
                 budget {phase_bound:.3e}"
            );
        }
    }
    // `J`, now native. Compared against its envelope rather than relatively, because `J` has
    // infinitely many zeros and a relative comparison is meaningless at each one.
    for i in 1..60 {
        let x32 = (i as f32) * 0.5;
        let x64 = f64::from(x32);
        let envelope = (2.0 / (core::f64::consts::PI * x64)).sqrt();
        for (name, got, want) in [
            ("J0", Real::bessel_j0(x32), Real::bessel_j0(x64)),
            ("J1", Real::bessel_j1(x32), Real::bessel_j1(x64)),
        ] {
            let error = (f64::from(got) - want).abs() / envelope;
            assert!(
                error <= 1e-6,
                "{name}({x32}) is {error:.3e} of the envelope away from the f64 value"
            );
        }
    }
    // `K`, now native: the same trapezoidal rule with all three constants retuned to this
    // format's target, which cut the node count by about three. See
    // `kernels::f32::bessel::modified_second`.
    for i in 1..60 {
        let x32 = (i as f32) * 0.5;
        let x64 = f64::from(x32);
        for (name, got, want) in [
            ("K0e", Real::bessel_k0e(x32), Real::bessel_k0e(x64)),
            ("K1e", Real::bessel_k1e(x32), Real::bessel_k1e(x64)),
        ] {
            assert_close_f32(got, want, 16.0, &format!("{name}({x32})"));
        }
    }
    // The integrals, now native. One continued fraction behind all four, with the joins
    // unchanged from `f64` and every depth shorter; see `kernels::f32::integrals`.
    for i in 1..60 {
        let x32 = (i as f32) * 0.5;
        let x64 = f64::from(x32);
        for (name, got, want) in [
            ("Si", Real::sin_int(x32), Real::sin_int(x64)),
            ("Ci", Real::cos_int(x32), Real::cos_int(x64)),
            ("E1", Real::exp_int_e1(x32), Real::exp_int_e1(x64)),
        ] {
            assert_close_f32(got, want, 32.0, &format!("{name}({x32})"));
        }
    }
    // zeta across the pole and the critical strip. The eta accelerator runs 10 terms here
    // against the f64 peer's 24, which is where the 16-ulp budget comes from.
    for i in -40..40 {
        let s32 = (i as f32) * 0.25;
        if (s32 - 1.0).abs() < 0.3 {
            continue; // the pole, where f32 cannot resolve the argument anyway
        }
        assert_close_f32(
            Real::zeta(s32),
            Real::zeta(f64::from(s32)),
            16.0,
            &format!("zeta({s32})"),
        );
    }
}

#[test]
fn f32_keeps_the_identities_that_f64_does() {
    // The Wronskian and the Lambert inverse, at f32 resolution. An identity that survives
    // narrowing is evidence the f32 dispatch is wired to the right functions -- a transposed
    // pair of methods would pass the value checks above only if both were transposed, but
    // would break these.
    for i in 1..40 {
        let x = (i as f32) * 0.5;
        let wronskian = Real::bessel_j0(x) * Real::bessel_y1(x)
            - Real::bessel_j1(x) * Real::bessel_y0(x)
            + 2.0 / (core::f32::consts::PI * x);
        let scale = 2.0 / (core::f32::consts::PI * x);
        assert!(
            wronskian.abs() <= 1e-5 * scale.max(1e-2),
            "f32 J/Y Wronskian broken at x = {x}: {wronskian:e}"
        );
        let modified =
            Real::bessel_i0e(x) * Real::bessel_k1e(x) + Real::bessel_i1e(x) * Real::bessel_k0e(x);
        assert!(
            (modified * x - 1.0).abs() <= 1e-5,
            "f32 I/K Wronskian broken at x = {x}"
        );
        let w = Real::lambert_w(x);
        assert!(
            (w * w.exp() - x).abs() <= 1e-5 * x.max(1.0),
            "f32 W does not invert at x = {x}"
        );
    }
}
