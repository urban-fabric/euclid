//! **Does the narrowed `f32` path carry the classical special functions faithfully?**
//!
//! Unlike the rest of the `f32` surface, the classical special functions — `J`, `Y`, `K`, `Si`,
//! `Ci`, `E₁`, `Ei`, `W` and `ζ` — deliberately evaluate in `f64` and narrow once, because
//! their methods have error floors set by the working precision and a native `f32` port would
//! be *less* accurate rather than more (see [`crate::kernels::f32`]). For those, agreement with
//! `f64` is near-exact by construction, and the test below asserts that rather than pretending
//! to measure independent drift.
//!
//! Split from [`super`] to keep both files inside the 400-line budget.

use crate::Real;
use euclid_utils::assert_close_f32;

// --- the classical special functions ------------------------------------

#[test]
fn f32_classical_special_functions_are_the_narrowed_f64_values() {
    // These narrow `f64` deliberately, so the bar is a single rounding rather than the several
    // ulp a native single-precision method would cost. Anything worse than ~1 ulp here means
    // the dispatch is not reaching the f64 reference at all.
    for i in 1..60 {
        let x32 = (i as f32) * 0.5;
        let x64 = f64::from(x32);
        for (name, got, want) in [
            ("J0", Real::bessel_j0(x32), Real::bessel_j0(x64)),
            ("J1", Real::bessel_j1(x32), Real::bessel_j1(x64)),
            ("Y0", Real::bessel_y0(x32), Real::bessel_y0(x64)),
            ("Y1", Real::bessel_y1(x32), Real::bessel_y1(x64)),
            ("K0e", Real::bessel_k0e(x32), Real::bessel_k0e(x64)),
            ("K1e", Real::bessel_k1e(x32), Real::bessel_k1e(x64)),
            ("Si", Real::sin_int(x32), Real::sin_int(x64)),
            ("Ci", Real::cos_int(x32), Real::cos_int(x64)),
            ("E1", Real::exp_int_e1(x32), Real::exp_int_e1(x64)),
            ("W", Real::lambert_w(x32), Real::lambert_w(x64)),
            // The Airy family narrows for the strongest reason in the module: it reaches its
            // values through a cancelling series, an outward march *and* a divergent
            // asymptotic, every one of which has an error floor set by the working precision.
            // Only the scaled forms are swept here -- the unscaled `Bi` overflows `f32` by
            // x = 26.5, long before this sweep ends.
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
            assert_close_f32(got, want, 1.0, &format!("{name}({x32})"));
        }
    }
    // The negative axis, where the Airy pair oscillates and the scaled forms are the unscaled
    // ones. `f32` cannot resolve the phase past about |x| = 20, so the sweep stops short of it.
    for i in 1..40 {
        let x32 = -(i as f32) * 0.5;
        let x64 = f64::from(x32);
        for (name, got, want) in [
            ("Ai", Real::airy_ai(x32), Real::airy_ai(x64)),
            ("Ai'", Real::airy_ai_prime(x32), Real::airy_ai_prime(x64)),
            ("Bi", Real::airy_bi(x32), Real::airy_bi(x64)),
            ("Bi'", Real::airy_bi_prime(x32), Real::airy_bi_prime(x64)),
        ] {
            assert_close_f32(got, want, 1.0, &format!("{name}({x32})"));
        }
    }
    // zeta over a sweep that crosses the pole and the critical strip.
    for i in -40..40 {
        let s32 = (i as f32) * 0.25;
        if (s32 - 1.0).abs() < 0.3 {
            continue; // the pole, where f32 cannot resolve the argument anyway
        }
        assert_close_f32(
            Real::zeta(s32),
            Real::zeta(f64::from(s32)),
            2.0,
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
