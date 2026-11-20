//! The error-function family at `f32`, against the `f64` kernel.
//!
//! Part of the [`cross_precision`](super) suite. `erfc` and `erf_inv` carry the widest budgets
//! here, and for opposite reasons: `erfc` is a tail that underflows, and the inverse drops the
//! Newton step its `f64` peer takes.

use super::*;

property! {
    #![config(settings("f32_precision_erf"))]

    /// `erf` at `f32`, including the linear region near the origin.
    fn the_f32_erf_tracks_f64(x in generate::float32::in_range(-6.0, 6.0)) {
        if !x.is_finite() {
            return Ok(());
        }
        tracks_f64!("erf", x, ALLOWED_ERF, Real::erf, Real::erf);
    }
}

property! {
    #![config(settings("f32_precision_erfc")
        .require("erfc complement form", 9.0)
        .require("erfc fraction form", 38.0))]

    /// `erfc` at `f32`, across the form crossover.
    ///
    /// This is the threshold the erf work moved — `1 − erf` below it, the continued fraction
    /// above — so both sides are labelled and required. The `f32` peer sites its own crossover
    /// independently of the `f64` one, which is exactly the kind of divergence a narrowing
    /// comparison catches and an identity does not.
    fn the_f32_erfc_tracks_f64(x in generate::float32::in_range(-3.0, 10.0)) {
        if !x.is_finite() {
            return Ok(());
        }
        classify!("erfc complement form", x < 1.6875);
        classify!("erfc fraction form", x >= 1.6875);
        // The allowance carries `x²`, which is the conditioning rather than slack. The tail
        // route is `e^{−x²}·erfcx(x)`, and forming `−x²` in `f32` costs `ε·x²` *absolute* in the
        // exponent; `exp` turns an absolute error in its argument into a relative error in its
        // result, so the output cannot be better than `ε·x²` relative however the kernel is
        // written. At `x = 9.15` that is `1.0e-5`, about 84 ulp — and the measured gap there is
        // 28, so the kernel is comfortably inside a bound it cannot escape. A flat contract
        // would have had to be 84+ ulp everywhere to admit this one argument, which would have
        // made it vacuous at `x = 1`.
        tracks_f64_conditioned!(
            "erfc",
            x,
            f64::from(ALLOWED_ERFC) + f64::from(x) * f64::from(x),
            Real::erfc,
            Real::erfc
        );
    }
}

property! {
    #![config(settings("f32_precision_erfcx")
        .require("erfcx below the envelope", 2.5)
        .require("erfcx fraction", 23.0))]

    /// `erfcx` at `f32`. The scaled form is where the continued fraction's convergence shows,
    /// and the `f32` port had to widen its own envelope rather than inherit `2.0`.
    fn the_f32_erfcx_tracks_f64(x in generate::float32::in_range(0.01, 40.0)) {
        if !x.is_finite() || x <= 0.0 {
            return Ok(());
        }
        classify!("erfcx below the envelope", x < 1.6875);
        classify!("erfcx fraction", x >= 1.6875);
        tracks_f64!("erfcx", x, ALLOWED_ERFCX, Real::erfcx, Real::erfcx);
    }
}

property! {
    #![config(settings("f32_precision_erf_inv"))]

    /// `erf⁻¹` at `f32`, which takes no Newton step because Giles fitted its coefficients *for*
    /// `f32` — so this is the one kernel here whose narrow version is not a narrowed wide one.
    fn the_f32_erf_inv_tracks_f64(x in generate::float32::in_range(-0.999, 0.999)) {
        if !x.is_finite() {
            return Ok(());
        }
        tracks_f64!("erf_inv", x, ALLOWED_ERF_INV, Real::erf_inv, Real::erf_inv);
    }
}

// ---- the modified Bessel functions --------------------------------------
