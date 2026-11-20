//! Airy, Lambert `W` and the non-elementary integrals at `f32`, against the `f64` kernel.
//!
//! Part of the [`cross_precision`](super) suite. Grouped together because each belongs to no
//! larger family in this tree, not because they share machinery.

use super::*;

property! {
    #![config(settings("f32_precision_airy_ai")
        .require("airy oscillatory", 12.0)
        .require("airy decaying", 35.0))]

    /// `Ai` at `f32`, over both signs, counted against its amplitude envelope.
    ///
    /// The oscillatory left half and the decaying right half are different algorithms, and the
    /// port rederived three of their four thresholds. On the left the envelope is the standard
    /// `π^{−1/2}|x|^{−1/4}`; on the right `Ai` decays monotonically with no zeros, so the value
    /// is its own scale and the criterion reduces to an ordinary ulp count.
    fn the_f32_airy_ai_tracks_f64(x in generate::float32::in_range(-15.0, 15.0)) {
        if !x.is_finite() {
            return Ok(());
        }
        classify!("airy oscillatory", x < 0.0);
        classify!("airy decaying", x >= 0.0);
        let envelope = if x < 0.0 {
            x.abs().powf(-0.25) / core::f32::consts::PI.sqrt()
        } else {
            0.0
        };
        tracks_f64_at_scale!(
            "airy_ai",
            x,
            ALLOWED_AIRY_AI,
            envelope,
            Real::airy_ai,
            Real::airy_ai
        );
    }
}

property! {
    #![config(settings("f32_precision_lambert_w")
        .require("W near the branch point", 3.0)
        .require("W ordinary", 25.0))]

    /// `W` at `f32`, from just above the branch point outwards.
    ///
    /// The branch point is `−1/e`, where `W` has a square-root singularity and the Halley
    /// iteration is seeded from a series rather than from the logarithm. `f32` leaves that
    /// neighbourhood about `1e4` times further out than `f64` does, so the window is labelled.
    fn the_f32_lambert_w_tracks_f64(x in generate::float32::in_range(-0.36, 300.0)) {
        if !x.is_finite() {
            return Ok(());
        }
        classify!("W near the branch point", x < -0.3);
        classify!("W ordinary", x >= -0.3);
        tracks_f64!("lambert_w", x, ALLOWED_LAMBERT_W, Real::lambert_w, Real::lambert_w);
    }
}

property! {
    #![config(settings("f32_precision_sin_int")
        .require("Si series", 3.5)
        .require("Si fraction", 45.0))]

    /// `Si` at `f32`, across `TRIG_SERIES_MAX`, which the measurement left at the `f64` value.
    fn the_f32_sin_int_tracks_f64(x in generate::float32::in_range(0.0, 60.0)) {
        if !x.is_finite() || x < 0.0 {
            return Ok(());
        }
        classify!("Si series", x < 4.0);
        classify!("Si fraction", x >= 4.0);
        tracks_f64!("sin_int", x, ALLOWED_SIN_INT, Real::sin_int, Real::sin_int);
    }
}

property! {
    #![config(settings("f32_precision_exp_int_ei")
        .require("Ei series", 8.0)
        .require("Ei asymptotic", 18.0))]

    /// `Ei` at `f32`, across `EI_ASYMPTOTIC_MIN`. Every term of the series is positive for
    /// `x > 0`, so the series cannot cancel — it can only be stopped too early, which is what
    /// the lower cap risks and this comparison would see.
    fn the_f32_exp_int_ei_tracks_f64(x in generate::float32::in_range(0.01, 60.0)) {
        if !x.is_finite() || x <= 0.0 {
            return Ok(());
        }
        classify!("Ei series", x < 20.0);
        classify!("Ei asymptotic", x >= 20.0);
        // `Ei` has a zero at `0.3725074`, reached by cancellation rather than analytically:
        // `Ei = γ + ln x + Σ`, all `O(1)` there, summing to `1e-3`. So the achievable absolute
        // error beside it is `ε·1`, not `ε·|Ei|`, and the fair unit is `ulp(1)` — the suite found
        // `x = 0.37225` at 310 ulp of the *value*, which is `1.85e-8` absolute, or a sixth of one
        // ulp of 1. Away from the zero `|Ei|` exceeds the envelope and this is an ordinary ulp
        // count.
        tracks_f64_at_scale!(
            "exp_int_ei",
            x,
            ALLOWED_EXP_INT_EI,
            1.0,
            Real::exp_int_ei,
            Real::exp_int_ei
        );
    }
}
