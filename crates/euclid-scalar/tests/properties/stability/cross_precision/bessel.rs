//! The Bessel family at `f32`, against the `f64` kernel.
//!
//! Part of the [`cross_precision`](super) suite. The scaled forms are compared rather than the
//! unscaled ones wherever the unscaled value leaves `f32`'s range long before `f64`'s.

use super::*;

property! {
    #![config(settings("f32_precision_i0e")
        .require("i0e no-log", 2.0)
        .require("i0e peak-factored", 27.0)
        .require("i0e asymptotic", 18.0))]

    /// `I₀ᵉ` at `f32`, across all three forms the regime split selects.
    ///
    /// The domain deliberately spans `I1_NO_LOG_MAX` at `1` and `I_ASYMPTOTIC_MIN` at `100`, and
    /// all three are required. The asymptotic side is the one worth insisting on: the series is
    /// wrong by a factor of 15 by `x = 1e8` while staying finite and plausible, so the handover
    /// is load bearing rather than an optimisation.
    fn the_f32_i0e_tracks_f64(x in generate::float32::in_range(0.0, 400.0)) {
        if !x.is_finite() || x < 0.0 {
            return Ok(());
        }
        classify!("i0e no-log", x < 1.0);
        classify!("i0e peak-factored", (1.0..100.0).contains(&x));
        classify!("i0e asymptotic", x >= 100.0);
        tracks_f64!("bessel_i0e", x, ALLOWED_I0E, Real::bessel_i0e, Real::bessel_i0e);
    }
}

property! {
    #![config(settings("f32_precision_i1e")
        .require("i1e no-log", 2.0)
        .require("i1e peak-factored", 27.0)
        .require("i1e asymptotic", 18.0))]

    /// `I₁ᵉ` at `f32`. Unlike `I₀ᵉ` this one really does take the no-logarithm route, where
    /// `I₁ ~ x/2` must come back exact.
    ///
    /// Swept on the positive axis only, deliberately: `I₁ᵉ` is odd, so the narrowing gap at `−x`
    /// is the mirror of the gap at `x` and a two-sided sweep would measure each argument twice.
    /// The *sign* is what a one-sided sweep cannot see, and `cargo mutants` duly found that it
    /// was unchecked here — it is now pinned by `the_first_kind_is_odd_in_every_form` beside the
    /// kernel, which is the stronger assertion since it is exact rather than budgeted.
    fn the_f32_i1e_tracks_f64(x in generate::float32::in_range(0.0, 400.0)) {
        if !x.is_finite() || x < 0.0 {
            return Ok(());
        }
        classify!("i1e no-log", x < 1.0);
        classify!("i1e peak-factored", (1.0..100.0).contains(&x));
        classify!("i1e asymptotic", x >= 100.0);
        tracks_f64!("bessel_i1e", x, ALLOWED_I1E, Real::bessel_i1e, Real::bessel_i1e);
    }
}

property! {
    #![config(settings("f32_precision_k0e")
        .require("k0e series", 2.0)
        .require("k0e quadrature", 46.0))]

    /// `K₀ᵉ` at `f32`, across the series/quadrature handover the port moved to `1.5`.
    fn the_f32_k0e_tracks_f64(x in generate::float32::in_range(0.01, 200.0)) {
        if !x.is_finite() || x <= 0.0 {
            return Ok(());
        }
        classify!("k0e series", x <= 1.5);
        classify!("k0e quadrature", x > 1.5);
        tracks_f64!("bessel_k0e", x, ALLOWED_K0E, Real::bessel_k0e, Real::bessel_k0e);
    }
}

// ---- oscillating, Airy, and the transcendentals -------------------------

property! {
    #![config(settings("f32_precision_j0")
        .require("j0 series", 3.0)
        .require("j0 march", 24.0)
        .require("j0 Hankel", 20.0))]

    /// `J₀` at `f32`, counted against its amplitude envelope `√(2/πx)`.
    ///
    /// `J₀` has infinitely many zeros in this domain, so an ulp count against the value measures
    /// distance to the nearest zero rather than the quality of the kernel — see
    /// [`narrowing_margin_at_scale`]. The envelope is the Hankel amplitude, capped at `1` because
    /// it diverges as `x → 0` while `J₀(0) = 1`.
    fn the_f32_j0_tracks_f64(x in generate::float32::in_range(0.0, 40.0)) {
        if !x.is_finite() || x < 0.0 {
            return Ok(());
        }
        classify!("j0 series", x < 2.0);
        classify!("j0 march", (2.0..7.0).contains(&x));
        classify!("j0 Hankel", x >= 7.0);
        let envelope = if x > 0.0 {
            (2.0 / (core::f32::consts::PI * x)).sqrt().min(1.0)
        } else {
            1.0
        };
        tracks_f64_at_scale!(
            "bessel_j0",
            x,
            ALLOWED_J0,
            envelope,
            Real::bessel_j0,
            Real::bessel_j0
        );
    }
}
