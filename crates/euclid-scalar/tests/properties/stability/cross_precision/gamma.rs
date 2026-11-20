//! The gamma family at `f32`, against the `f64` kernel.
//!
//! Part of the [`cross_precision`](super) suite, whose preamble holds the shared ulp-budget
//! machinery and explains what a cross-precision budget does and does not claim.

use super::*;

property! {
    #![config(settings("f32_precision_gamma"))]

    /// `Γ` at `f32`, against the `f64` kernel rounded.
    ///
    /// The domain crosses the small-shape route, the Lanczos core and the recurrence, and stops
    /// below the overflow at `Γ(35)` so the comparison stays defined.
    fn the_f32_gamma_tracks_f64(x in generate::float32::in_range(0.02, 34.0)) {
        if !x.is_finite() || x <= 0.0 {
            return Ok(());
        }
        tracks_f64!("gamma", x, ALLOWED_GAMMA, Real::gamma, Real::gamma);
    }
}

property! {
    #![config(settings("f32_precision_ln_gamma")
        .require("ln_gamma recurrence", 2.5)
        .require("ln_gamma asymptotic", 45.0))]

    /// `lnΓ` at `f32`, across the recurrence/asymptotic handover at `10`.
    ///
    /// Both sides are required: the recurrence walks up to the threshold and the asymptotic
    /// takes over above it, and a climb that stopped reaching one of them would be reporting a
    /// clean pass over half the kernel.
    fn the_f32_ln_gamma_tracks_f64(x in generate::float32::in_range(0.02, 400.0)) {
        if !x.is_finite() || x <= 0.0 {
            return Ok(());
        }
        classify!("ln_gamma recurrence", x < 10.0);
        classify!("ln_gamma asymptotic", x >= 10.0);
        // `lnΓ` has two zeros in this domain, at `x = 1` and `x = 2`, and they are the reason the
        // fixed grid quotes 72 ulp for this kernel while the climb measured 4: the grid happens
        // to land near one of them and a hill climb on a thin feature does not have to. Both are
        // cancellation zeros — `Γ` is near 1 and `ln` of a value near 1 has absolute accuracy,
        // not relative — so `ulp(1)` is the honest unit. Above `x ≈ 2.4` the value exceeds the
        // envelope and this reverts to an ordinary ulp count.
        tracks_f64_at_scale!(
            "ln_gamma",
            x,
            ALLOWED_LN_GAMMA,
            1.0,
            Real::ln_gamma,
            Real::ln_gamma
        );
    }
}

property! {
    #![config(settings("f32_precision_digamma")
        .require("psi recurrence", 4.0)
        .require("psi asymptotic", 22.0))]

    /// `ψ` at `f32`, across `PSI_ASYMPTOTIC_MIN`, which the port rederived at `6` rather than
    /// inheriting the `f64` value.
    fn the_f32_digamma_tracks_f64(x in generate::float32::in_range(0.02, 60.0)) {
        if !x.is_finite() || x <= 0.0 {
            return Ok(());
        }
        classify!("psi recurrence", x < 6.0);
        classify!("psi asymptotic", x >= 6.0);
        // `ψ` has a zero at `1.4616`, and like `Ei`'s it is a cancellation of `O(1)` terms rather
        // than an analytic one, so the envelope is `ulp(1)`. This was a latent flake rather than
        // an observed failure: the measured worst was 4 ulp because *this* seed's climb never
        // got close enough to the zero. A kernel whose test passes only while the search misses
        // a thin region is a kernel with no test there.
        tracks_f64_at_scale!(
            "digamma",
            x,
            ALLOWED_DIGAMMA,
            1.0,
            Real::digamma,
            Real::digamma
        );
    }
}

property! {
    #![config(settings("f32_precision_trigamma"))]

    /// `ψ'` at `f32`. Steeper than `ψ` near the origin, where it diverges like `1/x²`.
    fn the_f32_trigamma_tracks_f64(x in generate::float32::in_range(0.02, 60.0)) {
        if !x.is_finite() || x <= 0.0 {
            return Ok(());
        }
        tracks_f64!("trigamma", x, ALLOWED_TRIGAMMA, Real::trigamma, Real::trigamma);
    }
}

// ---- the error function family ------------------------------------------
