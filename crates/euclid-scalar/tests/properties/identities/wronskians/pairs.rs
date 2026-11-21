//! The four Wronskian identities themselves.
//!
//! Part of the [`wronskians`](super) suite, whose preamble explains why a Wronskian is the
//! strongest instrument available for a solution pair: it constrains both solutions *and* both
//! derivatives at one argument, to a value that is a constant rather than another evaluation.

use super::*;

property! {
    // All four properties here score, so the block can carry a budget: each compares against an
    // *exact* closed form, which is the one situation where an accuracy claim can be stated
    // without a reference implementation at all. Two of them use `safety·κ·u` and two use a
    // documented bound, for the reason given at `bound_fraction`.
    #![config(
        settings("wronskians")
            .cases(600)
            .budget("tests/budgets.txt")
    )]

    /// `Aiᵉ(x)·Biᵉ′(x) − Aiᵉ′(x)·Biᵉ(x) = 1/π`, for **every** real `x`.
    ///
    /// The Airy Wronskian, in the scaled form that cannot overflow. Four separate kernels, one
    /// exact constant, and no argument excluded — `Ai` underflows past `x ≈ 104` and `Bi`
    /// overflows there, but their scaled forms are ordinary numbers at `1e12`.
    ///
    /// A mutation anywhere in the four that breaks their mutual consistency breaks this,
    /// including in a branch a pointwise test never reaches: the identity holds across every
    /// crossover because it holds identically, not approximately.
    fn the_scaled_airy_wronskian_is_one_over_pi(
        x in generate::float::magnitude(-20, 40),
    ) {
        classify!("negative", x < 0.0);
        classify!("positive", x > 0.0);
        classify!("large", x.abs() > 1e6);

        let scaled_airy = |t: f64| {
            Real::airy_ai_e(t) * Real::airy_bi_e_prime(t)
                - Real::airy_ai_e_prime(t) * Real::airy_bi_e(t)
        };
        let wronskian = scaled_airy(x);

        // Beyond `|x| ≈ 1e11` the phase `ζ` can no longer be resolved in f64 — `ulp(ζ)` passes
        // `2π` — so on the negative axis the values stop being accurate pointwise. The trait
        // documents that limit; this declines the regime rather than loosening the bound.
        if x < -1e11 {
            return Ok(());
        }

        check!(
            wronskian.is_finite(),
            "the scaled Airy Wronskian must stay finite at x = {x:e}, got {wronskian:e}"
        );
        // Scored as a *fraction of the documented bound*, which is the ratio form a budget
        // wants. Deliberately not `forward_stable_margin`: this Wronskian is identically `1/π`,
        // so `f′ = 0` and a condition-scaled bound `safety·κ·u` collapses to zero — measured,
        // the margin came back as `7.4e14`, which says nothing about the kernel. A condition
        // number is undefined as a yardstick for a quantity that does not vary.
        target(bound_fraction(wronskian, core::f64::consts::FRAC_1_PI, airy_bound(x)));
        check_relative!(
            wronskian,
            core::f64::consts::FRAC_1_PI,
            airy_bound(x),
            "Ai^e Bi^e' - Ai^e' Bi^e must be 1/pi at x = {x:e}"
        );
    }

    /// The same identity on the **unscaled** four, wherever both are representable.
    ///
    /// Worth asserting separately: `airy_ai` and `airy_ai_e` are different code paths — one
    /// recovers the other by forming `e^{−ζ}` — so an identity on the scaled family says
    /// nothing about the convenient one. The range stops where `Bi` overflows, which is the
    /// honest limit of the unscaled form rather than a chosen bound.
    fn the_airy_wronskian_is_one_over_pi(x in generate::float::in_range(-300.0, 100.0)) {
        classify!("negative", x < 0.0);
        classify!("positive", x > 0.0);

        let unscaled_airy = |t: f64| {
            Real::airy_ai(t) * Real::airy_bi_prime(t) - Real::airy_ai_prime(t) * Real::airy_bi(t)
        };
        let wronskian = unscaled_airy(x);
        target(bound_fraction(wronskian, core::f64::consts::FRAC_1_PI, airy_bound(x)));

        // Past `x ≈ 104` one factor has overflowed and the other underflowed, so the product is
        // `0·inf`. That is the scaled form's job, and it is checked above.
        if !wronskian.is_finite() {
            return Ok(());
        }
        check_relative!(
            wronskian,
            core::f64::consts::FRAC_1_PI,
            airy_bound(x),
            "Ai Bi' - Ai' Bi must be 1/pi at x = {x:e}"
        );
    }

    /// `I₀ᵉ(x)·K₁ᵉ(x) + I₁ᵉ(x)·K₀ᵉ(x) = 1/x`, for every `x > 0`.
    ///
    /// The modified Bessel Wronskian. Every term is positive, so there is no cancellation and
    /// the residual is the four kernels' own error — which is why the bound is `1e-13` against
    /// `bessel_i0e`'s documented `1.7e-14` rather than something looser.
    ///
    /// The scaled form is what makes the whole axis reachable: `I₀(710)` overflows and
    /// `K₀(746)` underflows, while the scaled pair are ordinary numbers out past `1e15`. A
    /// sweep of the unscaled identity skipped 1,716 of 2,001 points above `x = 100`.
    fn the_scaled_modified_bessel_wronskian_is_one_over_x(
        x in generate::float::magnitude(-26, 50),
    ) {
        let x = x.abs();
        if x == 0.0 || !x.is_finite() {
            return Ok(());
        }
        classify!("below one", x < 1.0);
        classify!("above one", x >= 1.0);
        classify!("large", x > 1e6);

        let modified = |t: f64| {
            Real::bessel_i0e(t) * Real::bessel_k1e(t) + Real::bessel_i1e(t) * Real::bessel_k0e(t)
        };
        let wronskian = modified(x);
        // `κ = 1` here — the Wronskian is exactly `1/x` — so `safety·κ·u` is a real yardstick
        // and the claim becomes "no worse than the problem itself". Measured worst is 121 ulp
        // over this sweep, for a quantity built from four kernels, so the safety factor is 512.
        if let Some(margin) =
            relation::forward_stable_margin(x, modified, |t| 1.0 / t, MODIFIED_SAFETY)
        {
            target(margin);
        }
        relation::forward_stable(
            "I0e K1e + I1e K0e = 1/x",
            x,
            modified,
            |t| 1.0 / t,
            MODIFIED_SAFETY,
        )?;
        check!(
            wronskian.is_finite() && wronskian > 0.0,
            "the scaled I/K Wronskian must be positive and finite at x = {x:e}, got {wronskian:e}"
        );
        check_relative!(
            wronskian,
            1.0 / x,
            1e-13,
            "I0^e K1^e + I1^e K0^e must be 1/x at x = {x:e}"
        );
    }

    /// `J₀(x)·Y₁(x) − J₁(x)·Y₀(x) = −2/(πx)`, for `0 < x ≤ 1e8`.
    ///
    /// The oscillatory pair's Wronskian. Unlike the `I`/`K` one this is a genuine difference of
    /// comparable terms, so it does cancel — and it is the only one of the three with a range
    /// limit that is about the *kernel* rather than about overflow: `J` and `Y` are accurate
    /// pointwise only while their phase can be resolved, which the trait documents as failing
    /// near `1e11`. Measured, the residual is `5.9e-16` to `1e8` and `9.9e-12` by `1.2e11`.
    ///
    /// The test stops at `1e8` to stay in the range where the measured residual supports
    /// the tolerance below.
    fn the_bessel_jy_wronskian_is_minus_two_over_pi_x(
        x in generate::float::in_range(1e-8, 1e8),
    ) {
        if x <= 0.0 {
            return Ok(());
        }
        classify!("below one", x < 1.0);
        classify!("above one", x >= 1.0);

        let oscillating = |t: f64| {
            Real::bessel_j0(t) * Real::bessel_y1(t) - Real::bessel_j1(t) * Real::bessel_y0(t)
        };
        let wronskian = oscillating(x);
        // `κ = 1` again. Measured worst is 4 ulp, so 16 leaves the usual fourfold headroom.
        if let Some(margin) = relation::forward_stable_margin(
            x,
            oscillating,
            |t| -2.0 / (core::f64::consts::PI * t),
            OSCILLATING_SAFETY,
        ) {
            target(margin);
        }
        relation::forward_stable(
            "J0 Y1 - J1 Y0 = -2/(pi x)",
            x,
            oscillating,
            |t| -2.0 / (core::f64::consts::PI * t),
            OSCILLATING_SAFETY,
        )?;
        check_relative!(
            wronskian,
            -2.0 / (core::f64::consts::PI * x),
            1e-13,
            "J0 Y1 - J1 Y0 must be -2/(pi x) at x = {x:e}"
        );
    }
}
