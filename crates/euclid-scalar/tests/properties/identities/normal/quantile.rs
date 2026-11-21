//! The quantile, forward and inverse, including the log-CDF round trip into the far tail.
//!
//! Part of the [`normal`](super) suite. Antisymmetry and monotonicity are the laws that hold
//! where no reference does: both are exact statements about a function whose values out there
//! are only known to a few digits.

use super::*;

property! {
    // Its own block, because the coverage requirement below names a label only this property
    // applies and a block-level config reaches every property in its block. Sharing one block
    // made three siblings fail for "required coverage was not met … applied no labels", which is
    // the same footgun `two_sum` and `two_product` are kept apart for.
    #![config(
        settings("normal_quantile_tail")
            .cases(600)
            .require("beyond the erf_inv seed", 15.0)
    )]

    /// `ln Φ(Φ⁻¹(p)) = ln p`, swept from `1e-300` upward — the solver's own law, in the domain
    /// where its seed has to be right.
    ///
    /// Measured at `4.3e-16` relative across `[1e-300, 1e-100]` and no worse than `2.8e-15`
    /// anywhere, so the bound is `1e-14` rather than something that would admit a broken
    /// iteration. Stated in the **log domain** deliberately: that is where both the residual and
    /// the derivative stay finite, and it is the formulation the kernel itself iterates on, so
    /// agreement here is agreement about the thing being solved.
    ///
    /// The coverage requirement is what makes this reach the seed at all. Below `p = 1e-15` the
    /// kernel abandons `erf_inv` — `2p − 1` has rounded to exactly `−1` there — and switches to
    /// an asymptotic start, and that branch is where the eleven surviving seed mutants live. A
    /// generator that mostly produced ordinary probabilities would exercise it a few percent of
    /// the time and silently stop if a range were ever narrowed.
    fn the_quantile_inverts_the_log_cdf_into_the_far_tail(
        p in generate::float::magnitude(-1000, -2),
    ) {
        let p = p.abs();
        if !(p > 0.0 && p <= 0.5) {
            return Ok(());
        }
        classify!("beyond the erf_inv seed", p < 1e-15);
        classify!("within the erf_inv seed", p >= 1e-15);
        classify!("subnormal probability", p < f64::MIN_POSITIVE);

        let x = Real::ndtri(p);
        check!(
            x.is_finite() && x < 0.0,
            "ndtri({p:e}) = {x:e} must be finite and negative"
        );
        relation::agree(
            "ln Phi(ndtri(p)) = ln p",
            (),
            |()| Real::log_ndtr(x),
            |()| p.ln(),
            &Relative(1e-14),
        )?;
    }

}

property! {
    #![config(settings("normal_quantile").cases(600))]

    /// `Φ(Φ⁻¹(p)) = p`, to the accuracy the composition is entitled to.
    ///
    /// The same round trip in probability space, which is the one a caller sees — and it is
    /// *necessarily* looser than the log-domain form above, by an amount worth stating because
    /// it looks like a defect otherwise. `ndtri`'s result carries a relative error of about `ε`,
    /// so an absolute error of about `ε|x|`; `d ln p/dx` is `|x|` in the tail, so that emerges
    /// as a relative error of `ε·x²` in `p`. At `p = 2.9e-291`, `x ≈ −36.6` and the prediction
    /// is `2.9e-13` against a measured `3.7e-13`.
    ///
    /// That is the bound below. A flat `1e-15` would fail honestly and a flat `1e-12` would be
    /// vacuous near `p = ½`.
    fn the_quantile_round_trips_through_the_cdf(p in generate::float::magnitude(-1000, -2)) {
        let p = p.abs();
        if !(p > 0.0 && p <= 0.5) {
            return Ok(());
        }
        let x = Real::ndtri(p);
        if !x.is_finite() {
            return Ok(());
        }
        let back = Real::ndtr(x);
        if back == 0.0 {
            // `Phi` has underflowed; the log-domain property above covers this regime.
            return Ok(());
        }
        relation::agree(
            "Phi(ndtri(p)) = p",
            (),
            |()| back,
            |()| p,
            &Relative(16.0 * f64::EPSILON * (1.0 + x * x)),
        )?;
    }

    /// `Φ⁻¹(1 − p) = −Φ⁻¹(p)`, the quantile's antisymmetry — and a bound set by the *argument*
    /// rather than by the solver.
    ///
    /// # What this property is really measuring
    ///
    /// Written with a flat bound over `[1e-12, 0.5]` it fails at `p = 1e-12` by `3.1e-6`, and
    /// the kernel is not at fault. `1 − p` is representable to an ulp of one, so recovering `p`
    /// from it carries a relative error of `ε/p` — `2.2e-4` at `1e-12` — and the quantile's
    /// `dx/d ln p = 1/|x|` turns that into an absolute `ε/(p|x|)`, about `3.1e-5`. The measured
    /// `3.1e-6` sits an order inside that.
    ///
    /// [`Real::ndtri`]'s docstring says exactly this and says what to do instead: "for an
    /// extreme upper quantile, ask for `-ndtri(q)` with the small `q` directly rather than
    /// `ndtri(1 - q)`; the subtraction is what loses the digits, not the solver." So the first
    /// draft of this property was asserting the thing the documentation warns against. It now
    /// carries the `ε/(p|x|)` term explicitly, which keeps the law meaningful near the median —
    /// where it is `1e-13` — and honest in the tail, where no implementation could do better.
    fn the_quantile_is_antisymmetric(p in generate::float::in_range(1e-9, 0.5)) {
        if !(p > 0.0 && p < 0.5) {
            return Ok(());
        }
        classify!("near the median", p > 0.4);
        classify!("where the subtraction bites", p < 1e-6);

        let upper = Real::ndtri(1.0 - p);
        let lower = Real::ndtri(p);
        if !upper.is_finite() || !lower.is_finite() || lower == 0.0 {
            return Ok(());
        }
        // The error in reconstructing `p` from `1 - p`, carried through the quantile's slope.
        let from_the_subtraction = 8.0 * f64::EPSILON / (p * lower.abs());
        relation::agree(
            "ndtri(1 - p) = -ndtri(p)",
            (),
            |()| upper,
            |()| -lower,
            &Absolute(from_the_subtraction + 1e-13),
        )?;
    }

    /// The quantile's endpoints, its domain, and that it stays monotone and finite into the
    /// subnormal probabilities.
    ///
    /// `ndtri(1e-320)` is `−38.27` — the solver works on a *subnormal* `p`, which is worth
    /// pinning because it is the sort of range a rewrite quietly loses. Outside `[0, 1]` the
    /// answer is `NaN` rather than a plausible number.
    fn the_quantile_respects_its_domain_and_stays_monotone(
        p in generate::float::magnitude(-1070, -2),
    ) {
        check_ulps!(Real::ndtri(0.5f64), 0.0, 0, "the median is exactly zero");
        check!(
            Real::ndtri(0.0f64) == f64::NEG_INFINITY && Real::ndtri(1.0f64) == f64::INFINITY,
            "the quantile must reach both infinities"
        );
        check!(
            Real::ndtri(1.5f64).is_nan()
                && Real::ndtri(-0.5f64).is_nan()
                && Real::ndtri(f64::NAN).is_nan(),
            "outside [0, 1] the quantile must be NaN"
        );

        let p = p.abs();
        if !(p > 0.0 && p < 0.25) {
            return Ok(());
        }
        classify!("subnormal probability", p < f64::MIN_POSITIVE);

        // Monotone increasing in `p`, which a solver that failed to converge would break.
        let x = Real::ndtri(p);
        let x2 = Real::ndtri(p * 2.0);
        check!(
            x.is_finite() && x2.is_finite() && x < x2,
            "ndtri must increase from p = {p:e}: {x:e} then {x2:e}"
        );
    }
}
