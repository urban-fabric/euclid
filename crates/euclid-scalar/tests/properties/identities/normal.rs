//! The standard normal family: CDF, log-CDF, quantile, and the inverse complementary error
//! function.
//!
//! Part of the [`identities`](super) suite.
//!
//! Four public methods — `ndtr`, `log_ndtr`, `ndtri`, `erfc_inv` — with real kernels behind them
//! and **no generated property** before this file. `kernels/normal.rs` and its f32 peer were
//! measured at 33 survivors, the largest cluster being the quantile's Newton seed.
//!
//! # The seed is the point of the deep-tail round trip
//!
//! `ndtri` solves `ln Φ(x) = ln p` by Newton from a seed, and eleven of those survivors are in
//! the seed function: perturb it and the iteration still converges, so the output cannot see it.
//! That is [cause 8](super::super) and the reason an accuracy band on a round trip is the wrong
//! instrument — a band wide enough for the real kernel is wide enough for the mutant.
//!
//! What a *generated* property adds is not a tighter tolerance but a **wider domain**. The fixed
//! suite checks five values of `p`; the property below sweeps `1e-300` upward in log space, and
//! six Newton steps from a bad seed do not converge everywhere. The seed also has a direct
//! accuracy test in the normal kernel.
//!
//! # Two identities that are true and untestable, and the saturation behind both
//!
//! Measured over 4,000-point sweeps before being written:
//!
//! - `log_ndtr(x) = ln(ndtr(x))` reaches a relative error of **0.11** at `x = 7.98`. `Φ(x)`
//!   rounds to `1.0` once `x` passes about `8.3`, so its log is `0` while the true value is
//!   `−8e-17`. That is the exact failure `log_ndtr`'s own docstring describes itself as
//!   existing to prevent, so the identity is asserted only on `x ≤ 0`, where `Φ ≤ ½` and
//!   nothing saturates.
//! - `ndtri(ndtr(x)) = x` reaches **1.5e-3** at `x = 7.97`, for the same reason from the other
//!   side: once `1 − Φ(x)` drops below `ulp(1)` the quantile cannot recover what the
//!   representation threw away.
//!
//! This is the same lesson as the log-domain suite's three rejected laws, and it is worth
//! stating once more as a rule: **where a stable kernel exists, the naive identity it replaces
//! is not available as a test.** The honest check is the one that avoids the regime the kernel
//! was written for, or an independent asymptotic — both are below.

use super::*;

/// `½ln(2π)`, the normal density's log normalizer.
fn half_ln_tau() -> f64 {
    0.5 * (2.0 * core::f64::consts::PI).ln()
}

property! {
    #![config(settings("normal_cdf").cases(500))]

    /// `Φ(x) + Φ(−x) = 1`, with the CDF staying inside `[0, 1]` and hitting its ends.
    ///
    /// The defining symmetry of a symmetric distribution, exact to a single ulp over
    /// `[−40, 40]`, and it needs no reference value. It is also the law that catches a CDF
    /// computed through the wrong tail: using `1 − erf` rather than `erfc` is accurate on one
    /// side and cancels on the other, and the reflection sees the difference.
    fn the_normal_cdf_reflects_about_a_half(x in generate::float::in_range(-40.0, 40.0)) {
        classify!("left tail", x < -5.0);
        classify!("right tail", x > 5.0);

        check_ulps!(
            Real::ndtr(x) + Real::ndtr(-x),
            1.0,
            1,
            "Phi(x) + Phi(-x) must be 1"
        );
        let p = Real::ndtr(x);
        check!(
            (0.0..=1.0).contains(&p),
            "ndtr({x:e}) = {p:e} left [0, 1]"
        );
        // Monotone, and strictly so wherever the format can still tell.
        check!(
            Real::ndtr(x + 1.0) >= p,
            "ndtr must be non-decreasing at {x:e}"
        );
    }

    /// The CDF's endpoints and the log-CDF's, including a signed zero that is a real claim.
    ///
    /// `log_ndtr(+∞)` is `−0.0`, not `+0.0`: `ln Φ` approaches zero *from below* because `Φ < 1`
    /// everywhere, and the sign bit is the only thing left recording that at the limit. A
    /// carrier that returned `+0.0` would be indistinguishable by comparison and wrong by
    /// convention, which is exactly the case [`Real::signbit`] exists to express.
    fn the_normal_endpoints_are_exact(x in generate::float::in_range(-1e3, 1e3)) {
        check_ulps!(Real::ndtr(0.0f64), 0.5, 0, "Phi(0) is exactly a half");
        check_ulps!(
            Real::log_ndtr(0.0f64),
            -core::f64::consts::LN_2,
            0,
            "ln Phi(0) is exactly -ln 2"
        );
        check!(
            Real::ndtr(f64::NEG_INFINITY) == 0.0 && Real::ndtr(f64::INFINITY) == 1.0,
            "the CDF must reach both ends"
        );
        check!(
            Real::ndtr(f64::NAN).is_nan() && Real::log_ndtr(f64::NAN).is_nan(),
            "NaN must propagate"
        );

        let limit = Real::log_ndtr(f64::INFINITY);
        check!(
            limit == 0.0 && Real::signbit(&limit),
            "ln Phi(+inf) must be -0.0, approaching zero from below; got {limit:e}"
        );

        // And the log-CDF is never positive, anywhere, because `Phi <= 1`.
        let l = Real::log_ndtr(x);
        check!(
            l <= 0.0 || l.is_nan(),
            "ln Phi({x:e}) = {l:e} must not be positive"
        );
    }

    /// `log_ndtr(x) = ln(Φ(x))` on `x ≤ 0`, where the CDF has not saturated.
    ///
    /// The range is the whole content of this property: see the module header for the `0.11`
    /// relative error the unrestricted form reaches on the right. Below zero `Φ ≤ ½`, the log is
    /// well conditioned, and the two must agree — which pins the direct branch against the
    /// `erfcx` one through their shared boundary.
    ///
    /// The lower limit is where `Φ` underflows to zero and `ln` of it is `−∞` while `log_ndtr`
    /// is still a perfectly ordinary number. That divergence is the function's entire purpose,
    /// so it is declined rather than asserted.
    fn log_ndtr_agrees_with_the_log_of_the_cdf_where_that_is_representable(
        x in generate::float::in_range(-37.0, 0.0),
    ) {
        let p = Real::ndtr(x);
        if !(p > 0.0 && p < 1.0) {
            return Ok(());
        }
        classify!("below minus one", x < -1.0);
        classify!("near zero", x > -1.0);

        relation::agree(
            "log_ndtr = ln(ndtr) on the left",
            (),
            |()| Real::log_ndtr(x),
            |()| p.ln(),
            &Relative(1e-14),
        )?;
    }

    /// `ln Φ(x) → −x²/2 − ln|x| − ln√(2π)` in the left tail, against the asymptotic's own
    /// error term.
    ///
    /// The one genuinely **independent** reference available here: it shares no code with the
    /// kernel, and it reaches `x = −1e6` where `Φ` underflowed hundreds of orders ago. This is
    /// the regime `log_ndtr` was written for, so it is the regime most worth pinning.
    ///
    /// # The bound is the asymptotic's truncation, not the kernel's error
    ///
    /// The next term of the series is `ln(1 − 1/x² + 3/x⁴ − …)`, so the absolute error is about
    /// `1/x²` against a value of about `x²/2` — a relative error of `2/x⁴`. At `x = −50` that
    /// predicts `3.2e-7` and the sweep measures `3.184e-7`, which is the agreement that makes
    /// this a bound rather than a fudge. A constant tolerance here would be vacuous at `1e6`
    /// and wrong at `50`.
    fn log_ndtr_matches_its_left_tail_asymptotic(
        x in generate::float::in_range(-1e6, -50.0),
    ) {
        classify!("deep tail", x < -1e4);

        let asymptotic = -0.5 * x * x - (-x).ln() - half_ln_tau();
        relation::agree(
            "ln Phi ~ -x^2/2 - ln|x| - ln sqrt(2 pi)",
            (),
            |()| Real::log_ndtr(x),
            |()| asymptotic,
            &Relative(4.0 / (x * x * x * x) + 64.0 * f64::EPSILON),
        )?;
    }
}

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

property! {
    #![config(settings("erfc_inv").cases(500))]

    /// `erfc⁻¹(erfc(x)) = x` for `x ≥ 0`, and `erfc(erfc⁻¹(y)) = y` into the far tail.
    ///
    /// Restricted to `x ≥ 0` for the reason the module header gives: `erfc(x) → 2` on the left
    /// and the information in `2 − erfc(x)` falls below `ulp(2)`, so the inverse cannot recover
    /// it — measured `1.3e-6` at `x = −4.98`. On the right `erfc` decays to zero, which loses
    /// nothing in relative terms, and the round trip holds all the way out.
    ///
    /// The reverse trip carries the same `ε·x²` amplification as the quantile's, and for the
    /// same reason: `erfc_inv` is `ndtri` in disguise — `erfc⁻¹(y) = −Φ⁻¹(y/2)/√2` — which is
    /// also why it reaches `y = 1.6e-288`.
    fn erfc_inv_inverts_erfc_on_the_side_that_keeps_its_digits(
        x in generate::float::in_range(0.0, 25.0),
    ) {
        classify!("tail", x > 5.0);

        let y = Real::erfc(x);
        if y > 0.0 && y < 2.0 {
            relation::agree(
                "erfc_inv(erfc(x)) = x",
                (),
                |()| Real::erfc_inv(y),
                |()| x,
                &Absolute(16.0 * f64::EPSILON * (1.0 + x * x)),
            )?;
        }

        // The endpoints are the contract's, and out-of-domain is NaN rather than a guess.
        check!(
            Real::erfc_inv(0.0f64) == f64::INFINITY
                && Real::erfc_inv(2.0f64) == f64::NEG_INFINITY,
            "erfc_inv must reach both infinities"
        );
        check!(
            Real::erfc_inv(2.5f64).is_nan()
                && Real::erfc_inv(-0.1f64).is_nan()
                && Real::erfc_inv(f64::NAN).is_nan(),
            "outside [0, 2] erfc_inv must be NaN"
        );
    }

    /// The same inverse from the other side, swept in log space so it reaches the tail.
    fn erfc_recovers_what_its_inverse_was_given(y in generate::float::magnitude(-1000, -1)) {
        let y = y.abs();
        if !(y > 0.0 && y < 1.0) {
            return Ok(());
        }
        classify!("deep tail", y < 1e-100);

        let x = Real::erfc_inv(y);
        if !x.is_finite() {
            return Ok(());
        }
        relation::agree(
            "erfc(erfc_inv(y)) = y",
            (),
            |()| Real::erfc(x),
            |()| y,
            &Relative(16.0 * f64::EPSILON * (1.0 + x * x)),
        )?;
    }
}
