//! The sigmoid, softplus and logit.
//!
//! Part of the [`log_domain`](super) suite. `logit` inverts the sigmoid only to the precision
//! the *representation* allows — the sigmoid saturates to one long before its argument runs out
//! — which is why that law carries its limit in its name.

use super::*;

property! {
    #![config(settings("sigmoid").cases(500))]

    /// The sigmoid's reflections, one exact and one to a single ulp.
    ///
    /// `ln_1m_sigmoid(x) = ln_sigmoid(−x)` is **bit-exact** — measured at 0 ulps across
    /// `[−700, 700]` — because `1 − σ(x) = σ(−x)` identically and both are routed through the
    /// same `ln_1p_exp`. Asserting it at 0 ulps pins that routing: an implementation that
    /// computed `ln(1 − σ(x))` instead would still be *accurate*, and would no longer be exact.
    ///
    /// `σ(x) + σ(−x) = 1` is within one ulp over the same range, which is as close as a sum of
    /// two rounded halves can come.
    fn the_sigmoid_reflects_about_a_half(x in generate::float::in_range(-700.0, 700.0)) {
        classify!("saturated", x.abs() > 36.0);

        check_ulps!(
            Real::ln_1m_sigmoid(&x),
            Real::ln_sigmoid(&-x),
            0,
            "ln(1 - sigmoid(x)) must be bit-exactly ln(sigmoid(-x))"
        );
        check_ulps!(
            Real::sigmoid(&x) + Real::sigmoid(&-x),
            1.0,
            1,
            "sigmoid(x) + sigmoid(-x) must be 1"
        );
        // And it stays in its range, which is what "logistic" means.
        let s = Real::sigmoid(&x);
        check!(
            (0.0..=1.0).contains(&s),
            "sigmoid({x:e}) = {s:e} left [0, 1]"
        );
    }

    /// `ln_1p_exp(x) = x + ln_1p_exp(−x)`, for `x ≥ 0`.
    ///
    /// Softplus's reflection, stated as a **sum**. As a difference it is not a law in `f64` at
    /// all — see this module's header — because `ln_1p_exp(x)` rounds to exactly `x` once
    /// `e^{−x}` falls below `ulp(x)`, around `x = 34`. Written this way both sides are
    /// `x + something small` and the comparison is well conditioned everywhere, including in
    /// exactly that regime: the small term simply stops mattering to both sides at once.
    fn softplus_reflects_through_its_argument(x in generate::float::in_range(0.0, 700.0)) {
        classify!("beyond the rounding point", x > 34.0);

        relation::agree(
            "ln_1p_exp(x) = x + ln_1p_exp(-x)",
            (),
            |()| Real::ln_1p_exp(&x),
            |()| x + Real::ln_1p_exp(&-x),
            &Relative(1e-15),
        )?;
        // Softplus is above both `0` and `x`, which brackets it the way `ln 2` brackets the
        // log-sum-exp — and indeed it *is* `ln_add_exp(x, 0)`.
        let softplus = Real::ln_1p_exp(&x);
        check!(
            softplus >= x && softplus >= 0.0,
            "ln_1p_exp({x:e}) = {softplus:e} must dominate both 0 and x"
        );
        check_ulps!(
            softplus,
            Real::ln_add_exp(x, 0.0),
            0,
            "ln_1p_exp(x) must agree with ln_add_exp(x, 0)"
        );
    }

    /// `logit(sigmoid(x)) = x`, to the accuracy the **representation** of `σ(x)` allows.
    ///
    /// # Why the bound carries `e^{|x|}`
    ///
    /// The composition is perfectly conditioned in exact arithmetic — `logit` is `sigmoid`'s
    /// inverse, so the derivative of the round trip is one. What degrades is not the
    /// arithmetic but the *storage*: `σ(x) = 1 − e^{−x} + …` approaches `1`, and once
    /// `1 − σ(x)` falls below `ulp(1)` the information `logit` needs is simply not in the
    /// number any more. `logit`'s derivative is `1/(p(1−p))`, so an `ε`-sized error in `p`
    /// emerges as `ε/(1−p) ≈ ε·e^{|x|}`.
    ///
    /// Measured, that is exactly what happens: `2.2e-13` on `[−1,1]`, `3.2e-13` on `[−10,10]`,
    /// `2.4e-9` on `[−20,20]`, `3.7e-5` on `[−30,30]`. A constant tolerance would be vacuous at
    /// one end or wrong at the other — the same lesson `exp_and_ln_round_trip` records — so the
    /// bound is absolute and carries the amplification.
    fn logit_inverts_the_sigmoid_to_the_precision_the_representation_allows(
        x in generate::float::in_range(-30.0, 30.0),
    ) {
        classify!("near saturation", x.abs() > 20.0);
        classify!("central", x.abs() < 1.0);

        let round_trip = Real::logit(&Real::sigmoid(&x));
        if !round_trip.is_finite() {
            return Ok(());
        }
        // `ε·e^{|x|}` is the information lost to representing `σ(x)` near one, plus a floor for
        // the two evaluations themselves.
        let bound = 8.0 * f64::EPSILON * (1.0 + x.abs().exp());
        relation::agree(
            "logit(sigmoid(x)) = x",
            (),
            |()| round_trip,
            |()| x,
            &Absolute(bound),
        )?;
    }
}
