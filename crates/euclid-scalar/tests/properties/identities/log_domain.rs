//! The log-domain family: log-sum-exp, the sigmoid/logit pair, and the absorbing-zero products.
//!
//! Part of the [`identities`](super) suite.
//!
//! These twenty-odd methods are the ones a probabilistic consumer actually calls, every one of
//! them is real logic rather than a `std` delegation, and **none of them had a generated
//! property** before this file.
//!
//! # Two of these laws are exact, which is rarer than it sounds
//!
//! Measured over 4,000-point sweeps before being written here, `ln_add_exp` is **bit-exactly
//! symmetric** and `ln_1m_sigmoid(x)` is **bit-exactly** `ln_sigmoid(−x)` across `[−700, 700]`.
//! Both are asserted at zero ulps, which is the strongest statement available and the one a
//! future refactor is most likely to break: either could become "accurate to a few ulps" without
//! any accuracy test noticing.
//!
//! `max(a,b) ≤ ln_add_exp(a,b) ≤ max(a,b) + ln 2` is exact too, and holds at *every* argument
//! including the infinities — a bound rather than a value, so nothing can round it away.
//!
//! # Three laws that are true in mathematics and false in `f64`
//!
//! The measurement rejected three identities this file was drafted with, and they are recorded
//! because the mistake is the same one each time: **an identity that subtracts two nearly-equal
//! quantities is not a test, it is a cancellation.**
//!
//! - `ln_sigmoid(x) = ln(sigmoid(x))` reaches a relative error of **0.49** at `x = 36.7`.
//!   `sigmoid` saturates to exactly `1.0` there, so `ln(sigmoid(x))` is `0` while the true value
//!   is about `−1.1e-16`. That saturation is the entire reason `ln_sigmoid` exists, so asserting
//!   the naive form would be asserting the bug.
//! - `ln_1p_exp(x) − x = ln_1p_exp(−x)` reaches a relative error of **1.0** at `x = 33.6`, where
//!   `ln_1p_exp(x)` has rounded to exactly `x` and the left side is exactly zero. Restated as a
//!   *sum* — `ln_1p_exp(x) = x + ln_1p_exp(−x)` — it is well conditioned for `x ≥ 0` and is
//!   asserted below.
//! - `logit(sigmoid(x)) = x` degrades from `2.2e-13` on `[−1,1]` to `1.1e-2` on `[−36,36]`,
//!   because `1 − sigmoid(x)` falls below `ulp(1)` and the round trip cannot recover what the
//!   representation discarded. Its bound carries `e^{|x|}` explicitly for that reason.
//!
//! # And one convention worth pinning because it is a choice
//!
//! `xlogy(0, y)` is `0` for **every** `y` — including `±∞` and `NaN`. The absorbing zero wins
//! over NaN propagation, which is defensible and is not the only defensible answer, so it is
//! asserted rather than left to be rediscovered.

use super::*;

property! {
    #![config(settings("log_sum_exp").cases(500))]

    /// `ln_add_exp` is symmetric and sits in `[max, max + ln 2]`, both exactly.
    ///
    /// The bound is the whole content of the stable formulation: `ln(eᵃ + eᵇ)` is
    /// `max + ln(1 + e^{−|a−b|})`, and the second term lies in `(0, ln 2]` by construction. So
    /// the law holds at every argument, needs no tolerance, and is false for any implementation
    /// that overflows or picks the wrong operand as the pivot.
    ///
    /// Symmetry is bit-exact because `max` and `|a − b|` both are. Measured at 0 ulps over
    /// 40,000 pairs spanning `[−600, 600]`, so it is asserted at 0 rather than "a few".
    fn the_log_sum_exp_is_symmetric_and_sits_in_its_bracket(
        a in generate::float::in_range(-600.0, 600.0),
        b in generate::float::in_range(-600.0, 600.0),
    ) {
        classify!("far apart", (a - b).abs() > 50.0);
        classify!("close together", (a - b).abs() <= 1.0);

        let result = Real::ln_add_exp(a, b);
        check_ulps!(
            result,
            Real::ln_add_exp(b, a),
            0,
            "ln_add_exp must be bit-exactly symmetric"
        );

        let pivot = a.max(b);
        check!(
            result >= pivot,
            "ln_add_exp({a:e}, {b:e}) = {result:e} fell below max = {pivot:e}"
        );
        check!(
            result <= pivot + core::f64::consts::LN_2,
            "ln_add_exp({a:e}, {b:e}) = {result:e} exceeded max + ln 2"
        );
    }

    /// `ln_add_exp(a + c, b + c) = ln_add_exp(a, b) + c` — shift invariance.
    ///
    /// The defining property of a log-sum-exp, and the reason it is the stable form: shifting by
    /// the pivot is exactly what the implementation does internally, so an implementation that
    /// is *not* shift-invariant has lost the plot entirely. No reference value is involved.
    ///
    /// # The comparison must be absolute, not relative
    ///
    /// A first draft compared relatively and measured `1.8e-1`, which says nothing about the
    /// kernel: `ln_add_exp(a,b) + c` passes through zero, and relative error against a reference
    /// that is nearly zero is meaningless. The bound here is absolute and scaled by the inputs,
    /// because the error is the rounding of `a + c` and `b + c` — each `ε` relative to its own
    /// magnitude — carried through a function whose derivative in each argument is at most one.
    fn the_log_sum_exp_is_shift_invariant(
        a in generate::float::in_range(-300.0, 300.0),
        b in generate::float::in_range(-300.0, 300.0),
        c in generate::float::in_range(-300.0, 300.0),
    ) {
        let base = Real::ln_add_exp(a, b);
        let shifted = Real::ln_add_exp(a + c, b + c);
        if !base.is_finite() || !shifted.is_finite() {
            return Ok(());
        }
        let scale = a.abs().max(b.abs()).max(c.abs()).max(1.0);
        relation::agree(
            "ln_add_exp is shift invariant",
            (),
            |()| shifted,
            |()| base + c,
            &Absolute(16.0 * f64::EPSILON * scale),
        )?;
    }

    /// Against the naive `ln(eᵃ + eᵇ)`, wherever the naive form is representable at all.
    ///
    /// The differential check: same answer, wider domain. Restricted to `|a|, |b| ≤ 300` so the
    /// exponentials exist, and absolute for the same reason as above — the reference crosses
    /// zero. Outside that range the naive form is `inf` or `0` and the stable one still works,
    /// which the bracket property above covers.
    fn the_log_sum_exp_agrees_with_the_naive_form_where_that_exists(
        a in generate::float::in_range(-300.0, 300.0),
        b in generate::float::in_range(-300.0, 300.0),
    ) {
        let naive = (a.exp() + b.exp()).ln();
        if !naive.is_finite() {
            return Ok(());
        }
        let scale = a.abs().max(b.abs()).max(1.0);
        relation::agree(
            "ln_add_exp agrees with ln(exp a + exp b)",
            (),
            |()| Real::ln_add_exp(a, b),
            |()| naive,
            &Absolute(16.0 * f64::EPSILON * scale),
        )?;
    }

    /// The identities at the edges: `−∞` is the additive identity, and a `NaN` in **either**
    /// operand propagates.
    ///
    /// `e^{−∞}` is zero, so `ln_add_exp(a, −∞)` must be exactly `a` — the identity element of a
    /// log-domain sum, which is what makes it usable as a fold's initial value.
    ///
    /// The NaN half is drawn with `pair_with_edge_cases` rather than written as fixed points,
    /// because `ln_add_exp` opens with `if self.is_nan() || other.is_nan()` and that is the guard
    /// shape this repository has now found two real defects in: weakened to `&&`, it needs a
    /// test at *each* operand to catch.
    fn the_log_sum_exp_handles_its_identity_and_both_nans(
        pair in generate::float::pair_with_edge_cases(-20, 20),
    ) {
        let (a, y) = pair;
        classify!("nan operand", y.is_nan());
        classify!("infinite operand", y.is_infinite());

        if a.is_nan() || y.is_nan() {
            check!(
                Real::ln_add_exp(a, y).is_nan() && Real::ln_add_exp(y, a).is_nan(),
                "a NaN operand must propagate: ln_add_exp({a:e}, {y:e})"
            );
            return Ok(());
        }

        // `-inf` is the identity, from either side.
        check_ulps!(
            Real::ln_add_exp(a, f64::NEG_INFINITY),
            a,
            0,
            "ln_add_exp(a, -inf) must be exactly a"
        );
        check_ulps!(
            Real::ln_add_exp(f64::NEG_INFINITY, a),
            a,
            0,
            "ln_add_exp(-inf, a) must be exactly a"
        );
        // And `+inf` absorbs.
        check!(
            Real::ln_add_exp(a, f64::INFINITY) == f64::INFINITY,
            "ln_add_exp(a, +inf) must be +inf"
        );
    }
}

property! {
    #![config(settings("log_sub_exp").cases(500))]

    /// `ln_add_exp(ln_sub_exp(a, b), b) = a` for `a > b` — subtraction inverts addition.
    ///
    /// Measured **exact** over 40,000 ordered pairs, at every gap, which is a strong statement
    /// about both functions at once: `(eᵃ − eᵇ) + eᵇ = eᵃ` can only survive in the log domain if
    /// each of the two is doing its pivot and its `ln_1p`/`ln_1m` correctly. No reference value,
    /// and it reaches arguments where `eᵃ` does not exist.
    ///
    /// The degenerate cases are the contract's edges and are asserted directly: equal arguments
    /// give `ln 0 = −∞`, and `a < b` is outside the documented domain ("valid for
    /// `self >= other`") and returns `NaN` rather than a complex result.
    fn log_sub_exp_inverts_log_add_exp(
        a in generate::float::in_range(-300.0, 300.0),
        b in generate::float::in_range(-300.0, 300.0),
    ) {
        let (high, low) = if a >= b { (a, b) } else { (b, a) };
        classify!("close together", (high - low) < 1.0);
        classify!("far apart", (high - low) > 50.0);

        // Equal arguments: ln(e^a - e^a) = ln 0.
        check!(
            Real::ln_sub_exp(high, high) == f64::NEG_INFINITY,
            "ln_sub_exp(a, a) must be -inf"
        );
        // Out of domain, by the trait's own statement of it.
        if high > low {
            check!(
                Real::ln_sub_exp(low, high).is_nan(),
                "ln_sub_exp below the diagonal must be NaN: ({low:e}, {high:e})"
            );
        }

        let difference = Real::ln_sub_exp(high, low);
        if !difference.is_finite() {
            return Ok(());
        }
        // The result is a log of something smaller than `e^high`, so it cannot exceed it.
        check!(
            difference <= high,
            "ln_sub_exp({high:e}, {low:e}) = {difference:e} exceeded its larger argument"
        );
        // Mixed rather than relative, and for a reason the exploration behind this property
        // managed to hide from itself: it skipped the `a == 0` cases while measuring, and
        // `a == 0` is precisely where a relative comparison is undefined. At `(0, −36.4)` the
        // round trip returns `−4.9e-32` where the reference is exactly `0` — an absolute error
        // of nothing and a relative error of infinity. Scaling by the argument is the honest
        // criterion, the same one `euclid_utils` uses throughout.
        relation::agree(
            "ln_add_exp(ln_sub_exp(a, b), b) = a",
            (),
            |()| Real::ln_add_exp(difference, low),
            |()| high,
            &Absolute(16.0 * f64::EPSILON * high.abs().max(1.0)),
        )?;
    }
}

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

property! {
    #![config(settings("absorbing_zero").cases(400))]

    /// `xlogy(0, y)` is `0` for **every** `y`, including `±∞` and `NaN`.
    ///
    /// The absorbing zero, and a genuine convention rather than a forced answer: `0 · ln 0` is
    /// an indeterminate form, and this surface resolves it to `0` because that is what makes an
    /// entropy or a log-likelihood sum work when a category has no observations. Propagating
    /// `NaN` would be equally defensible in the abstract and would break every one of those
    /// callers.
    ///
    /// Measured and asserted here because it is the kind of choice a refactor silently reverses:
    /// a rewrite that evaluates `x * y.ln()` first and patches afterwards gets `±∞` and `NaN`
    /// wrong while passing every finite test. The second argument is drawn by
    /// `pair_with_edge_cases`, so the non-finite cases arrive by construction rather than by
    /// being listed.
    fn the_absorbing_zero_wins_over_every_second_argument(
        pair in generate::float::pair_with_edge_cases(-40, 40),
    ) {
        let (_, y) = pair;
        classify!("nan", y.is_nan());
        classify!("infinite", y.is_infinite());
        classify!("zero", y == 0.0);

        for (name, got) in [
            ("xlogy", Real::xlogy(0.0, y)),
            ("xlog1py", Real::xlog1py(0.0, y)),
            ("xlog1my", Real::xlog1my(0.0, y)),
        ] {
            check!(
                got == 0.0,
                "{name}(0, {y:e}) must be 0, got {got:e}"
            );
        }
    }

    /// Away from the absorbing zero, each product is its plain definition.
    ///
    /// Which is the other half of the contract: the special case must be a special case, not a
    /// different function. `xlogy(x, y) = x·ln y`, `xlog1py(x, y) = x·ln1p(y)` and
    /// `xlog1my(x, y) = x·ln1m(y)`, each on the domain where its log exists.
    fn each_product_is_its_definition_away_from_zero(
        x in generate::float::in_range(-1e6, 1e6),
        y in generate::float::in_range(-0.9, 0.9),
    ) {
        if x == 0.0 {
            return Ok(());
        }
        classify!("negative multiplier", x < 0.0);

        // `y + 1` and `1 - y` are both comfortably positive on this range, so all three logs
        // exist and none of them is near its singularity.
        if y > 0.0 {
            relation::agree(
                "xlogy = x ln y",
                (),
                |()| Real::xlogy(x, y),
                |()| x * y.ln(),
                &Relative(1e-14),
            )?;
        }
        relation::agree(
            "xlog1py = x ln1p(y)",
            (),
            |()| Real::xlog1py(x, y),
            |()| x * Real::ln_1p(&y),
            &Relative(1e-14),
        )?;
        relation::agree(
            "xlog1my = x ln1m(y)",
            (),
            |()| Real::xlog1my(x, y),
            |()| x * Real::ln_1m(&y),
            &Relative(1e-14),
        )?;
    }
}
