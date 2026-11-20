//! Monotonicity properties.

use super::*;

// ===== Monotonicity =======================================================

property! {
    #![config(settings("monotone"))]

    /// `erf`, `tanh`, `exp`, `ln` and `sinh` are increasing; `erfc` is decreasing.
    ///
    /// Order preservation is checkable without any notion of accuracy, which makes it the one
    /// property that survives a kernel being a few ulps off. A non-monotone CDF is a bug that
    /// a pointwise accuracy test will not see, because every individual value is within
    /// tolerance — it is the *relationship* between two of them that is wrong.
    fn the_increasing_functions_preserve_order(
        a in generate::float::in_range(-10.0, 10.0),
        b in generate::float::in_range(-10.0, 10.0),
    ) {
        relation::monotone("erf", a, b, Real::erf)?;
        relation::monotone("tanh", a, b, f64::tanh)?;
        relation::monotone("exp", a, b, f64::exp)?;
        relation::monotone("sinh", a, b, f64::sinh)?;

        // `erfc` decreases, which is monotonicity of its negation.
        relation::monotone("negated erfc", a, b, |v| -Real::erfc(v))?;
    }

    /// `ln` is increasing on the positives, over the whole exponent range rather than a narrow
    /// window — the region where an argument-reduction mistake would show.
    fn ln_preserves_order_across_the_exponent_range(
        a in generate::float::positive(),
        b in generate::float::positive(),
    ) {
        relation::monotone("ln", a, b, f64::ln)?;
    }
}

// ===== Monotonicity in the *parameter*, not the argument ==================

property! {
    #![config(settings("cdf_shape_monotone")
        .require("strict", 20.0)
        .require("saturated", 10.0))]

    /// `P(a, x)` decreases and `Q(a, x)` increases as the shape grows, at fixed argument.
    ///
    /// # A different axis from everything above
    ///
    /// Every other monotonicity property in this crate varies the *argument*. These vary the
    /// **shape**, and nothing had checked that axis at all — which matters because it is the
    /// structural fact that makes the distribution a distribution: more shape moves mass to the
    /// right, so less of it lies below any fixed `x`. A kernel that got this backwards in one
    /// branch would still pass every pointwise accuracy test and every argument-axis ordering.
    ///
    /// `Q` is asserted separately rather than derived as `1 − P`, because
    /// [`Real::gamma_ur`] has its own route through the series and the continued fraction and is
    /// not computed as the complement. A sign error in one of its branches is invisible from `P`.
    ///
    /// # Why the shapes are held apart, and what happens when they are not
    ///
    /// Asserted **exactly**, with no tolerance — and that is only honest because the shapes are
    /// separated. Measured over 40,000 pairs at each of the ratios `1 + 1e-7`, `1.001`, `1.1` and
    /// `2`: about 17,000 strict decreases, the rest saturated at `0` or `1`, and **zero
    /// inversions** at every ratio.
    ///
    /// At *adjacent* shapes it is a different story: stepping the shape by one ulp gives 7,985
    /// inversions in 160,000 pairs. That is not a defect but the familiar limit — the true
    /// difference `(∂P/∂a)·ulp(a)` drops below the kernel's own error, so the two values are no
    /// longer distinguishable and their order is noise. The same effect the wiggle budgets in
    /// `properties/stability.rs` record on the argument axis.
    ///
    /// The `saturated` floor is the one that keeps this honest. Over half of all pairs have `P`
    /// pinned at `0` or `1`, where the comparison holds trivially, so without a floor on the
    /// strict cases the property could pass having compared nothing but saturation.
    fn the_gamma_cdf_is_monotone_in_its_shape(
        x in generate::float::magnitude(-3, 3),
        shape in generate::float::magnitude(-2, 8),
        ratio in generate::float::in_range(1.001, 1e4),
    ) {
        let (x, shape) = (x.abs(), shape.abs());
        if x <= 0.0 || shape <= 0.0 || !x.is_finite() || !shape.is_finite() || ratio <= 1.0 {
            return Ok(());
        }
        let larger = shape * ratio;
        if !larger.is_finite() {
            return Ok(());
        }

        let (p_small, p_large) = (Real::gamma_lr(x, shape), Real::gamma_lr(x, larger));
        let (q_small, q_large) = (Real::gamma_ur(x, shape), Real::gamma_ur(x, larger));
        if !(p_small.is_finite() && p_large.is_finite()) {
            return Ok(());
        }
        classify!("strict", p_large < p_small);
        classify!("saturated", p_large == p_small);

        check!(
            p_large <= p_small,
            "P must not increase with the shape: P({shape:e}, {x:e}) = {p_small:e} but \
             P({larger:e}, {x:e}) = {p_large:e}"
        );
        if q_small.is_finite() && q_large.is_finite() {
            check!(
                q_large >= q_small,
                "Q must not decrease with the shape: Q({shape:e}, {x:e}) = {q_small:e} but \
                 Q({larger:e}, {x:e}) = {q_large:e}"
            );
        }
    }
}

property! {
    #![config(settings("beta_shape_monotone")
        .require("strict in a", 10.0)
        .require("strict in b", 10.0))]

    /// `I_x(a, b)` decreases in `a` and increases in `b`, at fixed argument.
    ///
    /// The two directions are opposite and both are exact: raising `a` moves the beta density
    /// towards `1` and so takes mass from below `x`, while raising `b` moves it towards `0`. A
    /// kernel that confused the two parameters anywhere — in the Edgeworth arm, say, where the
    /// skewness carries a `b − a` — would satisfy every symmetric test and fail this one.
    ///
    /// Both directions are measured clean over 14,400 pairs each, so these are asserted exactly,
    /// on the same reasoning as the gamma property above.
    fn the_beta_cdf_is_monotone_in_each_shape(
        x in generate::float::in_range(1e-6, 1.0 - 1e-6),
        shape in generate::float::magnitude(-1, 5),
        other in generate::float::magnitude(-1, 5),
        ratio in generate::float::in_range(1.001, 1e3),
    ) {
        let (shape, other) = (shape.abs(), other.abs());
        if !(x > 0.0 && x < 1.0) || shape <= 0.0 || other <= 0.0 || ratio <= 1.0 {
            return Ok(());
        }
        if !shape.is_finite() || !other.is_finite() {
            return Ok(());
        }
        let larger = shape * ratio;
        if !larger.is_finite() {
            return Ok(());
        }

        // Raising `a` can only take mass from below `x`.
        let (by_a_small, by_a_large) = (
            Real::beta_reg(x, shape, other),
            Real::beta_reg(x, larger, other),
        );
        if by_a_small.is_finite() && by_a_large.is_finite() {
            classify!("strict in a", by_a_large < by_a_small);
            check!(
                by_a_large <= by_a_small,
                "I_x must not increase with a: I_{x:e}({shape:e}, {other:e}) = {by_a_small:e} \
                 but I_{x:e}({larger:e}, {other:e}) = {by_a_large:e}"
            );
        }

        // Raising `b` can only add mass below `x`.
        let (by_b_small, by_b_large) = (
            Real::beta_reg(x, other, shape),
            Real::beta_reg(x, other, larger),
        );
        if by_b_small.is_finite() && by_b_large.is_finite() {
            classify!("strict in b", by_b_large > by_b_small);
            check!(
                by_b_large >= by_b_small,
                "I_x must not decrease with b: I_{x:e}({other:e}, {shape:e}) = {by_b_small:e} \
                 but I_{x:e}({other:e}, {larger:e}) = {by_b_large:e}"
            );
        }
    }
}
