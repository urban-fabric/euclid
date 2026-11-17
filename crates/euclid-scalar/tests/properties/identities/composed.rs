//! Composed properties.

use super::*;

// ===== Composed identities ================================================

property! {
    #![config(settings("composed"))]

    /// `sin² + cos² = 1`.
    ///
    /// Absolute tolerance against a known scale, and two ulps of room rather than one: the sum
    /// accumulates the error of both kernels plus two squarings.
    fn the_pythagorean_identity_holds(x in generate::float::in_range(-100.0, 100.0)) {
        let total = x.sin() * x.sin() + x.cos() * x.cos();
        relation::agree("sin^2 + cos^2 = 1", x, |_| total, |_| 1.0, &Absolute(4e-16))?;
    }

    /// `cosh² − sinh² = 1`.
    ///
    /// The range is narrow on purpose, and the first attempt at it was still too wide. Both
    /// terms grow like `e^{2x}/4`, so their difference is a cancellation whose absolute error is
    /// an ulp of `cosh²(x)` — against a true result of `1`, the relative error is
    /// `cosh²(x) · 2⁻⁵²`. At `x = 5` that is `1.8e-12`, which is what the suite reported, and
    /// holding `1e-13` needs `cosh²(x) ≤ 450`, so `|x| ≤ 3`.
    ///
    /// This is a property of the identity rather than of the kernels. The test states the range
    /// where it is meaningful instead of loosening the tolerance until it passes everywhere,
    /// because the range *is* the finding.
    fn the_hyperbolic_identity_holds_where_it_is_well_conditioned(
        x in generate::float::in_range(-3.0, 3.0),
    ) {
        let total = x.cosh() * x.cosh() - x.sinh() * x.sinh();
        relation::agree("cosh^2 - sinh^2 = 1", x, |_| total, |_| 1.0, &Relative(1e-13))?;
    }

    /// `hypot` is symmetric in its arguments, and exactly so — the norm does not care which
    /// component it was given first.
    fn hypot_is_commutative(
        x in generate::float::magnitude(-200, 200),
        y in generate::float::magnitude(-200, 200),
    ) {
        relation::commutative("hypot", x, y, f64::hypot, &Ulps(0))?;
    }

    /// `ulp` is positive and finite for every finite input, and is the gap to the next
    /// representable value **away from zero** — which `next_after` computes independently.
    ///
    /// # The direction is part of the law, and this property used to get it wrong
    ///
    /// It read `next_after(x, f64::INFINITY)` for every `x`, which for a negative argument steps
    /// *toward* zero — the opposite of what [`Real::ulp`] documents. The property passed anyway,
    /// and would have kept passing, because the two directions have the same width everywhere
    /// except on a binade boundary, and `magnitude` draws a full 52-bit mantissa so it never
    /// produces one.
    ///
    /// So this asserted a false law that was saved by its generator, and `ulp`'s real direction
    /// bug — found by mutation testing, not by this property — would have made the *corrected*
    /// law fail here while the original kept quiet. At every negative power of two the original
    /// `step` is exactly half the true gap. See the sibling property below, which draws the
    /// input this one cannot.
    fn ulp_agrees_with_next_after(x in generate::float::magnitude(-1_000, 1_000)) {
        let gap = Real::ulp(x);
        check!(gap > 0.0 && gap.is_finite(), "ulp({x}) = {gap} is not a positive finite gap");

        let away = if x < 0.0 { f64::NEG_INFINITY } else { f64::INFINITY };
        let step = (Real::next_after(x, away) - x).abs();
        check_ulps!(gap, step, 0, "ulp disagreed with the step away from zero");
    }

    /// The same law at the one input that can distinguish its direction: an exact power of two.
    ///
    /// On a binade boundary the step away from zero is twice the step toward it, so this is
    /// where `ulp`'s documented direction is observable and the only place a direction error
    /// can be caught. `generate::float::binade_boundary` exists for this: `magnitude` reaches a
    /// mantissa of zero with probability `2^-52`.
    ///
    /// The second check is the one that keeps the law honest. It asserts the two directions
    /// *differ by exactly two*, which fails if anyone simplifies the first check back to the
    /// unconditional `next_after(x, INFINITY)` this property's sibling used to use.
    fn ulp_steps_away_from_zero_at_a_binade_boundary(
        x in generate::float::binade_boundary(-1_000, 1_000),
    ) {
        classify!("negative", x < 0.0);
        classify!("positive", x > 0.0);

        let gap = Real::ulp(x);
        let away = if x < 0.0 { f64::NEG_INFINITY } else { f64::INFINITY };
        check_ulps!(
            gap,
            (Real::next_after(x, away) - x).abs(),
            0,
            "ulp must be the step away from zero at a binade boundary"
        );

        // Above the first normal binade the inward step is half the outward one. At and below
        // `MIN_POSITIVE` it is not: the subnormals share one spacing, so the boundary stops
        // being a boundary and the factor is one.
        if x.abs() > f64::MIN_POSITIVE {
            let toward = if x < 0.0 { f64::INFINITY } else { f64::NEG_INFINITY };
            let inward = (Real::next_after(x, toward) - x).abs();
            check_ulps!(
                gap,
                inward * 2.0,
                0,
                "the step away from zero must be twice the step toward it"
            );
        }
    }

    /// `mul_add` must be at least as accurate as the separate operations, which is the entire
    /// reason it exists. Checked by comparing both against the exact product, recovered with
    /// `two_product`.
    fn mul_add_is_at_least_as_accurate_as_the_separate_operations(
        a in generate::float::magnitude(-100, 100),
        b in generate::float::magnitude(-100, 100),
        c in generate::float::magnitude(-100, 100),
    ) {
        let fused = Real::mul_add(a, b, c);
        let separate = a * b + c;
        if !fused.is_finite() || !separate.is_finite() {
            return Ok(());
        }

        // The exact value of a*b + c, to the extent f64 can express it: the rounded product
        // plus its exact remainder, then c.
        let (product, remainder) = Real::two_product(a, b);
        let reference = product + c + remainder;
        if !reference.is_finite() {
            return Ok(());
        }

        let fused_error = (fused - reference).abs();
        let separate_error = (separate - reference).abs();
        classify!("fma strictly better", fused_error < separate_error);

        check!(
            fused_error <= separate_error || separate_error == 0.0,
            "mul_add was less accurate than a*b + c: fma off by {fused_error:e}, \
             separate off by {separate_error:e}"
        );
    }
}
