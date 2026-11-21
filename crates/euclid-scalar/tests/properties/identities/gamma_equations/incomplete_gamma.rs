//! The incomplete gamma pair `P(a, x)` and `Q(a, x)`.
//!
//! Part of the [`gamma_equations`](super) suite. Two laws that need no reference implementation
//! at all: the pair are complementary probabilities, so `P + Q = 1` everywhere, and at integer
//! shape the pair collapse to closed forms in `exp` alone.

use super::*;

property! {
    #![config(settings("incomplete_gamma").cases(600))]

    /// `P(a, x) + Q(a, x) = 1`, which the trait states as a contract and nothing checked.
    ///
    /// [`Real::gamma_lr`]'s docstring says "`P` and `Q` are probabilities at every shape: both
    /// lie in `[0, 1]`, sum to `1`, and are monotone in `self`." That is three testable claims
    /// in one sentence, and `gamma_lr` and `gamma_ur` held **24 survivors** between the two
    /// precisions while no property mentioned either.
    ///
    /// The sum is measured exact for `a ≤ 100` and one ulp to `a = 1e8` — which is the strongest
    /// thing it could be, since `Q` is formed so a small upper tail keeps its relative accuracy
    /// rather than being computed as `1 − P`. If it *were* `1 - p` this law would be vacuous;
    /// it is not, so the law constrains two separate evaluations.
    fn the_incomplete_gamma_pair_are_complementary_probabilities(
        a in generate::float::magnitude(-6, 26),
        ratio in generate::float::in_range(0.0, 4.0),
    ) {
        let a = a.abs();
        if a <= 0.0 || !a.is_finite() || a.is_nan() || ratio < 0.0 {
            return Ok(());
        }
        let x = a * ratio;
        if !x.is_finite() {
            return Ok(());
        }
        classify!("below the mode", ratio < 1.0);
        classify!("above the mode", ratio > 1.0);
        classify!("large shape", a > 1e3);

        let lower = Real::gamma_lr(x, a);
        let upper = Real::gamma_ur(x, a);
        if !lower.is_finite() || !upper.is_finite() {
            return Ok(());
        }
        check!(
            (0.0..=1.0).contains(&lower) && (0.0..=1.0).contains(&upper),
            "P = {lower:e} and Q = {upper:e} must both be probabilities at a = {a:e}, x = {x:e}"
        );
        check_ulps!(
            lower + upper,
            1.0,
            1,
            "P + Q must be one at a = {a:e}, x = {x:e}"
        );
        // Monotone in `x`, the other half of the docstring's claim.
        let further = Real::gamma_lr(x + (x * 1e-6).max(1e-12), a);
        if further.is_finite() {
            check!(
                further >= lower,
                "P must not decrease in x at a = {a:e}, x = {x:e}"
            );
        }
    }

    /// The closed forms at integer shape, which are exact and need no reference.
    ///
    /// At `a = 1` the incomplete gamma collapses to the exponential: `Q(1, x) = e^{−x}` and
    /// `P(1, x) = −expm1(−x)`. At integer `n` it is the truncated exponential series,
    /// `Q(n, x) = e^{−x}Σ_{k<n} x^k/k!`. Both are elementary, so they pin the whole
    /// series/continued-fraction/Edgeworth machinery against arithmetic that shares none of it.
    ///
    /// Measured `5.7e-14`, `1.4e-15` and `1.1e-14` respectively.
    fn the_incomplete_gamma_collapses_to_the_exponential_at_integer_shape(
        x in generate::float::in_range(1e-3, 30.0),
        shape in generate::int(1, 12),
    ) {
        if x <= 0.0 {
            return Ok(());
        }
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let n = shape as u32;

        // a = 1: the exponential itself.
        let upper_one = Real::gamma_ur(x, 1.0);
        if upper_one > 0.0 {
            relation::agree(
                "Q(1, x) = exp(-x)",
                (),
                |()| upper_one,
                |()| (-x).exp(),
                &Relative(1e-12),
            )?;
        }
        let lower_one = Real::gamma_lr(x, 1.0);
        if lower_one > 0.0 {
            relation::agree(
                "P(1, x) = -expm1(-x)",
                (),
                |()| lower_one,
                |()| -((-x).exp_m1()),
                &Relative(1e-13),
            )?;
        }

        // Integer shape: the truncated series, summed directly.
        let mut term = 1.0f64;
        let mut series = 1.0f64;
        for k in 1..n {
            term *= x / f64::from(k);
            series += term;
        }
        let truncated = (-x).exp() * series;
        let upper = Real::gamma_ur(x, f64::from(n));
        if truncated > 0.0 && upper > 0.0 {
            classify!("higher shape", n > 4);
            relation::agree(
                "Q(n, x) = exp(-x) sum_{k<n} x^k/k!",
                (),
                |()| upper,
                |()| truncated,
                &Relative(1e-12),
            )?;
        }

        // And the endpoints, exactly.
        check_ulps!(Real::gamma_lr(0.0f64, 2.5), 0.0, 0, "P(a, 0) is zero");
        check_ulps!(Real::gamma_ur(0.0f64, 2.5), 1.0, 0, "Q(a, 0) is one");
        check!(
            Real::gamma_lr(f64::INFINITY, 2.5) == 1.0
                && Real::gamma_ur(f64::INFINITY, 2.5) == 0.0,
            "the pair must reach their limits exactly"
        );
    }
}
