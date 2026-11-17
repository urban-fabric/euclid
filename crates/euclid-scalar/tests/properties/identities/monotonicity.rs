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
