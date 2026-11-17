//! Bounds properties.

use super::*;

// ===== Bounds =============================================================

property! {
    #![config(settings("bounds").require("saturated", 2.0))]

    /// The bounded functions stay in range, on **any** input including the non-finite ones.
    ///
    /// The cheapest property in the suite and one of the most effective, because a kernel that
    /// goes wrong usually goes far wrong. `any()` is used rather than `finite()` because the
    /// range claim covers infinities too: `erf(∞)` is `1`, not something larger.
    fn the_bounded_functions_stay_bounded(x in generate::float::any()) {
        if x.is_nan() {
            // Every one of these must propagate NaN rather than inventing a bound value.
            classify!("nan");
            check!(Real::erf(x).is_nan(), "erf(NaN) must be NaN");
            check!(x.tanh().is_nan(), "tanh(NaN) must be NaN");
            return Ok(());
        }

        classify!("saturated", x.abs() > 6.0);
        classify!("infinite", x.is_infinite());

        relation::bounded("erf", x, Real::erf, -1.0, 1.0)?;
        relation::bounded("erfc", x, Real::erfc, 0.0, 2.0)?;
        relation::bounded("tanh", x, f64::tanh, -1.0, 1.0)?;
        relation::bounded("sin", x, |v| if v.is_finite() { v.sin() } else { 0.0 }, -1.0, 1.0)?;
    }

}

property! {
    // Its own block: the `saturated` requirement above is the bounds property's, and a
    // block-level config applies to every property in it.
    #![config(settings("complementary"))]

    /// `erf(x) + erfc(x) = 1`, exactly the pair's defining relationship.
    ///
    /// Absolute rather than relative: the sum is compared against `1`, but the *error* lives at
    /// the scale of the smaller term, and for large `x` that term is `1e-100`. An absolute
    /// tolerance is the right instrument here because the target has a known scale.
    fn erf_and_erfc_are_complementary(x in generate::float::in_range(-6.0, 6.0)) {
        let total = Real::erf(x) + Real::erfc(x);
        relation::agree("erf + erfc = 1", x, |_| total, |_| 1.0, &Absolute(1e-15))?;
    }
}
