//! Small-shape gamma tails at single precision.
//!
//! Part of the [`distribution_extremes`](super) suite. The complementarity `P + Q = 1` is the
//! one law that still holds where both tails have saturated, which is why it is the instrument
//! for a format whose probabilities run out three hundred orders earlier than `f64`'s.

use super::*;

property! {
    // Floors on both sides of `SMALL_SHAPE`, at roughly half the coverage measured over six
    // seeds (the split runs near even, 45.8–54.2% either way). The small-shape arm is the one
    // that replaced a widening to `f64`, and its `x = 4` defect — `P + Q = 1.0000009537` — was
    // only reachable because cases land on both sides of `a = 0.01`.
    #![config(settings("f32_native_small_shape")
        .cases(500)
        .require("small-shape branch", 25.0)
        .require("regular series", 25.0))]

    /// The native `f32` incomplete gamma at a small shape, where it used to widen to `f64`.
    ///
    /// `IncompleteGammaMethod::SmallShape` is the branch that replaced the widening, and it is
    /// the only one of the six that keeps the `O(a)` terms the other series drop. Three laws
    /// reach it, none of which the widened version could meaningfully have been tested against:
    ///
    /// - **`P + Q = 1` exactly to a tolerance**, which is the pair's defining relation and the
    ///   one a mis-sited branch breaks first, because `gamma_lr` and `gamma_ur` choose their
    ///   routes independently and a disagreement shows up here immediately.
    /// - **Monotonicity in `x`**, which no series identity implies and which catches a branch
    ///   returning a value from the wrong regime at a boundary.
    /// - **`P(a, x) → 1` as the argument grows**, the limit the `Limit` arm must supply
    ///   explicitly — without it the series return `NaN` at an infinite argument and poison
    ///   `erf` and the whole normal family built on it.
    ///
    /// The budget on `P + Q` is a few ulps rather than exact: the two tails are formed by
    /// different expressions, so the sum rounds twice, and at a small shape one of them is near
    /// `1` where its own ulp is `1.2e-7`.
    ///
    /// Written in receiver form — `x.gamma_lr(a)` — deliberately. The trait takes the *argument*
    /// as the receiver and the shape as the parameter, which is the opposite order from the
    /// kernel function behind it, and the first draft of this property had them swapped. It then
    /// reported a monotonicity violation that looked exactly like a branch defect and was
    /// `P(shape = x, at = a)` decreasing in its shape, which is correct behaviour.
    fn the_native_f32_small_shape_gamma_tails_are_complementary(
        a in generate::float32::in_range(1e-4, 0.02),
        x in generate::float32::in_range(1e-6, 4.0),
    ) {
        if !a.is_finite() || !x.is_finite() || a <= 0.0 || x <= 0.0 {
            return Ok(());
        }
        classify!("small-shape branch", a <= 0.01);
        classify!("regular series", a > 0.01);

        let lower = x.gamma_lr(a);
        let upper = x.gamma_ur(a);
        check!(
            (0.0..=1.0).contains(&lower) && (0.0..=1.0).contains(&upper),
            "f32 gamma tails must be probabilities: P({a:e},{x:e}) = {lower:e}, Q = {upper:e}"
        );
        relation::agree(
            "f32 small shape: P + Q = 1",
            (),
            |()| f64::from(lower) + f64::from(upper),
            |()| 1.0,
            &Absolute(8.0 * f64::from(f32::EPSILON)),
        )?;

        // P is nondecreasing in x. Compared at a doubled argument so the step is never lost to
        // the argument's own rounding.
        let further = (x * 2.0).gamma_lr(a);
        check!(
            further >= lower,
            "f32 P must be nondecreasing in x: P({a:e},{x:e}) = {lower:e} but \
             P({a:e},{:e}) = {further:e}",
            x * 2.0
        );

        // The limits the `Limit` arm owns.
        check!(
            f32::INFINITY.gamma_lr(a) == 1.0 && f32::INFINITY.gamma_ur(a) == 0.0,
            "f32 gamma tails must saturate at an infinite argument for a = {a:e}"
        );
        check!(
            0.0f32.gamma_lr(a) == 0.0 && 0.0f32.gamma_ur(a) == 1.0,
            "f32 gamma tails must be exact at a zero argument for a = {a:e}"
        );
    }
}
