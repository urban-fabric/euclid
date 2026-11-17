//! Round trips properties.

use super::*;

// ===== Round trips ========================================================

property! {
    #![config(settings("round_trips"))]

    /// `exp` and `ln` invert each other, to the bound the **composition** is entitled to.
    ///
    /// # Why this is not a fixed ulp count any more
    ///
    /// It was four ulps, and four ulps is wrong. The property passed 400 cases for three
    /// releases and failed on the first 25x soak, at `x = -0.0624`, where the round trip costs
    /// more than four ulps for a perfectly good reason.
    ///
    /// The docstring it replaces already named the mechanism — "`exp`'s relative error is
    /// amplified by `ln`'s derivative near 1" — and then picked a constant anyway. That is
    /// exactly the mistake [`two_stage_error_bound`] exists to prevent, and the amplification is
    /// not constant: with `κ_exp(x) = |x|` and `κ_ln(exp x) = 1/|x|`, the two-stage bound's
    /// `κ_outer·κ_inner + κ_outer` comes to `1 + 1/|x|`. At `x = -0.0624` that is about 17, so
    /// seventeen ulps is the honest allowance and four was a claim about a different function.
    ///
    /// It diverges as `x` approaches zero, which is not a defect to be clamped away: `ln` is
    /// genuinely ill conditioned at 1, and a round trip through it near zero *cannot* be
    /// ulp-exact. Any constant tolerance here is either wrong near zero or vacuous away from
    /// it — the two-stage bound is the only spelling that is neither, and away from zero it
    /// settles to about `8u`, which is where the original four ulps was aiming.
    ///
    /// [`two_stage_error_bound`]: euclid_stability::numeric::two_stage_error_bound
    fn exp_and_ln_round_trip(x in generate::float::in_range(-700.0, 700.0)) {
        classify!("near zero", x.abs() < 1.0);
        classify!("ordinary", x.abs() >= 1.0);

        // Undefined where `ln`'s conditioning has no value — `x = 0` exactly, where the trip is
        // exact anyway and there is no relative error to bound.
        let Some(bound) = numeric::two_stage_error_bound(f64::exp, f64::ln, x, 8.0) else {
            return Ok(());
        };
        relation::round_trip("exp then ln", x, f64::exp, f64::ln, &Relative(bound))?;
    }

    /// `sqrt` and squaring invert each other, to a relative tolerance rather than ULP: squaring
    /// doubles the relative error, so the trip cannot be ulp-exact.
    fn sqrt_and_square_round_trip(x in generate::float::positive()) {
        relation::round_trip("sqrt then square", x, f64::sqrt, |v| v * v, &Relative(1e-14))?;
    }

    /// `cbrt` likewise, and the sign is part of the contract — unlike `sqrt`, the cube root is
    /// defined for negatives, and a kernel built on `exp(ln(x)/3)` silently is not.
    fn cbrt_round_trips_including_negatives(x in generate::float::magnitude(-100, 100)) {
        classify!("negative", x < 0.0);
        relation::round_trip(
            "cbrt then cube",
            x,
            Real::cbrt,
            |v| v * v * v,
            &Relative(1e-13),
        )?;
    }

    /// `frexp` and `ldexp` invert each other **exactly**: the decomposition into mantissa and
    /// exponent is a relabelling of the bits, not a computation, so anything short of bit
    /// equality is a bug.
    fn frexp_and_ldexp_round_trip_exactly(x in generate::float::finite()) {
        let (mantissa, exponent) = Real::frexp(x);
        classify!("zero", x == 0.0);
        classify!("subnormal", x != 0.0 && x.abs() < f64::MIN_POSITIVE);

        check_ulps!(
            Real::ldexp(mantissa, exponent),
            x,
            0,
            "frexp/ldexp must round-trip exactly"
        );

        // And the mantissa is in its documented range, which is what makes the pair canonical.
        if x != 0.0 && x.is_finite() {
            check!(
                (0.5..1.0).contains(&mantissa.abs()),
                "frexp mantissa {mantissa} is outside [0.5, 1)"
            );
        }
    }

    /// `erf` and `erf_inv` invert each other. A looser tolerance than the others, and honestly
    /// so: `erf` saturates, so its inverse is steep near `±1` and amplifies any error there.
    fn erf_and_its_inverse_round_trip(x in generate::float::in_range(-2.0, 2.0)) {
        classify!("near saturation", x.abs() > 1.5);
        relation::round_trip("erf then erf_inv", x, Real::erf, Real::erf_inv, &Relative(1e-10))?;
    }

    /// `next_after` steps one representable value and back, which must return exactly. The
    /// cheapest possible check that the bit-level navigation is right.
    ///
    /// Note what it fixes about the *target*: `±INFINITY`, every time. That is why the sibling
    /// property below exists — a law over the first argument alone cannot see a guard on the
    /// second.
    fn next_after_steps_and_returns(x in generate::float::magnitude(-300, 300)) {
        let up = Real::next_after(x, f64::INFINITY);
        let back = Real::next_after(up, f64::NEG_INFINITY);
        check_ulps!(back, x, 0, "stepping up then down must return exactly");
        check!(up > x, "next_after toward +inf must increase");
    }
}

property! {
    // Its own block: the coverage requirements name labels only this property applies, and a
    // block-level config applies to every property in its block — the footgun `two_sum` and
    // `two_product` are kept apart for.
    #![config(
        settings("next_after_targets")
            .cases(500)
            .require("nan target", 5.0)
            .require("equal target", 5.0)
            .require("zero target", 5.0)
    )]

    /// `next_after`'s contract over **every kind of target**, which is the argument its other
    /// properties hold fixed.
    ///
    /// # Why the second argument needs its own generator
    ///
    /// The guard is `if x.is_nan() || y.is_nan()`, and weakening that `||` to `&&` survived the
    /// whole suite. A NaN *source* cannot show it: the fall-through walks `x`'s bits by one and
    /// a NaN's neighbour is another NaN, so `is_nan()` holds either way and the mutant looks
    /// correct. A NaN *target* reaches the direction test, where every comparison against NaN
    /// is false, and `next_after(1.0, NaN)` returns `1.0 - 1ulp` — a plausible neighbour for a
    /// direction nobody asked for. In a bracketing search that is a step taken on a poisoned
    /// bound, with nothing non-finite left downstream to notice.
    ///
    /// `generate::float::pair_with_edge_cases` draws the target from the values that make a
    /// guard fire, including `x` itself for the equality arm, which a random pair never hits.
    /// The requirements above are what stop a reweighting from silently starving a case.
    fn next_after_handles_every_kind_of_target(
        pair in generate::float::pair_with_edge_cases(-300, 300),
    ) {
        let (x, y) = pair;
        classify!("nan target", y.is_nan());
        classify!("equal target", x == y);
        classify!("zero target", y == 0.0);
        classify!("infinite target", y.is_infinite());

        let result = Real::next_after(x, y);

        // A NaN in either argument propagates. This is the guard above.
        if x.is_nan() || y.is_nan() {
            check!(
                result.is_nan(),
                "next_after({x:e}, {y:e}) = {result:e} must be NaN"
            );
            return Ok(());
        }

        // Equal arguments return the target rather than stepping off it.
        if x == y {
            check_ulps!(result, y, 0, "equal arguments must return the target");
            return Ok(());
        }

        // Otherwise: exactly one step, toward the target, and never past it.
        check!(
            result != x,
            "next_after({x:e}, {y:e}) did not move"
        );
        if y > x {
            check!(
                result > x && result <= y,
                "next_after({x:e}, {y:e}) = {result:e} left the interval"
            );
        } else {
            check!(
                result < x && result >= y,
                "next_after({x:e}, {y:e}) = {result:e} left the interval"
            );
        }
        // One step means the trip back returns exactly, whatever the target was.
        let back = Real::next_after(result, x);
        check_ulps!(back, x, 0, "stepping toward a target and back must return");
    }
}
