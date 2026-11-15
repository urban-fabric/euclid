//! The `Real` surface's identities, as named metamorphic relations.
//!
//! Where `conditioning.rs` hunts for the worst input, this suite states the laws that must hold
//! for *every* input: a round trip returns, an odd function is odd, a bounded function stays in
//! range, a monotone function preserves order. None of them needs a known answer, which is why
//! they can be checked over generated floats at all.
//!
//! # Written through `relation`, deliberately
//!
//! Nearly every property here is one call into [`euclid_stability::relation`] rather than a
//! hand-written pair of evaluations and a comparison. That is the standardization this suite
//! exists to demonstrate, and the reason is not brevity:
//!
//! - The comparison is a **parameter**, so the tolerance appears once and visibly. A
//!   hand-written identity check reaches for `==`, which is wrong for a computed float, or for
//!   an absolute epsilon, which is wrong at unknown magnitude.
//! - The degenerate cases are handled once. `relation::monotone` sorts its own arguments;
//!   `Ulps` refuses `NaN` rather than silently comparing it equal to itself.
//! - The relation is **named**, so a failure says "f(-x) = -f(x) does not hold" instead of
//!   leaving a reader to reconstruct the law from two calls.
//!
//! # Tolerances, and why they differ
//!
//! A tolerance is a claim about the implementation, so each one below is justified where it is
//! written. The pattern: a single kernel evaluation should be within an ulp or two; a composed
//! identity accumulates each step's error, so `sin² + cos²` gets more room than `sin` alone;
//! and a round trip through two kernels gets the sum of both.
//!
//! Where a relation is checked against zero — `erf(x) + erfc(x) - 1`, `cosh² - sinh²` — the
//! comparison must be absolute rather than relative, because relative error against zero is
//! undefined and `within_relative` correctly refuses it.

use euclid_numeric::Real;
use euclid_stability::relation::{Absolute, Exact, Relative, Ulps};
use euclid_stability::{Config, check, check_ulps, classify, generate, property, relation};

fn settings(suite: &str) -> Config {
    Config::default()
        .cases(400)
        .regressions(format!("tests/regressions/{suite}.txt"))
}

// ===== Symmetry ===========================================================

property! {
    #![config(settings("symmetry"))]

    /// `sin`, `tan`, `sinh`, `tanh`, `asinh` and `erf` are all **odd**: `f(-x) = -f(x)`.
    ///
    /// Catches a kernel that takes `abs` too early, or that handles the sign by a branch which
    /// disagrees with itself across zero. One ulp, since each is a single evaluation.
    fn the_odd_functions_are_odd(x in generate::float::in_range(-20.0, 20.0)) {
        classify!("near zero", x.abs() < 1e-8);
        classify!("ordinary", x.abs() >= 1e-8);

        relation::odd("sin", x, f64::sin, &Ulps(1))?;
        relation::odd("tan", x, f64::tan, &Ulps(1))?;
        relation::odd("sinh", x, f64::sinh, &Ulps(1))?;
        relation::odd("tanh", x, f64::tanh, &Ulps(1))?;
        relation::odd("asinh", x, Real::asinh, &Ulps(1))?;
        relation::odd("erf", x, Real::erf, &Ulps(1))?;
    }

    /// `cos`, `cosh` and `abs` are **even**: `f(-x) = f(x)`.
    ///
    /// Exact, not approximate: an even function of a negated argument should be computed from
    /// the same reduced argument, so the results ought to be bit-identical.
    fn the_even_functions_are_even(x in generate::float::in_range(-20.0, 20.0)) {
        relation::even("cos", x, f64::cos, &Ulps(0))?;
        relation::even("cosh", x, f64::cosh, &Ulps(0))?;
        relation::even("abs", x, f64::abs, &Ulps(0))?;
    }

    /// `erfc(-x) = 2 - erfc(x)`, the reflection that follows from `erf` being odd.
    ///
    /// **Absolute**, and the reason is a conditioning result the suite found rather than one
    /// anyone wrote down first. A relative tolerance of `1e-14` fails at `x = -5`, where the
    /// true value is `1.5e-12`: the right-hand side computes it by subtracting two numbers
    /// either side of 2, so its absolute error is an ulp of 2 — about `4.4e-16` — and against a
    /// result of `1.5e-12` that is a relative error of `3e-4`.
    ///
    /// The kernels are fine. The *identity* is ill-conditioned in this direction, and its honest
    /// accuracy claim is absolute at the scale of the larger operand. Loosening the relative
    /// tolerance until it passed would have hidden exactly the fact worth knowing: do not
    /// compute a small `erfc` by reflecting a large one.
    fn erfc_reflects_about_one(x in generate::float::in_range(-5.0, 5.0)) {
        classify!("reflection cancels", x < -2.0);
        relation::agree(
            "erfc(-x) = 2 - erfc(x)",
            x,
            |v: f64| Real::erfc(-v),
            |v: f64| 2.0 - Real::erfc(v),
            &Absolute(2e-15),
        )?;
    }
}

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

// ===== Round trips ========================================================

property! {
    #![config(settings("round_trips"))]

    /// `exp` and `ln` invert each other. Four ulps, because the trip passes through two kernels
    /// and `exp`'s relative error is amplified by `ln`'s derivative near 1.
    fn exp_and_ln_round_trip(x in generate::float::in_range(-700.0, 700.0)) {
        classify!("near zero", x.abs() < 1.0);
        relation::round_trip("exp then ln", x, f64::exp, f64::ln, &Ulps(4))?;
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
    fn next_after_steps_and_returns(x in generate::float::magnitude(-300, 300)) {
        let up = Real::next_after(x, f64::INFINITY);
        let back = Real::next_after(up, f64::NEG_INFINITY);
        check_ulps!(back, x, 0, "stepping up then down must return exactly");
        check!(up > x, "next_after toward +inf must increase");
    }
}

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
    /// representable value — which `next_after` computes independently.
    fn ulp_agrees_with_next_after(x in generate::float::magnitude(-1_000, 1_000)) {
        let gap = Real::ulp(x);
        check!(gap > 0.0 && gap.is_finite(), "ulp({x}) = {gap} is not a positive finite gap");

        let step = (Real::next_after(x, f64::INFINITY) - x).abs();
        check_ulps!(gap, step, 0, "ulp disagreed with the step to the next value");
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

// ===== The tier contract, over generated input ============================

property! {
    #![config(settings("tier"))]

    /// `Exact` is the right comparison for the discrete parts of the surface, and this pins
    /// them: classification is a predicate, not an approximation.
    fn classification_is_exact(x in generate::float::any()) {
        relation::agree(
            "is_finite is the negation of is_nan-or-infinite",
            x,
            |v: f64| v.is_finite(),
            |v: f64| !(v.is_nan() || v.is_infinite()),
            &Exact,
        )?;
    }

    /// Negation is an involution — but only where `PartialEq` can see it.
    ///
    /// `Exact` compares with `==`, and `NaN != NaN`, so `involution` correctly reports that the
    /// law fails on `NaN`. That is not a bug in either the relation or in negation: it is the
    /// distinction between "the same value" and "the same bits", and `Exact` means the former.
    /// So the finite case goes through the relation and the `NaN` case is stated separately, in
    /// the comparison that can actually express it.
    fn negation_is_an_involution_on_the_finite_floats(x in generate::float::finite()) {
        relation::involution("negate", x, |v: f64| -v, &Exact)?;
    }

    /// And on `NaN`, negation round-trips *bitwise* — including the sign bit, which negation
    /// must flip rather than ignore.
    fn negating_nan_twice_restores_its_bits(_ in generate::just(0u8)) {
        let nan = f64::NAN;
        check!(
            (-(-nan)).to_bits() == nan.to_bits(),
            "double negation must restore NaN's bits"
        );
        check!(
            (-nan).is_sign_negative() != nan.is_sign_negative(),
            "negation must flip NaN's sign bit"
        );
    }
}
