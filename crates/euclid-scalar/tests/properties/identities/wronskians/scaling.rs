//! The scaled forms against their unscaled twins, and the derivative's one zero crossing.
//!
//! Part of the [`wronskians`](super) suite. These are the laws that pin the *scaling* rather
//! than the solutions: a scaled kernel is only useful if it is the unscaled one divided by a
//! factor the caller can reconstruct, and that factor is what is asserted here.

use super::*;

property! {
    #![config(settings("scalings").cases(500))]

    /// Each scaled Bessel is its unscaled twin times the documented factor.
    ///
    /// The *defining* relation of the scaled family, and the one thing an identity among the
    /// scaled functions alone cannot establish: the Wronskian above is satisfied by any
    /// consistent pair, including one scaled by the wrong exponent. This pins which exponent.
    ///
    /// # The tolerance is the exponential's conditioning, not a constant
    ///
    /// Checking `I₀ᵉ(x) = I₀(x)·e^{−|x|}` requires *forming* `e^{−|x|}`, whose relative error is
    /// `|x|·ε` because that is `exp`'s condition number — so the comparison is necessarily worse
    /// than either function. At `x = 602` the sweep measures `5.7e-14` against a predicted
    /// `1.3e-13`. A constant bound would either fail at the top of the range or be vacuous at
    /// the bottom, and the quantity it would be hiding is exactly the error the scaled form
    /// exists to avoid.
    fn each_scaled_bessel_is_its_unscaled_twin_times_the_factor(
        x in generate::float::in_range(1e-6, 690.0),
    ) {
        if x <= 0.0 {
            return Ok(());
        }
        classify!("small", x < 1.0);
        classify!("large", x > 100.0);

        // `exp`'s relative error at the argument, plus a floor for the multiply and the two
        // kernels. `numeric::condition_at` reports |x| for exp, which is the amplification.
        let conditioning = numeric::condition_at(f64::exp, -x).unwrap_or(x.abs());
        let bound = 64.0 * f64::EPSILON + conditioning.abs() * f64::EPSILON * 2.0;

        let decay = (-x).exp();
        let growth = x.exp();

        for (name, scaled, unscaled, factor) in [
            ("I0", Real::bessel_i0e(x), Real::bessel_i0(x), decay),
            ("I1", Real::bessel_i1e(x), Real::bessel_i1(x), decay),
            ("K0", Real::bessel_k0e(x), Real::bessel_k0(x), growth),
            ("K1", Real::bessel_k1e(x), Real::bessel_k1(x), growth),
        ] {
            let want = unscaled * factor;
            // Only where the unscaled form is still in range: above `x ≈ 710` `I` overflows and
            // `K` underflows, and `0·inf` is not a disagreement about the scaling.
            if !want.is_finite() || want == 0.0 || !unscaled.is_finite() || unscaled == 0.0 {
                continue;
            }
            check_relative!(
                scaled,
                want,
                bound,
                "{name}^e must equal {name} times its factor at x = {x:e}"
            );
        }
    }

    /// `I₁/I₀` and `ln I₀` are the two quantities the scaled form exists to serve, and both must
    /// agree with the ratio and log computed from it.
    ///
    /// `bessel_i0e`'s own docstring names them: "the overflow-free route to `ln I₀(x) =
    /// ln I₀ᵉ(x) + |x|` and `I₁/I₀ = I₁ᵉ/I₀ᵉ`". Those are claims about two *other* trait
    /// methods, so they are checkable — and they reach arguments where the unscaled ratio is
    /// `inf/inf`.
    fn the_scaled_pair_delivers_the_ratio_and_the_log(
        x in generate::float::magnitude(-20, 20),
    ) {
        let x = x.abs();
        if x == 0.0 || !x.is_finite() {
            return Ok(());
        }
        classify!("beyond the unscaled range", x > 710.0);

        let i0e = Real::bessel_i0e(x);
        let i1e = Real::bessel_i1e(x);
        if i0e == 0.0 || !i0e.is_finite() || !i1e.is_finite() {
            return Ok(());
        }

        // `I1/I0` has its own kernel, and the scaled ratio is the identity it is built on.
        check_relative!(
            Real::bessel_i1_over_i0(x),
            i1e / i0e,
            1e-13,
            "i1_over_i0 must equal I1^e / I0^e at x = {x:e}"
        );

        // `ln I₀` has two honest routes, and each is valid only where it is well conditioned.
        // Between them they cover the axis, which is the same two-halves split
        // `identities/cross_precision` uses for the same reason.
        //
        // The crossover is not a tuning knob. Above `x = 1` the overflow-free route is the one
        // that works, and it is the reason the scaled form exists. Below `x = 1` that route
        // *cannot* work: it adds `ln(I₀ᵉ) ≈ −x` to `x` to produce `ln I₀ ≈ x²/4`, so two
        // quantities of size `x` cancel down to one of size `x²/4` and the digits go with them.
        // At `x = 9.5e-7` that costs six digits.
        //
        // Below one the comparison is **exponentiated instead**, and that is the whole trick:
        // `ln I₀(x) ≈ x²/4` is `2.3e-13` there, far below `ulp(1)`, so *every* route that goes
        // through `I₀` has an absolute error of about `ε` and no relative claim about the log
        // can be tight. Two drafts of this property learned that the hard way — first against
        // `ln(I₀ᵉ) + x`, then against `ln(I₀)` — both failing against a kernel that was right.
        // Comparing `e^{ln I₀}` to `I₀` moves the check onto quantities of size one, where the
        // format has digits to spare, and the tiny log is then pinned to a relative `1e-14` by
        // proxy.
        if x <= 1.0 {
            check_relative!(
                Real::ln_bessel_i0(x).exp(),
                Real::bessel_i0(x),
                1e-14,
                "exp(ln_bessel_i0) must equal I0 below one, at x = {x:e}"
            );
        } else if i0e > 0.0 {
            check_relative!(
                Real::ln_bessel_i0(x),
                i0e.ln() + x,
                1e-13,
                "ln_bessel_i0 must equal ln(I0^e) + |x| above one, at x = {x:e}"
            );
        }
    }
}

property! {
    #![config(settings("airy_limits").cases(300))]

    /// `Ai′` approaches zero **from below**, and the sign bit is the only record of it.
    ///
    /// Found by triaging the Airy survivors. Above `LEADING_MIN = 1e10` the kernel returns the
    /// limits directly rather than multiplying a finite scaled value by an infinite exponential,
    /// and that tuple is `(0.0, -0.0, +∞, +∞)`. Deleting the minus — making `Ai′` a *positive*
    /// zero — survived 1,060 mutants, because no comparison can see it: `-0.0 == 0.0`.
    ///
    /// The sign is not decoration. `Ai` is positive and decreasing on the positive axis, so
    /// `Ai′ < 0` everywhere there and the limit is approached from below. A caller dividing by
    /// `Ai′`, or taking its sign to choose a direction, gets the wrong answer from `+0.0` — the
    /// same class of defect as `log_ndtr(+∞)` returning `+0.0` instead of `-0.0`, which this
    /// suite's sibling pins for the same reason.
    fn the_airy_derivative_reaches_zero_from_below(
        x in generate::float::in_range(1e10, 1e300),
    ) {
        if x < 1e10 || x.is_nan() {
            return Ok(());
        }
        classify!("very large", x > 1e100);

        // Past the threshold the unscaled values are the limits themselves.
        let derivative = Real::airy_ai_prime(x);
        check!(
            derivative == 0.0 && Real::signbit(&derivative),
            "Ai'({x:e}) must be -0.0, approaching zero from below; got {derivative:e}"
        );
        check!(
            Real::airy_ai(x) == 0.0 && !Real::signbit(&Real::airy_ai(x)),
            "while Ai itself reaches +0.0, since Ai > 0 throughout"
        );
        check!(
            Real::airy_bi(x) == f64::INFINITY && Real::airy_bi_prime(x) == f64::INFINITY,
            "and Bi and Bi' both diverge upward"
        );

        // Below the threshold the sign is a *computed* value rather than a literal, and must
        // still be negative — which is what makes the limit above consistent rather than
        // arbitrary.
        for &moderate in &[1.0f64, 5.0, 20.0, 100.0] {
            let slope = Real::airy_ai_prime(moderate);
            check!(
                slope < 0.0,
                "Ai' must be negative at {moderate}, got {slope:e}"
            );
        }
    }
}
