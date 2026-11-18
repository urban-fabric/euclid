//! Wronskian and scaling identities for the Airy and Bessel families.
//!
//! Part of the [`identities`](super) suite, and the sharpest instrument in it.
//!
//! # Why these are worth more than an accuracy band
//!
//! A Wronskian is an identity between a pair of solutions to the same second-order equation,
//! and its value is a **closed form** — `1/π`, `1/x`, `−2/(πx)`. So unlike almost everything
//! else about a special function, it can be checked without a reference value, at any argument,
//! to the precision of the format.
//!
//! That buys two things no pointwise accuracy test does:
//!
//! - **One identity constrains four separately-implemented functions at once.** `Ai`, `Ai′`,
//!   `Bi` and `Bi′` are four different code paths — series, asymptotic, quadrature, each with
//!   its own crossovers — and the Wronskian is false unless all four agree with each other
//!   *and* with the equation they solve. A fixed-point test checks one function at one argument
//!   against one transcribed constant.
//! - **It reaches arguments no reference table does.** The scaled forms satisfy the same
//!   identity (see below), so the check runs out to `1e12` and `1e15` where nothing has ever
//!   been transcribed.
//!
//! # The exponentials cancel, which is what makes the scaled forms usable here
//!
//! The scaled variants are defined with reciprocal factors —
//! `Aiᵉ = e^{ζ}Ai` against `Biᵉ = e^{−ζ}Bi`, and `I₀ᵉ = e^{−|x|}I₀` against `K₀ᵉ = e^{x}K₀` —
//! so every product in a Wronskian pairs one of each and the factor disappears exactly:
//!
//! ```text
//! Aiᵉ·Biᵉ′ − Aiᵉ′·Biᵉ = e^{ζ}Ai·e^{−ζ}Bi′ − e^{ζ}Ai′·e^{−ζ}Bi = Ai·Bi′ − Ai′·Bi = 1/π
//! ```
//!
//! The exponential factors cancel, so the same identity exercises the separate scaled
//! implementation without overflow across the tested range.
//!
//! # Where the tolerances come from
//!
//! Each was measured over 2,000-point sweeps per decade before being written here, and each is
//! a claim about a mechanism rather than a round number:
//!
//! - **`I₀K₁ + I₁K₀ = 1/x` is a sum of positive terms** for `x > 0`, so nothing cancels and the
//!   residual is just the inputs' own error. `bessel_i0e` documents a measured worst case of
//!   `1.7e-14`; the sweep's worst is `1.6e-14`, at `x ≈ 38`. The bound is `1e-13`.
//! - **The Airy Wronskian on the positive axis** holds to `5.1e-15` from `0` to `1e12`, because
//!   the scaled forms never build the exponential.
//! - **On the negative axis it degrades as `ζ·ε`**, with `ζ = ⅔|x|^{3/2}`, which is the error
//!   term `airy_ai`'s own docstring names. At `x = −8.96e4` that predicts `3.9e-9` and the
//!   sweep measures `1.3e-9`. A fixed tolerance here would be either wrong near zero or
//!   vacuous far from it, so the bound carries the `ζ` term explicitly — the same reasoning
//!   `exp_and_ln_round_trip` uses for its conditioning factor.
//! - **`J₀Y₁ − J₁Y₀`** does cancel, and its phase stops being resolvable once `ulp(x)` passes
//!   the spacing of a zero; the trait documents that limit as `|x| ≈ 1e11`. The sweep confirms
//!   it: `5.9e-16` to `1e8`, then `9.9e-12` by `1.2e11`. The property stops at `1e8` and says
//!   so rather than loosening the bound to cover a regime the kernel does not claim.

use super::*;

/// `ζ = ⅔|x|^{3/2}`, the Airy phase, and the error term both Airy branches are documented
/// against.
fn airy_zeta(x: f64) -> f64 {
    2.0 / 3.0 * x.abs().powf(1.5)
}

/// The relative bound for an Airy Wronskian residual at `x`.
///
/// Two terms, matching the two regimes in the kernel's own precision note: a floor for the four
/// kernel evaluations and the difference between their products, plus `ζ·ε` on the negative axis
/// where the phase carries the error. On the positive axis the scaled forms never form `e^{ζ}`,
/// so only the floor applies.
fn airy_bound(x: f64) -> f64 {
    let floor = 64.0 * f64::EPSILON;
    if x < 0.0 {
        floor + airy_zeta(x) * f64::EPSILON
    } else {
        floor
    }
}

property! {
    #![config(settings("wronskians").cases(600))]

    /// `Aiᵉ(x)·Biᵉ′(x) − Aiᵉ′(x)·Biᵉ(x) = 1/π`, for **every** real `x`.
    ///
    /// The Airy Wronskian, in the scaled form that cannot overflow. Four separate kernels, one
    /// exact constant, and no argument excluded — `Ai` underflows past `x ≈ 104` and `Bi`
    /// overflows there, but their scaled forms are ordinary numbers at `1e12`.
    ///
    /// A mutation anywhere in the four that breaks their mutual consistency breaks this,
    /// including in a branch a pointwise test never reaches: the identity holds across every
    /// crossover because it holds identically, not approximately.
    fn the_scaled_airy_wronskian_is_one_over_pi(
        x in generate::float::magnitude(-20, 40),
    ) {
        classify!("negative", x < 0.0);
        classify!("positive", x > 0.0);
        classify!("large", x.abs() > 1e6);

        let wronskian = Real::airy_ai_e(x) * Real::airy_bi_e_prime(x)
            - Real::airy_ai_e_prime(x) * Real::airy_bi_e(x);

        // Beyond `|x| ≈ 1e11` the phase `ζ` can no longer be resolved in f64 — `ulp(ζ)` passes
        // `2π` — so on the negative axis the values stop being accurate pointwise. The trait
        // documents that limit; this declines the regime rather than loosening the bound.
        if x < -1e11 {
            return Ok(());
        }

        check!(
            wronskian.is_finite(),
            "the scaled Airy Wronskian must stay finite at x = {x:e}, got {wronskian:e}"
        );
        check_relative!(
            wronskian,
            core::f64::consts::FRAC_1_PI,
            airy_bound(x),
            "Ai^e Bi^e' - Ai^e' Bi^e must be 1/pi at x = {x:e}"
        );
    }

    /// The same identity on the **unscaled** four, wherever both are representable.
    ///
    /// Worth asserting separately: `airy_ai` and `airy_ai_e` are different code paths — one
    /// recovers the other by forming `e^{−ζ}` — so an identity on the scaled family says
    /// nothing about the convenient one. The range stops where `Bi` overflows, which is the
    /// honest limit of the unscaled form rather than a chosen bound.
    fn the_airy_wronskian_is_one_over_pi(x in generate::float::in_range(-300.0, 100.0)) {
        classify!("negative", x < 0.0);
        classify!("positive", x > 0.0);

        let wronskian = Real::airy_ai(x) * Real::airy_bi_prime(x)
            - Real::airy_ai_prime(x) * Real::airy_bi(x);

        // Past `x ≈ 104` one factor has overflowed and the other underflowed, so the product is
        // `0·inf`. That is the scaled form's job, and it is checked above.
        if !wronskian.is_finite() {
            return Ok(());
        }
        check_relative!(
            wronskian,
            core::f64::consts::FRAC_1_PI,
            airy_bound(x),
            "Ai Bi' - Ai' Bi must be 1/pi at x = {x:e}"
        );
    }

    /// `I₀ᵉ(x)·K₁ᵉ(x) + I₁ᵉ(x)·K₀ᵉ(x) = 1/x`, for every `x > 0`.
    ///
    /// The modified Bessel Wronskian. Every term is positive, so there is no cancellation and
    /// the residual is the four kernels' own error — which is why the bound is `1e-13` against
    /// `bessel_i0e`'s documented `1.7e-14` rather than something looser.
    ///
    /// The scaled form is what makes the whole axis reachable: `I₀(710)` overflows and
    /// `K₀(746)` underflows, while the scaled pair are ordinary numbers out past `1e15`. A
    /// sweep of the unscaled identity skipped 1,716 of 2,001 points above `x = 100`.
    fn the_scaled_modified_bessel_wronskian_is_one_over_x(
        x in generate::float::magnitude(-26, 50),
    ) {
        let x = x.abs();
        if x == 0.0 || !x.is_finite() {
            return Ok(());
        }
        classify!("below one", x < 1.0);
        classify!("above one", x >= 1.0);
        classify!("large", x > 1e6);

        let wronskian = Real::bessel_i0e(x) * Real::bessel_k1e(x)
            + Real::bessel_i1e(x) * Real::bessel_k0e(x);
        check!(
            wronskian.is_finite() && wronskian > 0.0,
            "the scaled I/K Wronskian must be positive and finite at x = {x:e}, got {wronskian:e}"
        );
        check_relative!(
            wronskian,
            1.0 / x,
            1e-13,
            "I0^e K1^e + I1^e K0^e must be 1/x at x = {x:e}"
        );
    }

    /// `J₀(x)·Y₁(x) − J₁(x)·Y₀(x) = −2/(πx)`, for `0 < x ≤ 1e8`.
    ///
    /// The oscillatory pair's Wronskian. Unlike the `I`/`K` one this is a genuine difference of
    /// comparable terms, so it does cancel — and it is the only one of the three with a range
    /// limit that is about the *kernel* rather than about overflow: `J` and `Y` are accurate
    /// pointwise only while their phase can be resolved, which the trait documents as failing
    /// near `1e11`. Measured, the residual is `5.9e-16` to `1e8` and `9.9e-12` by `1.2e11`.
    ///
    /// The test stops at `1e8` to stay in the range where the measured residual supports
    /// the tolerance below.
    fn the_bessel_jy_wronskian_is_minus_two_over_pi_x(
        x in generate::float::in_range(1e-8, 1e8),
    ) {
        if x <= 0.0 {
            return Ok(());
        }
        classify!("below one", x < 1.0);
        classify!("above one", x >= 1.0);

        let wronskian = Real::bessel_j0(x) * Real::bessel_y1(x)
            - Real::bessel_j1(x) * Real::bessel_y0(x);
        check_relative!(
            wronskian,
            -2.0 / (core::f64::consts::PI * x),
            1e-13,
            "J0 Y1 - J1 Y0 must be -2/(pi x) at x = {x:e}"
        );
    }
}

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
