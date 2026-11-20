//! The CDF and log-CDF, including the left tail where the probability underflows.
//!
//! Part of the [`normal`](super) suite. The log-CDF is the half that needs the generated laws
//! most: below about `x = -38` the probability itself is gone in `f64`, so `ln Φ` can only be
//! checked against its own asymptotic, and the two routes have to agree across the join.

use super::*;

property! {
    #![config(settings("normal_cdf").cases(500))]

    /// `Φ(x) + Φ(−x) = 1`, with the CDF staying inside `[0, 1]` and hitting its ends.
    ///
    /// The defining symmetry of a symmetric distribution, exact to a single ulp over
    /// `[−40, 40]`, and it needs no reference value. It is also the law that catches a CDF
    /// computed through the wrong tail: using `1 − erf` rather than `erfc` is accurate on one
    /// side and cancels on the other, and the reflection sees the difference.
    fn the_normal_cdf_reflects_about_a_half(x in generate::float::in_range(-40.0, 40.0)) {
        classify!("left tail", x < -5.0);
        classify!("right tail", x > 5.0);

        // Zero ulps, not one. `erfc` reflects on the negatives — `2 − erfc(|x|)` — so this
        // identity is exact by construction rather than by the two sides sharing an error, which
        // is what it used to be and what made it drift when one side improved. See
        // `stability::symmetry_exact::the_complementary_error_function_reflects_exactly`.
        check_ulps!(
            Real::ndtr(x) + Real::ndtr(-x),
            1.0,
            0,
            "Phi(x) + Phi(-x) must be 1"
        );
        let p = Real::ndtr(x);
        check!(
            (0.0..=1.0).contains(&p),
            "ndtr({x:e}) = {p:e} left [0, 1]"
        );
        // Monotone, and strictly so wherever the format can still tell.
        check!(
            Real::ndtr(x + 1.0) >= p,
            "ndtr must be non-decreasing at {x:e}"
        );
    }

    /// The CDF's endpoints and the log-CDF's, including a signed zero that is a real claim.
    ///
    /// `log_ndtr(+∞)` is `−0.0`, not `+0.0`: `ln Φ` approaches zero *from below* because `Φ < 1`
    /// everywhere, and the sign bit is the only thing left recording that at the limit. A
    /// carrier that returned `+0.0` would be indistinguishable by comparison and wrong by
    /// convention, which is exactly the case [`Real::signbit`] exists to express.
    fn the_normal_endpoints_are_exact(x in generate::float::in_range(-1e3, 1e3)) {
        check_ulps!(Real::ndtr(0.0f64), 0.5, 0, "Phi(0) is exactly a half");
        check_ulps!(
            Real::log_ndtr(0.0f64),
            -core::f64::consts::LN_2,
            0,
            "ln Phi(0) is exactly -ln 2"
        );
        check!(
            Real::ndtr(f64::NEG_INFINITY) == 0.0 && Real::ndtr(f64::INFINITY) == 1.0,
            "the CDF must reach both ends"
        );
        check!(
            Real::ndtr(f64::NAN).is_nan() && Real::log_ndtr(f64::NAN).is_nan(),
            "NaN must propagate"
        );

        let limit = Real::log_ndtr(f64::INFINITY);
        check!(
            limit == 0.0 && Real::signbit(&limit),
            "ln Phi(+inf) must be -0.0, approaching zero from below; got {limit:e}"
        );

        // And the log-CDF is never positive, anywhere, because `Phi <= 1`.
        let l = Real::log_ndtr(x);
        check!(
            l <= 0.0 || l.is_nan(),
            "ln Phi({x:e}) = {l:e} must not be positive"
        );
    }

    /// `log_ndtr(x) = ln(Φ(x))` on `x ≤ 0`, where the CDF has not saturated.
    ///
    /// The range is the whole content of this property: see the module header for the `0.11`
    /// relative error the unrestricted form reaches on the right. Below zero `Φ ≤ ½`, the log is
    /// well conditioned, and the two must agree — which pins the direct branch against the
    /// `erfcx` one through their shared boundary.
    ///
    /// The lower limit is where `Φ` underflows to zero and `ln` of it is `−∞` while `log_ndtr`
    /// is still a perfectly ordinary number. That divergence is the function's entire purpose,
    /// so it is declined rather than asserted.
    fn log_ndtr_agrees_with_the_log_of_the_cdf_where_that_is_representable(
        x in generate::float::in_range(-37.0, 0.0),
    ) {
        let p = Real::ndtr(x);
        if !(p > 0.0 && p < 1.0) {
            return Ok(());
        }
        classify!("below minus one", x < -1.0);
        classify!("near zero", x > -1.0);

        relation::agree(
            "log_ndtr = ln(ndtr) on the left",
            (),
            |()| Real::log_ndtr(x),
            |()| p.ln(),
            &Relative(1e-14),
        )?;
    }

    /// `ln Φ(x) → −x²/2 − ln|x| − ln√(2π)` in the left tail, against the asymptotic's own
    /// error term.
    ///
    /// The one genuinely **independent** reference available here: it shares no code with the
    /// kernel, and it reaches `x = −1e6` where `Φ` underflowed hundreds of orders ago. This is
    /// the regime `log_ndtr` was written for, so it is the regime most worth pinning.
    ///
    /// # The bound is the asymptotic's truncation, not the kernel's error
    ///
    /// The next term of the series is `ln(1 − 1/x² + 3/x⁴ − …)`, so the absolute error is about
    /// `1/x²` against a value of about `x²/2` — a relative error of `2/x⁴`. At `x = −50` that
    /// predicts `3.2e-7` and the sweep measures `3.184e-7`, which is the agreement that makes
    /// this a bound rather than a fudge. A constant tolerance here would be vacuous at `1e6`
    /// and wrong at `50`.
    fn log_ndtr_matches_its_left_tail_asymptotic(
        x in generate::float::in_range(-1e6, -50.0),
    ) {
        classify!("deep tail", x < -1e4);

        let asymptotic = -0.5 * x * x - (-x).ln() - half_ln_tau();
        relation::agree(
            "ln Phi ~ -x^2/2 - ln|x| - ln sqrt(2 pi)",
            (),
            |()| Real::log_ndtr(x),
            |()| asymptotic,
            &Relative(4.0 / (x * x * x * x) + 64.0 * f64::EPSILON),
        )?;
    }
}
