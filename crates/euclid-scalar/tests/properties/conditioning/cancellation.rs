//! Cancellation properties.

use super::*;

// ===== The cancellation-prone elementary forms ============================

property! {
    #![config(settings("exp_m1").require("naive loses 8+ digits", 5.0))]

    /// `exp_m1(x)` against `exp(x) - 1`.
    ///
    /// Near zero `exp x` is within an ulp of 1, so the subtraction leaves only rounding noise.
    ///
    /// Reference: `x + x²/2 + x³/6 + x⁴/24`, whose next term is `x⁵/120` — a relative error of
    /// `x⁴/120`. Bounding that by `1e-15` needs `|x| ≤ 2^-11`, which is the generator's upper
    /// limit.
    fn exp_m1_survives_where_the_naive_difference_cannot(
        x in generate::float::magnitude(-80, -13),
    ) {
        let taylor = x + x * x / 2.0 + x * x * x / 6.0 + x * x * x * x / 24.0;
        let safe = Real::exp_m1(&x);
        let naive = x.exp() - 1.0;

        let loss = digits_destroyed(naive, taylor);
        classify!("naive loses 8+ digits", loss >= 8.0);
        classify!("naive loses everything", loss >= 15.0);
        target(loss);

        relation::agree(
            "exp_m1 against its Taylor series",
            x,
            |_| safe,
            |_| taylor,
            &Relative(1e-14),
        )?;
    }
}

property! {
    #![config(settings("ln_1p").require("naive loses 8+ digits", 5.0))]

    /// `ln_1p(x)` against `(1.0 + x).ln()`.
    ///
    /// The cancellation is in the *argument* rather than the result: for tiny `x`, `1.0 + x`
    /// rounds to exactly 1, and the logarithm of that is zero — so the naive form returns zero
    /// for a nonzero answer.
    ///
    /// Reference: `x - x²/2 + x³/3 - x⁴/4`, next term `x⁵/5`, relative error `x⁴/5`. Bounding
    /// that by `1e-15` needs `|x| ≤ 2^-12`, so the generator gets `-14`: `magnitude` reaches
    /// `2^(max+1)`, not `2^max`.
    fn ln_1p_survives_where_the_naive_argument_rounds_away(
        x in generate::float::magnitude(-80, -14),
    ) {
        let taylor = x - x * x / 2.0 + x * x * x / 3.0 - x * x * x * x / 4.0;
        let safe = Real::ln_1p(&x);
        let naive = (1.0 + x).ln();

        let loss = digits_destroyed(naive, taylor);
        classify!("naive loses 8+ digits", loss >= 8.0);
        classify!("naive returns exactly zero", naive == 0.0 && taylor != 0.0);
        target(loss);

        check_relative!(safe, taylor, 1e-14, "ln_1p disagreed with its Taylor series");
    }
}

property! {
    #![config(settings("cos_m1").require("naive loses 8+ digits", 5.0))]

    /// `cos_m1(x)` against `cos(x) - 1` — the haversine kernel.
    ///
    /// `cos x - 1` is `-x²/2` to leading order, so for `x` near `1e-8` the true answer is around
    /// `1e-16`, the ulp of 1, and the naive subtraction has no digits left to express it. This
    /// is the one that silently destroys short-distance accuracy in a geodesic calculation.
    ///
    /// Reference: `-x²/2 + x⁴/24 - x⁶/720`, next term `x⁸/40320`. Relative to the leading
    /// `x²/2` that is `x⁶/20160`; bounding it by `1e-15` needs `|x| ≤ 2^-6`, so the generator
    /// gets `-9` for the mantissa octave plus margin.
    fn cos_m1_survives_the_haversine_cancellation(
        x in generate::float::magnitude(-80, -9),
    ) {
        let x2 = x * x;
        let taylor = -x2 / 2.0 + x2 * x2 / 24.0 - x2 * x2 * x2 / 720.0;
        let safe = Real::cos_m1(x);
        let naive = x.cos() - 1.0;

        let loss = digits_destroyed(naive, taylor);
        classify!("naive loses 8+ digits", loss >= 8.0);
        classify!("naive returns exactly zero", naive == 0.0 && taylor != 0.0);
        target(loss);

        check_relative!(safe, taylor, 1e-14, "cos_m1 disagreed with its Taylor series");
    }
}

property! {
    #![config(settings("sinc"))]

    /// `sinc(x)` against `1 - x²/6 + x⁴/120`, next term `x⁶/5040`; `|x| ≤ 2^-6` keeps that
    /// below `1e-15`.
    fn sinc_is_continuous_through_the_origin(x in generate::float::magnitude(-80, -9)) {
        let x2 = x * x;
        let taylor = 1.0 - x2 / 6.0 + x2 * x2 / 120.0;
        check_relative!(Real::sinc(x), taylor, 1e-14, "sinc disagreed with its Taylor series");
    }

    /// And `sinc` is exactly one at the origin, where the naive form is not a number at all.
    fn sinc_fills_in_the_removable_singularity(_ in generate::just(0u8)) {
        check_ulps!(Real::sinc(0.0_f64), 1.0, 0, "sinc(0) must be exactly 1");
        let naive = 0.0_f64.sin() / 0.0;
        check!(naive.is_nan(), "precondition: the naive form is NaN at the origin");
    }
}
