//! Searching for numerical instability, rather than asserting its absence at fixed points.
//!
//! `src/tests/accuracy` already checks this crate's kernels against known values at chosen
//! arguments. That is the right way to pin a published constant, and it has a blind spot: the
//! arguments were chosen by whoever wrote the test, so the worst argument is found only if
//! someone thought of it. This suite is the complement — it *searches*, over a float
//! distribution built to reach the awkward regions, and it hill-climbs on observed error.
//!
//! # Every test here has two halves
//!
//! Each cancellation-prone form in this crate exists because a naive expression of the same
//! quantity is catastrophically wrong somewhere. So the tests assert both directions:
//!
//! 1. **The safe form is accurate**, checked against an independent reference.
//! 2. **The naive form really is bad**, enforced by
//!    [`Config::require`](euclid_stability::Config::require): the property *requires* that some
//!    fraction of generated cases exhibit severe naive error, so the test fails if the
//!    generator ever stops reaching the ill-conditioned region.
//!
//! The second half is what makes this a demonstration rather than a hope. Without it, a change
//! that narrowed the generator would leave a green suite checking nothing interesting — the
//! safe form would agree with the naive one everywhere, because nowhere hard was ever tried.
//!
//! # Choosing the reference, and the range that makes it valid
//!
//! Near zero the reference is a truncated Taylor series: it converges in a few terms, has no
//! cancellation, and is not this crate's own answer, so the check is not circular. But a
//! truncated series is only accurate over the range where its next term is negligible, and that
//! range is **much smaller than it looks**. The first draft of this file generated `|x|` up to
//! `2^-2` and compared against a four-term series at a `1e-13` tolerance; the truncation error
//! there is around `1e-5`, so the suite failed immediately — correctly, and against the test
//! rather than the code.
//!
//! Each generator below is therefore bounded by the exponent at which its reference is good to
//! about `1e-15` relative, and the arithmetic is recorded next to it. That is a real constraint
//! and not a workaround: outside the stated range the naive form is well conditioned anyway, so
//! nothing interesting is being skipped.
//!
//! Set `EUCLID_STABILITY_STATS=1` to see the distribution each property reached and the worst
//! error its climb found.

use euclid_numeric::Real;
use euclid_numeric::compensated;
use euclid_stability::relation::Relative;
use euclid_stability::{
    Config, check, check_relative, check_ulps, classify, generate, property, relation, target,
    tolerance,
};

/// How many of an `f64`'s ~16 significant digits `approximate` lost against `reference`.
///
/// Returned as `f64` so it can be handed to [`target`] directly: the search then climbs towards
/// the input that destroys the most digits, which is exactly the question "how bad can this
/// get".
fn digits_destroyed(approximate: f64, reference: f64) -> f64 {
    if reference == 0.0 || !reference.is_finite() || !approximate.is_finite() {
        return 0.0;
    }
    let relative = ((approximate - reference) / reference).abs();
    if relative <= 0.0 {
        return 0.0;
    }
    (16.0 + relative.log10()).max(0.0)
}

fn settings(suite: &str) -> Config {
    Config::default()
        .cases(400)
        .regressions(format!("tests/regressions/{suite}.txt"))
}

// ===== The exact primitives ===============================================

property! {
    #![config(settings("two_sum").cases(600).require("remainder is nonzero", 20.0))]

    /// `two_sum` must be **exact**: `s + e` is `a + b` with no error at all. Not "accurate to a
    /// few ulps" — exact, because every compensated algorithm in the crate is built on that
    /// guarantee and silently degrades if it only almost holds.
    ///
    /// Verifying exactness needs arithmetic wider than the thing under test, which is the whole
    /// difficulty. The first attempt here reconstructed the remainder as `(s - a) - b` and
    /// failed, correctly: those subtractions are themselves rounded in general, so the
    /// "reconstruction" was just another approximation. Sterbenz's lemma makes `s - a` exact
    /// only when the two are within a factor of two, which is not the interesting case.
    ///
    /// The fix is to generate operands that are **integers**, so `a + b` is exactly computable
    /// in `i128` and the comparison is against true arithmetic rather than against more `f64`.
    /// Integer operands still exercise the hard part — the magnitudes differ by up to 2^53, so
    /// the sum rounds and the remainder is genuinely nonzero.
    fn two_sum_is_exact_on_integer_operands(
        a_bits in generate::int(-(1i64 << 53), 1i64 << 53),
        b_shift in generate::int(0, 60),
    ) {
        // `a` is an integer; `b` is an integer scaled far enough away that `a + b` must round.
        #[allow(clippy::cast_precision_loss)]
        let a = a_bits as f64;
        let b = (a_bits.rotate_left(17) % (1i64 << 40)) as f64 * 2.0_f64.powi(b_shift as i32);
        if !a.is_finite() || !b.is_finite() {
            return Ok(());
        }

        let (sum, error) = Real::two_sum(a, b);
        if !sum.is_finite() {
            check_ulps!(error, 0.0, 0, "a non-finite sum must carry no remainder");
            return Ok(());
        }

        classify!("remainder is nonzero", error != 0.0);
        classify!("remainder is zero", error == 0.0);

        // Both `sum` and `error` are integers here (sums of integers, exactly representable or
        // not), so the true total can be compared in i128 with no rounding anywhere.
        let exact_total = i128::from(a_bits) + (b as i128);
        let reconstructed = (sum as i128) + (error as i128);
        check!(
            reconstructed == exact_total,
            "two_sum({a}, {b}) = ({sum}, {error}) reconstructs {reconstructed}, not {exact_total}"
        );
    }

}

property! {
    // Its own block: the requirement above names `two_sum`'s label, and a block-level config
    // applies to every property in it. Sharing one block made this property fail for having
    // never applied a label it does not use, which is the footgun worth keeping separated.
    #![config(settings("two_product"))]

    /// `two_product` must likewise be exact: `p + e` is `a * b`, to the last bit.
    ///
    /// Here there *is* a wider primitive available — `fma(a, b, -p)` is the exact remainder by
    /// definition — so it serves as the reference directly.
    fn two_product_is_exact(
        a in generate::float::magnitude(-200, 200),
        b in generate::float::magnitude(-200, 200),
    ) {
        let (product, error) = Real::two_product(a, b);
        if !product.is_finite() || (error != 0.0 && !error.is_finite()) {
            return Ok(());
        }
        let exact_remainder = Real::mul_add(a, b, -product);
        check_ulps!(error, exact_remainder, 0, "two_product lost part of the remainder");
    }
}

// ===== Compensated summation ==============================================

property! {
    #![config(
        settings("compensated_sum")
            .require("ill conditioned", 5.0)
            .require("naive loses 6+ digits", 2.0)
    )]

    /// Compensated summation must survive cancellation that destroys the naive fold.
    ///
    /// The terms are built so their true total is known by construction: a large value, its
    /// negation, and a handful of small ones sum to exactly the small ones. The naive
    /// left-to-right fold cannot see them, because each is below the running total's ulp.
    ///
    /// The coverage requirements are what make this a search rather than an anecdote.
    fn compensated_summation_beats_naive_under_cancellation(
        big in generate::float::magnitude(20, 200),
        smalls in generate::vec(generate::float::in_range(-4.0, 4.0), 1, 6),
    ) {
        let mut terms = vec![big];
        terms.extend_from_slice(&smalls);
        terms.push(-big);

        let exact: f64 = smalls.iter().sum();
        let condition = tolerance::condition_number(&terms);

        classify!("ill conditioned", condition > 1e12);
        classify!("benign", condition < 1e3);

        let naive = terms.iter().fold(0.0_f64, |acc, t| acc + t);
        let compensated = compensated::sum(&terms);

        let naive_loss = digits_destroyed(naive, exact);
        classify!("naive loses 6+ digits", naive_loss >= 6.0);
        target(naive_loss);

        check_relative!(
            compensated,
            exact,
            1e-12,
            "compensated summation failed at condition number {condition:e}"
        );
    }
}

property! {
    // A block of its own, because the requirements name *this* property's labels. A shared
    // block-level config applies its requirements to every property in the block, which fails
    // confusingly when two properties classify differently — the first draft of this file hit
    // exactly that, and the coverage report said so.
    #![config(settings("compensated_dot").require("naive loses 4+ digits", 2.0))]

    /// The compensated dot product must be exact for inputs whose cancellation is known by
    /// construction, where the naive form accumulates the loss term by term.
    fn compensated_dot_survives_cancellation(
        scale in generate::float::magnitude(20, 150),
        tail in generate::float::in_range(-2.0, 2.0),
    ) {
        // Order matters, and the first draft got it wrong: `[scale, -scale, tail]` cancels
        // *first*, so the naive fold loses nothing and the coverage requirement failed. The
        // small term has to sit between the large ones to be rounded away.
        let a = [scale, tail, -scale];
        let b = [1.0, 1.0, 1.0];

        let naive: f64 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        let compensated = compensated::dot(&a, &b);

        let loss = digits_destroyed(naive, tail);
        classify!("naive loses 4+ digits", loss >= 4.0);
        target(loss);

        check_relative!(compensated, tail, 1e-12, "compensated dot failed");
    }
}

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

// ===== Overflow and underflow rather than cancellation ====================

property! {
    #![config(settings("hypot").require("naive overflows", 50.0))]

    /// `hypot(x, y)` against `(x*x + y*y).sqrt()`.
    ///
    /// A different failure from cancellation: squaring a component above `2^512` overflows
    /// `f64` even when the norm is perfectly representable, so the naive form returns infinity
    /// for a finite answer.
    ///
    /// The exponent bound matters and is easy to get wrong — the first draft generated
    /// `magnitude(120, 300)`, reasoning in decimal. Those are *binary* exponents, so the
    /// largest value produced was about `4e90`, whose square is `1.6e181` and perfectly finite:
    /// the naive form never overflowed at all, and the coverage requirement caught it. Overflow
    /// needs `|x| > 2^512 ≈ 1.3e154`.
    ///
    /// Reference: the naive form *rescaled*. Factoring out the larger component is exactly what
    /// `hypot` is supposed to do, and the rescaled expression cannot overflow.
    fn hypot_avoids_intermediate_overflow(
        x in generate::float::magnitude(515, 1_020),
        y in generate::float::magnitude(515, 1_020),
    ) {
        let safe = Real::hypot(x, y);
        let naive = (x * x + y * y).sqrt();

        classify!("naive overflows", !naive.is_finite());
        classify!("naive survives", naive.is_finite());

        let m = x.abs().max(y.abs());
        let reference = m * ((x / m) * (x / m) + (y / m) * (y / m)).sqrt();

        check!(safe.is_finite(), "hypot overflowed on a representable norm");
        check_relative!(safe, reference, 1e-14, "hypot disagreed with the rescaled norm");
    }
}

property! {
    #![config(settings("hypot_underflow").require("naive underflows to zero", 50.0))]

    /// The underflow direction, which is the one people forget: squaring a component below
    /// `2^-537` flushes to zero, so the naive form returns zero for a nonzero norm.
    fn hypot_avoids_intermediate_underflow(
        x in generate::float::magnitude(-1_020, -540),
        y in generate::float::magnitude(-1_020, -540),
    ) {
        let safe = Real::hypot(x, y);
        let naive = (x * x + y * y).sqrt();

        classify!("naive underflows to zero", naive == 0.0);

        let m = x.abs().max(y.abs());
        let reference = m * ((x / m) * (x / m) + (y / m) * (y / m)).sqrt();

        check!(safe > 0.0, "hypot underflowed on a nonzero norm");
        check_relative!(safe, reference, 1e-14, "hypot disagreed with the rescaled norm");
    }
}

property! {
    #![config(settings("hypot3").require("naive overflows", 50.0))]

    /// `hypot3` has the same contract in three dimensions.
    fn hypot3_avoids_intermediate_overflow(
        x in generate::float::magnitude(515, 1_020),
        y in generate::float::magnitude(515, 1_020),
        z in generate::float::magnitude(515, 1_020),
    ) {
        let safe = Real::hypot3(x, y, z);
        let naive = (x * x + y * y + z * z).sqrt();
        classify!("naive overflows", !naive.is_finite());

        let m = x.abs().max(y.abs()).max(z.abs());
        let reference = m * ((x / m) * (x / m) + (y / m) * (y / m) + (z / m) * (z / m)).sqrt();

        check!(safe.is_finite(), "hypot3 overflowed on a representable norm");
        check_relative!(safe, reference, 1e-14, "hypot3 disagreed with the rescaled norm");
    }
}

// ===== Conditioning as a measurement ======================================

/// The conditioning curve: naive summation's error as a function of the condition number, with
/// compensated summation's error alongside it.
///
/// Not a property but a *measurement*, and the clearest statement of what this suite is for. A
/// fixed-point accuracy test can say "the sum was right at these terms"; only a sweep can say
/// "the naive algorithm loses every digit past this condition number, and the compensated one
/// does not".
///
/// The terms are `[κ/2, 1, -κ/2]`, whose exact sum is `1` and whose condition number is about
/// `κ`. The interesting threshold is `2^53`: below it the `1` still fits in the running total's
/// ulp and the naive fold is fine, above it the `1` is rounded away entirely.
///
/// A previous version of this test tried to show `exp(a+b) = exp(a)·exp(b)` degrading with
/// magnitude. It does not, and the measurement said so — `exp`'s error is *relative*, so the
/// product compounds it by a constant factor rather than a growing one, and the worst relative
/// error sat at `2.2e-16` across every decade. Keeping the honest negative result here because
/// the wrong intuition is a natural one: not every identity is ill-conditioned at scale, and
/// the way to find out is to measure.
#[test]
fn naive_summation_degrades_with_condition_number_and_compensated_does_not() {
    let mut rows = Vec::new();

    for exponent in [0i32, 4, 8, 12, 15, 16, 18, 20] {
        let half = 10f64.powi(exponent);
        let terms = [half, 1.0, -half];
        let exact = 1.0;

        let condition = tolerance::condition_number(&terms);
        let naive = terms.iter().fold(0.0_f64, |acc, t| acc + t);
        let compensated = compensated::sum(&terms);

        let naive_error = (naive - exact).abs() / exact;
        let compensated_error = (compensated - exact).abs() / exact;
        rows.push((exponent, condition, naive_error, compensated_error));
    }

    eprintln!("summation error against condition number:");
    eprintln!("    terms         condition      naive error   compensated error");
    for (exponent, condition, naive, compensated) in &rows {
        eprintln!(
            "    [1e{exponent:<2}, 1, -1e{exponent:<2}]  {condition:>9.2e}  {naive:>13.2e}  {compensated:>17.2e}"
        );
    }

    // Compensated summation is exact at every condition number tested. This is the claim worth
    // making, and it is much stronger than "accurate to a tolerance".
    for (exponent, condition, _, compensated_error) in &rows {
        assert_eq!(
            *compensated_error, 0.0,
            "compensated summation lost accuracy at 1e{exponent} (condition {condition:e})"
        );
    }

    // And the naive fold really does fail, which is what makes the comparison meaningful. Past
    // 2^53 the small term is below the running total's ulp and is discarded outright.
    let worst_naive = rows
        .iter()
        .map(|(_, _, naive, _)| *naive)
        .fold(0.0_f64, f64::max);
    assert!(
        worst_naive >= 1.0,
        "the naive fold never lost the small term, so this sweep is not exercising \
         cancellation at all; worst relative error was {worst_naive:e}"
    );

    // The degradation is monotone in the condition number rather than noisy: error never
    // improves as conditioning worsens.
    for window in rows.windows(2) {
        let (before, after) = (&window[0], &window[1]);
        assert!(
            after.2 >= before.2,
            "naive error fell from {:e} to {:e} as the condition number rose from {:e} to {:e}",
            before.2,
            after.2,
            before.1,
            after.1
        );
    }
}
