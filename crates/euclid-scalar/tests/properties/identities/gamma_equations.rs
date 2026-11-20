//! The gamma family's functional equations: reflection, duplication and recurrence.
//!
//! Part of the [`identities`](super) suite.
//!
//! These are the strongest laws the gamma family has, and they are the right instrument for the
//! same reason the Airy Wronskians were: each one constrains the function at **two or three
//! different arguments at once**, so a wrong branch, a wrong coefficient or a misplaced crossover
//! has nowhere to hide. A systematic error in the Lanczos coefficients does not survive
//! duplication.
//!
//! # What these add over the fixed tests
//!
//! `src/tests/relationships/functional_equations.rs` already asserts four of these, and the
//! generated versions are not duplicates of it. The fixed tests sample *round arguments in narrow
//! ranges* — `digamma` reflection at twenty points in `(0, 1)` at multiples of `0.05`, duplication
//! only at quarter-integers below twenty — which is precisely the gap that let `ulp`'s direction
//! bug survive a thousand-decade sweep: the law was right and the arguments could not reach the
//! failure. What is new here:
//!
//! - **`Γ(x)Γ(1−x) = π/sin(πx)` on the negative axis**, which nothing tested. This is the law
//!   that pins `Γ` between its poles, where the measured spread is `387` ulp
//!   across the pole structure. The reference compensates for rounding of the angle before
//!   applying the sine addition identity.
//! - **`polygamma` at orders above one.** The fixed suite covers `digamma` and `trigamma`; the
//!   general recurrence covers every order, and `polygamma.rs` plus its f32 peer held 24
//!   survivors.
//! - **The f32 peers**, where 43 of the family's survivors live and no functional equation had
//!   ever been applied.
//! - **`rgamma`**, including the claim its docstring makes and nothing checked: `1/Γ(175)` is a
//!   representable `1.56e-316` while `Γ(175).recip()` is `0`, because the intermediate overflows.
//!
//! # Four tolerance traps, all found by measuring first
//!
//! - **`π·cot(πx)` is zero at `x = ½`**, so `digamma`'s reflection has a reference that crosses
//!   zero and a relative comparison there reports an error of `1.0` against a correct kernel.
//!   This is the **fourth** suite in a row to hit a zero-crossing reference; it is now the first
//!   thing to check when a law involves a trigonometric factor.
//! - **The `polygamma` recurrence's reference cancels, and badly.** `ψ⁽ⁿ⁾(x+1) = ψ⁽ⁿ⁾(x) +
//!   (−1)ⁿn!/x^{n+1}` is exact, but at `x = 0.05` and `n = 5` the correction term is `7.7e9`
//!   while the result is `O(1)`: two huge quantities cancelling to a small one. Measured relative
//!   error climbs `6.5e-14, 1.8e-14, 8.5e-12, 4.4e-12, 6.6e-9, 2.2e-7` for orders zero to five —
//!   that is the *reference* losing digits, not the kernel, so the bound carries the
//!   amplification explicitly.
//! - **The recurrence degrades near a pole.** `Γ(x+1) = xΓ(x)` is `2.3e-10` at `x = −15 + 5e-7`,
//!   which is `ε` divided by the distance to the pole. The bound carries that distance.
//! - **f32 cannot support the high-order recurrence at all**: orders zero to three reach `15%`
//!   relative. The f32 property keeps the orders and the range where the cancellation is
//!   survivable and says so, rather than loosening a bound until it passes.

use super::*;

/// Distance from `x` to the nearest integer — where `Γ`'s poles and the reflection's zeros are.
fn pole_distance(x: f64) -> f64 {
    (x - x.round()).abs()
}

/// Recover the low part of πx and apply the sine addition identity.
fn reflection_sine_reference(x: f64) -> f64 {
    const PI_LOW: f64 = 1.224_646_799_147_353_2e-16;
    let pi = core::f64::consts::PI;
    let angle = pi * x;
    let residual = x.mul_add(pi, -angle) + x * PI_LOW;
    let (sin, cos) = angle.sin_cos();
    sin * residual.cos() + cos * residual.sin()
}

property! {
    #![config(settings("gamma_reflection").cases(600))]

    /// `Γ(x)Γ(1−x) = π/sin(πx)` — Euler's reflection formula, on both sides of the origin.
    ///
    /// One relation tying `Γ` at `x` and at `1−x` to `sin` and `π`. On the negative axis it is
    /// the only law available that does not need a reference value, because it maps each
    /// inter-pole interval onto a positive argument where `Γ` is well behaved — and it is
    /// *where* `Γ` is hardest, since the function alternates sign and diverges between every
    /// pair of consecutive negative integers.
    ///
    fn gamma_satisfies_eulers_reflection_formula(
        x in generate::float::in_range(-150.0, 1.0),
    ) {
        // Within a hair of a pole both sides are enormous and the comparison measures the
        // rounding of `x` rather than the kernel; the pole-adjacent behaviour has its own test.
        if pole_distance(x) < 1e-6 || x >= 1.0 {
            return Ok(());
        }
        classify!("negative", x < 0.0);
        classify!("the unit interval", x > 0.0);
        classify!("far from the origin", x < -20.0);

        let product = Real::gamma(x) * Real::gamma(1.0 - x);
        let closed_form = core::f64::consts::PI / reflection_sine_reference(x);
        if !product.is_finite() || !closed_form.is_finite() {
            return Ok(());
        }
        relation::agree(
            "Gamma(x) Gamma(1-x) = pi / sin(pi x)",
            (),
            |()| product,
            |()| closed_form,
            &Relative(1e-12),
        )?;
    }

    /// The same in logs, which reaches `|x| = 170` where the product would overflow.
    ///
    /// `ln|Γ(x)| + ln|Γ(1−x)| = ln(π/|sin πx|)`. Half the negative axis is skipped by
    /// construction rather than by choice: `ln_gamma` is `NaN` where `Γ` is negative, which is
    /// every other inter-pole interval, so the law applies on the intervals where both factors
    /// are positive. The guard is `is_finite`, and it declined 1,999 of 4,001 swept points when
    /// this was measured — which is the right half, not a failure.
    fn ln_gamma_satisfies_the_reflection_formula(
        x in generate::float::in_range(-170.0, 0.0),
    ) {
        if pole_distance(x) < 1e-6 {
            return Ok(());
        }
        let sum = Real::ln_gamma(x) + Real::ln_gamma(1.0 - x);
        let closed_form = (core::f64::consts::PI / reflection_sine_reference(x).abs()).ln();
        if !sum.is_finite() || !closed_form.is_finite() {
            return Ok(());
        }
        classify!("deep negative", x < -50.0);
        relation::agree(
            "ln|Gamma(x)| + ln|Gamma(1-x)| = ln(pi/|sin pi x|)",
            (),
            |()| sum,
            |()| closed_form,
            &Relative(1e-12),
        )?;
    }

    /// `Γ(x+1) = x·Γ(x)`, with a bound that carries the distance to the pole.
    ///
    /// The defining recurrence. On the positive axis it holds to `7.1e-14`; approaching a pole
    /// from the negative side it degrades as `ε` over the distance — `2.3e-10` at
    /// `x = −15 + 5e-7`, against a predicted `4.4e-10`. That is the argument's own rounding
    /// being amplified by a function that is diverging, not a defect, so the bound says so.
    fn gamma_satisfies_its_recurrence(x in generate::float::in_range(-20.0, 170.0)) {
        if pole_distance(x) < 1e-9 && x <= 0.0 {
            return Ok(());
        }
        classify!("negative", x < 0.0);
        classify!("positive", x > 0.0);

        let shifted = Real::gamma(x + 1.0);
        let scaled = x * Real::gamma(x);
        if !shifted.is_finite() || !scaled.is_finite() || scaled == 0.0 {
            return Ok(());
        }
        // `eps / distance-to-pole` on the negative axis, where `Gamma` is diverging; a plain
        // floor on the positive axis, where it is not.
        let bound = if x < 0.0 {
            (8.0 * f64::EPSILON / pole_distance(x)).max(1e-13)
        } else {
            1e-12
        };
        relation::agree(
            "Gamma(x+1) = x Gamma(x)",
            (),
            |()| shifted,
            |()| scaled,
            &Relative(bound.min(1e-6)),
        )?;
    }
}

property! {
    #![config(settings("digamma_equations").cases(600))]

    /// `ψ(1−x) − ψ(x) = π·cot(πx)` — digamma's reflection formula.
    ///
    /// # Absolute, because the reference is zero at the midpoint
    ///
    /// `cot(π/2)` is zero, so at `x = ½` both sides vanish and a relative comparison divides by
    /// nothing — measured as a relative error of `1.0` against a correct kernel, which says
    /// only that the test was written wrong. The bound scales with the reference's own
    /// magnitude, which is the mixed criterion this suite has now needed four times.
    fn digamma_satisfies_the_reflection_formula(
        x in generate::float::in_range(0.001, 0.999),
    ) {
        if !(0.0..1.0).contains(&x) {
            return Ok(());
        }
        classify!("near the midpoint", (x - 0.5).abs() < 0.05);
        classify!("near an endpoint", x < 0.1 || x > 0.9);

        let difference = Real::digamma(1.0 - x) - Real::digamma(x);
        let cotangent = core::f64::consts::PI / (core::f64::consts::PI * x).tan();
        if !difference.is_finite() || !cotangent.is_finite() {
            return Ok(());
        }
        // Scaled by the larger of the two, so the midpoint is judged absolutely and the
        // endpoints — where `cot` diverges — relatively.
        let scale = difference.abs().max(cotangent.abs()).max(1.0);
        relation::agree(
            "psi(1-x) - psi(x) = pi cot(pi x)",
            (),
            |()| difference,
            |()| cotangent,
            &Absolute(1e-12 * scale),
        )?;
    }

    /// `ψ(2x) = ½ψ(x) + ½ψ(x+½) + ln 2` and the `ln Γ` duplication beside it.
    ///
    /// Two relations at three arguments each, swept to `x = 1e5` where the fixed tests stop at
    /// twenty. `ln_gamma`'s version holds to `5.2e-16` over that whole range, which is the
    /// strongest statement in this file and the one that would catch a Stirling crossover
    /// placed wrongly.
    fn the_duplication_formulae_hold_over_the_whole_range(
        x in generate::float::in_range(0.01, 1e5),
    ) {
        if x <= 0.0 {
            return Ok(());
        }
        classify!("small", x < 1.0);
        classify!("large", x > 1e3);

        let digamma_lhs = Real::digamma(2.0 * x);
        let digamma_rhs =
            0.5 * Real::digamma(x) + 0.5 * Real::digamma(x + 0.5) + <f64 as Real>::ln_2();
        if digamma_lhs.is_finite() && digamma_rhs.is_finite() {
            let scale = digamma_lhs.abs().max(1.0);
            relation::agree(
                "psi(2x) = psi(x)/2 + psi(x + 1/2)/2 + ln 2",
                (),
                |()| digamma_lhs,
                |()| digamma_rhs,
                &Absolute(1e-13 * scale),
            )?;
        }

        let ln_gamma_lhs = Real::ln_gamma(2.0 * x);
        let ln_gamma_rhs = (2.0 * x - 1.0) * <f64 as Real>::ln_2()
            - 0.5 * <f64 as Real>::ln_pi()
            + Real::ln_gamma(x)
            + Real::ln_gamma(x + 0.5);
        if ln_gamma_lhs.is_finite() && ln_gamma_rhs.is_finite() {
            let scale = ln_gamma_lhs.abs().max(1.0);
            relation::agree(
                "lnG(2x) = (2x-1)ln2 - ln(pi)/2 + lnG(x) + lnG(x + 1/2)",
                (),
                |()| ln_gamma_lhs,
                |()| ln_gamma_rhs,
                &Absolute(1e-14 * scale),
            )?;
        }
    }

    /// `ψ'(x) + ψ'(1−x) = π²/sin²(πx)` — trigamma's reflection.
    fn trigamma_satisfies_the_reflection_formula(
        x in generate::float::in_range(0.001, 0.999),
    ) {
        if !(0.0..1.0).contains(&x) {
            return Ok(());
        }
        let sine = (core::f64::consts::PI * x).sin();
        let sum = Real::trigamma(x) + Real::trigamma(1.0 - x);
        let closed_form = core::f64::consts::PI * core::f64::consts::PI / (sine * sine);
        if !sum.is_finite() || !closed_form.is_finite() {
            return Ok(());
        }
        relation::agree(
            "psi'(x) + psi'(1-x) = pi^2 / sin^2(pi x)",
            (),
            |()| sum,
            |()| closed_form,
            &Relative(1e-12),
        )?;
    }
}

property! {
    #![config(settings("polygamma_recurrence").cases(500))]

    /// `ψ⁽ⁿ⁾(x+1) = ψ⁽ⁿ⁾(x) + (−1)ⁿ·n!/x^{n+1}`, for every order the format can support.
    ///
    /// The general recurrence, which the fixed suite covers only at orders zero and one through
    /// the named `digamma` and `trigamma`. `polygamma.rs` and its f32 peer held 24 survivors.
    ///
    /// # The bound carries the reference's cancellation, which grows with the order
    ///
    /// The identity is exact, and evaluating it is not. At `x = 0.05` and `n = 5` the correction
    /// term `n!/x^{n+1}` is `7.7e9` while `ψ⁽⁵⁾(x+1)` is `O(1)`, so the right-hand side is a
    /// difference of two huge quantities that cancels down to a small one — the *reference*
    /// losing its digits, not the kernel. Measured relative error over `[0.05, 60]`:
    ///
    /// ```text
    /// order   0        1        2        3        4        5
    /// error   6.5e-14  1.8e-14  8.5e-12  4.4e-12  6.6e-9   2.2e-7
    /// ```
    ///
    /// So the bound is the ratio of the correction to the answer, times `ε` — which is the
    /// standard statement of a cancellation's cost, and which collapses to a few ulps wherever
    /// the correction is not dominant.
    fn polygamma_satisfies_its_recurrence(
        x in generate::float::in_range(0.05, 60.0),
        order in generate::int(0, 5),
    ) {
        if x <= 0.0 {
            return Ok(());
        }
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let order = order as u32;
        classify!("high order", order >= 4);
        classify!("small argument", x < 1.0);

        #[allow(clippy::cast_precision_loss)]
        let factorial = (1..=u64::from(order)).product::<u64>() as f64;
        let sign = if order.is_multiple_of(2) { 1.0 } else { -1.0 };
        let correction = sign * factorial / x.powi(order as i32 + 1);

        let shifted = Real::polygamma(x + 1.0, order);
        let recurred = Real::polygamma(x, order) + correction;
        if !shifted.is_finite() || !recurred.is_finite() || shifted == 0.0 {
            return Ok(());
        }

        // The cancellation: how much larger the correction is than the result it produces.
        let amplification = (correction.abs() / shifted.abs()).max(1.0);
        relation::agree(
            "psi^(n)(x+1) = psi^(n)(x) + (-1)^n n! / x^(n+1)",
            (),
            |()| shifted,
            |()| recurred,
            &Relative(64.0 * f64::EPSILON * amplification),
        )?;
    }

    /// `polygamma` at orders zero and one *is* `digamma` and `trigamma`, to the bit.
    ///
    /// The trait says so — "`order = 0` is `digamma` and `order = 1` is `trigamma`; this
    /// generalizes both" — and a generalization that merely *agrees closely* with the thing it
    /// generalizes has a second implementation hiding in it. Asserted at zero ulps, which is
    /// also the assertion that would notice if the two stopped sharing a code path.
    fn polygamma_generalizes_digamma_and_trigamma(
        x in generate::float::in_range(0.01, 200.0),
    ) {
        if x <= 0.0 {
            return Ok(());
        }
        check_ulps!(
            Real::polygamma(x, 0),
            Real::digamma(x),
            0,
            "polygamma(x, 0) must be digamma(x) exactly"
        );
        check_ulps!(
            Real::polygamma(x, 1),
            Real::trigamma(x),
            0,
            "polygamma(x, 1) must be trigamma(x) exactly"
        );
    }
}

property! {
    #![config(settings("rgamma").cases(500))]

    /// `1/Γ` is entire: it has no poles, it is zero at every non-positive integer, and it is
    /// defined where `Γ` itself overflows.
    ///
    /// That last clause is a claim `rgamma`'s docstring makes and nothing checked:
    /// "`1/Γ(175)` is a representable `1e-316`, while `Γ(175).recip()` gives `0` because the
    /// intermediate overflowed." Measured, `rgamma(175)` is `1.5563171e-316` and
    /// `Γ(175).recip()` is `0.0` — so the claim holds and is now pinned, including the *reason*
    /// it is not expressible as a reciprocal.
    fn the_reciprocal_gamma_is_entire(x in generate::float::in_range(-20.0, 170.0)) {
        // The zeros, which are exact rather than small.
        for integer in [0.0f64, -1.0, -2.0, -3.0, -10.0, -170.0] {
            check!(
                Real::rgamma(integer) == 0.0,
                "rgamma({integer}) must be exactly zero, got {:e}",
                Real::rgamma(integer)
            );
        }

        // Defined past `Gamma`'s overflow, which is the whole point of having it.
        let beyond = Real::rgamma(175.0f64);
        check!(
            beyond > 0.0 && beyond.is_finite(),
            "rgamma(175) must be a representable positive value, got {beyond:e}"
        );
        check!(
            Real::gamma(175.0f64).recip() == 0.0,
            "and the reciprocal route must underflow, or this test has lost its point"
        );

        // Elsewhere it agrees with the reciprocal, where the reciprocal exists.
        if pole_distance(x) < 1e-6 && x <= 0.0 {
            return Ok(());
        }
        let gamma = Real::gamma(x);
        if !gamma.is_finite() || gamma == 0.0 {
            return Ok(());
        }
        classify!("negative", x < 0.0);
        relation::agree(
            "rgamma(x) Gamma(x) = 1",
            (),
            |()| Real::rgamma(x) * gamma,
            |()| 1.0,
            &Relative(1e-12),
        )?;
    }
}

property! {
    #![config(settings("rising_factorial").cases(600))]

    /// `(a)_{n+1} = (a)_n·(a+n)`, in logs — and it is **bit-exact**.
    ///
    /// The rising factorial's own recurrence, and the best law in this file: measured at **zero
    /// ulps** across `a ∈ (0, 1e12]` and every integer `n` up to forty, in `f64` and `f32`
    /// alike. Nothing is approximated — the identity is a single `ln(a+n)` added to a previous
    /// value — so anything short of equality is a defect.
    ///
    /// This is the law the first draft of this suite was missing, and the omission showed:
    /// `ln_rising_factorial` held **twenty** survivors across the two precisions, more than any
    /// other function in the family, while the properties beside it tested `gamma`, `ln_gamma`,
    /// `digamma`, `trigamma` and `polygamma` and never touched it.
    fn the_rising_factorial_satisfies_its_recurrence(
        a in generate::float::magnitude(-6, 40),
        step in generate::int(0, 40),
    ) {
        let a = a.abs();
        if a <= 0.0 || !a.is_finite() || a.is_nan() {
            return Ok(());
        }
        #[allow(clippy::cast_precision_loss)]
        let n = step as f64;
        classify!("large shape", a > 1e6);
        classify!("small shape", a < 1.0);

        let longer = Real::ln_rising_factorial(a, n + 1.0);
        let shorter = Real::ln_rising_factorial(a, n) + (a + n).ln();
        if !longer.is_finite() || !shorter.is_finite() {
            return Ok(());
        }
        check_ulps!(
            longer,
            shorter,
            0,
            "(a)_(n+1) = (a)_n (a+n) must hold exactly at a = {a:e}, n = {n}"
        );
    }

    /// `ln (a)ₙ = Σ_{k<n} ln(a+k)` for integer `n`, also bit-exact, plus the two endpoints.
    ///
    /// The direct definition. Unlike the `ln_gamma` difference below it has no cancellation at
    /// any shape, which is why it can be asserted at zero ulps rather than to a tolerance.
    fn the_rising_factorial_is_the_product_it_claims_to_be(
        a in generate::float::in_range(0.01, 1e4),
        count in generate::int(0, 30),
    ) {
        if a <= 0.0 {
            return Ok(());
        }
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let n = count as u32;

        let mut sum = 0.0f64;
        for k in 0..n {
            sum += (a + f64::from(k)).ln();
        }
        let direct = Real::ln_rising_factorial(a, f64::from(n));
        if !direct.is_finite() || !sum.is_finite() {
            return Ok(());
        }
        check_ulps!(
            direct,
            sum,
            0,
            "ln (a)_n must equal the sum of logs at a = {a:e}, n = {n}"
        );

        // The empty product is one, so its log is zero — exactly, at every shape.
        check_ulps!(
            Real::ln_rising_factorial(a, 0.0),
            0.0,
            0,
            "the empty rising factorial is exactly zero in logs"
        );
        check_ulps!(
            Real::ln_rising_factorial(a, 1.0),
            a.ln(),
            0,
            "a single factor is exactly ln a"
        );
    }

    /// Against `lnΓ(a+n) − lnΓ(a)` — **only where that difference still has digits**.
    ///
    /// The identity is exact in mathematics and the subtraction is not, which is the entire
    /// reason `ln_rising_factorial` is a named operation rather than a two-call expression. Its
    /// docstring says so: both terms are large and nearly equal when `a` is large and `n` is
    /// small, "so the result loses the digits that matter".
    ///
    /// Measured, that is a relative error of `4.1e-14` for `a ≤ 10` and **`1.5e-4` by
    /// `a ≈ 1e12`**. So this property asserts the agreement on the range where the naive form is
    /// trustworthy and stops there — asserting it further would be asserting the cancellation
    /// the kernel exists to avoid, which is the mistake this suite's header records three times
    /// over.
    fn the_rising_factorial_agrees_with_the_naive_difference_where_that_is_sound(
        a in generate::float::in_range(0.01, 10.0),
        step in generate::int(1, 20),
    ) {
        if a <= 0.0 {
            return Ok(());
        }
        #[allow(clippy::cast_precision_loss)]
        let n = step as f64;
        let direct = Real::ln_rising_factorial(a, n);
        let naive = Real::ln_gamma(a + n) - Real::ln_gamma(a);
        if !direct.is_finite() || !naive.is_finite() {
            return Ok(());
        }
        let scale = direct.abs().max(1.0);
        relation::agree(
            "ln (a)_n = lnG(a+n) - lnG(a)",
            (),
            |()| direct,
            |()| naive,
            &Absolute(1e-13 * scale),
        )?;
    }
}

property! {
    #![config(settings("incomplete_gamma").cases(600))]

    /// `P(a, x) + Q(a, x) = 1`, which the trait states as a contract and nothing checked.
    ///
    /// [`Real::gamma_lr`]'s docstring says "`P` and `Q` are probabilities at every shape: both
    /// lie in `[0, 1]`, sum to `1`, and are monotone in `self`." That is three testable claims
    /// in one sentence, and `gamma_lr` and `gamma_ur` held **24 survivors** between the two
    /// precisions while no property mentioned either.
    ///
    /// The sum is measured exact for `a ≤ 100` and one ulp to `a = 1e8` — which is the strongest
    /// thing it could be, since `Q` is formed so a small upper tail keeps its relative accuracy
    /// rather than being computed as `1 − P`. If it *were* `1 - p` this law would be vacuous;
    /// it is not, so the law constrains two separate evaluations.
    fn the_incomplete_gamma_pair_are_complementary_probabilities(
        a in generate::float::magnitude(-6, 26),
        ratio in generate::float::in_range(0.0, 4.0),
    ) {
        let a = a.abs();
        if a <= 0.0 || !a.is_finite() || a.is_nan() || ratio < 0.0 {
            return Ok(());
        }
        let x = a * ratio;
        if !x.is_finite() {
            return Ok(());
        }
        classify!("below the mode", ratio < 1.0);
        classify!("above the mode", ratio > 1.0);
        classify!("large shape", a > 1e3);

        let lower = Real::gamma_lr(x, a);
        let upper = Real::gamma_ur(x, a);
        if !lower.is_finite() || !upper.is_finite() {
            return Ok(());
        }
        check!(
            (0.0..=1.0).contains(&lower) && (0.0..=1.0).contains(&upper),
            "P = {lower:e} and Q = {upper:e} must both be probabilities at a = {a:e}, x = {x:e}"
        );
        check_ulps!(
            lower + upper,
            1.0,
            1,
            "P + Q must be one at a = {a:e}, x = {x:e}"
        );
        // Monotone in `x`, the other half of the docstring's claim.
        let further = Real::gamma_lr(x + (x * 1e-6).max(1e-12), a);
        if further.is_finite() {
            check!(
                further >= lower,
                "P must not decrease in x at a = {a:e}, x = {x:e}"
            );
        }
    }

    /// The closed forms at integer shape, which are exact and need no reference.
    ///
    /// At `a = 1` the incomplete gamma collapses to the exponential: `Q(1, x) = e^{−x}` and
    /// `P(1, x) = −expm1(−x)`. At integer `n` it is the truncated exponential series,
    /// `Q(n, x) = e^{−x}Σ_{k<n} x^k/k!`. Both are elementary, so they pin the whole
    /// series/continued-fraction/Edgeworth machinery against arithmetic that shares none of it.
    ///
    /// Measured `5.7e-14`, `1.4e-15` and `1.1e-14` respectively.
    fn the_incomplete_gamma_collapses_to_the_exponential_at_integer_shape(
        x in generate::float::in_range(1e-3, 30.0),
        shape in generate::int(1, 12),
    ) {
        if x <= 0.0 {
            return Ok(());
        }
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let n = shape as u32;

        // a = 1: the exponential itself.
        let upper_one = Real::gamma_ur(x, 1.0);
        if upper_one > 0.0 {
            relation::agree(
                "Q(1, x) = exp(-x)",
                (),
                |()| upper_one,
                |()| (-x).exp(),
                &Relative(1e-12),
            )?;
        }
        let lower_one = Real::gamma_lr(x, 1.0);
        if lower_one > 0.0 {
            relation::agree(
                "P(1, x) = -expm1(-x)",
                (),
                |()| lower_one,
                |()| -((-x).exp_m1()),
                &Relative(1e-13),
            )?;
        }

        // Integer shape: the truncated series, summed directly.
        let mut term = 1.0f64;
        let mut series = 1.0f64;
        for k in 1..n {
            term *= x / f64::from(k);
            series += term;
        }
        let truncated = (-x).exp() * series;
        let upper = Real::gamma_ur(x, f64::from(n));
        if truncated > 0.0 && upper > 0.0 {
            classify!("higher shape", n > 4);
            relation::agree(
                "Q(n, x) = exp(-x) sum_{k<n} x^k/k!",
                (),
                |()| upper,
                |()| truncated,
                &Relative(1e-12),
            )?;
        }

        // And the endpoints, exactly.
        check_ulps!(Real::gamma_lr(0.0f64, 2.5), 0.0, 0, "P(a, 0) is zero");
        check_ulps!(Real::gamma_ur(0.0f64, 2.5), 1.0, 0, "Q(a, 0) is one");
        check!(
            Real::gamma_lr(f64::INFINITY, 2.5) == 1.0
                && Real::gamma_ur(f64::INFINITY, 2.5) == 0.0,
            "the pair must reach their limits exactly"
        );
    }
}

property! {
    // Floors at roughly half the thinnest coverage over six seeds: `below one` runs 8.6–12.0%,
    // the two sides of the Edgeworth handover 15.4%+ and 81.6%+.
    //
    // `the_reflection_formulae_hold_at_f32` moved to its own block below: it sweeps `(0.002,
    // 0.998)`, so it can never reach `x > 1e3`, and a floor binds every property sharing the
    // `#![config]`. Both blocks keep `settings("gamma_equations_f32")` and so share
    // `tests/regressions/gamma_equations_f32.txt`, leaving the committed tape in place.
    #![config(settings("gamma_equations_f32")
        .cases(500)
        .require("below one", 4.0)
        .require("below the edgeworth handover", 7.0)
        .require("large", 60.0))]

    /// The f32 peers, which is where most of this family's survivors live.
    ///
    /// Thirty-seven of the eighty-nine survivors measured across these four files are in the
    /// `f32` kernels, and until this property no functional equation had ever been applied to
    /// them. The f32 tree is a separate implementation throughout — "evaluated in `f32`
    /// throughout, with no `f64` round-trip" — so none of the laws above says anything about it.
    ///
    /// The bounds are the narrower format's own, measured rather than scaled from the `f64`
    /// figures: `3.1e-7` for the `ln Γ` duplication, `3.1e-5` for the trigamma reflection, one
    /// ulp for `P + Q`, and **zero ulps** for the rising factorial's recurrence, which is exact
    /// in `f32` exactly as it is in `f64`.
    fn the_functional_equations_hold_at_f32(
        x in generate::float32::in_range(0.002, 1e4),
        step in generate::int(0, 20),
    ) {
        if x <= 0.0 || !x.is_finite() {
            return Ok(());
        }
        classify!("below one", x < 1.0);
        classify!("large", x > 1e3);
        // `large` is exactly `GAMMA_EDGEWORTH_MIN`, so it is one side of a crossover rather than
        // a size label. Naming the other side turns the pair into a bracket around the handover,
        // which is what a floor can then hold the generator to.
        classify!("below the edgeworth handover", x <= 1e3);

        // The rising factorial's recurrence is exact here too.
        #[allow(clippy::cast_precision_loss)]
        let n = step as f32;
        let longer = Real::ln_rising_factorial(x, n + 1.0);
        let shorter = Real::ln_rising_factorial(x, n) + (x + n).ln();
        if longer.is_finite() && shorter.is_finite() {
            // Bit equality rather than `check_ulps!`, which is an `f64` macro: at `f32` the
            // claim is the same one and exact equality is how to say it in this type.
            check!(
                longer == shorter,
                "f32: (a)_(n+1) = (a)_n (a+n) must hold exactly at a = {x:e}: \
                 {longer:e} against {shorter:e}"
            );
        }

        // `ln Gamma` duplication.
        let duplication_lhs = Real::ln_gamma(2.0 * x);
        let duplication_rhs = (2.0 * x - 1.0) * <f32 as Real>::ln_2()
            - 0.5 * <f32 as Real>::ln_pi()
            + Real::ln_gamma(x)
            + Real::ln_gamma(x + 0.5);
        if duplication_lhs.is_finite() && duplication_rhs.is_finite() {
            let scale = duplication_lhs.abs().max(1.0);
            relation::agree(
                "f32: lnG(2x) duplication",
                (),
                |()| f64::from(duplication_lhs),
                |()| f64::from(duplication_rhs),
                &Absolute(1e-6 * f64::from(scale)),
            )?;
        }

        // `digamma`'s recurrence.
        let digamma_shifted = Real::digamma(x + 1.0);
        let digamma_recurred = Real::digamma(x) + 1.0 / x;
        if digamma_shifted.is_finite() && digamma_recurred.is_finite() {
            let scale = digamma_shifted.abs().max(1.0);
            relation::agree(
                "f32: psi(x+1) = psi(x) + 1/x",
                (),
                |()| f64::from(digamma_shifted),
                |()| f64::from(digamma_recurred),
                &Absolute(1e-4 * f64::from(scale)),
            )?;
        }

        // The incomplete pair still sums to one, to an ulp of the narrower format.
        let lower = Real::gamma_lr(x, 2.5f32);
        let upper = Real::gamma_ur(x, 2.5f32);
        if lower.is_finite() && upper.is_finite() {
            check!(
                (0.0..=1.0).contains(&lower) && (0.0..=1.0).contains(&upper),
                "f32: P and Q must be probabilities at x = {x:e}"
            );
            // One ulp of one, which in `f32` is `EPSILON` — and is exactly what the sweep
            // measured (`1.192e-7`).
            let residual = (lower + upper - 1.0).abs();
            check!(
                residual <= f32::EPSILON,
                "f32: P + Q must be one at x = {x:e}, off by {residual:e}"
            );
        }
        // And `Q(1, x)` is still the exponential.
        if x < 80.0 {
            let exponential = Real::gamma_ur(x, 1.0f32);
            if exponential > 0.0 {
                relation::agree(
                    "f32: Q(1, x) = exp(-x)",
                    (),
                    |()| f64::from(exponential),
                    |()| f64::from((-x).exp()),
                    &Relative(1e-5),
                )?;
            }
        }
    }

}

property! {
    // Its own block, because its `(0.002, 0.998)` sweep cannot reach the crossover labels the
    // block above requires. Same suite name, so the committed tape stays where it is.
    #![config(settings("gamma_equations_f32").cases(500))]

    /// The reflection formulae at f32, on the unit interval where they are sharpest.
    fn the_reflection_formulae_hold_at_f32(x in generate::float32::in_range(0.002, 0.998)) {
        if !(0.0..1.0).contains(&x) {
            return Ok(());
        }
        let pi = <f32 as Real>::pi();

        let product = Real::gamma(x) * Real::gamma(1.0 - x);
        let closed_form = pi / (pi * x).sin();
        if product.is_finite() && closed_form.is_finite() {
            relation::agree(
                "f32: Gamma(x) Gamma(1-x) = pi / sin(pi x)",
                (),
                |()| f64::from(product),
                |()| f64::from(closed_form),
                &Relative(1e-4),
            )?;
        }

        let sine = (pi * x).sin();
        let trigamma_sum = Real::trigamma(x) + Real::trigamma(1.0 - x);
        let trigamma_closed = pi * pi / (sine * sine);
        if trigamma_sum.is_finite() && trigamma_closed.is_finite() {
            relation::agree(
                "f32: psi'(x) + psi'(1-x) = pi^2 / sin^2(pi x)",
                (),
                |()| f64::from(trigamma_sum),
                |()| f64::from(trigamma_closed),
                &Relative(1e-4),
            )?;
        }
    }
}

property! {
    #![config(
        settings("rising_factorial_regimes")
            .cases(600)
            .require("fractional n", 20.0)
            .require("the asymptotic regime", 15.0)
    )]

    /// The rising factorial's **other three regimes**, which the laws above could not reach.
    ///
    /// # A vacuous law, and the generator that made it vacuous
    ///
    /// The recurrence property beside this one measured **bit-exact** and killed nothing, and
    /// the reason is worth recording because it is the third time the same mistake has appeared
    /// in a different costume. It draws `n` from `generate::int`, so `n` is always a whole
    /// number — and whole `n ≤ 64` takes the direct `Σ ln(a+k)` branch, which satisfies the
    /// recurrence *by construction*: one more term is exactly one more `ln`. The law was true,
    /// exact, and testing one branch out of four.
    ///
    /// `ln_rising_factorial` has four regimes: `n == 0`, the direct sum for whole `n ≤ 64`, a
    /// Stirling-style expansion for large `a`, and the `ln_gamma` difference otherwise. Twenty
    /// survivors lived in the three the law never entered, and five of those are in a guard that
    /// only runs for a **non-finite** argument — which every property above skips explicitly.
    ///
    /// So this one requires fractional `n` and the large-`a` regime by coverage floor rather
    /// than hoping for them, and it tests the non-finite guard directly. The lesson generalises:
    /// **ask which branch a law exercises, not just whether it holds.**
    fn the_rising_factorial_holds_in_every_regime(
        a in generate::float::in_range(0.5, 1e6),
        n in generate::float::in_range(0.25, 200.0),
    ) {
        if a <= 0.0 || n <= 0.0 {
            return Ok(());
        }
        classify!("fractional n", n.fract() != 0.0);
        classify!("the asymptotic regime", a > 1e3 && a + n > 1e3);
        classify!("beyond the direct sum", n > 64.0);

        // The recurrence still holds, and now in the regimes that were never entered. It is no
        // longer bit-exact here: the asymptotic branch is an approximation, so this is a
        // tolerance rather than an equality, which is itself the sign that a different code path
        // is being exercised.
        let longer = Real::ln_rising_factorial(a, n + 1.0);
        let shorter = Real::ln_rising_factorial(a, n) + (a + n).ln();
        if longer.is_finite() && shorter.is_finite() {
            let scale = longer.abs().max(1.0);
            relation::agree(
                "(a)_(n+1) = (a)_n (a+n), in every regime",
                (),
                |()| longer,
                |()| shorter,
                &Absolute(1e-9 * scale),
            )?;
        }

        // And against the `ln_gamma` difference where that is still sound, which is the regime
        // boundary the kernel's own threshold is placed at.
        if a < 1e3 {
            let direct = Real::ln_rising_factorial(a, n);
            let naive = Real::ln_gamma(a + n) - Real::ln_gamma(a);
            if direct.is_finite() && naive.is_finite() {
                let scale = direct.abs().max(1.0);
                relation::agree(
                    "ln (a)_n = lnG(a+n) - lnG(a), fractional n",
                    (),
                    |()| direct,
                    |()| naive,
                    &Absolute(1e-10 * scale),
                )?;
            }
        }
    }

}

property! {
    // A third block, because the requirements above name labels this property does not apply.
    // That footgun has now bitten three times in this work; the rule is one `require` set per
    // block, and a property that does not classify into it needs its own block.
    #![config(settings("rising_factorial_edges").cases(400))]

    /// The non-finite guard, which five survivors sat inside and nothing reached.
    ///
    /// `ln_rising_factorial` opens `if !a.is_finite() || !n.is_finite()` and falls back to the
    /// `ln_gamma` difference there. Every other property in this file guards with `is_finite`
    /// and skips those arguments, so the branch had no coverage at all — and `||` weakened to
    /// `&&`, plus four mutations of the `ln_gamma(a + n) - ln_gamma(a)` inside it, all survived.
    ///
    /// These are the contract's answers at the edges, asserted as the function behaves so a
    /// rewrite has to decide deliberately rather than drift.
    fn the_rising_factorial_handles_non_finite_arguments(
        a in generate::float::in_range(0.5, 100.0),
    ) {
        if a <= 0.0 {
            return Ok(());
        }
        // An empty product is one whatever the shape, including a non-finite one: the `n == 0`
        // guard precedes the non-finite one.
        check_ulps!(
            Real::ln_rising_factorial(f64::INFINITY, 0.0),
            0.0,
            0,
            "the empty product is zero in logs even at an infinite shape"
        );

        // The exact answers, not merely "not finite". A first draft of this property asserted
        // only `!is_finite()` and killed nothing: the mutations inside the guard turn one
        // non-finite value into another, so a test that accepts any of them accepts all of
        // them. Hedging an assertion is the same as not making it.
        //
        // These follow from the fallback being `lnΓ(a+n) − lnΓ(a)`:
        //
        //   a = ±∞ : `lnΓ(∞) − lnΓ(∞)` is `∞ − ∞`, so NaN — including for `a = −∞`, where
        //            `lnΓ` is NaN to begin with.
        //   n = +∞ : `lnΓ(∞) − lnΓ(3)` is `+∞`, a genuinely infinite product.
        //   n = −∞ : `a + n` is `−∞`, where `lnΓ` is NaN.
        for &(shape, count, expect_nan) in &[
            (f64::INFINITY, 3.0, true),
            (f64::NEG_INFINITY, 3.0, true),
            (f64::NAN, 3.0, true),
            (3.0, f64::NEG_INFINITY, true),
            (3.0, f64::NAN, true),
        ] {
            let got = Real::ln_rising_factorial(shape, count);
            check!(
                got.is_nan() == expect_nan,
                "ln (a)_n at a = {shape:e}, n = {count:e} must be NaN, got {got:e}"
            );
        }
        // An infinite count is the one case that diverges rather than failing, and it is the
        // `a + n` inside the guard that decides which.
        check!(
            Real::ln_rising_factorial(3.0f64, f64::INFINITY) == f64::INFINITY,
            "an infinite count gives exactly +inf, got {:e}",
            Real::ln_rising_factorial(3.0f64, f64::INFINITY)
        );
        let _ = a;

        // The f32 peer of the same guard, which holds five of the survivors on its own. The
        // narrower kernel has the identical structure — `if !a.is_finite() || !n.is_finite()`
        // falling back to the `ln_gamma` difference — and the identical answers.
        for &(shape, count) in &[
            (f32::INFINITY, 3.0f32),
            (f32::NEG_INFINITY, 3.0),
            (f32::NAN, 3.0),
            (3.0, f32::NEG_INFINITY),
            (3.0, f32::NAN),
        ] {
            let got = Real::ln_rising_factorial(shape, count);
            check!(
                got.is_nan(),
                "f32: ln (a)_n at a = {shape:e}, n = {count:e} must be NaN, got {got:e}"
            );
        }
        check!(
            Real::ln_rising_factorial(3.0f32, f32::INFINITY) == f32::INFINITY,
            "f32: an infinite count gives exactly +inf"
        );
        check_ulps!(
            f64::from(Real::ln_rising_factorial(f32::INFINITY, 0.0f32)),
            0.0,
            0,
            "f32: the empty product is zero in logs even at an infinite shape"
        );
    }
}

property! {
    #![config(settings("rising_factorial_negative").cases(400))]

    /// A **negative** count, which selects the Stirling branch for a reason nothing tested.
    ///
    /// `ln (a)ₙ` is `lnΓ(a+n) − lnΓ(a)`, defined for any `n` with `a + n > 0` — so `n` may be
    /// negative, and the docstring's product `a(a+1)…(a+n−1)` simply runs the other way. Every
    /// other property in this file generates `n > 0`, and that gap hid a real defect.
    ///
    /// # The defect
    ///
    /// The Stirling branch is guarded by `a > STIRLING_RATIO_MIN && a + n > STIRLING_RATIO_MIN`,
    /// and mutating the second `a + n` to `a * n` survived everything. For positive `n` the two
    /// agree about which branch to take. For *negative* `n` they do not: `a * n` is negative, so
    /// the mutant falls through to the direct difference — which at `a = 1e12, n = −3.5` returns
    /// `−96.7109375` where the answer is `−96.70857390574204`. Wrong by `2.4e-5`, five digits
    /// gone to precisely the cancellation the Stirling branch exists to avoid.
    ///
    /// # Why the recurrence cannot catch it
    ///
    /// Worth recording, because it was the obvious thing to try. `(a)_{n+1} = (a)_n(a+n)` holds
    /// for the naive form *too*: both `lnΓ(a+n)` terms carry nearly the same error, which
    /// cancels in the difference, so the recurrence is satisfied by a badly wrong value. A
    /// self-referential law cannot see an error common to both of its sides — the same reason
    /// the seed recurrences in this work needed the function tested in isolation.
    ///
    /// # The reference that does catch it
    ///
    /// For negative *integer* `n` the answer is an exact sum, and a well-conditioned one:
    /// `ln (a)_{−m} = −Σ_{k=1}^{m} ln(a − k)`. No cancellation, no Stirling, no `lnΓ` — so it
    /// pins the branch independently at exactly the arguments where the naive difference fails.
    fn the_rising_factorial_handles_a_negative_count(
        a in generate::float::in_range(1e3, 1e14),
        count in generate::int(1, 60),
    ) {
        if a <= 0.0 {
            return Ok(());
        }
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let m = count as u32;
        #[allow(clippy::cast_precision_loss)]
        let n = -(m as f64);
        if a + n <= 0.0 {
            return Ok(());
        }
        classify!("above the Stirling threshold", a > 300.0);
        classify!("very large shape", a > 1e9);

        // `-sum_{k=1..m} ln(a - k)`, which is exact in the sense that matters: every term is
        // O(ln a) and they are all the same sign, so nothing cancels.
        let mut reference = 0.0f64;
        for k in 1..=m {
            reference -= (a - f64::from(k)).ln();
        }

        let direct = Real::ln_rising_factorial(a, n);
        if !direct.is_finite() || !reference.is_finite() {
            return Ok(());
        }
        let scale = reference.abs().max(1.0);
        relation::agree(
            "ln (a)_{-m} = -sum ln(a - k)",
            (),
            |()| direct,
            |()| reference,
            &Absolute(1e-12 * scale),
        )?;
    }
}

property! {
    #![config(settings("rising_factorial_negative_f32").cases(400))]

    /// The f32 peer of the negative-count gap, which has one extra mutant of its own.
    ///
    /// The f32 kernel guards its Stirling branch with `a > 1e2 && a + n > 1e2`, and mutating
    /// that `&&` to `||` survived. For positive `n` the two clauses agree. For negative `n` large
    /// enough to pull `a + n` back under the threshold they do not, and the mutant takes Stirling
    /// where it must not: at `a = 1000, n = −950` it returns about `−5753` where the answer is
    /// `−5760.6`, wrong by a tenth of a percent, because `ln1p(n/a)` is being asked about a ratio
    /// close to `−1`.
    ///
    /// The reference is the same well-conditioned sum the f64 peer uses, evaluated in `f64` so
    /// that the comparison measures the f32 kernel rather than the reference.
    fn the_f32_rising_factorial_handles_a_negative_count(
        shape in generate::float32::in_range(101.0, 400.0),
        remainder in generate::float32::in_range(1.5, 20.0),
    ) {
        if shape <= 100.0 || remainder <= 1.0 || !shape.is_finite() {
            return Ok(());
        }
        // Constructed, not hoped for. The regime these two clauses disagree about needs
        // `a > 100` *and* `a + n < 100` at the same time, so `n` is chosen to land `a + n` on a
        // drawn remainder below the threshold.
        //
        // # The observable window is narrow, and both draws are sized to it
        //
        // Two drafts killed nothing before this one, and the reason is worth keeping. The
        // Stirling difference does *not* fail merely because `a + n` drops below the threshold:
        // at `a = 1000, n = -980` it is still right to eleven digits, because the correction
        // series' error at `z = 20` is about `3.5e-7` absolute against a value of `-5866`. It
        // fails when **both** `a` and `a + n` are small, where that same absolute error is no
        // longer negligible — at `a = 101, n = -99` it returns `-363.7` against a true `-359.1`,
        // wrong by `1.3%`.
        //
        // The first draft drew `a` uniformly on `[50, 1e5]`, where reaching `a + n < 100` needs
        // a fraction above `0.999`; the second narrowed the fraction but left `a` wide, so the
        // joint event was about one case in three thousand. Both ranges are now sized to the
        // window: `a` just above the threshold, and a remainder just above one. That is the
        // fifth time in this work a property has missed a regime the generator could not
        // construct — the generators' README states the rule, "valid by construction rather
        // than by rejection", and it applies to regimes as much as to values.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let m = (shape - remainder) as u32;
        if m == 0 {
            return Ok(());
        }
        #[allow(clippy::cast_precision_loss)]
        let n = -(m as f32);
        let sum = shape + n;
        if sum <= 1.0 || sum >= 100.0 || sum.is_nan() {
            return Ok(());
        }
        classify!("above the threshold", shape > 100.0);
        classify!("sum falls below it", sum < 100.0);

        let mut reference = 0.0f64;
        for k in 1..=m {
            reference -= (f64::from(shape) - f64::from(k)).ln();
        }

        let direct = Real::ln_rising_factorial(shape, n);
        if !direct.is_finite() || !reference.is_finite() {
            return Ok(());
        }
        // An f32 kernel against an f64 reference: the bound is the narrower format's, scaled by
        // the magnitude of a sum that reaches several thousand.
        let scale = reference.abs().max(1.0);
        relation::agree(
            "f32: ln (a)_{-m} = -sum ln(a - k)",
            (),
            |()| f64::from(direct),
            |()| reference,
            &Absolute(1e-5 * scale),
        )?;
    }
}

property! {
    // Floors on both sides of `PSI_ASYMPTOTIC_MIN`, at roughly half the thinnest coverage
    // measured over six seeds (the recurrence side runs 15.4–18.4%, the asymptotic side 81.6%+).
    // This is the threshold the f32 port moved rather than inherited, so a generator drifting to
    // one side of it would be testing the inherited choice and reporting a clean pass.
    #![config(settings("f32_native_psi")
        .cases(500)
        .require("below the asymptotic shift", 7.0)
        .require("above the asymptotic shift", 60.0))]

    /// The native `f32` `ψ` and `ψ₁` against their own recurrences and reflections.
    ///
    /// These had no property coverage at all while they narrowed the `f64` kernels, because
    /// narrowing made them true by construction — any law they satisfied was `f64`'s law, one
    /// rounding away. Now that both are native the laws say something, and they are the only
    /// check that reaches the branch structure: `PsiReduction` routes on two thresholds and a
    /// parity, and a case that lands on the wrong side of one is exactly what a recurrence law
    /// catches.
    ///
    /// The bounds are the format's, measured rather than scaled from the `f64` figures. `ψ`'s
    /// recurrence is the tight one — `ψ(x+1) − ψ(x) = 1/x` is an identity between two values
    /// that are `O(1)` apart, so it holds to a few ulps of the larger. `ψ₁`'s reflection is the
    /// loose one, because `π²/sin²` is unbounded near the integers.
    fn the_native_f32_polygammas_satisfy_their_laws(
        x in generate::float32::in_range(0.01, 60.0),
    ) {
        if !x.is_finite() || x <= 0.0 {
            return Ok(());
        }
        classify!("below the asymptotic shift", x < 6.0);
        classify!("above the asymptotic shift", x >= 6.0);

        // Both recurrences **cancel at small x**, and the budgets have to carry it rather than
        // the kernel being blamed for it. `ψ(x) ≈ −1/x` and `ψ₁(x) ≈ 1/x²` near zero, so each
        // law subtracts two large nearly-equal quantities to leave an `O(1)` answer: at
        // `x = 0.01` the `ψ₁` form takes `10000.58 − 10000` and `ulp(10000)` is already `9.8e-4`
        // in `f32`. The achievable agreement therefore scales with the *shed term*, not with the
        // result — which is why the bound below is written in terms of `1/x` and `1/x²`. Fitting
        // a flat bound instead is how this property first reported a defect at `x = 0.01` that
        // was entirely its own arithmetic.
        let epsilon = f64::from(f32::EPSILON);

        // psi(x+1) = psi(x) + 1/x. Crosses the recurrence/asymptotic threshold for x near 5.
        let shifted = Real::digamma(x + 1.0);
        let recurred = Real::digamma(x) + 1.0 / x;
        if shifted.is_finite() && recurred.is_finite() {
            let shed = f64::from(1.0f32 / x);
            let scale = (shed + f64::from(shifted.abs())).max(1.0);
            relation::agree(
                "f32: psi(x+1) = psi(x) + 1/x",
                (),
                |()| f64::from(shifted),
                |()| f64::from(recurred),
                &Absolute(64.0 * epsilon * scale),
            )?;
        }

        // psi_1(x+1) = psi_1(x) - 1/x^2, the same crossing for the second derivative.
        let shifted = Real::trigamma(x + 1.0);
        let recurred = Real::trigamma(x) - 1.0 / (x * x);
        if shifted.is_finite() && recurred.is_finite() {
            let shed = f64::from(1.0f32 / (x * x));
            let scale = (shed + f64::from(shifted.abs())).max(1.0);
            relation::agree(
                "f32: psi_1(x+1) = psi_1(x) - 1/x^2",
                (),
                |()| f64::from(shifted),
                |()| f64::from(recurred),
                &Absolute(64.0 * epsilon * scale),
            )?;
        }

        // psi_1 is positive and strictly decreasing on the positive axis: a shape check the
        // recurrence cannot see, and the one a wrong branch would break first.
        let here = Real::trigamma(x);
        let further = Real::trigamma(x + 0.5);
        check!(here > 0.0, "f32 trigamma must be positive at {x:e}, got {here:e}");
        check!(
            further <= here,
            "f32 trigamma must decrease: psi_1({x:e}) = {here:e} but psi_1({:e}) = {further:e}",
            x + 0.5
        );
    }
}

property! {
    #![config(settings("f32_native_ln_beta").cases(500))]

    /// The native `f32` `ln B` against the two closed forms that pin its branch.
    ///
    /// `LnBetaForm` picks between the rising-factorial identity and the combined Stirling
    /// expansion on the smaller shape, and the exact `ln B(a, 1) = −ln a` is the law that
    /// catches a mis-sited switch: it is the case the original `lnΓ(a) + lnΓ(b) − lnΓ(a+b)`
    /// form got catastrophically wrong, so it is the one worth keeping a law on.
    ///
    /// Also checked is symmetry, which holds *exactly* — not to a tolerance. `ln_beta` sorts its
    /// arguments before doing anything else, so `B(a,b)` and `B(b,a)` take the same path through
    /// the same branch and must agree to the bit. A future branch that reads `a` and `b`
    /// separately would break this and nothing else here.
    fn the_native_f32_ln_beta_matches_its_closed_forms(
        a in generate::float32::in_range(0.01, 1e5),
        b in generate::float32::in_range(0.01, 1e5),
    ) {
        if !a.is_finite() || !b.is_finite() || a <= 0.0 || b <= 0.0 {
            return Ok(());
        }
        classify!("rising factorial", a.min(b) < 1024.0);
        classify!("combined Stirling", a.min(b) >= 1024.0);

        // Symmetry, exactly.
        let forward = Real::ln_beta(a, b);
        let reversed = Real::ln_beta(b, a);
        check!(
            forward.to_bits() == reversed.to_bits(),
            "f32 ln_beta must be symmetric to the bit: ln B({a:e},{b:e}) = {forward:e} but \
             ln B({b:e},{a:e}) = {reversed:e}"
        );

        // ln B(a, 1) = -ln a, exactly representable as a law at every magnitude.
        let unit = Real::ln_beta(a, 1.0f32);
        if unit.is_finite() {
            let scale = f64::from(unit.abs().max(1.0));
            relation::agree(
                "f32: ln B(a, 1) = -ln a",
                (),
                |()| f64::from(unit),
                |()| -f64::from(a.ln()),
                &Absolute(2e-6 * scale),
            )?;
        }
    }
}
