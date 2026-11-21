//! The rising factorial `(x)_n`, across every regime and both argument signs.
//!
//! Part of the [`gamma_equations`](super) suite. The largest group here, and deliberately so:
//! `(x)_n` is formed as a ratio of gammas where that is safe and as a product where it is not,
//! so the law has to hold across a crossover *and* at the non-finite arguments and negative
//! counts where neither form applies.

use super::*;

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
