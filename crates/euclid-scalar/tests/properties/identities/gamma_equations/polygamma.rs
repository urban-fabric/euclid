//! Arbitrary-order `polygamma`, and the reciprocal gamma's entirety.
//!
//! Part of the [`gamma_equations`](super) suite. Separated from the reflection laws because the
//! instrument differs: the recurrence `ψ⁽ⁿ⁾(x+1) = ψ⁽ⁿ⁾(x) + (−1)ⁿ n!/x^{n+1}` holds at every
//! order, so it reaches the orders above one that the fixed suite never covered — and
//! `polygamma.rs` plus its `f32` peer held 24 surviving mutants between them before it did.

use super::*;

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
