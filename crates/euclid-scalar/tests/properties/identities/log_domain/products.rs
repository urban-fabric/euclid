//! Log-domain products, and the absorbing zero.
//!
//! Part of the [`log_domain`](super) suite. The absorbing zero is the case an ordinary product
//! gets wrong: `0 · ∞` is a `NaN` in the arithmetic and a zero in the measure, and these laws
//! pin which one this surface promises.

use super::*;

property! {
    #![config(settings("absorbing_zero").cases(400))]

    /// `xlogy(0, y)` is `0` for **every** `y`, including `±∞` and `NaN`.
    ///
    /// The absorbing zero, and a genuine convention rather than a forced answer: `0 · ln 0` is
    /// an indeterminate form, and this surface resolves it to `0` because that is what makes an
    /// entropy or a log-likelihood sum work when a category has no observations. Propagating
    /// `NaN` would be equally defensible in the abstract and would break every one of those
    /// callers.
    ///
    /// Measured and asserted here because it is the kind of choice a refactor silently reverses:
    /// a rewrite that evaluates `x * y.ln()` first and patches afterwards gets `±∞` and `NaN`
    /// wrong while passing every finite test. The second argument is drawn by
    /// `pair_with_edge_cases`, so the non-finite cases arrive by construction rather than by
    /// being listed.
    fn the_absorbing_zero_wins_over_every_second_argument(
        pair in generate::float::pair_with_edge_cases(-40, 40),
    ) {
        let (_, y) = pair;
        classify!("nan", y.is_nan());
        classify!("infinite", y.is_infinite());
        classify!("zero", y == 0.0);

        for (name, got) in [
            ("xlogy", Real::xlogy(0.0, y)),
            ("xlog1py", Real::xlog1py(0.0, y)),
            ("xlog1my", Real::xlog1my(0.0, y)),
        ] {
            check!(
                got == 0.0,
                "{name}(0, {y:e}) must be 0, got {got:e}"
            );
        }
    }

    /// Away from the absorbing zero, each product is its plain definition.
    ///
    /// Which is the other half of the contract: the special case must be a special case, not a
    /// different function. `xlogy(x, y) = x·ln y`, `xlog1py(x, y) = x·ln1p(y)` and
    /// `xlog1my(x, y) = x·ln1m(y)`, each on the domain where its log exists.
    fn each_product_is_its_definition_away_from_zero(
        x in generate::float::in_range(-1e6, 1e6),
        y in generate::float::in_range(-0.9, 0.9),
    ) {
        if x == 0.0 {
            return Ok(());
        }
        classify!("negative multiplier", x < 0.0);

        // `y + 1` and `1 - y` are both comfortably positive on this range, so all three logs
        // exist and none of them is near its singularity.
        if y > 0.0 {
            relation::agree(
                "xlogy = x ln y",
                (),
                |()| Real::xlogy(x, y),
                |()| x * y.ln(),
                &Relative(1e-14),
            )?;
        }
        relation::agree(
            "xlog1py = x ln1p(y)",
            (),
            |()| Real::xlog1py(x, y),
            |()| x * Real::ln_1p(&y),
            &Relative(1e-14),
        )?;
        relation::agree(
            "xlog1my = x ln1m(y)",
            (),
            |()| Real::xlog1my(x, y),
            |()| x * Real::ln_1m(&y),
            &Relative(1e-14),
        )?;
    }
}
