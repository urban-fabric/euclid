//! The inverse complementary error function, in both directions.
//!
//! Part of the [`normal`](super) suite. Split out because the round trip is only sound on one
//! side: `erfc` loses its digits above `y = 1`, so the identity is asserted in the direction
//! that keeps them and the other direction is checked by recovering the *input* instead.

use super::*;

property! {
    // Floors at roughly half the thinnest coverage over six seeds: the complement side of the
    // crossover runs 8.8–13.2%, the fraction side 86.8%+, the tail 75.2%+.
    //
    // This is a separate block from `erfc_recovers_what_its_inverse_was_given` below, which
    // applies none of these labels — a floor binds every property sharing the `#![config]`, so a
    // property that does not classify belongs in its own block. Both blocks keep
    // `settings("erfc_inv")`, so they share `tests/regressions/erfc_inv.txt` and the committed
    // tape that pins the `y = 4.957e-3` case is untouched: a regression file is per-suite rather
    // than per-property by design, and saving appends.
    #![config(settings("erfc_inv")
        .cases(500)
        .require("erfc complement form", 4.0)
        .require("erfc fraction form", 60.0)
        .require("tail", 50.0))]

    /// `erfc⁻¹(erfc(x)) = x` for `x ≥ 0`, and `erfc(erfc⁻¹(y)) = y` into the far tail.
    ///
    /// Restricted to `x ≥ 0` for the reason the module header gives: `erfc(x) → 2` on the left
    /// and the information in `2 − erfc(x)` falls below `ulp(2)`, so the inverse cannot recover
    /// it — measured `1.3e-6` at `x = −4.98`. On the right `erfc` decays to zero, which loses
    /// nothing in relative terms, and the round trip holds all the way out.
    ///
    /// The reverse trip carries the same `ε·x²` amplification as the quantile's, and for the
    /// same reason: `erfc_inv` is `ndtri` in disguise — `erfc⁻¹(y) = −Φ⁻¹(y/2)/√2` — which is
    /// also why it reaches `y = 1.6e-288`.
    fn erfc_inv_inverts_erfc_on_the_side_that_keeps_its_digits(
        x in generate::float::in_range(0.0, 25.0),
    ) {
        classify!("tail", x > 5.0);
        // Both sides of `ERFCX_CF_MIN`, the `erfc` form crossover — `1 − erf` below it and the
        // continued fraction above. Spelled as a literal because the constant is private to the
        // kernel, as the other crossover labels in these suites are. This is the threshold the
        // erf work moved from `2.0` to `27/16`, and the stranded band it closed was `[1.6875, 2)`
        // — which only a run landing on both sides of the join can speak to.
        classify!("erfc complement form", x < 1.6875);
        classify!("erfc fraction form", x >= 1.6875);

        let y = Real::erfc(x);
        if y > 0.0 && y < 2.0 {
            relation::agree(
                "erfc_inv(erfc(x)) = x",
                (),
                |()| Real::erfc_inv(y),
                |()| x,
                &Absolute(16.0 * f64::EPSILON * (1.0 + x * x)),
            )?;
        }

        // The endpoints are the contract's, and out-of-domain is NaN rather than a guess.
        check!(
            Real::erfc_inv(0.0f64) == f64::INFINITY
                && Real::erfc_inv(2.0f64) == f64::NEG_INFINITY,
            "erfc_inv must reach both infinities"
        );
        check!(
            Real::erfc_inv(2.5f64).is_nan()
                && Real::erfc_inv(-0.1f64).is_nan()
                && Real::erfc_inv(f64::NAN).is_nan(),
            "outside [0, 2] erfc_inv must be NaN"
        );
    }
}

property! {
    // Its own block, and its own floor: this property classifies `deep tail` and none of the
    // crossover labels above, so sharing a `#![config]` with them would require a label it never
    // applies. The suite name is deliberately the same, so the committed tape stays in place.
    #![config(settings("erfc_inv").cases(500).require("deep tail", 40.0))]

    /// The same inverse from the other side, swept in log space so it reaches the tail.
    fn erfc_recovers_what_its_inverse_was_given(y in generate::float::magnitude(-1000, -1)) {
        let y = y.abs();
        if !(y > 0.0 && y < 1.0) {
            return Ok(());
        }
        classify!("deep tail", y < 1e-100);

        let x = Real::erfc_inv(y);
        if !x.is_finite() {
            return Ok(());
        }
        relation::agree(
            "erfc(erfc_inv(y)) = y",
            (),
            |()| Real::erfc(x),
            |()| y,
            &Relative(16.0 * f64::EPSILON * (1.0 + x * x)),
        )?;
    }
}
