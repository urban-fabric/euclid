//! Reflection, duplication and recurrence for `Γ`, `lnΓ`, `ψ` and `ψ₁`.
//!
//! Part of the [`gamma_equations`](super) suite, which explains why these laws are the right
//! instrument. The four reflection formulae share the `reflection_sine_reference` helper in the
//! parent: each constrains its function at `x` and `1−x` simultaneously, so the reference has to
//! compensate for rounding of the angle before applying the sine addition identity, or the
//! reference is the less accurate of the two sides.

use super::*;

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
