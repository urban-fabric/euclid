//! The same functional equations at single precision.
//!
//! Part of the [`gamma_equations`](super) suite. Grouped by *format* rather than by law, because
//! what they share is the thing under test: the `f32` kernels are native rather than widened, so
//! each law has to hold in a format with 24 bits of mantissa and its own crossovers. The budgets
//! here are this format's, not the `f64` ones loosened.

use super::*;

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
