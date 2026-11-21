//! The expansion table, and every kernel measured against it. Part of [`super`].

use super::*;

/// The seventeen expansions, each with the window where it is a valid reference.
///
/// Windows are base-ten exponents. The lower end is where the expansion has converged — checked
/// per point, not asserted — and the upper is where the *kernel* leaves the format: `airy_*_e`
/// stops at `1e80` because `ζ = ⅔x^{3/2}` overflows past it, `trigamma` at `1e150` because `1/x`
/// underflows, and `zeta` at `1e1.7 ≈ 50` because `2^{-s}` falls under an ulp of `1` and the
/// kernel then correctly returns exactly `1`.
fn asymptotes() -> Vec<Asymptote> {
    let (i0, i1) = (hankel(0.0, true), hankel(1.0, true));
    let (k0, k1) = (hankel(0.0, false), hankel(1.0, false));
    let (i0r, i1r, i0l) = (i0.clone(), i1.clone(), i0.clone());
    let (ai, bi) = (airy(true), airy(false));
    let (aip, bip) = (airy_derivative(true), airy_derivative(false));
    let erfcx = erfcx_coefficients();
    let tau = core::f64::consts::TAU;
    let pi = core::f64::consts::PI;
    let two_thirds = 2.0 / 3.0;

    // `ψ`, `ψ'` and `lnΓ` share the Bernoulli tail shape, with most odd powers vanishing.
    // `ψ(x) ~ ln x − 1/(2x) − Σ B₂ₖ/(2k·x^{2k})`, carried to `B₁₂`. The tail length is not
    // cosmetic: at `x = 100` a list stopping at `B₈` leaves a remainder of `107` ulp, and the
    // convergence guard rejects the whole window rather than let it be read as kernel error.
    let digamma = padded(&[
        (1, -0.5),
        (2, -1.0 / 12.0),
        (4, 1.0 / 120.0),
        (6, -1.0 / 252.0),
        (8, 1.0 / 240.0),
        (10, -1.0 / 132.0),
        (12, 691.0 / 32_760.0),
    ]);
    // `ψ'(x) ~ 1/x + 1/(2x²) + Σ B₂ₖ/x^{2k+1}`.
    let trigamma = padded(&[
        (1, 1.0),
        (2, 0.5),
        (3, 1.0 / 6.0),
        (5, -1.0 / 30.0),
        (7, 1.0 / 42.0),
        (9, -1.0 / 30.0),
        (11, 5.0 / 66.0),
    ]);
    // `lnΓ(x) ~ (x−½)ln x − x + ½ln 2π + Σ B₂ₖ/(2k(2k−1)·x^{2k−1})`.
    let stirling = padded(&[
        (1, 1.0 / 12.0),
        (3, -1.0 / 360.0),
        (5, 1.0 / 1260.0),
        (7, -1.0 / 1680.0),
        (9, 1.0 / 1188.0),
        (11, -691.0 / 360_360.0),
    ]);
    // `asinh(x) = ln 2x + 1/(4x²) − 3/(32x⁴) + 5/(96x⁶)`, and `acosh` the same with *every*
    // correction negative. Those signs are how `acosh` first "failed", at `x = 2.6e3`.
    let asinh = padded(&[
        (2, 0.25),
        (4, -3.0 / 32.0),
        (6, 5.0 / 96.0),
        (8, -35.0 / 1024.0),
    ]);
    let acosh = padded(&[
        (2, -0.25),
        (4, -3.0 / 32.0),
        (6, -5.0 / 96.0),
        (8, -35.0 / 1024.0),
    ]);

    vec![
        // `erfcx ~ 1/(x√π)`. It reaches the regime where its own expansion is exact, which `erfc`
        // cannot: `erfc` underflows at `x ≈ 27`, long before `1/(2x²)` falls under an ulp.
        row(
            "erfcx",
            Real::erfcx,
            move |x| scaled(summed(&erfcx, x * x), 1.0 / (x * SQRT_PI)),
            (1.0, 150.0),
            4.0,
        ),
        row(
            "bessel_i0e",
            Real::bessel_i0e,
            move |x| scaled(summed(&i0, x), 1.0 / (tau * x).sqrt()),
            (3.0, 300.0),
            24.0,
        ),
        row(
            "bessel_i1e",
            Real::bessel_i1e,
            move |x| scaled(summed(&i1, x), 1.0 / (tau * x).sqrt()),
            (3.0, 300.0),
            24.0,
        ),
        row(
            "bessel_k0e",
            Real::bessel_k0e,
            move |x| scaled(summed(&k0, x), (pi / (2.0 * x)).sqrt()),
            (3.0, 300.0),
            8.0,
        ),
        row(
            "bessel_k1e",
            Real::bessel_k1e,
            move |x| scaled(summed(&k1, x), (pi / (2.0 * x)).sqrt()),
            (3.0, 300.0),
            8.0,
        ),
        // The two expansions' prefactors cancel, so the ratio needs no `√(2πx)` at all.
        row(
            "bessel_i1_over_i0",
            Real::bessel_i1_over_i0,
            move |x| ratio(summed(&i1r, x), summed(&i0r, x)),
            (3.0, 300.0),
            32.0,
        ),
        row(
            "ln_bessel_i0",
            Real::ln_bessel_i0,
            move |x| shifted(log_of(summed(&i0l, x)), x - 0.5 * (LN_TAU + x.ln())),
            (2.0, 300.0),
            4.0,
        ),
        row(
            "digamma",
            Real::digamma,
            move |x| shifted(summed(&digamma, x), x.ln()),
            (2.0, 300.0),
            4.0,
        ),
        row(
            "trigamma",
            Real::trigamma,
            move |x| summed(&trigamma, x),
            (2.0, 150.0),
            4.0,
        ),
        row(
            "ln_gamma",
            Real::ln_gamma,
            move |x| shifted(summed(&stirling, x), (x - 0.5) * x.ln() - x + 0.5 * LN_TAU),
            (2.0, 300.0),
            4.0,
        ),
        row(
            "airy_ai_e",
            Real::airy_ai_e,
            move |x| {
                let scale = 1.0 / (2.0 * SQRT_PI * x.powf(0.25));
                scaled(summed(&ai, two_thirds * x.powf(1.5)), scale)
            },
            (3.0, 80.0),
            8.0,
        ),
        row(
            "airy_bi_e",
            Real::airy_bi_e,
            move |x| {
                let scale = 1.0 / (SQRT_PI * x.powf(0.25));
                scaled(summed(&bi, two_thirds * x.powf(1.5)), scale)
            },
            (3.0, 80.0),
            4.0,
        ),
        row(
            "asinh",
            Real::asinh,
            move |x| shifted(summed(&asinh, x), x.ln() + core::f64::consts::LN_2),
            (2.0, 150.0),
            4.0,
        ),
        row(
            "acosh",
            Real::acosh,
            move |x| shifted(summed(&acosh, x), x.ln() + core::f64::consts::LN_2),
            (2.0, 150.0),
            4.0,
        ),
        // The derivatives grow like `∓x^{1/4}` rather than decaying, and ride a *different*
        // coefficient series — see [`airy_derivative`]. Before this they had no accuracy test at
        // any large argument, only the `NaN`/finiteness rows in `super::super::range`.
        row(
            "airy_ai_e_prime",
            Real::airy_ai_e_prime,
            move |x| {
                let scale = -x.powf(0.25) / (2.0 * SQRT_PI);
                scaled(summed(&aip, two_thirds * x.powf(1.5)), scale)
            },
            (1.0, 79.0),
            8.0,
        ),
        row(
            "airy_bi_e_prime",
            Real::airy_bi_e_prime,
            move |x| {
                let scale = x.powf(0.25) / SQRT_PI;
                scaled(summed(&bip, two_thirds * x.powf(1.5)), scale)
            },
            (1.0, 79.0),
            8.0,
        ),
        row("zeta", Real::zeta, dirichlet, (1.0, 1.7), 12.0),
    ]
}

/// Every kernel matches its asymptotic expansion across the large-argument regime.
///
/// Two failure modes are separated deliberately, because they have opposite fixes. A kernel over
/// its budget is a kernel problem. A reference whose own error bar is not inside the budget is a
/// *test* problem — the window bottom is too low — and is reported as such rather than absorbed
/// into a wider tolerance. Lumping the two together is what makes a sweep like this rot: every
/// unexplained disagreement becomes a reason to loosen the number.
#[test]
fn every_kernel_matches_its_asymptotic_expansion() {
    let rows = asymptotes();
    let mut failures = Vec::new();
    let mut unconverged = Vec::new();
    let mut compared = 0usize;

    for (name, kernel, reference, low, high, budget) in &rows {
        let (mut worst, mut worst_at) = (0.0_f64, 0.0_f64);
        for i in 0..=SAMPLES {
            let exponent = low + (high - low) * (i as f64) / (SAMPLES as f64);
            let x = 10f64.powf(exponent);
            let (got, (want, bar)) = (kernel(x), reference(x));
            if !got.is_finite() || !want.is_finite() || want == 0.0 {
                continue;
            }
            // The reference has to be sharper than the thing it is being used to measure.
            if !bar.is_finite() || bar / want.abs() > 0.25 * budget * f64::EPSILON {
                unconverged.push(format!(
                    "  {name} at x = {x:e}: the expansion's own truncation is {:.2} ulp against a \
                     {budget}-ulp budget, so this window bottom cannot measure a kernel",
                    bar / want.abs() / f64::EPSILON
                ));
                break;
            }
            compared += 1;
            let ulps = ((got - want) / want).abs() / f64::EPSILON;
            if ulps > worst {
                (worst, worst_at) = (ulps, x);
            }
        }
        if worst > *budget {
            failures.push(format!(
                "  {name} is {worst:.2} ulp from its expansion at x = {worst_at:e}, budget \
                 {budget}"
            ));
        }
    }

    assert_eq!(rows.len(), 17, "the sweep lost kernels");
    // Reported before the sample-count floor, because an unconverged reference *causes* a short
    // sweep — the guard stops that kernel's scan — and the cause is the more useful message.
    assert!(
        unconverged.is_empty(),
        "asymptotic references were not converged — raise the window, do not raise the \
         budget:\n{}",
        unconverged.join("\n")
    );
    assert!(
        compared >= 25_000,
        "only {compared} comparisons made; the windows or guards have shrunk the sweep"
    );
    assert!(
        failures.is_empty(),
        "kernels disagreed with their asymptotic expansion:\n{}",
        failures.join("\n")
    );
}
