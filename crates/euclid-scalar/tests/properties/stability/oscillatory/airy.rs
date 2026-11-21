//! The negative-Airy modulus and Wronskian: phase-free checks in `ζ = ⅔|x|^{3/2}`.
//!
//! Split from the parent, which carries the `J`/`Y` pair and the pointwise statement. The
//! same two questions are asked here — does the envelope match its asymptotic model, and is
//! the Wronskian blind to a phase error the four calls share — of an oscillation in `ζ`
//! rather than in `x`.

use super::*;
// ===== The negative Airy axis, where the same trick does not work =========

/// `ζ = ⅔|x|^{3/2}`, the Airy phase on the negative axis.
fn airy_phase(z: f64) -> f64 {
    2.0 / 3.0 * z.powf(1.5)
}

/// `Ai(−z)² + Bi(−z)²` and the derivative pair, against a **two-term** model.
///
/// # Why this is not the `J`/`Y` test with different letters
///
/// The modulus works for `J`/`Y` because `sin²θ + cos²θ = 1` holds for *any* `θ`, including a
/// wrong one — so a phase error shared between the two functions cancels out of the sum and the
/// remaining quantity is exact. That argument needs the phase to be **shared**.
///
/// # The phase is shared now, and this test used to say it was not
///
/// It was not, when this was written: `hankel` formed `ω = ζ − (ν/2 + ¼)π` as a single `f64`,
/// so the four evaluations at orders `±1/3` and `±2/3` each rounded their own `ω` to `ulp(ζ)`
/// and the errors did not cancel. The model below carried a `ζ·ε` term for exactly that, and
/// the measured residual reached `4.9e-8` by `z = 1e6` and `70%` by `1e15`.
///
/// `hankel` now evaluates `sin ζ` and `cos ζ` at the exact argument and applies the `(ν/2 + ¼)π`
/// shift through the angle-subtraction identities, so all four calls derive from the *same* two
/// trigonometric values and their phase error is common. It cancels out of the modulus, and the
/// `ζ·ε` term is gone:
///
/// ```text
/// z             1e3      1e5      1e8      1e11     1e15
/// residual    1.3e-10  0.0e+00  2.2e-16  6.3e-16  1.6e-16
/// old ζ·ε     5.1e-12  5.0e-09  1.5e-04  4.6e+00  4.4e+06
/// ```
///
/// Past `z ≈ 1e5` the old model was not a bound at all — at `1e11` it permitted a residual of
/// `464%`. What is left is the expansion's own truncation plus round-off.
///
/// # The two terms that remain
///
/// ```text
/// rel|M²|  ≲  0.0694/ζ²   +   8ε
///             asymptotic      round-off
/// ```
///
/// The first is the expansion's own `O(ζ^{-2})` correction. `Ai` and `Bi` carry
/// `A = Σ(−1)^k u_{2k} ζ^{-2k}` and `B = Σ(−1)^k u_{2k+1} ζ^{-2k-1}` against orthogonal
/// trigonometric factors, so the cross terms cancel and `M² = (A² + B²)/(π√z)`, giving
/// `1 + (u₁² − 2u₂)/ζ²` with `u₁ = 5/72` and `u₂ = 385/10368` — that is `−0.0694/ζ²`. At `z = 10`,
/// `ζ = 21.08`, the prediction is `1.56e-4` and the measurement is `1.56e-4`.
///
/// The second is a floor rather than a growing term: eight ulp of accumulated round-off in four
/// evaluations and two squarings. `16ε` gives the same worst ratio, so the binding point is the
/// asymptotic term and not the floor — which is how it should be.
///
/// Together they fit the whole axis to a factor of **1.401** over `[5, 1e200]`, across 8,002
/// comparisons. Note what this no longer does: the old model "independently reproduced the
/// trait's precision note" that accuracy departs around `|x| ≈ 1e11`. That note is still right
/// about the individual *values* — `ζ` itself is an `f64`, so `Ai(−z)` at `z = 1e11` is the `Ai`
/// of a measurably different argument — but it is no longer what limits this *identity*, because
/// an error in `ζ` shared by all four values cancels here.
#[test]
fn the_negative_airy_modulus_matches_its_two_term_model() {
    /// Worst measured ratio is `1.401`; `4` admits that and a doubling.
    const BUDGET: f64 = 4.0;

    let mut failures = Vec::new();
    let mut compared = 0usize;
    for i in 0..=SAMPLES {
        // `1e0.7` up, because below about `z = 5` the expansion is not the route at all.
        let z = 10f64.powf(0.7 + (200.0 - 0.7) * (i as f64) / (SAMPLES as f64));
        let zeta = airy_phase(z);
        if !zeta.is_finite() {
            continue;
        }
        let model = 0.0694 / (zeta * zeta) + 8.0 * f64::EPSILON;
        for (name, modulus, want) in [
            (
                "Ai(-z)^2 + Bi(-z)^2",
                Real::airy_ai(-z).powi(2) + Real::airy_bi(-z).powi(2),
                1.0 / (core::f64::consts::PI * z.sqrt()),
            ),
            (
                "Ai'(-z)^2 + Bi'(-z)^2",
                Real::airy_ai_prime(-z).powi(2) + Real::airy_bi_prime(-z).powi(2),
                z.sqrt() / core::f64::consts::PI,
            ),
        ] {
            if !modulus.is_finite() || want == 0.0 || !want.is_finite() {
                continue;
            }
            compared += 1;
            let relative = ((modulus - want) / want).abs();
            if relative > BUDGET * model {
                failures.push(format!(
                    "  {name} at z = {z:e} is {relative:e} from its envelope, against the \
                     {BUDGET}·(0.0694/zeta^2 + 8*eps) = {:e} the model allows",
                    BUDGET * model
                ));
            }
        }
    }

    assert!(compared >= 7_000, "only {compared} comparisons made");
    assert!(
        failures.is_empty(),
        "the negative Airy modulus left its two-term model:\n{}",
        failures.join("\n")
    );
}

/// `Ai·Bi′ − Ai′·Bi = 1/π` holds to round-off across the whole negative axis.
///
/// # This test used to assert the opposite, and the old version is why the defect was found
///
/// It was `the_airy_wronskian_does_see_a_phase_error`, and it did: the four values came from four
/// `hankel` calls that each rounded their own `ω = ζ − (ν/2 + ¼)π`, so the errors did not cancel.
/// At `z = 1e8` the residual was `1.48e-4`, tracking `ζ·ε`, against `6.98e-16` now.
///
/// The old assertion was two-sided — it required the residual to sit *within a decade of `ζ·ε`*
/// rather than at round-off — and its failure message said what to do if it ever collapsed: "the
/// four Bessel calls now share a phase and the modulus test above can be sharpened". That is
/// exactly what happened when `hankel` stopped forming `ω` directly, and it is the reason this
/// change was caught rather than silently absorbed. A one-sided `residual < tolerance` would have
/// passed and said nothing.
///
/// # What it pins now
///
/// The identity is a *constant*, so with a shared phase it is blind to any error in `ζ` — the
/// same insensitivity [`the_wronskian_cannot_see_a_common_phase_error`] measures for `J`/`Y`.
/// Measured worst **`1.22e-15`** at `z = 1.9e35`, over 2,001 points across `[5, 1e200]`; the
/// bound here is `1e-14`.
///
/// Being blind to phase is what makes it a *useful* check on everything else: the four values
/// are combined with four different coefficients, so an error in any one of them — an amplitude,
/// a connection-formula constant, a transposed `J` order — does not cancel and shows up here.
#[test]
fn the_airy_wronskian_is_blind_to_a_shared_phase() {
    let wronskian = |x: f64| {
        Real::airy_ai(x) * Real::airy_bi_prime(x) - Real::airy_ai_prime(x) * Real::airy_bi(x)
    };
    let one_over_pi = 1.0 / core::f64::consts::PI;

    let mut worst = 0.0f64;
    let mut worst_at = 0.0f64;
    for i in 0..=2_000 {
        let z = 10f64.powf(0.7 + (200.0 - 0.7) * f64::from(i) / 2_000.0);
        if !airy_phase(z).is_finite() {
            continue;
        }
        let residual = ((wronskian(-z) - one_over_pi) / one_over_pi).abs();
        if residual > worst {
            worst = residual;
            worst_at = z;
        }
    }

    // Round-off, and independent of `ζ` — which is the whole content of the claim. If this grows
    // back towards `ζ·ε`, the four `hankel` calls have stopped sharing a phase.
    assert!(
        worst < 1e-14,
        "the Airy Wronskian residual reached {worst:e} at z = {worst_at:e}, against a measured \
         worst of 1.22e-15. If it has grown towards ζ·ε the four Bessel calls no longer share a \
         phase; see `hankel`, which must not form ω as a single f64."
    );
}
