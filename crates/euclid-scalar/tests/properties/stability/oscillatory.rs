//! `J` and `Y` at large argument, where no pointwise test can reach them.
//!
//! # The blind spot
//!
//! Every other accuracy test in this suite compares a kernel against a reference *value*. For the
//! oscillating Bessels that stops working early, because their asymptotic form is
//! `√(2/πx)·cos(x − π/4)` and the phase needs `x` to a precision the format does not have: one
//! ulp of `x` at `1e8` is `1.5e-8` of phase, and at `1e17` it is larger than `2π`. So
//! [`super::asymptotics`] excludes all four, and `identities::wronskians` stops its `J`/`Y`
//! Wronskian at `1e8` for the same reason, saying so explicitly.
//!
//! That leaves `J₀`, `J₁`, `Y₀`, `Y₁` with **no accuracy statement at all above `1e8`**, across
//! nearly three hundred decades of their domain.
//!
//! # The Wronskian cannot fill it, and that is measured rather than argued
//!
//! The obvious objection is that `J₀Y₁ − J₁Y₀ = −2/(πx)` already constrains the pair. It does,
//! but not in the way that matters: the identity's value depends only on `x`, so a phase error
//! **common to both functions** very nearly cancels out of it. Injecting a shift `d` into all four
//! evaluations and comparing what the residual sees against what `J₀` actually does:
//!
//! ```text
//! x      d        Wronskian residual    J0's own relative change    dilution
//! 1e3    1 ulp    5.1e-16               2.2e-14                     43x
//! 1e6    1 ulp    0 (exactly)           2.6e-10                     unbounded
//! 1e6    10 ulp   1.2e-15               2.6e-9                      2.2e6
//! 1e8    1 ulp    0 (exactly)           3.4e-8                      unbounded
//! 1e8    10 ulp   1.3e-15               3.4e-7                      2.6e8
//! ```
//!
//! At `x = 1e8` a one-ulp common phase error is **exactly invisible** to the Wronskian while
//! costing `J₀` eight digits. A thousand ulps of phase there — `3.4e-5` relative in `J₀` — moves
//! the residual to `1.5e-13`, which still sits inside the `1e-13` tolerance that test asserts. So
//! the Wronskian is not a weak check on amplitude; it is almost no check on phase, and the
//! dilution grows like `x`.
//!
//! # What does work for `J`/`Y`: the modulus, which has no phase in it at all
//!
//! `M_ν²(x) = J_ν²(x) + Y_ν²(x)` is the squared amplitude of the oscillation, and the `cos` and
//! `sin` that carry the phase square and sum to one. What remains is a function of `x` alone:
//!
//! ```text
//! M_ν²(x) ~ (2/πx)·[1 + ½(4ν² − 1)/(2x)² + …]
//! ```
//!
//! — so `−1/(8x²)` at `ν = 0` and `+3/(8x²)` at `ν = 1`. The signs differ, which is worth stating
//! because getting them wrong is silent: a first pass used `+1/(8x²)` for both and `M₀²` reported
//! `1.1e7` ulp while `M₁²` reported `3.9`, the discrepancy between the two being the only clue.
//!
//! Above `1e8` the correction is already under an ulp, so the **leading term alone** is an exact
//! reference, and it stays exact to the top of the format. Measured worst, over 4,001 points per
//! window:
//!
//! ```text
//! window              M0^2       M1^2
//! [1e8,   1e16]      2.58 ulp   2.80 ulp
//! [1e16,  1e20]      2.42       2.42
//! [1e20,  1e100]     2.87       2.87
//! [1e100, 1e300]     2.27       2.27
//! ```
//!
//! **Under three ulps across two hundred and ninety decades**, for four kernels that had no
//! accuracy test above `1e8` at all. That is the strongest statement available about them, and it
//! is available only because the quantity asserted has no phase in it.
//!
//! # What this does and does not establish
//!
//! # The negative Airy axis needs the opposite tools
//!
//! `Ai(−z)` and `Bi(−z)` oscillate too, and the obvious move — the same modulus — **does not
//! work**, for a structural reason worth knowing before reaching for it:
//! [`the_negative_airy_modulus_matches_its_two_term_model`]. `J`/`Y` share one phase, so the
//! modulus cancels any error in it and the Wronskian cannot see one. Airy's four values come from
//! four separate Bessel evaluations, so the modulus *carries* the phase error while the Wronskian
//! *reports* it — the reverse of this file's first two tests, pinned in
//! [`the_airy_wronskian_does_see_a_phase_error`].
//!
//! # What this does and does not establish
//!
//! It pins the **amplitude**. It says nothing about the phase, and cannot: the phase genuinely is
//! unrecoverable at these arguments, so there is no reference to compare against and no defect to
//! find. A caller needing `J₀(1e12)` to a given accuracy needs an argument reduction this crate
//! does not promise — which is what the trait documents — and what this file adds is that the
//! *envelope* those oscillations sit in is correct to a couple of ulps regardless.

use super::*;

/// Sample points per window.
const SAMPLES: usize = 4_000;

/// `2/(πx)`, the phase-free leading term of both moduli.
fn leading(x: f64) -> f64 {
    2.0 / (core::f64::consts::PI * x)
}

/// A common phase error is invisible to the `J`/`Y` Wronskian, by a factor that grows with `x`.
///
/// This is the justification for the rest of the file, asserted rather than left in prose. It
/// injects a shift into all four evaluations — which is exactly what an argument reduction losing
/// bits would do — and checks that the Wronskian residual moves far less than the function values.
///
/// Asserting a *lower bound on the dilution* is deliberate. The claim being pinned is not that
/// the Wronskian is inaccurate, but that it is **insensitive** to this failure mode, so the thing
/// to protect against is someone later concluding that the Wronskian covers the oscillating
/// kernels at large argument and deleting the modulus test below.
#[test]
fn the_wronskian_cannot_see_a_common_phase_error() {
    let wronskian_at =
        |x: f64| Real::bessel_j0(x) * Real::bessel_y1(x) - Real::bessel_j1(x) * Real::bessel_y0(x);

    // (argument, phase shift in ulps of x, the dilution the Wronskian must at least achieve)
    for &(x, ulps, least_dilution) in &[(1e6_f64, 10.0_f64, 1e5_f64), (1e8, 10.0, 1e7)] {
        let shift = ulps * (f64::from_bits(x.to_bits() + 1) - x);

        // What the identity sees when both functions are shifted together.
        let residual = ((wronskian_at(x + shift) - (-leading(x))) / leading(x)).abs();
        // What `J₀` itself does over the same shift.
        let moved = ((Real::bessel_j0(x + shift) - Real::bessel_j0(x)) / Real::bessel_j0(x)).abs();

        assert!(
            moved > 1e-9,
            "a {ulps}-ulp shift at x = {x:e} should move J0 appreciably, but it moved {moved:e}; \
             this test's premise has gone"
        );
        let dilution = moved / residual;
        assert!(
            dilution >= least_dilution,
            "at x = {x:e} a {ulps}-ulp common phase error moved J0 by {moved:e} and the \
             Wronskian residual by {residual:e}, a dilution of only {dilution:e}. If the \
             Wronskian has become sensitive to phase, this file's premise needs rewriting — do \
             not simply relax the number."
        );
    }
}

/// `J_ν² + Y_ν²` matches its phase-free asymptotic from `1e8` to the top of the format.
///
/// The only accuracy statement this suite makes about the oscillating Bessels above `1e8`, and it
/// reaches `1e300`. See the module header for why the modulus is the right quantity and for the
/// measured table these budgets come from.
///
/// The budget is `8` ulp against a measured worst of `2.87`, the usual twofold-plus headroom. It
/// is a single constant rather than a per-window column because the measurement is flat — the
/// correction term is long gone by `1e8`, so there is nothing left that varies with `x`.
#[test]
fn the_bessel_modulus_matches_its_phase_free_asymptotic() {
    /// Measured worst is `2.87`; `8` admits that and a doubling.
    const BUDGET: f64 = 8.0;

    let mut failures = Vec::new();
    let mut compared = 0usize;

    for &(low, high) in &[
        (8.0_f64, 16.0_f64),
        (16.0, 20.0),
        (20.0, 100.0),
        (100.0, 300.0),
    ] {
        for i in 0..=SAMPLES {
            let exponent = low + (high - low) * (i as f64) / (SAMPLES as f64);
            let x = 10f64.powf(exponent);
            let want = leading(x);
            if want == 0.0 || !want.is_finite() {
                continue;
            }
            for (name, modulus) in [
                (
                    "J0^2 + Y0^2",
                    Real::bessel_j0(x).powi(2) + Real::bessel_y0(x).powi(2),
                ),
                (
                    "J1^2 + Y1^2",
                    Real::bessel_j1(x).powi(2) + Real::bessel_y1(x).powi(2),
                ),
            ] {
                if !modulus.is_finite() {
                    failures.push(format!("  {name} at x = {x:e} is {modulus:e}"));
                    continue;
                }
                compared += 1;
                let ulps = ((modulus - want) / want).abs() / f64::EPSILON;
                if ulps > BUDGET {
                    failures.push(format!(
                        "  {name} at x = {x:e} is {ulps:.2} ulp from 2/(pi x), budget {BUDGET}"
                    ));
                }
            }
        }
    }

    assert!(
        compared >= 30_000,
        "only {compared} comparisons made; the windows have shrunk"
    );
    assert!(
        failures.is_empty(),
        "the Bessel modulus left its phase-free asymptotic:\n{}",
        failures.join("\n")
    );
}

/// Below `1e8` the correction term is needed, and its sign differs between the two orders.
///
/// The complement to the test above, covering the decades where `½(4ν² − 1)/(2x)²` is still above
/// an ulp. Worth asserting separately rather than widening the window, because it is the only
/// place the `ν`-dependence of the sign is visible: using `+1/(8x²)` for both orders passes for
/// `M₁²` and fails for `M₀²` by seven orders of magnitude, so a single shared coefficient would
/// look almost right.
#[test]
fn the_bessel_modulus_needs_its_correction_term_below_1e8() {
    // `-1/8` at order zero and `+3/8` at order one, both from `½(4ν² − 1)/4`.
    for &(name, coefficient, order) in &[("M0^2", -0.125_f64, 0u32), ("M1^2", 0.375, 1)] {
        let mut worst = 0.0_f64;
        let mut worst_at = 0.0_f64;
        for i in 0..=SAMPLES {
            let x = 10f64.powf(4.0 + 4.0 * (i as f64) / (SAMPLES as f64));
            let want = leading(x) * (1.0 + coefficient / (x * x));
            let modulus = if order == 0 {
                Real::bessel_j0(x).powi(2) + Real::bessel_y0(x).powi(2)
            } else {
                Real::bessel_j1(x).powi(2) + Real::bessel_y1(x).powi(2)
            };
            let ulps = ((modulus - want) / want).abs() / f64::EPSILON;
            if ulps > worst {
                worst = ulps;
                worst_at = x;
            }
        }
        assert!(
            worst <= 16.0,
            "{name} is {worst:.2} ulp from its corrected asymptotic at x = {worst_at:e} over \
             [1e4, 1e8]. If the coefficient's sign is the suspect, note that it is {coefficient} \
             here and differs between the two orders."
        );
    }
}

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
/// remaining quantity is exact. That argument needs the phase to be **shared**, and on the
/// negative Airy axis it is not: `negative_asymptotic` builds all four values from four separate
/// Bessel evaluations at orders `±1/3` and `±2/3`, each with its own rounding of `ζ`. So the
/// modulus carries the phase error instead of cancelling it, and
/// `Ai(−z)² + Bi(−z)² ~ 1/(π√z)` is **not** an exact reference at large `z` the way
/// `J₀² + Y₀² ~ 2/(πx)` is.
///
/// Measured, the difference is stark. `J`/`Y`'s modulus holds to under three ulp from `1e8` to
/// `1e300`; Airy's reaches `4.9e-8` relative by `z = 1e6` and `70%` by `1e15`. Reporting that as
/// a defect would have been wrong, which is why this file models it instead.
///
/// # The two terms, both confirmed
///
/// ```text
/// rel|M²|  ≲  0.0694/ζ²   +   ζ·ε
///             asymptotic      phase rounding
///             (below ~3e3)    (above ~3e3)
/// ```
///
/// The first is the expansion's own `O(ζ^{-2})` correction. `Ai` and `Bi` carry
/// `A = Σ(−1)^k u_{2k} ζ^{-2k}` and `B = Σ(−1)^k u_{2k+1} ζ^{-2k-1}` against orthogonal
/// trigonometric factors, so the cross terms cancel and `M² = (A² + B²)/(π√z)`, giving
/// `1 + (u₁² − 2u₂)/ζ²` with `u₁ = 5/72` and `u₂ = 385/10368` — that is `−0.0694/ζ²`. At `z = 10`,
/// `ζ = 21.08`, the prediction is `1.56e-4` and the measurement is `1.56e-4`.
///
/// The second is `ulp(ζ)`: the phase is an `f64` of magnitude `ζ`, so it carries `ζ·ε` radians of
/// uncertainty, and an unshared phase error of that size shows up directly in the modulus.
///
/// Together they fit the whole axis to a factor of `1.4` — worst ratio `1.183` for `M²` at
/// `z = 3.5e3` and `1.406` for `M′²` at `z = 2.3e5`, over `[5, 1e200]`. And the model
/// independently reproduces the trait's own precision note: `ζ·ε` passes `1` at `z ≈ 6e10`, which
/// is where the documentation says "accuracy in fact departs … around `|x| ≈ 1e11`".
#[test]
fn the_negative_airy_modulus_matches_its_two_term_model() {
    /// Worst measured ratio is `1.406`; `4` admits that and a doubling.
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
        let model = 0.0694 / (zeta * zeta) + zeta * f64::EPSILON;
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
                     {BUDGET}·(0.0694/zeta^2 + zeta*eps) = {:e} the model allows",
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

/// The Airy Wronskian *does* see a phase error — the exact opposite of the `J`/`Y` one.
///
/// A pleasing consequence of the structural difference above, and worth pinning because it
/// reverses the conclusion of
/// [`the_wronskian_cannot_see_a_common_phase_error`] for this family.
///
/// `Ai·Bi′ − Ai′·Bi = 1/π` is a *constant*, so one might expect it to be even blinder to phase
/// than `−2/(πx)` was. It is not, and for the same reason the modulus fails: the four values come
/// from four separate Bessel evaluations, so their phase errors do **not** share a sign and do not
/// cancel. Measured at `z = 1e8`, the Wronskian residual is `5.27e-5` while `Ai`'s own relative
/// change under a one-ulp shift is `2.18e-5` — a dilution of `0.41`, meaning the identity sees the
/// error slightly *more* clearly than the function does. The `J`/`Y` dilution at the same
/// argument is `2.6e8`.
///
/// So the two families need opposite tools, which is the useful thing to record: share the phase
/// and the modulus is exact while the Wronskian goes blind; split it and the modulus carries the
/// error while the Wronskian reports it.
#[test]
fn the_airy_wronskian_does_see_a_phase_error() {
    let z = 1e8_f64;
    let wronskian = |x: f64| {
        Real::airy_ai(x) * Real::airy_bi_prime(x) - Real::airy_ai_prime(x) * Real::airy_bi(x)
    };
    let one_over_pi = 1.0 / core::f64::consts::PI;

    let residual = ((wronskian(-z) - one_over_pi) / one_over_pi).abs();
    let phase_rounding = airy_phase(z) * f64::EPSILON;

    // The residual is the phase rounding reaching the identity, not an unrelated defect: it
    // should sit within a decade of `ζ·ε` rather than at round-off.
    assert!(
        residual > 0.01 * phase_rounding && residual < 10.0 * phase_rounding,
        "the Airy Wronskian residual at z = {z:e} is {residual:e}, which is not the \
         {phase_rounding:e} of phase rounding this family's split evaluation predicts. If it has \
         collapsed to round-off, the four Bessel calls now share a phase and the modulus test \
         above can be sharpened; if it has grown, something else is wrong."
    );
}
