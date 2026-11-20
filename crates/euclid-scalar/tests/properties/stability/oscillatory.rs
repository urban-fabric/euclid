//! `J` and `Y` at large argument: the phase-free checks, and the pointwise one they preceded.
//!
//! # The blind spot this file was written for
//!
//! Every other accuracy test in this suite compares a kernel against a reference *value*. For the
//! oscillating Bessels that was thought to stop working early, because their asymptotic form is
//! `√(2/πx)·cos(x − π/4)` and the phase needs `x` to a precision the format does not have: one
//! ulp of `x` at `1e8` is `1.5e-8` of phase, and at `1e17` it is larger than `2π`. So
//! [`super::asymptotics`] excludes all four, and `identities::wronskians` stops its `J`/`Y`
//! Wronskian at `1e8` for the same reason, saying so explicitly.
//!
//! That left `J₀`, `J₁`, `Y₀`, `Y₁` with **no accuracy statement at all above `1e8`**, across
//! nearly three hundred decades of their domain — which is what the phase-free tests below were
//! built to cover.
//!
//! # Two different things were being conflated, and one of them was a defect
//!
//! The reasoning above is sound about the *caller's* argument and was wrong about the
//! implementation. Those are separate claims:
//!
//! - `J₀` of a **real number** near `1e12` is unrecoverable, because rounding it to `f64` already
//!   moves the phase by `1.2e-4`. Still true, and nothing can fix it here.
//! - `J₀` of **the exact double it is handed** has a single right answer, and the kernel was not
//!   returning it. `hankel` formed `ω = x − (ν/2 + ¼)π` as one `f64`, which rounds to `ulp(x)`,
//!   so the values carried `ε√(2x/π)/2` — `4.1e-10` absolute at `x = 1e13` against a documented
//!   `1e-15`. Found by sampling the surface against MPFR at 256 bits rather than by any test in
//!   this tree, and fixed by applying the phase shift through the angle-subtraction identities.
//!
//! So there *is* a pointwise statement, and it now runs to `1e18`:
//! [`j_and_y_are_pointwise_accurate_to_the_top_of_the_format`]. The phase-free tests below are
//! kept, and not as history — they cover the whole axis to `1e300` where the pointwise test
//! carries six arguments, and the first of them is what makes the Wronskian's blindness explicit
//! rather than assumed.
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
//! # The negative Airy axis shares its phase too, since the `hankel` fix
//!
//! `Ai(−z)` and `Bi(−z)` oscillate as well, and this section used to say the same modulus
//! **does not work** for them — because `negative_asymptotic` builds all four values from four
//! separate Bessel evaluations at orders `±1/3` and `±2/3`, each of which rounded its own
//! `ω = ζ − (ν/2 + ¼)π` to `ulp(ζ)`. Unshared phase errors do not cancel, so the modulus carried
//! them and the Wronskian reported them: the reverse of this file's first two tests.
//!
//! `hankel` no longer forms `ω` directly. It takes `sin ζ` and `cos ζ` at the exact argument and
//! applies the order's shift through the angle-subtraction identities, so the four calls share
//! one phase and the Airy family now behaves like `J`/`Y`: the modulus cancels the error
//! ([`the_negative_airy_modulus_matches_its_two_term_model`], whose `ζ·ε` term is gone) and the
//! Wronskian is blind to it ([`the_airy_wronskian_is_blind_to_a_shared_phase`]).
//!
//! # What this does and does not establish
//!
//! The phase-free tests pin the **amplitude**, over the whole axis to `1e300`, and they say
//! nothing about the phase by construction — which is what makes them hold that far.
//!
//! The pointwise test pins the **value at the given double**, to `1e18`, which is as far as
//! exactly-representable powers of ten and 256-bit references were taken.
//!
//! Neither promises `J₀` of a real number near `1e12`: one ulp of the argument is `1.2e-4` of
//! phase there, so a caller needing that accuracy needs an argument reduction this crate does not
//! offer. That limit is a property of the caller's `x` rather than of the kernel, and the
//! distinction is worth keeping straight — collapsing the two is what let a genuine `ε√x` defect
//! sit behind an argument about conditioning.

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

/// `J` and `Y` **pointwise** at large argument, which this file previously could not do.
///
/// # Why this test can exist now
///
/// The module header above says the oscillating Bessels have "no accuracy statement at all above
/// `1e8`", and that the modulus is the strongest available because it is the only quantity with
/// no phase in it. That was true of the implementation it was written against: `hankel` formed
/// `ω = x − (ν/2 + ¼)π` as a single `f64`, so the phase carried `ulp(x)/2` radians of error and
/// the values were wrong by `ε√(2x/π)/2` — `4.1e-10` absolute at `x = 1e13`, against a
/// documented promise of `1e-15`.
///
/// `hankel` now evaluates `sin x` and `cos x` at the exact argument, where the library's own
/// reduction is good to an ulp however large `x` is, and applies the order's shift through the
/// angle-subtraction identities. The phase is as accurate as `x` is exact, so there *is* a
/// pointwise statement to make, and it runs to the top of the format.
///
/// # Provenance
///
/// MPFR at 256 bits through `ctypes`, rounded to nearest `f64`, evaluated at the exact `f64`
/// argument the test passes rather than at the real-valued expression it came from. That
/// distinction is the one CONTRIBUTING insists on and it is load-bearing here: these functions
/// oscillate, so `J₀` of the nearest double to `1e12` and `J₀` of `1e12` exactly are different
/// numbers. Every power of ten below is exactly representable, which removes the question.
///
/// Worst measured absolute error across the six arguments and four functions is **`3.5e-18`**,
/// which is below an ulp of every value in the table. The bound asserted is `1e-15`, the figure
/// the trait documents.
///
/// What this still does not promise is `J₀` of a *real number* near `1e12`: one ulp of the
/// argument is `1.2e-4` of phase there, so the conditioning of the input dominates anything the
/// implementation does. That is a statement about the caller's `x`, not about this kernel, and
/// the trait says so separately.
#[test]
fn j_and_y_are_pointwise_accurate_to_the_top_of_the_format() {
    // (x, J₀, J₁, Y₀, Y₁) at 256 bits, rounded to nearest `f64`.
    const REFERENCES: &[(f64, f64, f64, f64, f64)] = &[
        (
            1e3,
            0.024_786_686_152_420_176,
            0.004_728_311_907_089_524,
            0.004_715_917_977_622_813_5,
            -0.024_784_331_292_351_778,
        ),
        (
            1e6,
            0.000_331_043_013_739_873_76,
            -0.000_725_968_356_813_763,
            -0.000_725_968_522_335_179_1,
            -0.000_331_043_376_724_176_26,
        ),
        (
            1e9,
            2.468_747_188_626_919_5e-5,
            -5.210_422_641_553_878e-6,
            -5.210_422_653_897_613_5e-6,
            -2.468_747_188_887_440_7e-5,
        ),
        (
            1e12,
            1.016_712_505_004_068_2e-7,
            -7.913_802_683_850_441e-7,
            -7.913_802_683_850_949e-7,
            -1.016_712_505_008_025e-7,
        ),
        (
            1e15,
            6.156_638_646_885_021e-9,
            2.446_866_512_377_132_8e-8,
            2.446_866_512_377_132_4e-8,
            -6.156_638_646_885_01e-9,
        ),
        (
            1e18,
            -4.934_387_036_790_141e-10,
            -6.270_071_914_094_411e-10,
            -6.270_071_914_094_411e-10,
            4.934_387_036_790_141e-10,
        ),
    ];

    let mut worst = 0.0f64;
    let mut worst_name = "";
    let mut worst_at = 0.0f64;
    for &(x, j0, j1, y0, y1) in REFERENCES {
        for (name, got, want) in [
            ("J0", Real::bessel_j0(x), j0),
            ("J1", Real::bessel_j1(x), j1),
            ("Y0", Real::bessel_y0(x), y0),
            ("Y1", Real::bessel_y1(x), y1),
        ] {
            // Absolute, which is the unit the trait promises and the right one for a function
            // that passes through zero: these four have zeros densely packed out here, so a
            // relative bound would be meaningless at the arguments nearest them.
            let error = (got - want).abs();
            if error > worst {
                worst = error;
                worst_name = name;
                worst_at = x;
            }
        }
    }
    assert!(
        worst < 1e-15,
        "{worst_name} at x = {worst_at:e} is {worst:e} from its 256-bit reference, against a \
         measured worst of 3.5e-18 and the 1e-15 the trait documents. If this has grown, check \
         that `hankel` still applies the phase shift through the angle-subtraction identities \
         rather than forming ω as a single f64."
    );
}
