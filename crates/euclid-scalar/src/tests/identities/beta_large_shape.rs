//! The incomplete beta at large shape, where its continued fraction hands over.
//!
//! The peer of [`super::gamma_large_shape`], past the threshold where the continued fraction is
//! abandoned for an Edgeworth-corrected normal.
//!
//! # What was already covered, and what was not
//!
//! Worth stating precisely, because the gap here was not the obvious one.
//! [`super::beta_range`] already reached this branch: it sweeps `I_½(a, a) = ½` over
//! `a = 1e7 … 1e100` and checks the complement identity at pairs up to `(1e10, 1e10)`. The
//! branch executed, and on an exact identity. Two things were still missing:
//!
//! - **The top of the exponent range.** That sweep stops at `1e100`. The format goes to
//!   `1.8e308`, and `1e100` is a round number rather than a boundary — it reads as "the whole
//!   range" and leaves 206 decades of shape behind it. The defect below was in all of them.
//! - **`γ₁` at an asymmetric shape.** The skewness correction vanishes identically when `a = b`,
//!   so the symmetric sweep never evaluates it. The complement identity does run asymmetric
//!   pairs, but the Edgeworth form satisfies that identity for *any* antisymmetric `γ₁` —
//!   swapping the shapes negates `z` and `γ₁` together — so it constrains the term's sign and
//!   not its magnitude. Nothing had compared `γ₁` against an independent value.
//!
//! # The defect this found
//!
//! `I_x(a, b)` returned **NaN for every `x`** once `a + b` passed `5.64e102` — so for `a = b`
//! above `2.82e102`, two and a half decades past where the existing sweep stopped. The answers
//! it was failing to produce were not delicate: `I_½(a, a)` is exactly `½` at every shape, by
//! symmetry of `Beta(a, a)`. Three intermediates left `f64` while the values they encode stayed
//! well inside it:
//!
//! | Intermediate | Leaves `f64` at | What it encodes | True value there |
//! |---|---|---|---|
//! | `n·n·(n+1)` | `n > 5.6e102` | `σ` at `a = b = 1e150` | `3.5e-76` |
//! | `a·b` | `a·b > 1.8e308` | `γ₁` | `0` for `a = b` |
//! | the variance | `a = f64::MAX, b = 1e7` | `σ` | `1.8e-305` |
//!
//! Each is now formed from bounded ratios, and `z` is short-circuited once the normal tail has
//! saturated — which is also what stops `z·z` overflowing into `0 · ∞ = NaN`.
//!
//! The gamma peer never had this, because its variance is `a` rather than a quotient of three
//! `O(n)` factors. That asymmetry is why the beta needed its own file rather than a few more
//! rows in an existing one.
//!
//! # What the references are
//!
//! Three independent kinds, none of them euclid:
//!
//! - **An exact closed form.** `Beta(a, a)` is symmetric about `½`, so `I_½(a, a) = ½`
//!   identically. No approximation, no table, true at every `a` — and it is what the overflow
//!   broke.
//! - **60-digit `Decimal` values**, generated out-of-band from the all-positive series
//!   `I_x(a,b) = x^a (1−x)^b / (a·B(a,b)) · Σ (a+b)ₙ/(a+1)ₙ xⁿ`, which has no subtraction in it
//!   and so no cancellation. Its terms decay like `e^{−n²/4a}`, needing about `21√a` of them —
//!   68,000 at `a = 1e7`, which is why an in-branch reference is affordable at all. The
//!   generator was checked against `I_x(1,1) = x`, `I_x(a,1) = xᵃ`, `I_x(1,b) = 1−(1−x)ᵇ` and
//!   the exact polynomial `I_x(2,3)` before any value here was taken from it.
//! - **The converged continued fraction**, just below the threshold. Wholly unrelated to a
//!   normal CDF with a skewness correction, so agreement across the join is real evidence the
//!   join is sited correctly rather than a restatement.
//!
//! # Each `x` is written as the double the reference was evaluated at
//!
//! The density is `≈ 3568` at `a = b = 1e7`, so one ulp of `x` is worth `4e-13` of probability —
//! measured up to `4.4e-13` across this table. Computing a reference at the real-valued
//! `μ + zσ` and comparing it against euclid's value at the *rounded* `x` would charge the kernel
//! for that rounding, which is `44×` its actual error here.
//!
//! So each `x` below is in its shortest round-tripping decimal form, and the reference beside it
//! was computed from `Decimal(x)` — exact for a double — rather than from the expression `x` came
//! from. The two columns therefore describe the same argument exactly.

use crate::Real;
use euclid_utils::{assert_close, assert_exact};

use crate::kernels::beta::BETA_EDGEWORTH_MIN;

/// `(a, b, x, I_x(a,b))`, the `Decimal` references described in the module header.
///
/// **Regenerate with `python3 -B bin/references/incomplete.py beta-large-shape`**, which prints
/// these rows verbatim. The generator validates itself against `I_x(1,1) = x`, `I_x(a,1) = xᵃ`,
/// `I_x(1,b) = 1−(1−x)ᵇ`, the exact polynomial `I_x(2,3)` and `I_½(a,a) = ½` before emitting
/// anything, and exits non-zero if any of those fail. It is committed for the same reason
/// `bin/mutants.py` is: a reference whose instrument is missing is a number taken on faith.
///
/// Both symmetric and asymmetric shapes, because they test different code: `γ₁` is
/// `2(b−a)√(n+1)/((n+2)√(ab))`, which **vanishes identically when `a = b`**. A suite of
/// symmetric cases alone would leave the entire skewness correction unexercised, and it is the
/// one term that distinguishes this branch from a plain normal approximation.
const EDGEWORTH_REFERENCE: &[(f64, f64, f64, f64)] = &[
    // a = b: skewness is exactly zero, so these pin the mean/variance path alone.
    (1e7, 1e7, 0.49966458981176026, 0.0013498970344633411),
    (1e7, 1e7, 0.4998881966039201, 0.15865525998070798),
    (1e7, 1e7, 0.5, 0.5),
    (1e7, 1e7, 0.5001118033960799, 0.8413447400191719),
    (1e7, 1e7, 0.5003354101882397, 0.9986501029655345),
    // Ten to one, both orientations: `γ₁` is now ±5.4e-4 and must carry its sign.
    (1e7, 1e8, 0.09082686054276165, 0.001346692639660199),
    (1e7, 1e8, 0.09088168078698115, 0.15865525206114425),
    (1e7, 1e8, 0.09090909090909091, 0.5000360856871126),
    (1e7, 1e8, 0.09093650103120066, 0.8413447479377909),
    (1e7, 1e8, 0.09099132127542016, 0.9986468933492093),
    (1e8, 1e7, 0.9090086787245798, 0.001353106650783902),
    (1e8, 1e7, 0.9090634989687993, 0.15865525206220912),
    (1e8, 1e7, 0.9090909090909091, 0.4999639143124835),
    (1e8, 1e7, 0.9091183192130189, 0.8413447479392233),
    (1e8, 1e7, 0.9091731394572383, 0.9986533073603353),
    // A hundred to one, and a non-decade pair, so the grid is not all powers of ten.
    (1e7, 1e9, 0.00989164381026907, 0.0013462188497053026),
    (1e7, 1e9, 0.009897874669429624, 0.15865525013693643),
    (1e7, 1e9, 0.009900990099009901, 0.5000414250765153),
    (1e7, 1e9, 0.009904105528590178, 0.8413447498613683),
    (1e7, 1e9, 0.009910336387750733, 0.9986464180952248),
    (2e7, 5e7, 0.2855523009412959, 0.0013485579952529215),
    (2e7, 5e7, 0.2856602907899558, 0.15865525514140152),
    (2e7, 5e7, 0.2857142857142857, 0.5000150786007935),
    (2e7, 5e7, 0.28576828063861565, 0.841344744858726),
    (2e7, 5e7, 0.2858762704872755, 0.9986487618750133),
];

/// Measured worst absolute error `6.05e-9` over [`EDGEWORTH_REFERENCE`].
///
/// Absolute rather than relative, and `assert_close`'s mixed criterion makes it so for free
/// here: every reference is a probability, so `max(|want|, 1)` is 1. That is the right bar for
/// a CDF — a relative bound would demand `1e-9` of a `1.3e-3` tail value, which is `7.4e-7`
/// relative, and nothing about this approximation claims that.
///
/// It is also the error the kernel's own crossover analysis predicts: `≈ 0.0605/min(a,b)`, or
/// `6.1e-9` at `min = 1e7`. The agreement between that derivation and this measurement is the
/// substance of the check — an independent reference confirming the documented error model,
/// rather than a tolerance fitted to whatever came out.
const EDGEWORTH_TOL: f64 = 1e-8;

#[test]
fn beta_reg_matches_an_independent_reference_inside_the_edgeworth_branch() {
    for &(a, b, x, want) in EDGEWORTH_REFERENCE {
        assert!(
            a.min(b) >= BETA_EDGEWORTH_MIN,
            "({a:e}, {b:e}) does not reach the branch this table exists to test"
        );
        assert_close(
            Real::beta_reg(x, a, b),
            want,
            EDGEWORTH_TOL,
            &format!("I_x({a:e}, {b:e}) at x = {x:.17e}"),
        );
    }
}

#[test]
fn the_symmetric_median_is_exactly_one_half_at_every_representable_shape() {
    // `Beta(a, a)` is symmetric about ½, so this is an identity rather than an approximation,
    // and `exactly` is the honest bar: the skewness correction vanishes at `a = b` and `Φ(0)`
    // is ½ to the last bit, which is what makes the Edgeworth branch *exact* here.
    //
    // This is the assertion the overflow broke. It returned NaN from `1e103` up, so the
    // exponents past that point are the regression test; the ones below it guard the join.
    for exponent in [
        7.0f64, 7.5, 8.0, 9.0, 10.0, 12.0, 15.0, 20.0, 50.0, 102.0, 103.0, 150.0, 200.0, 300.0,
        307.0,
    ] {
        let a = 10f64.powf(exponent);
        assert_exact(
            Real::beta_reg(0.5, a, a),
            0.5,
            &format!("I_1/2(1e{exponent}, same) must be exactly 1/2"),
        );
    }
    // And where the sum itself overflows, which needs a shape within a half-decade of MAX.
    for &a in &[9e307f64, 1.5e308, f64::MAX] {
        assert!(
            !(a + a).is_finite(),
            "{a:e} was meant to overflow when doubled"
        );
        assert_exact(
            Real::beta_reg(0.5, a, a),
            0.5,
            &format!("I_1/2({a:e}, same) must be exactly 1/2 despite a+b overflowing"),
        );
    }
}

#[test]
fn an_overflowing_shape_sum_puts_its_point_mass_at_the_mean() {
    // The `a + b` overflow path has three arms — below the mean, above it, and exactly on it —
    // and the test above reaches only the third, because a symmetric shape puts the mean at
    // `0.5` and that is the `x` it asks about. Two mutants survived in precisely that blind
    // spot: swapping either comparison went unnoticed while `I_½(a, a)` stayed right.
    //
    // So this sweeps either side as well, and at an asymmetric ratio, where the mean is not at
    // the midpoint and getting the comparison backwards cannot coincidentally agree.
    for &(a, b) in &[
        (1.5e308f64, 1.5e308f64),
        (f64::MAX, f64::MAX),
        (1.5e308, 3.0e307),
        (3.0e307, 1.5e308),
    ] {
        assert!(!(a + b).is_finite(), "({a:e}, {b:e}) was meant to overflow");
        let mean = 1.0 / (1.0 + b / a);
        assert!(
            (0.0..=1.0).contains(&mean),
            "the ratio-form mean must stay a probability: {mean}"
        );
        for &x in &[0.0, 1e-9, 0.25, 0.5, 0.75, 1.0 - f64::EPSILON, 1.0] {
            let want = if x <= 0.0 {
                0.0 // the `x <= 0` guard in `beta_reg` takes this before the branch
            } else if x >= 1.0 {
                1.0
            } else if x < mean {
                0.0
            } else if x > mean {
                1.0
            } else {
                0.5
            };
            assert_exact(
                Real::beta_reg(x, a, b),
                want,
                &format!("I_{x}({a:e}, {b:e}) against a point mass at {mean}"),
            );
        }
        // And the arms are actually distinguishable at this ratio, or the loop proves nothing.
        assert!(
            mean > 1e-9 && mean < 1.0 - f64::EPSILON,
            "the sample must straddle the mean at ({a:e}, {b:e}), which sits at {mean}"
        );
    }
}

#[test]
fn beta_reg_resolves_the_side_of_an_extreme_mean() {
    let a = f64::MAX;
    let b = a.next_down();
    // The means straddle 1/2. The opposite tail is below 1e-270 in
    // each shape order by the one-sided variance bound.
    assert_eq!(Real::beta_reg(0.5, a, b), 0.0);
    assert_eq!(Real::beta_reg(0.5, b, a), 1.0);
    assert!(Real::beta_reg(f64::NAN, a, b).is_nan());

    let unit = 2.0f64.powi(1020);
    let (a, b) = (3.0 * unit, 15.0 * unit);
    // Binary64 rounds 1/6 down and 5/6 up. The adjacent value on
    // the other side of each fraction lies in the opposite tail.
    let x = 1.0f64 / 6.0;
    let y = 5.0f64 / 6.0;
    assert_eq!(Real::beta_reg(x, a, b), 0.0);
    assert_eq!(Real::beta_reg(x.next_up(), a, b), 1.0);
    assert_eq!(Real::beta_reg(y.next_down(), b, a), 0.0);
    assert_eq!(Real::beta_reg(y, b, a), 1.0);

    // The exact mean is representable for this asymmetric pair.
    assert_eq!(Real::beta_reg(0.25, 4.0 * unit, 12.0 * unit), 0.5);
    assert_eq!(Real::beta_reg(0.75, 12.0 * unit, 4.0 * unit), 0.5);

    // x = 1/2 + 2^-53 and b = a(1 - 4·2^-53), so x(a+b) - a
    // is negative even though the rounded product equals a.
    let a = 2.0f64.powi(1022);
    let b = f64::from_bits(a.to_bits() - 4);
    assert_eq!(Real::beta_reg(0.5f64.next_up(), a, b), 0.0);
}

#[test]
fn beta_reg_resolves_a_transition_finer_than_the_rounded_mean() {
    let a = 2.0f64.powi(104);
    let b = a.next_up();
    // Normal-limit references use the exact binary64 inputs and the
    // beta variance, evaluated with 100-digit Decimal arithmetic
    // before calling erfc. At these shapes skewness is below 1e-30.
    for (x, want) in [
        (0.5f64.next_down(), 0.5),
        (0.5, 0.760_249_938_906_523_3),
        (0.5f64.next_up(), 0.983_052_573_237_655_4),
    ] {
        assert_close(Real::beta_reg(x, a, b), want, 8e-16, "beta transition");
    }
    assert_close(
        Real::beta_reg(0.5, b, a),
        0.239_750_061_093_476_7,
        8e-16,
        "beta transition with reversed shapes",
    );
}

#[test]
fn beta_reg_resolves_the_transition_at_an_extreme_shape_ratio() {
    let a = 2.0f64.powi(104);
    let b = 3.0 * a * a;
    let x = (1.0 / a) / 3.0;
    // Binary64 rounds 1/3 down to (1 − 2^-54)/3. Thus b·x is
    // a − sqrt(a)/4 before rounding, and the beta variance gives
    // z = −1/4 to working precision. Φ(−1/4) is the normal limit.
    assert_close(
        Real::beta_reg(x, a, b),
        0.401_293_674_317_076_3,
        8e-16,
        "beta transition with widely separated shapes",
    );

    let a = 2.0f64.powi(200);
    let b = 3.0 * a * a;
    let x = (1.0 / a) / 3.0;
    assert_eq!(Real::beta_reg(x, a, b), 0.0);
    assert_eq!(Real::beta_reg(x.next_up(), a, b), 1.0);
}

#[test]
fn beta_reg_is_a_probability_and_monotone_at_every_large_shape() {
    // The invariant the overflow violated, stated before any question of accuracy — the same
    // shape of test as the gamma peer's `both_tails_stay_probabilities_at_every_shape`. Swept
    // over six decades of shape and three ratios, several deviations either side of the mean.
    for exponent in [7.0f64, 9.0, 20.0, 60.0, 150.0, 300.0] {
        for &ratio in &[1.0f64, 1e3, 1e-3] {
            let a = 10f64.powf(exponent);
            let b = a * ratio;
            if !(b.is_finite() && b.min(a) >= BETA_EDGEWORTH_MIN) {
                continue;
            }
            let n = a + b;
            let mean = a / n;
            let sd = (a / n).sqrt() * (b / n).sqrt() / (n + 1.0).sqrt();
            // Whether `f64` can even represent a point one deviation from the mean. `σ` falls
            // like `1/√a` while the spacing at the mean is fixed, so past `a ≈ 1e31` (where
            // `σ` drops under `ulp(½) = 1.1e-16`) the whole distribution lives inside one
            // representable step and the CDF is a *step function* in this format. That is the
            // format speaking, not the kernel, but it changes what the sweep below can claim —
            // and asserting the strong version anyway is how this test first failed.
            let spacing = Real::next_after(mean, f64::INFINITY) - mean;
            let resolvable = sd > spacing;
            let mut previous = -1.0f64;
            for step in -60..=60 {
                let x = mean + f64::from(step) * 0.1 * sd;
                if !(0.0..=1.0).contains(&x) {
                    continue;
                }
                let value = Real::beta_reg(x, a, b);
                assert!(
                    (0.0..=1.0).contains(&value),
                    "I_{x}(1e{exponent}, ratio {ratio:e}) = {value} is not a probability"
                );
                assert!(
                    value >= previous,
                    "I_x(1e{exponent}, ratio {ratio:e}) decreased at x = {x}: \
                     {previous} then {value}"
                );
                previous = value;
            }
            if resolvable {
                // The sweep has to actually cross the distribution, or it proves nothing: a
                // constant zero would satisfy every assertion above.
                assert!(
                    previous > 0.999,
                    "the sweep at 1e{exponent}, ratio {ratio:e} never reached the upper tail"
                );
            } else {
                // Unresolvable, so step in *ulps* instead and require the transition to be
                // sited at the mean rather than merely to exist somewhere.
                assert_exact(
                    Real::beta_reg(Real::next_after(mean, f64::NEG_INFINITY), a, b),
                    0.0,
                    &format!("one ulp below the mean at 1e{exponent}, ratio {ratio:e}"),
                );
                assert_exact(
                    Real::beta_reg(Real::next_after(mean, f64::INFINITY), a, b),
                    1.0,
                    &format!("one ulp above the mean at 1e{exponent}, ratio {ratio:e}"),
                );
            }
        }
    }
}

#[test]
fn the_reflection_identity_holds_inside_the_edgeworth_branch() {
    // `I_x(a,b) = 1 − I_{1−x}(b,a)` is exact, and the Edgeworth form satisfies it
    // analytically: swapping the shapes negates `z` and `γ₁` together, so the correction
    // changes sign with the tail it corrects. That makes this a test of the *implementation* —
    // in particular that the skewness keeps its sign — rather than of the approximation.
    //
    // Measured worst 7.0e-12, at (1e7, 1e9) where the mean sits closest to an endpoint.
    for &(a, b) in &[
        (1e7f64, 1e7f64),
        (1e7, 1e8),
        (1e8, 1e7),
        (1e7, 1e9),
        (2e7, 5e7),
    ] {
        let n = a + b;
        let sd = (a / n).sqrt() * (b / n).sqrt() / (n + 1.0).sqrt();
        for step in -40..=40 {
            let x = a / n + f64::from(step) * 0.1 * sd;
            if !(0.0..=1.0).contains(&x) {
                continue;
            }
            assert_close(
                Real::beta_reg(x, a, b),
                1.0 - Real::beta_reg(1.0 - x, b, a),
                5e-11,
                &format!("reflection at x = {x}, ({a:e}, {b:e})"),
            );
        }
    }
}

#[test]
fn the_shape_crossover_leaves_no_step() {
    // The continued fraction and the Edgeworth form share no code and no derivation — a
    // term-by-term Lentz recurrence against a normal CDF with a skewness correction — so
    // agreement at the join is the evidence that the join is in the right place. Checked at
    // three ratios, since the two routes have quite different dependence on it.
    //
    // Measured worst step 9.0e-9, which is both sides carrying their own ~6e-9: the fraction's
    // prefactor cancellation below, the expansion's truncation above. The kernel's threshold
    // derivation sets those equal here, and this is that prediction holding.
    for &z in &[-3.0f64, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0] {
        for &ratio in &[1.0f64, 10.0, 100.0] {
            let at = |a: f64| {
                let b = a * ratio;
                let n = a + b;
                let sd = (a / n).sqrt() * (b / n).sqrt() / (n + 1.0).sqrt();
                Real::beta_reg(a / n + z * sd, a, b)
            };
            let below = at(Real::next_after(BETA_EDGEWORTH_MIN, f64::NEG_INFINITY));
            let above = at(Real::next_after(BETA_EDGEWORTH_MIN, f64::INFINITY));
            assert_close(
                above,
                below,
                2e-8,
                &format!("the shape crossover steps at z = {z}, ratio {ratio}"),
            );
        }
    }
}

#[test]
fn a_degenerate_shape_ratio_still_gives_a_certainty() {
    // `f64::MAX` against `1e7` keeps `a + b` *finite* — adding `1e7` to `MAX` rounds back to
    // `MAX` — so this is not the overflowing-sum case but the underflowing-variance one: the
    // variance is `3e-610` and gone, while `σ` is `1.8e-305` and fine. The whole distribution
    // then sits inside one representable step, so every `x` is a certainty, and the question
    // is only whether it is the *right* certainty rather than a NaN.
    //
    // Which certainty is decided by the mean, not by the midpoint of `[0, 1]`, and the mean is
    // `5.6e-302` for `(1e7, MAX)` — not zero. `MIN_POSITIVE` is six decades *below* that, so
    // `I = 0` there is correct and an earlier version of this test was simply wrong to want 1.
    for &(a, b) in &[(f64::MAX, 1e7f64), (1e7, f64::MAX)] {
        let mean = a / (a + b);
        for &x in &[
            f64::MIN_POSITIVE,
            1e-300,
            1e-6,
            0.1,
            0.5,
            0.9,
            1.0 - f64::EPSILON,
        ] {
            let want = if x < mean {
                0.0
            } else if x > mean {
                1.0
            } else {
                0.5
            };
            assert_exact(
                Real::beta_reg(x, a, b),
                want,
                &format!("I_{x}({a:e}, {b:e}) against a point mass at {mean:e}"),
            );
        }
    }
    // The endpoints keep their exact values regardless of shape.
    for &(a, b) in &[(f64::MAX, 1e7f64), (1e7, f64::MAX), (f64::MAX, f64::MAX)] {
        assert_exact(Real::beta_reg(0.0, a, b), 0.0, "I_0 = 0");
        assert_exact(Real::beta_reg(1.0, a, b), 1.0, "I_1 = 1");
    }
}
