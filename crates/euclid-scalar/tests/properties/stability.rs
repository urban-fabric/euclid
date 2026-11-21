//! Shared helpers for the stability suites.
//!
//! Split by sub-topic, because one file carrying all of it went past this crate's 400-line
//! budget:
//!
//! - [`conditioned`] — error budgets set by `κ(x)·u` rather than by a literal, against closed
//!   forms and cross-family identities; and the six relations this suite had never used.
//! - [`recurrences`] — shift identities as metamorphic references: the same kernel at another
//!   argument, combined exactly, so no oracle and no invented tolerance.
//! - [`seams`] — no kernel may jump at an algorithm crossover. Forty-five kernels, each at the
//!   generic threshold list plus the constants its own source names.
//! - [`wiggle`] — ulp-level monotonicity: which kernels have none, and measured budgets for the
//!   ones that do.
//! - [`range`] — the contract a kernel owes over the *whole format*: no `NaN` where the value is
//!   real, no infinity where it is representable, and the documented bounds.
//! - [`log_domain`] — the same contract for the log-domain and logistic family, which `range`
//!   omitted entirely. These are the kernels that exist *because* the naive composition
//!   saturates, so "finite at every finite argument" is their specification rather than a
//!   quality metric, and it is asserted against the `std` forms they replace.
//! - [`limits`] — every kernel against its exact leading term as the argument approaches zero,
//!   which is an *analytic* reference and so needs no wider float.
//! - [`asymptotics`] — the mirror of [`limits`] at the other end of the format. Not the same test
//!   twice: a kernel's small- and large-argument branches are different code, so `limits`
//!   exercises the series route and this one the asymptotic route, and neither sees the other.
//! - [`log_pairs`] — the *two*-argument members of that family, on a `2,476 × 2,476` grid. Their
//!   failure mode lives in the pair rather than either argument, so a one-argument sweep cannot
//!   reach it: combining two numbers whose exponentials would each overflow is the entire job.
//! - [`f32_range`] — the `NaN`-and-bounds contract for **fifty-four** `f32` kernels, which is
//!   more of the narrow surface than any other file reaches. None of the numbers carry over from
//!   [`range`]: `f32` stops at `3.4e38` and its `EPSILON` is `1.19e-7`, so every boundary moves.
//! - [`f32_saturation`] — where those kernels reach the edge of the format, asserted as formulas
//!   in `f32`'s own parameters rather than as constants, plus a regression pin for the `I₀ᵉ(1e38)`
//!   overflow this crate once shipped.
//! - [`f32_exhaustive`] — every `f32` around each kernel's overflow and underflow thresholds,
//!   against the wide peer's *classification*. The one file here that found real defects: four
//!   premature saturations, all invisible to [`f32_range`]'s 628 probes.
//! - [`oscillatory`] — `J` and `Y` above `1e8`, where the phase is unrecoverable and every
//!   pointwise test gives out. Uses the phase-free modulus `J² + Y²`, and first measures that the
//!   `J`/`Y` Wronskian cannot stand in for it.
//! - [`scaled_siblings`] — each plain kernel's saturation judged by its own scaled sibling, which
//!   records the factor it divides out and so gives `ln|value|` free. The only oracle here that
//!   needs no reference at all and works identically in both precisions.
//! - [`narrow_models`] — the three `f32` kernels whose allowance is a *function* of the argument
//!   rather than a constant: `ndtr`'s `x²`, `rgamma`'s `|lnΓ|`, `cos_int`'s flat absolute error.
//!   All three topped a plain ulp ranking and none is a defect; the file records why.
//! - [`symmetry_exact`] — every odd kernel negates and every even one repeats, at **zero ulps**.
//!   The cheapest oracle here and the only one asserted exactly; it found `atanh` delegating to a
//!   `std` implementation that is not odd, in both precisions.
//! - [`exact_values`] — closed forms at exactly-representable arguments, where the reference
//!   needs no construction at all, plus the `±∞` and `NaN` limits that [`range`]'s finite probe
//!   set filters out. Found `lnΓ(1)` returning `−8.88e-16` for an exact zero.
//! - [`cross_precision`] — the `f32` kernels against their wide peer rounded to `f32`, which is
//!   the only bound that does not charge a narrow kernel for being narrow.
//! - [`inverses`] — the `f64` inverses against their own backward error, which is the only
//!   accuracy statement available for them without a wider float.
//! - `round_trip` — `erf_inv(erf(x))` against its two-stage forward-stability bound. Left here
//!   rather than moved, because it owns a budget baseline and a regression tape.
//!
//! What the suite is for, and why a pointwise accuracy test cannot replace it: both a seam and a
//! wiggle are about the *relationship between neighbouring* results rather than any single one.
//! A seam is a discontinuity at a crossover, so every value on one side is systematically off
//! while each looks plausible alone; wiggle is non-monotonicity inside tolerance, which breaks a
//! root-find built on top. See each child for what it found.

use euclid_scalar::Real;
use euclid_stability::numeric::{LIKELY_THRESHOLDS, hunt_seams, next_up, scan_monotone};

/// A named kernel with a scan start and whether it should be increasing there.
type MonotoneCase = (&'static str, fn(f64) -> f64, f64, bool);

/// Never — the kernel must be real-valued everywhere in its domain.
///
/// Shared, because "`NaN` is never the right answer here" is the common case: it is every row of
/// [`range`] bar three, and every row of [`log_domain`].
fn never(_: f64) -> bool {
    false
}

/// Inputs spanning the whole format: both signs, every decade, the subnormals, the extremes.
///
/// Shared by [`range`] and [`log_domain`], which both need the *whole format* rather than a
/// comfortable interval — that is the entire point of a range contract, and a second copy of this
/// list would be a second thing to keep in step.
fn whole_format_probes() -> Vec<f64> {
    let mut v: Vec<f64> = vec![
        0.0,
        -0.0,
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
        f64::from_bits(1),
        -f64::from_bits(1),
        f64::from_bits(2),
        f64::MAX,
        f64::MIN,
        1.0 - f64::EPSILON / 2.0,
        1.0 + f64::EPSILON,
        -1.0 + f64::EPSILON / 2.0,
    ];
    let mut e = -323i32;
    while e <= 308 {
        let decade = 10f64.powi(e);
        if decade.is_finite() && decade > 0.0 {
            // A non-integer multiplier matters: `1e3` and `3e3` are integers, so a sweep built
            // only from them tests `Γ` and `ψ` exclusively at their poles and never between.
            for scale in [1.0, 1.5, 3.0, 7.25] {
                let x = decade * scale;
                if x.is_finite() {
                    v.push(x);
                    v.push(-x);
                }
            }
        }
        e += 1;
    }
    v
}

/// The same sweep in `f32`: both signs, every decade, the subnormals, the extremes.
///
/// Not [`whole_format_probes`] cast down, which would be a different and much weaker test — the
/// `f64` list runs to `1e308`, and casting that to `f32` gives infinity, so three quarters of it
/// would be discarded as non-finite before the kernel saw it. `f32`'s own format has to be swept
/// on its own terms: `MAX` is `3.4e38`, the smallest subnormal is `1.4e-45`, and `EPSILON` is
/// `1.19e-7`, so every boundary worth probing sits somewhere different.
fn narrow_format_probes() -> Vec<f32> {
    let mut v: Vec<f32> = vec![
        0.0,
        -0.0,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        f32::from_bits(1),
        -f32::from_bits(1),
        f32::from_bits(2),
        f32::MAX,
        f32::MIN,
        1.0 - f32::EPSILON / 2.0,
        1.0 + f32::EPSILON,
        -1.0 + f32::EPSILON / 2.0,
        1.0,
        -1.0,
    ];
    let mut e = -45i32;
    while e <= 38 {
        let decade = 10f32.powi(e);
        if decade.is_finite() && decade > 0.0 {
            // Non-integer multipliers for the same reason as the wide list: `Γ` and `ψ` have
            // their poles at the negative integers, and a sweep built only from `±10^e` would
            // test them nowhere else.
            for scale in [1.0f32, 1.5, 3.0, 7.25] {
                let x = decade * scale;
                if x.is_finite() {
                    v.push(x);
                    v.push(-x);
                }
            }
        }
        e += 1;
    }
    v
}

/// Consecutive representable inputs scanned by each monotonicity test.
const STEPS: usize = 50_000;

/// Count inverting adjacent pairs and the worst inversion, in ulps of the output.
fn characterise(f: impl Fn(f64) -> f64, start: f64, increasing: bool) -> (usize, f64) {
    let mut x = start;
    let mut here = f(x);
    let mut inversions = 0usize;
    let mut worst = 0.0_f64;

    for _ in 0..STEPS {
        let next = next_up(x);
        let there = f(next);
        if here.is_finite() && there.is_finite() {
            let backwards = if increasing {
                here - there
            } else {
                there - here
            };
            if backwards > 0.0 {
                inversions += 1;
                let spacing = next_up(here.abs()) - here.abs();
                let ulps = backwards / spacing;
                if ulps > worst {
                    worst = ulps;
                }
            }
        }
        x = next;
        here = there;
    }
    (inversions, worst)
}

/// Assert a kernel's measured wiggle against its recorded budget.
fn assert_budget(
    name: &str,
    f: impl Fn(f64) -> f64,
    start: f64,
    increasing: bool,
    max_inversions: usize,
    max_ulps: f64,
) {
    let (inversions, worst) = characterise(f, start, increasing);
    assert!(
        inversions <= max_inversions,
        "{name} near {start} inverted on {inversions} of {STEPS} adjacent pairs, \
         budget {max_inversions}. The kernel got less monotone than when this budget was set."
    );
    assert!(
        worst <= max_ulps,
        "{name} near {start} had a {worst}-ulp inversion, budget {max_ulps}"
    );
}

// ===== Forward stability ==================================================

/// The round trip `erf_inv(erf(x))` must be forward stable — and getting that assertion right
/// took two attempts, which is the interesting part.
///
/// The first version bounded the error by the conditioning of the *composite map*. That is 1,
/// because the composite is the identity, and the test failed at `x = 1.76` by a factor of five:
/// the round trip's relative error there is about `3.4e-14`, roughly eleven times round-off.
///
/// The kernels are not unstable. The error enters the **intermediate**. `erf` saturates, so at
/// `x = 1.76` its derivative is `0.05`, and a relative error of `ε` in `erf(x)` becomes an
/// absolute error that `erf_inv` amplifies about twentyfold. The two stages' condition numbers
/// multiply to 1 exactly because they are inverses — which cancels the amplification out of the
/// bound instead of describing it.
///
/// `two_stage_error_bound` keeps both terms, `(κ_outer·κ_inner + κ_outer)·u`, where the second
/// is the inner stage's own rounding amplified by the outer. That comes to about 12 here, the
/// round trip fits inside it, and the computation is forward stable after all.
///
/// The generalisable lesson, and the reason this test is worth its length: **do not compose
/// through a saturating intermediate and expect the composite's conditioning to describe the
/// result.**
///
/// # Why this is a search with a budget rather than a grid
///
/// It began as a loop over 200 evenly spaced arguments, which found the `x = 1.76` case above and
/// was how the two-stage bound got written. Two things were wrong with it, and both are the
/// reason the engine grew the features this now uses.
///
/// A grid finds the worst case *on the grid*. The failure is a thin region — the round trip is
/// benign almost everywhere and bad where `erf` saturates — so whether the grid lands on it is
/// luck, and a 200-point grid over `(0, 2)` has a 0.01 resolution that could easily have stepped
/// over it. `target` replaces that with a hill climb towards the worst ratio it can construct,
/// which is the same question asked properly.
///
/// And the grid *discarded* the answer. `worst_ratio` was computed, compared against 1.0, and
/// dropped on the floor — so the suite could only ever notice the round trip becoming
/// *unstable*, never it becoming twice as inaccurate while still inside its bound. The margin is
/// around 0.5 today; a change taking it to 0.9 is a real loss of accuracy that the assertion
/// cannot see and `.budget()` can.
#[path = "stability/asymptotics.rs"]
mod asymptotics;
#[path = "stability/conditioned.rs"]
mod conditioned;
#[path = "stability/cross_precision.rs"]
mod cross_precision;
#[path = "stability/exact_values.rs"]
mod exact_values;
#[path = "stability/f32_exhaustive.rs"]
mod f32_exhaustive;
#[path = "stability/f32_range.rs"]
mod f32_range;
#[path = "stability/f32_saturation.rs"]
mod f32_saturation;
#[path = "stability/inverses.rs"]
mod inverses;
#[path = "stability/limits.rs"]
mod limits;
#[path = "stability/log_domain.rs"]
mod log_domain;
#[path = "stability/log_pairs.rs"]
mod log_pairs;
#[path = "stability/narrow_models.rs"]
mod narrow_models;
#[path = "stability/oscillatory.rs"]
mod oscillatory;
#[path = "stability/range.rs"]
mod range;
#[path = "stability/recurrences.rs"]
mod recurrences;
#[path = "stability/scaled_siblings.rs"]
mod scaled_siblings;
#[path = "stability/seams.rs"]
mod seams;
#[path = "stability/symmetry_exact.rs"]
mod symmetry_exact;
#[path = "stability/wiggle.rs"]
mod wiggle;

mod round_trip {
    use euclid_scalar::Real;
    use euclid_stability::numeric::{condition_at, two_stage_error_bound};
    use euclid_stability::{Config, check, classify, generate, property, target};

    /// How much of its admissible two-stage error the round trip used at `x`. `1.0` is exactly at
    /// the bound, which is the ratio form a budget baseline wants: comparable across inputs, and
    /// continuous in `x` so the climb has a gradient to follow.
    fn margin(x: f64) -> Option<f64> {
        let round_trip = Real::erf_inv(Real::erf(x));
        if !round_trip.is_finite() {
            return None;
        }
        let observed = ((round_trip - x) / x).abs();
        let bound = two_stage_error_bound(Real::erf, Real::erf_inv, x, 64.0)?;
        (bound > 0.0).then_some(observed / bound)
    }

    property! {
        #![config(
            Config::default()
                .cases(600)
                .regressions("tests/regressions/erf-round-trip.txt")
                .budget("tests/budgets.txt")
                .require("saturating", 5.0)
        )]

        /// The round trip stays inside its two-stage bound, and no less accurate than today.
        ///
        /// Two assertions in one property, deliberately. `check!` is the **contract** — forward
        /// stable, which is what a caller may rely on and what would still be true if the margin
        /// doubled. The budget is the **measurement** — and today it is 0.5 — which is what
        /// notices the doubling. Neither subsumes the other.
        fn the_erf_round_trip_is_forward_stable(x in generate::float::in_range(0.01, 2.0)) {
            let Some(margin) = margin(x) else {
                return Ok(());
            };

            // `erf` saturates above about 1.5, which is where the intermediate's rounding gets
            // amplified and the only region this property is really about. Required, because a
            // run that never got there would report a clean pass over the benign part.
            classify!("saturating", x > 1.5);
            classify!("benign", x <= 1.5);

            target(margin);

            let composite = condition_at(|v| Real::erf_inv(Real::erf(v)), x).unwrap_or(f64::NAN);
            let inner = condition_at(Real::erf, x).unwrap_or(f64::NAN);
            check!(
                margin <= 1.0,
                "the erf round trip exceeded its two-stage forward-stability bound by \
                 {margin:.2}x at x = {x}. The composite's condition number is {composite:e} \
                 and erf's own is {inner:e}, so if the composite looks benign the \
                 amplification is in the intermediate."
            );
        }
    }
}
