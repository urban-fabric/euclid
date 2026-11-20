//! Unit tests for the gamma quantile.
//!
//! Beside [`super`] rather than with the shared solver, because what they reach is this
//! module's: the seed, both brackets, the tail switch and the route table. The seed is checked
//! **in probability** rather than by its value — that is what a seed is for, since the bracket
//! search pays one CDF evaluation per octave between the seed and the root.

use super::*;
use crate::kernels::gamma::gamma_lr;

fn close(a: f64, b: f64, tol: f64) {
    assert!(
        (a - b).abs() <= tol * b.abs().max(1.0),
        "{a:e} vs {b:e} (tol {tol:e})"
    );
}

/// The deviation bracket must enclose the root, decline when it cannot help, and never be
/// trusted on a seed that missed.
///
/// The second and third are the ones that matter. Declining keeps the two probes off every
/// shape below the crossover, which is why the measured call counts at `a ≤ 1e2` are
/// unchanged to the last digit. Refusing a bracket that does not straddle `p` is what makes
/// the optimisation safe at all: the geometric expansion assumes nothing about the seed, so
/// falling back to it is always correct.
#[test]
fn the_deviation_bracket_encloses_the_root_or_declines() {
    // Identity CDF, so the root is `p` itself and the arithmetic is checkable by hand.
    let identity = |x: f64| x;

    // Declines where eight standard deviations is not inside the octave: `√a` is not small
    // against the seed.
    assert_eq!(deviation_bracket(&identity, 50.0, 50.0, 100.0), None);
    // Accepts at a shape where it is: half-width is `8·√1e6 = 8e3` against a seed of `1e6`.
    let bracket = deviation_bracket(&identity, 1e6, 1e6, 1e6);
    assert_eq!(bracket, Some((1e6 - 8e3, 1e6 + 8e3)));
    assert!(bracket.is_some_and(|(lo, hi)| lo <= 1e6 && 1e6 <= hi));

    // Declines when the seed is far enough off that the root falls outside the half-width,
    // which is the case the geometric expansion exists to handle.
    assert_eq!(deviation_bracket(&identity, 1e6, 2e6, 1e6), None);
    assert_eq!(deviation_bracket(&identity, 1e6, 1e6 + 9e3, 1e6), None);

    // A real CDF at a shape past the crossover: the accepted bracket must contain the root
    // the solver goes on to find.
    let a = 1e6_f64;
    for p in [0.05_f64, 0.3, 0.5, 0.95] {
        let cdf = |x: f64| crate::kernels::gamma::gamma_lr(a, x);
        let seed = gamma_seed(p, a);
        let bracket = deviation_bracket(&cdf, p, seed, a);
        assert!(
            bracket.is_some(),
            "declined at p={p}, where Wilson-Hilferty is accurate to O(1/a)"
        );
        let root = gamma_lr_inv(p, a);
        assert!(
            bracket.is_some_and(|(lo, hi)| lo <= root && root <= hi),
            "root {root} outside {bracket:?}"
        );
    }
}

#[test]
fn bracket_expansion_encloses_an_exact_root_in_either_direction() {
    for exponent in [-1_000, -100, -3, 0, 3, 100, 1_000] {
        let root = 2.0f64.powi(exponent);
        let bracket = expand_gamma_bracket(&|x| x, root, 1.0);
        assert!(bracket.is_some_and(|(lo, hi)| lo <= root && root <= hi));
    }
    assert_eq!(expand_gamma_bracket(&|x| x, 1.0, 1.0), Some((0.5, 1.0)));
    assert_eq!(expand_gamma_bracket(&|_| 0.0, 0.5, f64::MAX), None);
}

#[test]
fn gamma_seed_uses_the_transform_when_it_is_inside_the_support() {
    close(gamma_seed(0.5, 4.0), 4.0 * (35.0_f64 / 36.0).powi(3), 3e-16);
}

/// Below the support, the seed is checked **in probability** rather than by its value.
///
/// That is what a seed is for: `expand_gamma_bracket` pays one CDF evaluation per octave
/// between the seed and the root, so what matters is `P(a, seed)` against `p`, not the seed
/// itself. These assertions are what the previous mean-based fallback could not satisfy —
/// it returned `a`, landing at `P ≈ 0.96` to `0.999` *whatever `p` was*, because a
/// small-shape gamma keeps essentially all its mass many orders below its mean.
///
/// The bound is `1e-3` relative because `P(a, x) ≈ x^a/Γ(1+a)` drops a correction of order
/// `a`, so the seed's quality degrades as the shape rises towards where Wilson–Hilferty
/// takes over: measured `1.2e-4` at `a = 1/9`, the largest shape that reaches this branch,
/// against `6e-8` or better at `a ≤ 1e-2`. Even the loosest of those is within an octave of
/// the root, which is what the bracket search charges for. For scale, the mean-based
/// fallback this replaced missed by `0.92` relative at `a = 1e-3`.
///
/// `a = 1e-4` is deliberately absent: there the root underflows (`0.5^10000`), so the seed
/// floors at `MIN_POSITIVE` and no seed can land on `p`. That case is covered below.
#[test]
fn the_small_shape_seed_lands_on_its_own_probability() {
    for (p, a) in [
        (0.5, 1.0 / 9.0),
        (0.5, 1e-3),
        (0.25, 1e-2),
        (0.75, 1e-2),
        (0.1, 0.05),
    ] {
        let seed = gamma_seed(p, a);
        let at_seed = crate::kernels::gamma::gamma_lr(a, seed);
        assert!(
            (at_seed - p).abs() <= 1e-3 * p,
            "seed for p={p}, a={a:e} is {seed:e}, which sits at P = {at_seed:e}"
        );
    }
}

/// A root below the subnormal floor still has to produce a seed the expansion can move.
///
/// `exp` saturates to zero there, and a zero seed cannot be halved or doubled towards
/// anything — so the floor is `MIN_POSITIVE`, the smallest value the search can still work
/// from.
#[test]
fn an_unrepresentable_root_floors_the_seed_rather_than_returning_zero() {
    for a in [0.05, 1e-12] {
        let seed = gamma_seed(f64::MIN_POSITIVE, a);
        assert_eq!(seed, f64::MIN_POSITIVE, "a={a:e}");
    }
}

#[test]
fn gamma_quantile_round_trips_over_a_wide_parameter_range() {
    for &a in &[0.1f64, 0.5, 1.0, 2.5, 10.0, 100.0, 1e4] {
        for i in 1..40 {
            let p = f64::from(i) * 0.025;
            let x = gamma_lr_inv(p, a);
            assert!(x.is_finite() && x >= 0.0, "gamma_lr_inv({p},{a}) = {x}");
            close(gamma_lr(a, x), p, 1e-10);
        }
    }
}

#[test]
fn gamma_quantile_matches_the_exponential_closed_form() {
    // For a = 1 the CDF is 1 - e^-x, so the quantile is -ln(1-p).
    for i in 1..50 {
        let p = f64::from(i) * 0.02;
        close(gamma_lr_inv(p, 1.0), -(1.0 - p).ln(), 1e-10);
    }
}

/// The two-probe bracket, which is verified before it is used.
#[test]
fn the_deviation_bracket_is_rejected_unless_it_straddles_and_narrows() {
    let scaled = |x: f64| x / 200.0;
    // `half_width = 8√a` is `16` at `a = 4`, which beats an octave only above `seed = 64`.
    assert_eq!(
        deviation_bracket(&scaled, 0.5, 100.0, 4.0),
        Some((84.0, 116.0))
    );
    // Not narrower than the octave the expansion gives for free, so the two probes are skipped.
    assert_eq!(deviation_bracket(&scaled, 0.5, 50.0, 4.0), None);
    assert_eq!(deviation_bracket(&scaled, 0.5, 64.0, 4.0), None);
    // One step above the octave threshold it is worth taking, and the bracket is `seed ± 16`.
    // `p` has to be inside `[0.24, 0.40]` there, which is the straddle the probes check for.
    let narrowed = deviation_bracket(&scaled, 0.3, 64.0_f64.next_up(), 4.0);
    assert!(narrowed.is_some(), "above the threshold the probes pay off");
    if let Some((lo, hi)) = narrowed {
        assert!(
            (lo - 48.0).abs() < 1e-9 && (hi - 80.0).abs() < 1e-9,
            "{lo}..{hi}"
        );
    }
    // Straddles on neither side: a seed too low for `p` fails the upper probe, and vice versa.
    assert_eq!(deviation_bracket(&scaled, 0.9, 100.0, 4.0), None);
    assert_eq!(deviation_bracket(&scaled, 0.1, 100.0, 4.0), None);
    // A non-finite half width — an infinite shape — is rejected rather than bracketing to `inf`.
    assert_eq!(deviation_bracket(&scaled, 0.5, 100.0, f64::INFINITY), None);
    // Exactly on both probes is a straddle: the comparisons admit equality.
    let edge = |x: f64| if x <= 84.0 { 0.25 } else { 0.75 };
    assert_eq!(
        deviation_bracket(&edge, 0.25, 100.0, 4.0),
        Some((84.0, 116.0))
    );
    assert_eq!(
        deviation_bracket(&edge, 0.75, 100.0, 4.0),
        Some((84.0, 116.0))
    );
}

/// Both guards on the two-probe bracket, at the exact point each turns over.
#[test]
fn the_deviation_bracket_guard_is_strict_and_scales_the_seed() {
    let scaled = |x: f64| x / 200.0;
    // Exactly an octave's worth: `8√4 = 16` against `0.25·64 = 16`. The comparison is strict, so
    // this declines — the two probes would buy nothing the expansion does not give for free.
    // `p = 0.3` is inside `[cdf(48), cdf(80)] = [0.24, 0.40]`, so a widened comparison would
    // accept here and return a bracket. That is the only `p` at which the two differ.
    assert_eq!(deviation_bracket(&scaled, 0.3, 64.0, 4.0), None);
    // And the seed is *scaled* by a quarter, not offset by it: at `a = 16` the half-width is
    // `32`, which exceeds `0.25·100 = 25` and so declines, while `0.25 + 100` would admit it.
    assert_eq!(deviation_bracket(&scaled, 0.5, 100.0, 16.0), None);
    assert_eq!(8.0 * 16.0_f64.sqrt(), 32.0);
}

/// The gamma quantile's five routes, asserted as decisions.
///
/// Three of the five return a value no solve produces, and one — [`GammaQuantileRoute::AtZero`]
/// — returns the same `0.0` a sufficiently small interior `p` would converge to, so the decision
/// is what separates them rather than the root.
#[test]
fn the_gamma_quantile_routes_cover_the_probability_range() {
    use super::GammaQuantileRoute::{AtInfinity, AtZero, RoundsToShape, Solve, Undefined};

    // Shapes off the domain, and every `p` outside `[0, 1]` including `NaN`.
    for (p, a) in [
        (0.5, 0.0),
        (0.5, -1.0),
        (0.5, f64::NAN),
        (0.5, f64::INFINITY),
        (f64::NAN, 2.0),
        ((-0.0_f64).next_down(), 2.0),
        (1.0_f64.next_up(), 2.0),
        (-1.0, 2.0),
        (2.0, 2.0),
    ] {
        assert_eq!(
            super::GammaQuantileRoute::for_arguments(p, a),
            Undefined,
            "at p = {p:e}, a = {a:e}"
        );
    }
    // Both endpoints of the probability range.
    assert_eq!(super::GammaQuantileRoute::for_arguments(0.0, 2.0), AtZero);
    assert_eq!(
        super::GammaQuantileRoute::for_arguments(1.0, 2.0),
        AtInfinity
    );
    // Interior probabilities at an ordinary shape solve; at a shape whose spacing swallows the
    // whole distribution they round to it instead.
    //
    // The snap needs `ulp(a) ≥ 128√a`, which for `a = 2^k` is `k − 52 ≥ 7 + k/2`, so `k ≥ 118`.
    // `2^110` is deliberately *below* that — `residual_tolerance`'s own documentation turns on
    // the `1e-4` quantile still being distinguishable from the mean there — so it still solves.
    for p in [f64::MIN_POSITIVE, 1e-12, 0.25, 0.5, 0.75, 1.0 - 1e-12] {
        assert_eq!(super::GammaQuantileRoute::for_arguments(p, 2.0), Solve);
        assert_eq!(
            super::GammaQuantileRoute::for_arguments(p, 2f64.powi(110)),
            Solve
        );
        assert_eq!(
            super::GammaQuantileRoute::for_arguments(p, 2f64.powi(120)),
            RoundsToShape
        );
    }
}

/// The two tails, and the fact that they solve the same root through different equations.
#[test]
fn the_gamma_tail_switches_at_the_median_and_agrees_with_itself() {
    use super::GammaTail::{Lower, Upper};

    // Half itself is the lower tail: the comparison is strict.
    assert_eq!(super::GammaTail::for_probability(0.5), Lower);
    assert_eq!(super::GammaTail::for_probability(0.5_f64.next_up()), Upper);
    assert_eq!(
        super::GammaTail::for_probability(0.5_f64.next_down()),
        Lower
    );
    assert_eq!(super::GammaTail::for_probability(0.0), Lower);
    assert_eq!(super::GammaTail::for_probability(1.0), Upper);

    // The target moves with the arm, and only for the upper one.
    assert_eq!(Lower.target(0.3), 0.3);
    assert_eq!(Upper.target(0.75), -0.25);

    // `P = 1 − Q`, so the two residuals agree at every `x`: `cdf_lower − p` and
    // `cdf_upper − (p − 1)` are the same number to rounding. That identity is what makes the
    // switch free, and it is the only thing tying the two arms together.
    let a = 3.0;
    for x in [0.5_f64, 1.0, 3.0, 7.0, 20.0] {
        let p = 0.75;
        let lower = Lower.cdf(a, x) - Lower.target(p);
        let upper = Upper.cdf(a, x) - Upper.target(p);
        assert!(
            (lower - upper).abs() <= 1e-15,
            "at x = {x}: {lower} vs {upper}"
        );
    }
    // And the arms are genuinely different functions of `x`, so the pairing above is a
    // constraint rather than a tautology.
    assert!(Lower.cdf(a, 3.0) > 0.0);
    assert!(Upper.cdf(a, 3.0) < 0.0);
}
