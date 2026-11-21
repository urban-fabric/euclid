//! Unit tests for the normal CDF, log-CDF and quantile.
//!
//! A child module so implementation plus tests stay inside the crate's 400-line budget. The
//! inverse complementary error function keeps its own tests in [`inverse`], because what they
//! reach is its route table and its subnormal halving rather than anything here.
//!
//! The log-CDF tail is the bulk of what is asserted: it is the one regime where the probability
//! underflows while its logarithm stays representable, so no comparison against `ndtr` reaches
//! it, and the quantile has to come back through the same door.

mod inverse;

use super::*;
use crate::constants::f64::HALF_LN_TAU;
use crate::math::normal::central_start;

fn close(a: f64, b: f64, tol: f64) {
    assert!(
        (a - b).abs() <= tol * b.abs().max(1.0),
        "{a:e} vs {b:e} (tol {tol:e})"
    );
}

/// The upper quantile needs the symmetric route, and `ndtri_lower` alone will not do near 1.
///
/// At `p = 1 - ulp(1)`, a direct solve through `ndtri_lower` returns about `8.1824`.
/// The symmetric route returns about `8.2095`.
///
/// The reason is the one `log_ndtr`'s upper branch exists for. Newton's residual is
/// `ln Phi(x) - ln p`, and near `x = 8.2` both terms are about `-1e-16`, so the residual is
/// a difference of two quantities that have each lost all but a couple of digits. The
/// symmetric route evaluates the same root as `-ndtri_lower(1 - p)`, where `1 - p` is
/// `1.1e-16` and the lower tail is perfectly conditioned.
///
/// Nothing in the suite had evaluated the quantile this close to one — the fixed tests stop
/// at `0.99` and the generated property draws `p <= 0.5` — which is why a mutation that
/// removes the symmetric branch entirely went unnoticed.
#[test]
fn the_upper_quantile_needs_the_symmetric_route_near_one() {
    // Each of these is within a few ulps of 1, where the direct solve degrades.
    for &p in &[
        1.0 - f64::EPSILON / 2.0,
        1.0 - f64::EPSILON,
        1.0 - 2.0 * f64::EPSILON,
        1.0 - 1e-15,
        1.0 - 1e-13,
        0.999_999_999_999,
    ] {
        let complement = 1.0 - p;
        let got = ndtri(p);
        // The lower tail is well conditioned, so this is the reference.
        let want = -ndtri(complement);
        assert!(
            got.is_finite() && got > 0.0,
            "ndtri({p}) = {got} must be a finite positive abscissa"
        );
        close(got, want, 1e-12);
        // And it really is the upper tail: Phi of it recovers p to the format's resolution.
        close(log_ndtr(got), p.ln(), 1e-9);
    }
    // Symmetrically at the bottom, where the direct solve is the right one.
    for &p in &[f64::EPSILON / 2.0, 1e-15, 1e-300] {
        close(ndtri(p), -ndtri(1.0 - p + p), 1.0);
        assert!(ndtri(p) < 0.0, "the lower quantile stays negative at {p:e}");
    }
}

#[test]
fn ndtr_hits_its_known_values() {
    close(ndtr(0.0), 0.5, 1e-15);
    close(ndtr(1.0), 0.841_344_746_068_543, 1e-14);
    close(ndtr(-1.0), 0.158_655_253_931_457, 1e-14);
    close(ndtr(1.959_963_984_540_054), 0.975, 1e-12);
    close(ndtr(-3.0), 1.349_898_031_630_095e-3, 1e-12);
}

#[test]
fn log_ndtr_stays_finite_where_the_naive_form_dies() {
    assert_eq!(ndtr(-40.0), 0.0, "Phi underflows here");
    assert!(ndtr(-40.0).ln().is_infinite(), "so its log is -inf");
    let v = log_ndtr(-40.0);
    assert!(v.is_finite() && v < 0.0, "log_ndtr(-40) = {v}");
    // ln Phi(x) ~ -x^2/2 - ln|x| - ln sqrt(2 pi)
    close(v, -800.0 - 40.0f64.ln() - HALF_LN_TAU, 1e-3);
    for &x in &[-100.0f64, -1e3, -1e6] {
        let got = log_ndtr(x);
        assert!(got.is_finite(), "log_ndtr({x}) = {got}");
        close(got, -0.5 * x * x - (-x).ln() - HALF_LN_TAU, 1e-6);
    }
}

#[test]
fn ndtri_round_trips_through_ndtr() {
    for i in 1..100 {
        let p = f64::from(i) * 0.01;
        close(ndtr(ndtri(p)), p, 1e-12);
    }
}

#[test]
fn ndtri_round_trips_deep_in_the_tail_through_the_log_cdf() {
    for &p in &[1e-20f64, 1e-50, 1e-100, 1e-200, 1e-300] {
        let x = ndtri(p);
        assert!(x.is_finite() && x < 0.0, "ndtri({p}) = {x}");
        close(log_ndtr(x), p.ln(), 1e-12);
    }
}

/// `central_start` must be a good seed too, now that it is reachable.
///
/// Four mutants lived in this expression while it was inline inside `ndtri_lower` and all
/// four survived: `2p` to `2 + p` and to `2 / p`, `- 1` to `+ 1`, and `/ (1/sqrt 2)` to
/// `* (1/sqrt 2)`. Extracting it is the entire fix — the law is the same one that killed
/// `tail_start_from_log`'s eleven, and it could not be applied to an expression.
///
/// Three of the four put `2p - 1` outside `[-1, 1]`, so `erf_inv` returns a non-finite
/// value and the first assertion catches them. The fourth scales by `1/sqrt 2` instead of
/// dividing, which at `p = 0.01` seeds `-1.163` where the root is `-2.326` — `0.54` off in
/// `ln p`, against the `0.073` the real seed never exceeds.
///
/// # The bound has two tiers because the seed genuinely degrades
///
/// Measured per decade across `[1e-15, 0.5)`, the relative error in `ln p` runs
/// `7.2e-15` near `p = 0.1`, `1.2e-7` at `1e-9`, `1.2e-3` at `1e-12` and `7.3e-2` at
/// `1e-15`. That is not a defect: `Phi^{-1}(p) = sqrt 2 erf^{-1}(2p - 1)` is exact as an
/// identity, but `2p - 1` approaches `-1` where `erf_inv` is steep, so the *argument* loses
/// its information long before the formula does. It is also exactly why `ndtri_lower` stops
/// using this seed at `1e-15` and switches to [`tail_start_from_log`].
///
/// So a single tolerance would have to be `0.1` to cover the boundary, which is loose enough
/// to admit a badly wrong seed in the central region where the real error is `1e-14`. Two
/// tiers keep both ends honest.
#[test]
fn the_central_seed_is_accurate_where_it_is_used() {
    let mut worst_overall = 0.0f64;
    let mut worst_central = 0.0f64;
    for i in 0..=2_000 {
        // `[1e-15, 0.5)`, the range `ndtri_lower` selects this seed on.
        let exponent = -15.0 + 14.7 * f64::from(i) / 2_000.0;
        let p = 10f64.powf(exponent);
        if !(p > 0.0 && p < 0.5) {
            continue;
        }
        let seed = central_start(p);
        assert!(
            seed.is_finite() && seed < 0.0,
            "the central seed must be finite and negative at p = {p:e}, got {seed}"
        );

        // Reference-free, as for the tail seed: `ln Phi` of a good seed is near `ln p`.
        let relative = ((log_ndtr(seed) - p.ln()) / p.ln()).abs();
        worst_overall = worst_overall.max(relative);
        assert!(
            relative < 0.25,
            "the central seed is {relative:e} off in ln p at p = {p:e}, which is not a seed"
        );
        // Away from its own boundary it is not merely a seed, it is the answer.
        if p >= 1e-9 {
            worst_central = worst_central.max(relative);
            assert!(
                relative < 1e-6,
                "the central seed is {relative:e} off in ln p at p = {p:e}"
            );
        }
    }
    // Keep the measured figures visible, so a degradation to just inside the bounds shows.
    assert!(
        worst_overall < 0.1 && worst_central < 3e-7,
        "the central seed degraded: worst {worst_overall:e}, central {worst_central:e}"
    );

    // The exact identity it encodes, at the lower quartile.
    assert!(
        (central_start(0.25_f64) - (-0.674_489_750_196_081_7)).abs() < 1e-12,
        "sqrt(2) erf_inv(2p - 1) at p = 1/4 is the lower quartile"
    );
    // And the boundary behaviour `ndtri_lower` relies on: below ~1e-16, `2p - 1` is -1
    // exactly and this must go non-finite so the fallback takes over.
    assert!(
        !central_start(1e-17_f64).is_finite(),
        "the central seed must go non-finite once 2p - 1 rounds to -1"
    );
}

/// Newton reaches a common root from either usable seed.
#[test]
fn the_solver_reaches_the_same_root_from_either_seed() {
    for i in 0..=400 {
        let exponent = -14.0 + 13.7 * f64::from(i) / 400.0;
        let p = 10f64.powf(exponent);
        if !(p > 0.0 && p < 0.5) {
            continue;
        }
        let central = central_start(p);
        let tail = tail_start_from_log(p.ln());
        assert!(
            central.is_finite() && tail.is_finite(),
            "both seeds must be usable at p = {p:e}"
        );

        let from_central = refine_lower(p.ln(), central);
        let from_tail = refine_lower(p.ln(), tail);
        let truth = ndtri(p);
        // Six steps from either one, and the same answer to full precision.
        close(from_central, truth, 1e-13);
        close(from_tail, truth, 1e-13);
        assert!(
            (from_central - from_tail).abs() <= 1e-13 * truth.abs().max(1.0),
            "the two seeds must converge together at p = {p:e}: \
             {from_central:e} against {from_tail:e}"
        );
    }
}

/// `tail_start_from_log` must be a *good* seed, which nothing checked because Newton hides it.
///
/// Six Newton steps on `ln Phi(x) = ln p` can converge from a rough guess in the
/// right region. Checking the final quantile can therefore miss an inaccurate seed.
///
/// A generated property was tried first and killed none of them, which is the useful part of
/// the record: it swept `p` from `1e-300` in log space with a `1e-14` bound on the round
/// trip and a coverage floor on this very branch, and the survivor set came back **byte for
/// byte identical**. Widening the domain does not reach a seed; the iteration really is that
/// robust. The only way in is to test the seed *as a function*, which is what this does.
///
/// # The claim being pinned is the docstring's own
///
/// [`tail_start_from_log`] says its inversion is "good to a few digits — ample for Newton to finish
/// from". That is a testable statement and nothing tested it. Measured over 3,000 points
/// across `[1e-300, 1e-15]`: worst relative error `1.07e-3` in `ln p`, worst absolute
/// distance from the root `4.6e-3`. The bounds below are an order looser than both, which is
/// still three orders tighter than any of the mutations — the `/` for `*` one puts the seed
/// at `-2.34` where the root is `-7.94`.
///
/// The primary check needs no reference at all: a seed is good when `ln Phi` of it is close
/// to `ln p`, which is the equation being solved.
#[test]
fn the_tail_seed_is_close_enough_for_newton_to_finish_from() {
    let mut worst_relative = 0.0f64;
    let mut worst_distance = 0.0f64;
    for i in 0..=2_000 {
        let exponent = -300.0 + 285.0 * f64::from(i) / 2_000.0;
        let p = 10f64.powf(exponent);
        let seed = tail_start_from_log(p.ln());
        let ln_p = p.ln();

        assert!(
            seed.is_finite() && seed < 0.0,
            "the seed must be a finite lower-tail abscissa at p = {p:e}, got {seed}"
        );

        // Reference-free: the seed nearly solves the equation the iteration solves.
        let relative = ((log_ndtr(seed) - ln_p) / ln_p).abs();
        worst_relative = worst_relative.max(relative);
        assert!(
            relative < 1e-2,
            "the seed is {relative:e} off in ln p at p = {p:e}, which is not a few digits"
        );

        // And against the converged root, which the mutations move by whole units.
        let distance = (seed - ndtri(p)).abs();
        worst_distance = worst_distance.max(distance);
        assert!(
            distance < 5e-2,
            "the seed sits {distance:e} from the root at p = {p:e}"
        );
    }
    // Keep the measured figures visible: if a refactor degrades the seed to just inside the
    // bounds above, these say so.
    assert!(
        worst_relative < 2e-3 && worst_distance < 1e-2,
        "the seed degraded: worst relative {worst_relative:e}, worst distance {worst_distance:e}"
    );

    // Monotone in `p`, since `Phi` is: a seed that crossed over would start Newton on the
    // wrong side of the root.
    let mut previous = f64::NEG_INFINITY;
    for i in 0..=200 {
        let exponent = -300.0 + 285.0 * f64::from(i) / 200.0;
        let seed = tail_start_from_log(10f64.powf(exponent).ln());
        assert!(
            seed > previous,
            "the seed must increase with p, at 1e{exponent}"
        );
        previous = seed;
    }
}
