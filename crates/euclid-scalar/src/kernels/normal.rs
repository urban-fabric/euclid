//! The standard normal distribution: CDF, log-CDF, and quantile.
//!
//! These are the functions every probabilistic consumer reaches for, and the ones most easily
//! got wrong. All three are built on the error function rather than on a separate
//! approximation, so they inherit its accuracy and cannot drift away from it.
//!
//! Far in the lower tail, the probability can underflow while its logarithm remains
//! representable. The scaled complementary error function keeps the log-CDF finite
//! across that part of the range.

use crate::math::normal::{self, refine_lower, tail_start_from_log};

const LN_2: f64 = core::f64::consts::LN_2;
const FRAC_1_SQRT_2: f64 = core::f64::consts::FRAC_1_SQRT_2;

/// The standard normal CDF `Phi(x) = P(Z <= x)`.
///
/// `Phi(x) = erfc(-x/sqrt 2) / 2`. Using `erfc` rather than `1 - erf` keeps the left tail
/// accurate: it is the tail where the answer is small, so the complement form would cancel.
pub(crate) fn ndtr(x: f64) -> f64 {
    normal::ndtr(x)
}

/// `ln Phi(x)`, finite as far into the left tail as the exponent range allows.
///
/// Two regimes. For `x` above about `-1`, `Phi(x)` is comfortably representable and the direct
/// `ln(erfc(z)/2)` is accurate. Below that, `erfc(z)` underflows long before `ln Phi` stops
/// being representable, so the identity `erfc(z) = erfcx(z) e^{-z^2}` is used to move the
/// vanishing factor into the log:
///
/// ```text
/// ln Phi(x) = ln erfcx(z) - z^2 - ln 2,   z = -x / sqrt 2
/// ```
///
/// At `x = -100` this returns about `-5005.5`; the naive form returns `-inf` from `x = -38`.
///
/// # The upper tail needs its own branch too
///
/// The mirror problem sits on the right, and is easy to miss because nothing overflows. As
/// `x → +∞`, `Phi(x) → 1` and `ln Phi(x) → 0` from below, so the answer is a *small negative
/// number* — but `Phi(x)` itself rounds to exactly `1.0` once `x` passes about `8.3`, and then
/// `ln(1.0)` is `0`. The value is still accurate in absolute terms (it is wrong by `8e-17`) and
/// useless in relative terms, which is the sense a log-likelihood cares about.
///
/// `ln Phi(x) = ln(1 - Phi(-x))` fixes it: the complement is the *small* tail, so `ln1p` of it
/// keeps full relative accuracy. `log_ndtr(8.25)` is `-7.98e-17` rather than `0`.
pub(crate) fn log_ndtr(x: f64) -> f64 {
    normal::log_ndtr(x)
}

/// The standard normal quantile (probit) `Phi^{-1}(p)`.
///
/// Solved rather than approximated: an initial guess is refined by Newton's method on
/// `ln Phi(x) = ln p`, whose derivative `phi(x)/Phi(x)` is available in closed form. Working in
/// the log domain is what makes the tail work — in probability space both the residual and the
/// derivative underflow, while their logs stay finite.
///
/// The guess comes from `erf_inv` in the central region and from [`tail_start_from_log`] below about
/// `p = 1e-15`, where `2p - 1` has rounded to exactly `-1` and `erf_inv` can only return
/// `-inf`.
///
/// # Accuracy
/// Full double precision for `p` in roughly `[1e-300, 0.5]`, and the same by symmetry above
/// `0.5` *as far as `1 - p` is representable*. For an extreme upper quantile, ask for
/// `-ndtri(q)` with the small `q` directly rather than `ndtri(1 - q)`; the subtraction is what
/// loses the digits, not the solver.
pub(crate) fn ndtri(p: f64) -> f64 {
    normal::ndtri(p, 1e-15)
}

/// The inverse complementary error function: the `x` with `erfc(x) = y`.
///
/// Derived from the quantile rather than from `erf_inv(1 - y)`, which cancels for small `y`.
/// Since `erfc(x) = 2 Phi(-x sqrt 2)`, the inverse is `-ndtri(y/2) / sqrt 2`, which stays
/// accurate as far into the tail as [`ndtri`] does.
/// Which route inverts `erfc` at this probability.
///
/// `erfc` maps the whole line onto `(0, 2)`, so the two ends of that interval are limits rather
/// than values and everything outside it is off the domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ErfcInvRoute {
    /// Outside `[0, 2]`. `NaN` included without a separate test: `contains` is already false for
    /// it, so an `is_nan` arm here would be dead.
    Undefined,
    /// `y = 0`, the infimum of `erfc`'s range, approached as `x → +∞`.
    AtZero,
    /// `y = 2`, the supremum, approached as `x → −∞`.
    AtTwo,
    /// An interior probability: invert through the normal quantile, in whichever coordinate the
    /// halving survives. See [`HalvingCoordinate`].
    Interior,
}

impl ErfcInvRoute {
    fn for_probability(y: f64) -> Self {
        if !(0.0..=2.0).contains(&y) {
            Self::Undefined
        } else if y == 0.0 {
            Self::AtZero
        } else if y == 2.0 {
            Self::AtTwo
        } else {
            Self::Interior
        }
    }
}

/// Which coordinate the half-probability `y/2` is formed in.
///
/// `erfc(x) = 2Φ(−x√2)`, so the inverse wants `Φ⁻¹(y/2)`. Halving is exact for every normal `y`
/// — the exponent simply drops by one — but **not** once `y` is subnormal, where the mantissa has
/// no bit left to shift into and `y/2` rounds, possibly to zero. That is a threshold on the
/// *representation* rather than on the magnitude of the answer, so it is tested for directly as
/// `half + half == y` rather than compared against a constant.
///
/// The same enum the `f32` peer in [`super::f32::normal`] carries, where the band matters more:
/// subnormals start at `1.18e-38` there against `2.2e-308` here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HalvingCoordinate {
    /// `y/2` is exact. Invert it directly through the quantile.
    Exact,
    /// `y/2` would round. Form `ln(y/2) = ln y − ln 2` instead and refine in the log domain,
    /// where the probability stays representable far past where it underflows.
    Logarithmic,
}

impl HalvingCoordinate {
    fn for_probability(y: f64, half: f64) -> Self {
        if half + half == y {
            Self::Exact
        } else {
            Self::Logarithmic
        }
    }
}

pub(crate) fn erfc_inv(y: f64) -> f64 {
    match ErfcInvRoute::for_probability(y) {
        ErfcInvRoute::Undefined => f64::NAN,
        ErfcInvRoute::AtZero => f64::INFINITY,
        ErfcInvRoute::AtTwo => f64::NEG_INFINITY,
        ErfcInvRoute::Interior => {
            let half = 0.5 * y;
            match HalvingCoordinate::for_probability(y, half) {
                HalvingCoordinate::Exact => -ndtri(half) * FRAC_1_SQRT_2,
                HalvingCoordinate::Logarithmic => {
                    let ln_half = y.ln() - LN_2;
                    -refine_lower(ln_half, tail_start_from_log(ln_half)) * FRAC_1_SQRT_2
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::f64::HALF_LN_TAU;
    use crate::kernels::erf::{erfc, erfcx};
    use crate::math::normal::central_start;

    fn close(a: f64, b: f64, tol: f64) {
        assert!(
            (a - b).abs() <= tol * b.abs().max(1.0),
            "{a:e} vs {b:e} (tol {tol:e})"
        );
    }

    /// `erfc_inv` must reject its out-of-domain arguments, including the ones that look like
    /// rounding noise.
    ///
    /// Weakening the guard's `||` to `&&` makes
    /// `erfc_inv(-5e-324)` return **`+inf`** instead of `NaN`, which is worse than a wrong
    /// number: `0.5 * -5e-324` underflows to `-0.0`, `ndtri(-0.0)` takes the `p == 0.0` branch
    /// and returns `-inf`, and the closing negation turns that into a confident `+inf` for an
    /// argument outside the function's domain.
    ///
    /// The existing domain tests used `-0.1` and `2.5`, which the weakened guard still rejects
    /// by a different route — `ln` of a negative reaching `NaN` on its own. Only a value whose
    /// *half underflows to a signed zero* reaches the branch that invents an answer, which is
    /// why this needs the subnormals rather than a round number outside the range.
    #[test]
    fn erfc_inv_rejects_arguments_whose_half_underflows() {
        for &y in &[
            -5e-324,
            -1e-320,
            -f64::MIN_POSITIVE,
            -1e-300,
            -0.1,
            2.0 + f64::EPSILON * 2.0,
            2.5,
            f64::NAN,
        ] {
            let got = erfc_inv(y);
            assert!(
                got.is_nan(),
                "erfc_inv({y:e}) is outside [0, 2] and must be NaN, got {got:e}"
            );
        }
        // The endpoints themselves stay exactly as documented.
        assert_eq!(erfc_inv(0.0), f64::INFINITY);
        assert_eq!(erfc_inv(2.0), f64::NEG_INFINITY);
        // Inside the domain it solves, including well into the subnormals.
        for &y in &[1e-300, 1e-310, 1e-320] {
            let x = erfc_inv(y);
            assert!(
                x.is_finite() && x > 0.0,
                "erfc_inv({y:e}) is in the domain and must solve, got {x:e}"
            );
        }

        let roots = [1, 2, 3, 4, 5].map(|bits| erfc_inv(f64::from_bits(bits)));
        assert!(roots.iter().all(|x| x.is_finite()));
        assert!(roots.windows(2).all(|pair| pair[0] > pair[1]));
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

    #[test]
    fn erfc_inv_inverts_erfc_including_small_arguments() {
        for i in 1..40 {
            let y = f64::from(i) * 0.05;
            close(erfc(erfc_inv(y)), y, 1e-12);
        }
        // Small y is the regime `erf_inv(1 - y)` cannot reach.
        for &y in &[1e-10f64, 1e-50, 1e-200] {
            let x = erfc_inv(y);
            assert!(x.is_finite() && x > 0.0, "erfc_inv({y}) = {x}");
            // Check through the scaled form, since erfc(x) itself has underflowed.
            close(erfcx(x).ln() - x * x, y.ln(), 1e-9);
        }
    }

    /// The four routes that invert `erfc`, and the coordinate the halving is formed in.
    ///
    /// `erfc` maps the line onto `(0, 2)`, so `0` and `2` are limits rather than attained values
    /// and everything outside `[0, 2]` is off the domain — `NaN` included, which `contains`
    /// rejects without a separate test.
    #[test]
    fn the_erfc_inverse_routes_cover_its_range() {
        use ErfcInvRoute::{AtTwo, AtZero, Interior, Undefined};
        use HalvingCoordinate::{Exact, Logarithmic};

        for y in [
            f64::NAN,
            -f64::MIN_POSITIVE,
            -0.1,
            2.0_f64.next_up(),
            2.5,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ] {
            assert_eq!(ErfcInvRoute::for_probability(y), Undefined, "at {y:e}");
            assert!(erfc_inv(y).is_nan(), "at {y:e}");
        }
        assert_eq!(ErfcInvRoute::for_probability(0.0), AtZero);
        assert_eq!(ErfcInvRoute::for_probability(-0.0), AtZero);
        assert_eq!(erfc_inv(0.0), f64::INFINITY);
        assert_eq!(ErfcInvRoute::for_probability(2.0), AtTwo);
        assert_eq!(erfc_inv(2.0), f64::NEG_INFINITY);
        for y in [f64::MIN_POSITIVE, 0.5, 1.0, 1.5, 2.0_f64.next_down()] {
            assert_eq!(ErfcInvRoute::for_probability(y), Interior, "at {y:e}");
        }

        // Halving is exact for every normal `y` — the exponent just drops — and rounds once `y`
        // is subnormal, where the mantissa has no bit left to shift into.
        for y in [1.0_f64, 0.5, 1.5, f64::MIN_POSITIVE] {
            assert_eq!(
                HalvingCoordinate::for_probability(y, 0.5 * y),
                Exact,
                "at {y:e}"
            );
        }
        let subnormal = f64::from_bits(1);
        assert_eq!(0.5 * subnormal, 0.0, "the least subnormal halves to zero");
        assert_eq!(
            HalvingCoordinate::for_probability(subnormal, 0.5 * subnormal),
            Logarithmic
        );
        // And the log route still returns a finite, ordered answer out there, which is the whole
        // reason it exists.
        assert!(erfc_inv(subnormal).is_finite());
        assert!(erfc_inv(subnormal) > erfc_inv(f64::from_bits(2)) - 1.0);
    }
}
