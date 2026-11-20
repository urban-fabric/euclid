//! Single-precision peer of [`super::super::normal`]; see that module for the algorithms and
//! the reason the log-CDF is routed through `erfcx`.
//!
//! The CDF and normal quantile use native `f32`. The inverse complementary error
//! function widens its input to preserve accuracy through the tail transformation.

use crate::math::normal;

/// The standard normal CDF.
pub(crate) fn ndtr(x: f32) -> f32 {
    normal::ndtr(x)
}

/// Log-CDF, including tails outside the f32 probability range.
pub(crate) fn log_ndtr(x: f32) -> f32 {
    normal::log_ndtr(x)
}

/// The standard normal quantile.
pub(crate) fn ndtri(p: f32) -> f32 {
    // The expression 2p - 1 loses its low bits sooner in f32.
    normal::ndtri(p, 1e-6)
}

/// Which coordinate the half-probability `y/2` is formed in.
///
/// `erfc(x) = 2Φ(−x√2)`, so the inverse wants `Φ⁻¹(y/2)`. Halving is exact for every normal
/// `y` — the exponent simply drops by one — but **not** once `y` is subnormal, where the
/// mantissa has no bit left to shift into and `y/2` rounds, possibly to zero. That is a
/// threshold on the *representation* rather than on the magnitude of the answer, so it is tested
/// for directly as `p + p != y` rather than compared against a constant.
///
/// The band matters more in `f32` than in `f64`: subnormals start at `1.18e-38` here, well
/// inside the range of tail probabilities a caller will ask about.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HalvingCoordinate {
    /// `y/2` is exact. Invert it directly through the quantile.
    Exact,
    /// `y/2` would round. Form `ln(y/2) = ln y − ln 2` instead and refine in the log domain,
    /// where the probability stays representable far past where it underflows.
    Logarithmic,
}

impl HalvingCoordinate {
    fn for_probability(y: f32, half: f32) -> Self {
        if half + half == y {
            Self::Exact
        } else {
            Self::Logarithmic
        }
    }
}

/// The inverse complementary error function: the `x` with `erfc(x) = y`.
///
/// Native, where this used to widen to `f64` and narrow the result back. Both branches already
/// had `f32` machinery available — [`ndtri`] is native here and
/// [`refine_lower`](normal::refine_lower) is generic over the format — so the only thing the
/// widening bought was the subnormal halving, which [`HalvingCoordinate`] now handles in `f32`
/// by moving to logarithms instead of to a wider type.
pub(crate) fn erfc_inv(y: f32) -> f32 {
    match ErfcInvRoute::for_probability(y) {
        ErfcInvRoute::Undefined => f32::NAN,
        ErfcInvRoute::AtZero => f32::INFINITY,
        ErfcInvRoute::AtTwo => f32::NEG_INFINITY,
        ErfcInvRoute::Interior => {
            let half = 0.5 * y;
            match HalvingCoordinate::for_probability(y, half) {
                HalvingCoordinate::Exact => -ndtri(half) * core::f32::consts::FRAC_1_SQRT_2,
                HalvingCoordinate::Logarithmic => {
                    let ln_half = y.ln() - core::f32::consts::LN_2;
                    -normal::refine_lower(ln_half, normal::tail_start_from_log(ln_half))
                        * core::f32::consts::FRAC_1_SQRT_2
                }
            }
        }
    }
}

/// Which route inverts `erfc` at this probability.
///
/// The same four arms the `f64` peer in [`crate::kernels::normal`] carries. `erfc` maps the whole
/// line onto `(0, 2)`, so the two ends of that interval are limits rather than values and
/// everything outside it is off the domain.
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
    fn for_probability(y: f32) -> Self {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::f32::HALF_LN_TAU;
    use crate::kernels::f32::erf::erfc;
    use crate::math::normal::tail_start_from_log;

    fn close(a: f32, b: f32, tol: f32) {
        assert!(
            (a - b).abs() <= tol * b.abs().max(1.0),
            "{a:e} vs {b:e} (tol {tol:e})"
        );
    }

    #[test]
    fn ndtr_hits_its_known_values() {
        close(ndtr(0.0), 0.5, 1e-6);
        close(ndtr(1.0), 0.841_344_7, 1e-5);
        close(ndtr(-1.0), 0.158_655_25, 1e-5);
    }

    #[test]
    fn log_ndtr_stays_finite_past_the_f32_underflow() {
        assert_eq!(ndtr(-20.0), 0.0, "Phi underflows in f32 here");
        let v = log_ndtr(-20.0);
        assert!(v.is_finite() && v < 0.0, "log_ndtr(-20) = {v}");
        close(v, -200.0 - 20.0f32.ln() - HALF_LN_TAU, 1e-2);
    }

    /// The **upper** tail, which nothing in `f32` evaluated.
    ///
    /// `log_ndtr`'s first branch is `(-0.5 * erfc(x/sqrt 2)).ln_1p()`, and replacing that `*`
    /// with `+` survived: the f32 suite tested `log_ndtr(-20)` and nothing above zero, so the
    /// branch never ran. The generated `f64` property exercises the separate double-precision
    /// implementation.
    ///
    /// It matters for the reason the branch exists: above `x ~ 3.2` in `f32`, `Phi(x)` rounds to
    /// exactly `1` and `ln(Phi(x))` is `0`, losing a log-likelihood that is small but not zero.
    /// The mutated form returns `-0.69` where the answer is `-6.9e-4` — not a rounding problem,
    /// a different number.
    #[test]
    fn log_ndtr_resolves_the_upper_tail_where_the_cdf_has_saturated() {
        let mut saturated = false;
        for &x in &[0.5f32, 1.0, 2.0, 3.2, 5.0, 8.0] {
            let got = log_ndtr(x);
            // `ln Phi(x) = ln(1 - Phi(-x))`, and the complement is the small tail.
            let want = (-ndtr(-x)).ln_1p();
            assert!(
                got < 0.0 && got > -1.0,
                "ln Phi({x}) must be a small negative number, got {got}"
            );
            close(got, want, 1e-4);
            // And it is the *relative* accuracy that the branch buys. Stated as an implication
            // rather than at a guessed abscissa, because the saturation point is a property of
            // the format and not of this kernel: `Phi(-x)` has to fall below `ulp(1)/2 = 6e-8`,
            // which happens near `x = 5.3` — not at the `3.2` where the *relative* error starts
            // to matter, which is what the trait's note refers to. A first draft asserted
            // saturation at 3.2 and failed against a correct kernel.
            if ndtr(x) == 1.0 {
                saturated = true;
                assert_eq!(ndtr(x).ln(), 0.0, "f32: the naive log collapses at {x}");
                assert!(got < 0.0, "f32: while log_ndtr keeps the answer: {got:e}");
            }
        }
        assert!(
            saturated,
            "the sweep must reach the saturation this branch exists for"
        );
        // Monotone up to the saturation point and never positive.
        let mut previous = f32::NEG_INFINITY;
        for i in 0..=200 {
            #[allow(clippy::cast_precision_loss)]
            let x = -8.0 + 16.0 * (i as f32) / 200.0;
            let v = log_ndtr(x);
            assert!(v <= 0.0, "f32: ln Phi({x}) = {v} must not be positive");
            assert!(v >= previous, "f32: ln Phi must be non-decreasing at {x}");
            previous = v;
        }
    }

    #[test]
    fn ndtri_round_trips() {
        for i in 1..20 {
            let p = (i as f32) * 0.05;
            close(ndtr(ndtri(p)), p, 1e-4);
        }
        for &p in &[1e-10f32, 1e-20, 1e-30] {
            let x = ndtri(p);
            assert!(x.is_finite() && x < 0.0, "ndtri({p}) = {x}");
            close(log_ndtr(x), p.ln(), 1e-4);
        }
    }

    /// The f32 peer of `the_tail_seed_is_close_enough_for_newton_to_finish_from`.
    ///
    /// Six of the twelve surviving seed mutants were on this side, and one of them is the
    /// starkest evidence for the whole diagnosis: **replacing `tail_start_from_log` wholesale with the
    /// constant `-1.0` survived.** The same substitution is caught at `f64`, and the asymmetry
    /// is the narrower range — six Newton steps reach `p = 1e-30` from `-1.0` in `f32`, where
    /// they cannot reach `p = 1e-300` from it in `f64`. A seed that can be replaced by a
    /// constant is not being tested at all.
    ///
    /// Measured across `[1e-44, 1e-6]`, which is the range this seed is used on — above `1e-6`
    /// the kernel starts from `erf_inv` instead: worst relative error `4.5e-3` in `ln p`, worst
    /// distance from the root `1.26e-2`. Both are looser than the f64 figures by about the
    /// ratio of the two formats' precision, which is the expected amount.
    #[test]
    fn the_tail_seed_is_close_enough_for_newton_to_finish_from() {
        let mut worst_relative = 0.0f32;
        let mut worst_distance = 0.0f32;
        for i in 0..=1_000 {
            #[allow(clippy::cast_precision_loss)]
            let exponent = -44.0 + 38.0 * (i as f32) / 1_000.0;
            let p = 10f32.powf(exponent);
            if p <= 0.0 {
                continue;
            }
            let seed = tail_start_from_log(p.ln());
            let ln_p = p.ln();

            assert!(
                seed.is_finite() && seed < 0.0,
                "f32: the seed must be a finite lower-tail abscissa at p = {p:e}, got {seed}"
            );

            let relative = ((log_ndtr(seed) - ln_p) / ln_p).abs();
            worst_relative = worst_relative.max(relative);
            assert!(
                relative < 5e-2,
                "f32: the seed is {relative:e} off in ln p at p = {p:e}"
            );

            let distance = (seed - ndtri(p)).abs();
            worst_distance = worst_distance.max(distance);
            assert!(
                distance < 1e-1,
                "f32: the seed sits {distance:e} from the root at p = {p:e}"
            );
        }
        assert!(
            worst_relative < 1e-2 && worst_distance < 5e-2,
            "f32: the seed degraded: worst relative {worst_relative:e}, worst distance {worst_distance:e}"
        );
    }

    /// The f32 peers of the two defects the f64 differential harness found.
    ///
    /// Both survived here for the same reason they survived there, and both are real:
    ///
    /// - `p > 0.5` narrowed to `p == 0.5` removes the symmetric route, and `ndtri_lower` alone
    ///   degrades near `p = 1` because Newton's residual `ln Phi(x) - ln p` becomes a difference
    ///   of two quantities that have each lost their digits.
    /// - `erfc_inv`'s domain guard weakened from `||` to `&&` lets a tiny negative argument
    ///   through, where `0.5 * y` underflows to `-0.0` and the function returns a confident
    ///   `+inf` for something outside its domain.
    ///
    /// The f32 format reaches both regimes at far milder arguments than `f64` does, which is the
    /// usual reason its peer tests are worth writing separately rather than assumed from the
    /// wider one.
    #[test]
    fn the_quantile_and_its_inverse_guard_their_edges() {
        // Near one, the symmetric route is the accurate one.
        for &p in &[
            1.0 - f32::EPSILON / 2.0,
            1.0 - f32::EPSILON,
            1.0 - 1e-6,
            0.999_999,
        ] {
            let got = ndtri(p);
            let want = -ndtri(1.0 - p);
            assert!(
                got.is_finite() && got > 0.0,
                "f32: ndtri({p}) = {got} must be a finite positive abscissa"
            );
            close(got, want, 1e-4);
        }

        // Out of domain is NaN, including where the halving underflows.
        for &y in &[
            -1e-45f32,
            -f32::MIN_POSITIVE,
            -1e-30,
            -0.1,
            2.0 + f32::EPSILON * 2.0,
            2.5,
            f32::NAN,
        ] {
            let got = erfc_inv(y);
            assert!(
                got.is_nan(),
                "f32: erfc_inv({y:e}) is outside [0, 2] and must be NaN, got {got:e}"
            );
        }
        assert_eq!(erfc_inv(0.0f32), f32::INFINITY);
        assert_eq!(erfc_inv(2.0f32), f32::NEG_INFINITY);
        assert!(erfc_inv(f32::from_bits(1)).is_finite());
        assert!(erfc_inv(f32::from_bits(1)) > erfc_inv(f32::from_bits(2)));
    }

    #[test]
    fn erfc_inv_inverts_erfc() {
        for i in 1..20 {
            let y = (i as f32) * 0.1;
            close(erfc(erfc_inv(y)), y, 1e-4);
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
            f32::NAN,
            -f32::MIN_POSITIVE,
            -0.1,
            2.0_f32.next_up(),
            2.5,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ] {
            assert_eq!(ErfcInvRoute::for_probability(y), Undefined, "at {y:e}");
            assert!(erfc_inv(y).is_nan(), "at {y:e}");
        }
        assert_eq!(ErfcInvRoute::for_probability(0.0), AtZero);
        assert_eq!(ErfcInvRoute::for_probability(-0.0), AtZero);
        assert_eq!(erfc_inv(0.0), f32::INFINITY);
        assert_eq!(ErfcInvRoute::for_probability(2.0), AtTwo);
        assert_eq!(erfc_inv(2.0), f32::NEG_INFINITY);
        for y in [f32::MIN_POSITIVE, 0.5, 1.0, 1.5, 2.0_f32.next_down()] {
            assert_eq!(ErfcInvRoute::for_probability(y), Interior, "at {y:e}");
        }

        // Halving is exact for every normal `y` — the exponent just drops — and rounds once `y`
        // is subnormal, where the mantissa has no bit left to shift into.
        for y in [1.0_f32, 0.5, 1.5, f32::MIN_POSITIVE] {
            assert_eq!(
                HalvingCoordinate::for_probability(y, 0.5 * y),
                Exact,
                "at {y:e}"
            );
        }
        let subnormal = f32::from_bits(1);
        assert_eq!(0.5 * subnormal, 0.0, "the least subnormal halves to zero");
        assert_eq!(
            HalvingCoordinate::for_probability(subnormal, 0.5 * subnormal),
            Logarithmic
        );
        // And the log route still returns a finite, ordered answer out there, which is the whole
        // reason it exists.
        assert!(erfc_inv(subnormal).is_finite());
        assert!(erfc_inv(subnormal) > erfc_inv(f32::from_bits(2)) - 1.0);
    }
}
