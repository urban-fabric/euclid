//! The two steps of the Cohen–Rodriguez Villegas–Zagier acceleration, shared by both precisions.
//!
//! The Riemann zeta is reached through the Dirichlet eta on the critical strip, and the eta
//! through an accelerated alternating sum. Neither step below holds a format-dependent constant:
//! the coefficient recurrence is rational and the divisor's only constant is `ln 2`, which the
//! type supplies. The *term count* is per format and stays with the kernels — `f64` takes 24
//! terms and `f32` ten, because `5.83^{−10}` is already under half an `f32` ulp.

use crate::HostReal;

/// Advance the acceleration's coefficient by one index.
///
/// The recurrence is `cₖ₊₁ = (k+n)(k−n)cₖ / ((k+½)(k+1))`, which converges like
/// `(3 + √8)^{−n} ≈ 5.83^{−n}` — fast enough that a handful of terms reaches either format's
/// resolution, and the reason this sum is used instead of the defining series.
pub(crate) fn next_cvz_coefficient<T: HostReal>(k: T, n: T, coefficient: T) -> T {
    (k + n) * (k - n) * coefficient / ((k + T::HALF) * (k + T::ONE))
}

/// `1 − 2^{1−s}`, the factor relating the Dirichlet eta to the zeta.
///
/// Formed as `−expm1((1−s)·ln 2)` rather than `1 − 2^{1−s}`: the two agree in exact arithmetic,
/// but near `s = 1` the literal form is a difference of two quantities that both tend to one, and
/// `expm1` keeps the whole answer. At `s = 1` it is exactly zero, which is the zeta's simple pole
/// and the one argument the kernels must not divide by.
pub(crate) fn eta_divisor<T: HostReal>(s: T) -> T {
    -((T::ONE - s) * T::ln_2()).exp_m1()
}

/// Chebyshev coefficients of `ζ` on `[−1/8, 1/8]`, in `T₀ … T₁₅`.
///
/// Derived locally by `zeta_cheb.py` against `mpfr_zeta` at 400 bits, never transcribed. The
/// transform runs in `decimal` at 60 digits off node values and node *positions* carried as text,
/// because each of those floors the coefficients at `1e-16` and reads like convergence.
///
/// Sixteen terms reach `1e-18`, falling by about 16 per order — the convergence the geometry
/// predicts, since `ζ` is analytic at the origin and its nearest singularity is the pole at
/// `s = 1`, `0.875` from the centre against a half-width of `1/8`.
///
/// The interval was `[−1/8, 0]` and twelve terms when only the negative side was known to be
/// bad. It is symmetric now because the *amplification*-ranked sweep put the positive side at the
/// top of the whole crate: `5` ulp at `s = +0.00457` against `κ = 0.0084`, which is 19.7× the
/// floor. Five ulp reads as unremarkable and is not, because `ζ ≈ −0.504` there is about as
/// well-conditioned as a function gets — the same reason raw ulp is the wrong ranking.
const ZETA_ORIGIN_FIT: [f64; 16] = [
    -0.507_930_080_214_922_6,
    -0.116_352_648_601_810_38,
    -0.007_961_200_090_847_24,
    -0.000_498_360_352_399_564_9,
    -3.124_241_167_451_909e-5,
    -1.960_573_209_206_908_6e-6,
    -1.230_181_809_614_793_7e-7,
    -7.718_896_011_135_485e-9,
    -4.843_304_462_787_156e-10,
    -3.038_983_039_675_661_5e-11,
    -1.906_842_326_368_144e-12,
    -1.196_468_560_173_940_5e-13,
    -7.507_369_620_755_791e-15,
    -4.710_579_157_629_438e-16,
    -2.955_703_145_206_63e-17,
    -1.854_587_470_085_129_2e-18,
];

/// `ζ(s)` on `(−1/8, 1/8)` from [`ZETA_ORIGIN_FIT`], by Clenshaw.
///
/// # Why this interval has its own arm
///
/// The eta acceleration owns `(−1/8, 1)` and is not uniform across it: measured against a
/// 400-bit reference it is `4`–`9` ulp over the positive part and `15`–`20` over `(−1/8, 0)`.
/// The asymmetry is structural. `η(s) = Σ(−1)^{n−1}n^{−s}` converges for `s > 0`, and for `s < 0`
/// its terms *grow* with `n`, so the accelerator is summing a divergent series and cancels much
/// harder. Below zero it is at its own floor, which is why no movement of
/// `kernels::transcendental`'s `ZETA_ETA_MIN` could help and a third arm could.
///
/// The reflection cannot have the band either, and its own note says why: it needs `1 − s`, and
/// as `s → 0⁻` the pole of `ζ(1−s)` is cancelled against a vanishing `sin(πs/2)`. Measured, it
/// goes from `19` ulp to `60` by `−1/64`.
///
/// `t = 8s` maps the interval to `[−1, 1]` and is **exact** — a power of two — where the earlier
/// asymmetric interval needed `16s + 1` and one rounding. Clenshaw rather than a monomial Horner
/// for the usual reason: converting a degree-15 Chebyshev series to monomials gives alternating
/// growing coefficients whose sum discards most of what the fit bought.
pub(crate) fn zeta_near_origin<T: HostReal>(s: T) -> T {
    let t = s * T::from_f64(8.0);
    let twice = t + t;
    let mut second = T::ZERO;
    let mut first = T::ZERO;
    for &coefficient in ZETA_ORIGIN_FIT[1..].iter().rev() {
        let next = twice * first - second + T::from_f64(coefficient);
        second = first;
        first = next;
    }
    t * first - second + T::from_f64(ZETA_ORIGIN_FIT[0])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_coefficient_recurrence_is_the_published_ratio() {
        // `(k+n)(k−n)/((k+½)(k+1))` at the first few indices, exactly.
        assert_eq!(next_cvz_coefficient(1.0_f64, 0.0, 1.0), 1.0 / 3.0);
        assert_eq!(next_cvz_coefficient(2.0_f64, 1.0, 1.0), 0.4);
        // `k = n` annihilates the coefficient, which is what terminates the sum.
        assert_eq!(next_cvz_coefficient(1.0_f64, 1.0, 1.0), 0.0);
        assert_eq!(next_cvz_coefficient(5.0_f64, 5.0, 7.0), 0.0);
        // Linear in the incoming coefficient, so the recurrence cannot introduce a scale.
        assert_eq!(next_cvz_coefficient(2.0_f64, 1.0, 3.0), 1.2);
        assert_eq!(next_cvz_coefficient(2.0_f64, 1.0, 0.0), 0.0);
        // Above `k = n` the factor is positive and below one, which is the convergence.
        for k in [2.0_f64, 3.0, 10.0, 100.0] {
            let ratio = next_cvz_coefficient(k, 1.0, 1.0);
            assert!((0.0..1.0).contains(&ratio), "at k = {k}: {ratio}");
        }
        assert_eq!(next_cvz_coefficient(1.0_f32, 1.0, 1.0), 0.0);
        assert!((next_cvz_coefficient(2.0_f32, 1.0, 1.0) - 0.4).abs() < 1e-7);
    }

    #[test]
    fn the_eta_divisor_vanishes_exactly_at_the_pole() {
        // `1 − 2^{1−s}`: `−1` at the origin, `0` at the pole, `½` at two, `¾` at three.
        assert_eq!(eta_divisor(0.0_f64), -1.0);
        assert_eq!(eta_divisor(1.0_f64), 0.0);
        assert_eq!(eta_divisor(2.0_f64), 0.5);
        assert_eq!(eta_divisor(3.0_f64), 0.75);
        assert_eq!(eta_divisor(-1.0_f64), -3.0);
        // `expm1` rather than the literal difference: either side of the pole the divisor is
        // tiny, and the relative accuracy there is what the eta-to-zeta division depends on.
        // `1 + 2^{-40}` rather than `1 + 1e-12`: the subtraction `1 − s` is then *exact*, so
        // what is measured is this function's accuracy and not the argument's representation.
        let near = eta_divisor(1.0_f64 + 2.0_f64.powi(-40));
        assert!(near > 0.0, "{near:e}");
        assert!(
            (near / (2.0_f64.powi(-40) * core::f64::consts::LN_2) - 1.0).abs() < 1e-11,
            "{near:e}"
        );
        let below = eta_divisor(1.0_f64 - 2.0_f64.powi(-40));
        assert!(
            below < 0.0 && (below / near + 1.0).abs() < 1e-11,
            "{below:e}"
        );
        // Monotone, and it *reaches* one rather than approaching it: `2^{1−s}` falls below an
        // ulp of one by `s = 54`, so the divisor saturates there and stays exact above it.
        assert!(eta_divisor(2.0_f64) < eta_divisor(10.0_f64));
        assert!(eta_divisor(10.0_f64) < 1.0);
        assert_eq!(eta_divisor(1e3_f64), 1.0);
        assert_eq!(eta_divisor(f64::INFINITY), 1.0);
        assert_eq!(eta_divisor(1.0_f32), 0.0);
        assert_eq!(eta_divisor(2.0_f32), 0.5);
    }

    /// Every coefficient of the fit contributes, and each one's **sign** is pinned.
    ///
    /// Four mutants survived here, all `delete -` on a coefficient. They live through the
    /// functional-equation property because the later terms contribute below its tolerance:
    /// `c[11]` is `1.2e-13` against a `ζ` of `0.5`, so flipping it moves the answer by `2.4e-13`
    /// and the identity check is looser than that.
    ///
    /// A Chebyshev series cannot be spot-checked term by term through its own evaluation, so this
    /// compares Clenshaw against the **definition** — `Σ cₖ Tₖ(t)` with `Tₖ` by its own
    /// recurrence — at a `t` where no term vanishes. Any sign flip then shows up as a difference
    /// of twice that coefficient, which is far above the agreement two correct evaluations have.
    #[test]
    fn every_fit_coefficient_contributes_with_its_own_sign() {
        // `t = 0.3` keeps every `T_k(t)` away from zero, so no coefficient is masked.
        let t = 0.3_f64;
        let s = t / 8.0;
        let mut previous = 1.0_f64;
        let mut current = t;
        let mut direct = ZETA_ORIGIN_FIT[0] + ZETA_ORIGIN_FIT[1] * t;
        for &coefficient in &ZETA_ORIGIN_FIT[2..] {
            let next = 2.0 * t * current - previous;
            direct += coefficient * next;
            previous = current;
            current = next;
        }
        let clenshaw = zeta_near_origin(s);
        assert!(
            (clenshaw - direct).abs() <= 8.0 * f64::EPSILON * direct.abs(),
            "Clenshaw gives {clenshaw} against the direct sum {direct}"
        );
        // Comparing two evaluations cannot reach the tail, and the assertion above is why: its
        // floor is `8ε·|ζ|`, about `9e-16`, while the last coefficient is `1.9e-18`. Flipping
        // that one moves nothing either spelling of the series can see. The first draft of this
        // test asserted the opposite and failed, which is the useful kind of failure.
        //
        // What does reach every coefficient is its **sign**: all sixteen are negative on this
        // interval, because `ζ` is negative and monotone across it with no inflection. So a
        // deleted minus is a structural break rather than a numerical one, and this catches it
        // at `c[15]` exactly as well as at `c[0]`.
        assert!(
            ZETA_ORIGIN_FIT.iter().all(|c| *c < 0.0),
            "every fit coefficient is negative on [-1/8, 1/8]; a deleted sign breaks that"
        );
        // Monotone decay is the other structural fact, and it pins the ordering: a coefficient
        // moved or duplicated shows up here even where its magnitude is below the floor above.
        for pair in ZETA_ORIGIN_FIT.windows(2) {
            assert!(
                pair[1].abs() < pair[0].abs(),
                "the fit should decay monotonically: {:e} then {:e}",
                pair[0],
                pair[1]
            );
        }
    }
}
