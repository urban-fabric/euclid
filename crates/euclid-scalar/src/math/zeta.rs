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
}
