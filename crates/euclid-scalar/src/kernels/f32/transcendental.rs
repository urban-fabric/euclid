//! The `f32` peer of [`crate::kernels::transcendental`]: the Lambert `W` and the Riemann zeta.
//!
//! Native single precision throughout. Both used to widen to `f64` and narrow once, and the
//! reason given for it — the [module header](super)'s "classical special functions" exception —
//! does not actually apply to either: `W` is a Halley iteration on a well-conditioned equation,
//! and `ζ`'s three routes are a geometric series, an Euler–Maclaurin sum and a reflection, none
//! of which has the divergent-asymptotic or cancelling-march character that exception describes.
//!
//! Choosing `f32` is choosing speed at less accuracy, so both halves of that matter here. The
//! term counts are rederived at this format rather than inherited, and both come out **much**
//! shorter: the eta accelerator drops from 24 terms to 10, and `W`'s iteration cap from 60 to 8.

use super::gamma::ln_gamma;
use crate::math::gamma::reflection_sine;
mod lambert;

pub(crate) use lambert::lambert_w;

// ===== Riemann zeta ======================================================

/// Terms used by [`zeta_eta`].
///
/// The Cohen–Rodriguez Villegas–Zagier acceleration converges like `(3 + √8)^{−n} ≈ 5.83^{−n}`,
/// so the `f64` peer's 24 terms reach `≈2e-18` — eleven orders past anything `f32` can hold.
/// `5.83^{−10} = 5.4e-8` is already under half an `f32` ulp, so ten is the honest count and the
/// other fourteen were pure cost on a path chosen for speed.
///
/// There is a second reason not to inherit 24, and it is about correctness rather than cost: the
/// algorithm divides by `T_n(3)`, which grows like `5.83^n`. At `n = 24` that is `1.2e18`, far
/// past `f32`'s exactly-representable integers, so the normaliser itself would be rounded before
/// the sum even began. At `n = 10` it is `3.2e7`, just over `2^24` and so costing a single bit.
const CVZ_TERMS: u32 = 10;

/// `T_n(3)`, the integer Chebyshev normalisation for the eta accelerator.
///
/// Evaluated in `u64` from the Chebyshev recurrence and narrowed once at the end, so the integer
/// is exact before it is ever a float.
const fn cvz_normalizer(terms: u32) -> u64 {
    if terms == 0 {
        return 1;
    }
    let (mut previous, mut current) = (1, 3);
    let mut index = 1;
    while index < terms {
        (previous, current) = (current, 6 * current - previous);
        index += 1;
    }
    current
}

/// `ζ(s)` for `-1 < s < 1`, through the accelerated Dirichlet eta series.
///
/// `η(s) = Σ_{k≥1} (−1)^{k−1} k^{−s} = (1 − 2^{1−s})·ζ(s)` converges — conditionally — for every
/// `s > 0`, which is what reaches inside the critical strip where both other routes fail.
/// Conditional convergence is far too slow to use directly, so the sum is accelerated; see
/// [`CVZ_TERMS`]. The acceleration also continues eta across zero, which is why it covers
/// `−1 < s < 0` rather than reflection doing so: reflection would form `1 − s`, and for a tiny
/// negative `s` that rounds to exactly `1` and evaluates the pole even though `ζ(0) = −1/2` is
/// perfectly regular.
fn zeta_eta(s: f32) -> f32 {
    debug_assert!(s > -1.0 && s < 1.0);
    let n = CVZ_TERMS as f32;

    const NORMALIZER: u64 = cvz_normalizer(CVZ_TERMS);
    #[allow(clippy::cast_precision_loss)]
    let d = NORMALIZER as f32;

    let mut b = -1.0f32;
    let mut c = -d;
    let mut total = 0.0f32;
    for k in 0..CVZ_TERMS {
        let kf = k as f32;
        c = b - c;
        // The kth term of eta is (k+1)^{-s}.
        total += c * (kf + 1.0).powf(-s);
        b = crate::math::zeta::next_cvz_coefficient(kf, n, b);
    }
    let eta = total / d;
    eta / crate::math::zeta::eta_divisor(s)
}

/// Which of `ζ`'s routes evaluates a given `s`. The `f32` peer of the `f64` `ZetaRoute`.
#[derive(Clone, Copy, Debug, PartialEq)]
enum ZetaRoute {
    /// A value the function takes exactly: the limits, the pole, `ζ(0)`, and the trivial zeros.
    Value(f32),
    /// Euler–Maclaurin on the Hurwitz zeta, for `s > 1`.
    Hurwitz,
    /// The accelerated eta series, from [`ZETA_ETA_MIN`] up to `1`, including the critical strip.
    Eta,
    /// The functional equation, mapping `s ≤ −1` onto `1 − s ≥ 2`.
    Reflection,
}

/// Below this `s` the reflection replaces the eta acceleration.
///
/// **`−1/8`, raised from `−1`.** The eta route had all of `(−1, 0)` because the reflection needs
/// `1 − s` and that loses `s` entirely as `s → 0⁻`. True, and it does not need the whole interval
/// to avoid it — by `s = −1/8` the reflected argument is `9/8`, far enough from `ζ`'s pole at `1`
/// that the rounding of `1 − s` costs about an ulp.
///
/// # It is `f64`'s value and not `f64`'s measurement
///
/// The wide peer reached `−1/8` the same way and that is no reason to inherit it: `ε` is `1.2e-7`
/// here, so `ζ(1−s)`'s pole tells sooner and the crossover could easily have sat further out.
/// Measured against a 400-bit reference, worst `f32` ulp per band, with the handover forced to
/// each candidate:
///
/// ```text
/// s              [-0.99,-0.9]  [-0.9,-0.75]  [-0.75,-0.5]  [-0.5,-0.25]  [-0.25,-0.125]
/// handover −1           154.3         106.3          46.3          15.0             7.0   ulp
/// handover −1/4           5.0           5.1           5.1           6.2             7.0
/// handover −1/8           5.0           5.1           5.1           6.2             6.2
/// ```
///
/// `−1/8` is never worse than `−1/4` and strictly better on `[−1/4, −1/8)`, so it is the choice
/// rather than the inheritance. The `f32` figures also happen to be *better* than the `f64` ones
/// at the old handover — 154 against 171 — because this format's coarser inputs keep `sin(πs/2)`
/// away from zero, which is the same asymmetry `REFLECTED_ZERO_MIN` turns on.
///
/// What remains is `(−1/8, 0)`, which measures `6.8`–`7.1` ulp on every candidate because it is
/// the accelerator's own cancellation rather than a routing choice: `η`'s alternating series is
/// divergent for `s < 0`. The wide peer answers that with a Chebyshev fit on `|s| < 1/8`; this
/// format has no such arm yet, and that band is where it would go.
const ZETA_ETA_MIN: f32 = -0.125;

fn zeta_route(s: f32) -> ZetaRoute {
    if s.is_nan() || s == f32::NEG_INFINITY {
        ZetaRoute::Value(f32::NAN)
    } else if s == f32::INFINITY {
        ZetaRoute::Value(1.0)
    } else if s > 1.0 {
        ZetaRoute::Hurwitz
    } else if s == 1.0 {
        ZetaRoute::Value(f32::INFINITY)
    } else if s == 0.0 {
        ZetaRoute::Value(-0.5)
    } else if s > ZETA_ETA_MIN {
        ZetaRoute::Eta
    } else if s % 2.0 == 0.0 {
        // The negative even integers are the trivial zeros, returned exactly rather than left to
        // the `sin(πs/2)` factor — which at an integer argument evaluates to the rounding
        // residue of `πs/2` rather than to zero.
        ZetaRoute::Value(0.0)
    } else {
        ZetaRoute::Reflection
    }
}

/// The functional equation `ζ(s) = 2^s π^{s−1} sin(πs/2) Γ(1−s) ζ(1−s)`.
fn zeta_reflection(s: f32) -> f32 {
    // Formed in the log domain where it could overflow: `Γ(1−s)` grows without bound as
    // `s → −∞` and so does `ζ`, so the product has to be allowed to reach infinity rather than
    // producing `NaN` on the way. `f32` reaches that point at `s ≈ −35` rather than `f64`'s
    // `−171`, so the log domain is doing more of the work here, not less.
    let one_minus_s = 1.0 - s;
    let sin_term = reflection_sine(0.5 * s);
    let ln_magnitude = s * core::f32::consts::LN_2
        + (s - 1.0) * core::f32::consts::PI.ln()
        + sin_term.abs().ln()
        + ln_gamma(one_minus_s);
    // ζ(1-s) is positive throughout the reflection domain.
    (ln_magnitude.exp() * super::polygamma::hurwitz_zeta(one_minus_s, 1.0)).copysign(sin_term)
}

/// The Riemann zeta function `ζ(s)` for real `s`.
pub(crate) fn zeta(s: f32) -> f32 {
    match zeta_route(s) {
        ZetaRoute::Value(value) => value,
        ZetaRoute::Hurwitz => super::polygamma::hurwitz_zeta(s, 1.0),
        ZetaRoute::Eta => zeta_eta(s),
        ZetaRoute::Reflection => zeta_reflection(s),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::transcendental as reference;
    use euclid_utils::assert_close_f32;

    /// The native values against the `f64` kernel, at the budget a native path earns.
    ///
    /// The previous form of this test asserted one-ulp agreement, which was trivially true while
    /// both sides were the same `f64` computation rounded once. The budgets here are measured.
    #[test]
    fn the_native_values_track_the_f64_reference() {
        for &x in &[-0.3f32, -0.1, 0.5, 1.0, 10.0, 1e3, 1e20] {
            assert_close_f32(lambert_w(x), reference::lambert_w(f64::from(x)), 8.0, "W");
        }
        for &s in &[-5.5f32, -1.5, 0.5, 1.5, 2.0, 3.0, 10.0] {
            assert_close_f32(zeta(s), reference::zeta(f64::from(s)), 64.0, "zeta");
        }
    }

    #[test]
    fn the_f32_surface_keeps_the_f64_conventions() {
        assert_eq!(lambert_w(0.0), 0.0);
        assert!(lambert_w(-0.4).is_nan(), "below the branch point -1/e");
        assert!(lambert_w(f32::NAN).is_nan());
        assert_eq!(lambert_w(f32::INFINITY), f32::INFINITY);
        assert_eq!(zeta(1.0), f32::INFINITY, "the simple pole");
        assert!(zeta(f32::NAN).is_nan());
        // The trivial zeros are returned exactly, not left to the sine factor.
        for k in 1..6 {
            let s = -2.0 * (k as f32);
            assert_eq!(zeta(s), 0.0, "zeta({s}) must be exactly zero");
        }
        assert_eq!(zeta(0.0), -0.5, "zeta(0) is returned exactly");
        assert_close_f32(zeta(2.0), 1.644_934_066_848_226_4, 16.0, "zeta(2)");
    }

    /// Each route must be reached by the arguments it exists for.
    #[test]
    fn the_zeta_routes_cover_the_line() {
        assert_eq!(zeta_route(2.0), ZetaRoute::Hurwitz);
        assert_eq!(zeta_route(1.0), ZetaRoute::Value(f32::INFINITY));
        assert_eq!(zeta_route(0.5), ZetaRoute::Eta);
        assert_eq!(zeta_route(0.0), ZetaRoute::Value(-0.5));
        // `-0.5` is past `ZETA_ETA_MIN` now: the eta series is divergent for `s < 0` and the
        // reflection carries everything below `-1/8`, which took that band from 154 ulp to 5.
        assert_eq!(zeta_route(-0.5), ZetaRoute::Reflection);
        assert_eq!(zeta_route(ZETA_ETA_MIN), ZetaRoute::Reflection);
        assert_eq!(zeta_route(-0.0625), ZetaRoute::Eta);
        assert_eq!(zeta_route(-f32::MIN_POSITIVE), ZetaRoute::Eta);
        assert_eq!(zeta_route(-2.0), ZetaRoute::Value(0.0));
        assert_eq!(zeta_route(-3.0), ZetaRoute::Reflection);
        assert_eq!(zeta_route(f32::INFINITY), ZetaRoute::Value(1.0));
    }
}
