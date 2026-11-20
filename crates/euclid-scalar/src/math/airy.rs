//! Numerical steps shared by the two precisions of the Airy kernels.
//!
//! The three constants below are *mathematical* values, not measured thresholds, so one
//! definition serves both formats: each is stored at `f64` width and narrowed with
//! [`HostReal::from_f64`], which reproduces the single-precision literals the kernels used to
//! carry bit for bit. That is the distinction the tree's layout turns on — a constant that is the
//! same number in both formats belongs here, while one that is a different *measurement* in each
//! stays beside the code it governs.

use super::pair::{Slope, Solutions};
use crate::HostReal;

/// `Ai(0) = 3^{−2/3}/Γ(⅔)`.
pub(crate) const AI_0: f64 = 0.355_028_053_887_817_2;

/// `−Ai′(0) = 3^{−1/3}/Γ(⅓)`.
pub(crate) const NEG_AI_PRIME_0: f64 = 0.258_819_403_792_806_8;

/// `√3`, which relates `Bi` to `Ai` at the origin.
pub(crate) const SQRT_3: f64 = 1.732_050_807_568_877_2;

/// Combine the two entire solutions of `y″ = xy` into the Airy pair.
///
/// `f` and `g` are the even and odd Maclaurin solutions, each carrying its own derivative; the
/// Airy pair is a fixed linear combination of them, `Ai = c₁f − c₂g` and `Bi = √3(c₁f + c₂g)`,
/// with the same combination applied to the derivatives. The subtraction in `Ai` is the
/// cancellation that sites `AI_SERIES_MAX` in both kernels — it discards about `2ζ/ln 10` digits
/// — so the series is only used where that loss is under a digit, and each format measures that
/// point for itself.
///
/// In and out as [`Solutions<Slope<T>>`](Solutions): two independent solutions of a second-order
/// equation, each with its derivative, is what both sides of this are. It used to take four
/// loose scalars and return four more, in an order — `f`, `g`, `f′`, `g′` in, `Ai`, `Ai′`, `Bi`,
/// `Bi′` out — that *regrouped* silently in the middle, which is the kind of thing a reader has
/// to check rather than read.
pub(crate) fn combine<T: HostReal>(fg: Solutions<Slope<T>>) -> Solutions<Slope<T>> {
    let scaled_f = fg.first_kind.scaled_by(T::from_f64(AI_0));
    let scaled_g = fg.second_kind.scaled_by(T::from_f64(NEG_AI_PRIME_0));
    Solutions::new(
        scaled_f - scaled_g,
        (scaled_f + scaled_g).scaled_by(T::from_f64(SQRT_3)),
    )
}

/// Whether the oscillation still has a representable phase at this argument.
///
/// Both precisions' negative asymptotic carried a character-identical `if !z.is_finite()` guard
/// with a paragraph of comment attached, which is [`crate::math::scaling::PairGrowth`]'s
/// situation exactly: a branch that is the same decision in both formats, because it keys off
/// the range of a value the caller already computed rather than off any measured threshold.
///
/// Named so the honest answer in the second arm is stated once. `ζ = ⅔|x|^{3/2}` overflows above
/// `|x| ≈ 4.17e205` in `f64` and `6.39e25` in `f32`, and past that there is no argument to hand
/// to `sin` at all. Accuracy departs far earlier in both — around `|x| ≈ 1e11` and `1e5`, where
/// `ulp(ζ)` passes `2π` — so this is the point where the phase stops *existing*, not the point
/// where it stops being good.
///
/// Those two crossings are **measured here, and they are not the numbers this family used to
/// state.** Four doc comments across the two precisions gave `3e205` and `1.4e25`, which are both
/// comfortably inside the format: `ζ(3e205) = 1.1e308` against an `f64::MAX` of `1.8e308`, and
/// `ζ(1.4e25) = 3.5e37` against an `f32::MAX` of `3.4e38`. The test below bisects for the real
/// crossing rather than restating a figure, because a threshold nothing checks is how the first
/// four came to be wrong.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OscillatoryPhase<T> {
    /// `ζ` is finite: expand about this phase.
    Known(T),
    /// `ζ` has overflowed. Zero is finite, inside the decaying envelope, and the mean of the
    /// oscillation — which is the most that can be said once the phase is gone.
    Lost,
}

impl<T: HostReal> OscillatoryPhase<T> {
    /// The phase is formed once by the caller, and its range *is* the branch.
    pub(crate) fn for_zeta(z: T) -> Self {
        if HostReal::is_finite(z) {
            Self::Known(z)
        } else {
            Self::Lost
        }
    }
}

/// The leading-order scaled quadruple `(Aiᵉ, Aiᵉ′, Biᵉ, Biᵉ′)`, where every asymptotic correction
/// has fallen below rounding.
///
/// `Aiᵉ ~ 1/(2√π x^{1/4})` and `Biᵉ ~ 1/(√π x^{1/4})`, with the derivatives `∓x^{1/4}` over the
/// same factors — so the second pair is exactly twice the first and the derivatives carry the
/// opposite sign. Taking the fourth root as two square roots keeps it exact.
///
/// # Mutation testing is what asked for this being its own function
///
/// Fourteen mutants survived here, the largest single-function cluster in the crate: it is
/// selected only above each format's `LEADING_MIN`, where the *unscaled* surface returns
/// saturated limits and never consults it. Only the scaled forms reach it, and only a direct call
/// pins the arithmetic — which the two structural identities above now do without any reference.
pub(crate) fn leading_scaled<T: HostReal>(x: T) -> Solutions<Slope<T>> {
    let quarter = x.sqrt().sqrt();
    let root_pi = T::from_f64(crate::constants::f64::SQRT_PI);
    Solutions::new(
        Slope::new(
            T::ONE / (T::TWO * root_pi * quarter),
            -quarter / (T::TWO * root_pi),
        ),
        Slope::new(T::ONE / (root_pi * quarter), quarter / root_pi),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_narrowed_constants_are_the_literals_the_kernels_carried() {
        // The single-precision kernels used to spell these out; narrowing the `f64` value has to
        // reproduce them exactly, or centralising them would have moved a published number.
        assert_eq!(f32::from_f64(AI_0), 0.355_028_05_f32);
        assert_eq!(f32::from_f64(NEG_AI_PRIME_0), 0.258_819_4_f32);
        assert_eq!(f32::from_f64(SQRT_3), 1.732_050_8_f32);
        // And they are the values they claim to be.
        assert_eq!(SQRT_3, 3.0_f64.sqrt());
        assert!(
            (AI_0 - 3.0_f64.powf(-2.0 / 3.0) / crate::kernels::gamma::gamma(2.0 / 3.0)).abs()
                < 1e-15
        );
    }

    #[test]
    fn the_combination_is_the_airy_pair_at_the_origin() {
        // At the origin `f = 1`, `g = 0`, `f′ = 0`, `g′ = 1` — the initial conditions that pick
        // the two solutions out — so the combination returns the four values at zero directly:
        // `Ai(0) = c₁`, `Ai′(0) = −c₂`, and `Bi` is `√3` times each.
        let at_origin = Solutions::new(Slope::new(1.0_f64, 0.0), Slope::new(0.0, 1.0));
        let airy = combine(at_origin);
        assert_eq!(airy.first_kind.value, AI_0);
        assert_eq!(airy.first_kind.derivative, -NEG_AI_PRIME_0);
        assert_eq!(airy.second_kind.value, SQRT_3 * AI_0);
        assert_eq!(airy.second_kind.derivative, SQRT_3 * NEG_AI_PRIME_0);
        // `Bi(0)/Ai(0) = √3` and `Bi′(0)/Ai′(0) = −√3`, which is the relation the two signs
        // encode — and the only check that separates the subtraction from an addition.
        assert_eq!(airy.second_kind.value / airy.first_kind.value, SQRT_3);
        assert_eq!(
            airy.second_kind.derivative / airy.first_kind.derivative,
            -SQRT_3
        );
        // The same in `f32`.
        let airy = combine(Solutions::new(
            Slope::new(1.0_f32, 0.0),
            Slope::new(0.0, 1.0),
        ));
        assert_eq!(airy.first_kind.value, f32::from_f64(AI_0));
        assert_eq!(airy.first_kind.derivative, -f32::from_f64(NEG_AI_PRIME_0));
        let ratio = airy.second_kind.value / airy.first_kind.value;
        let ratio_prime = airy.second_kind.derivative / airy.first_kind.derivative;
        assert!((ratio - f32::from_f64(SQRT_3)).abs() < 1e-6);
        assert!((ratio_prime + f32::from_f64(SQRT_3)).abs() < 1e-6);
    }

    #[test]
    fn the_phase_is_known_until_zeta_leaves_the_format() {
        use OscillatoryPhase::{Known, Lost};
        assert_eq!(OscillatoryPhase::for_zeta(1.0_f64), Known(1.0));
        assert_eq!(OscillatoryPhase::for_zeta(0.0_f64), Known(0.0));
        assert_eq!(OscillatoryPhase::for_zeta(f64::MAX), Known(f64::MAX));
        assert_eq!(OscillatoryPhase::for_zeta(f64::INFINITY), Lost);
        assert_eq!(OscillatoryPhase::for_zeta(f64::NAN), Lost);
        assert_eq!(OscillatoryPhase::for_zeta(f32::MAX), Known(f32::MAX));
        assert_eq!(OscillatoryPhase::for_zeta(f32::INFINITY), Lost);

        // The argument at which each format actually loses the phase, bisected rather than
        // quoted. `ζ = ⅔|x|^{3/2}` is monotone, so the first `x` whose `ζ` is not finite is the
        // crossing, and the two sides of it are `Known` and `Lost` by construction.
        let crossing = |lost: &dyn Fn(f64) -> bool, mut lo: f64, mut hi: f64| {
            for _ in 0..200 {
                let mid = lo + (hi - lo) * 0.5;
                if mid == lo || mid == hi {
                    break;
                }
                if lost(mid) {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            lo
        };
        let zeta_64 = |x: f64| (2.0 / 3.0) * x * x.sqrt();
        let lost_64 = |x: f64| OscillatoryPhase::for_zeta(zeta_64(x)) == Lost;
        let edge = crossing(&lost_64, 1e205, 1e206);
        assert!(
            (edge / 4.173_881e205 - 1.0).abs() < 1e-5,
            "the f64 phase crossing moved to {edge:e}"
        );
        // Both sides of it, so the number above is a boundary and not just a magnitude.
        assert!(matches!(
            OscillatoryPhase::for_zeta(zeta_64(edge)),
            Known(_)
        ));
        assert!(lost_64(edge * 1.000_01));
        // `3e205`, which four doc comments used to claim, is well inside the format.
        assert!(matches!(
            OscillatoryPhase::for_zeta(zeta_64(3e205)),
            Known(_)
        ));

        let zeta_32 = |x: f32| (2.0 / 3.0) * x * x.sqrt();
        assert!(matches!(
            OscillatoryPhase::for_zeta(zeta_32(6.3e25)),
            Known(_)
        ));
        assert_eq!(OscillatoryPhase::for_zeta(zeta_32(6.4e25)), Lost);
        // And `1.4e25`, the figure the single-precision kernel used to carry.
        assert!(matches!(
            OscillatoryPhase::for_zeta(zeta_32(1.4e25)),
            Known(_)
        ));
    }

    #[test]
    fn the_leading_quadruple_is_the_asymptotic_envelope() {
        // `x = 1e4` makes `x^{1/4}` exactly `10`.
        let leading = leading_scaled(1e4_f64);
        assert_eq!(1e4_f64.sqrt().sqrt(), 10.0);
        let (ai, bi) = (leading.first_kind.value, leading.second_kind.value);
        assert!((ai - 0.028_209_479_177_387_815).abs() < 1e-15, "{ai}");
        assert!((bi - 0.056_418_958_354_775_63).abs() < 1e-15, "{bi}");
        // The four are not independent: `Biᵉ = 2Aiᵉ` and `Biᵉ′ = −2Aiᵉ′` exactly, since each pair
        // differs only by the factor of two in the denominator. That pins both prefactors
        // against each other with no reference at all.
        for x in [1e4_f64, 1e5, 1e6, 1e20, 1e300] {
            let (ai, bi) = (leading_scaled(x).first_kind, leading_scaled(x).second_kind);
            assert_eq!(bi.value / ai.value, 2.0, "at {x:e}");
            assert_eq!(bi.derivative / ai.derivative, -2.0, "at {x:e}");
            // `Aiᵉ` decays like `x^{-1/4}` and its derivative grows like `x^{1/4}`, so their
            // product is the constant `−1/(4π)`.
            let product = ai.value * ai.derivative;
            assert!(
                (product + 1.0 / (4.0 * core::f64::consts::PI)).abs() < 1e-15,
                "at {x:e}: {product}"
            );
            assert!(ai.value > 0.0 && bi.value > 0.0);
            assert!(ai.derivative < 0.0 && bi.derivative > 0.0);
        }
        // Monotone: the values fall and the derivatives rise as the argument grows.
        assert!(
            leading_scaled(1e4_f64).first_kind.value > leading_scaled(1e6_f64).first_kind.value
        );
        assert!(
            leading_scaled(1e4_f64).second_kind.derivative
                < leading_scaled(1e6_f64).second_kind.derivative
        );
        // And the same structure in `f32`, whose own `LEADING_MIN` is `1e4`.
        for x in [1e4_f32, 1e6, 1e20] {
            let leading = leading_scaled(x);
            let (ai, bi) = (leading.first_kind, leading.second_kind);
            assert_eq!(bi.value / ai.value, 2.0, "at {x:e}");
            assert_eq!(bi.derivative / ai.derivative, -2.0, "at {x:e}");
            let product = ai.value * ai.derivative;
            assert!((product + 1.0 / (4.0 * core::f32::consts::PI)).abs() < 1e-7);
        }
    }
}
