//! Numerical steps shared by the two precisions of the Airy kernels.
//!
//! The three constants below are *mathematical* values, not measured thresholds, so one
//! definition serves both formats: each is stored at `f64` width and narrowed with
//! [`HostReal::from_f64`], which reproduces the single-precision literals the kernels used to
//! carry bit for bit. That is the distinction the tree's layout turns on — a constant that is the
//! same number in both formats belongs here, while one that is a different *measurement* in each
//! stays beside the code it governs.

use crate::HostReal;

/// `Ai(0) = 3^{−2/3}/Γ(⅔)`.
pub(crate) const AI_0: f64 = 0.355_028_053_887_817_2;

/// `−Ai′(0) = 3^{−1/3}/Γ(⅓)`.
pub(crate) const NEG_AI_PRIME_0: f64 = 0.258_819_403_792_806_8;

/// `√3`, which relates `Bi` to `Ai` at the origin.
pub(crate) const SQRT_3: f64 = 1.732_050_807_568_877_2;

/// Combine the two entire solutions of `y″ = xy` into the four Airy values.
///
/// `f` and `g` are the even and odd Maclaurin solutions; the Airy pair is a fixed linear
/// combination of them, `Ai = c₁f − c₂g` and `Bi = √3(c₁f + c₂g)`, with the same combination
/// applied to the derivatives. The subtraction in `Ai` is the cancellation that sites
/// `AI_SERIES_MAX` in both kernels — it discards about `2ζ/ln 10` digits — so the series is only
/// used where that loss is under a digit, and each format measures that point for itself.
pub(crate) fn combine<T: HostReal>(f: T, g: T, f_prime: T, g_prime: T) -> (T, T, T, T) {
    let (c1, c2) = (T::from_f64(AI_0), T::from_f64(NEG_AI_PRIME_0));
    let root_three = T::from_f64(SQRT_3);
    (
        c1 * f - c2 * g,
        c1 * f_prime - c2 * g_prime,
        root_three * (c1 * f + c2 * g),
        root_three * (c1 * f_prime + c2 * g_prime),
    )
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
pub(crate) fn leading_scaled<T: HostReal>(x: T) -> (T, T, T, T) {
    let quarter = x.sqrt().sqrt();
    let root_pi = T::from_f64(crate::constants::f64::SQRT_PI);
    (
        T::ONE / (T::TWO * root_pi * quarter),
        -quarter / (T::TWO * root_pi),
        T::ONE / (root_pi * quarter),
        quarter / root_pi,
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
        // At the origin `f = 1`, `g = 0`, `f′ = 0`, `g′ = 1`, so the combination returns the four
        // values at zero directly: `Ai(0) = c₁`, `Ai′(0) = −c₂`, and `Bi` is `√3` times each.
        let (ai, ai_prime, bi, bi_prime) = combine(1.0_f64, 0.0, 0.0, 1.0);
        assert_eq!(ai, AI_0);
        assert_eq!(ai_prime, -NEG_AI_PRIME_0);
        assert_eq!(bi, SQRT_3 * AI_0);
        assert_eq!(bi_prime, SQRT_3 * NEG_AI_PRIME_0);
        // `Bi(0)/Ai(0) = √3` and `Bi′(0)/Ai′(0) = −√3`, which is the relation the two signs
        // encode — and the only check that separates the subtraction from an addition.
        assert_eq!(bi / ai, SQRT_3);
        assert_eq!(bi_prime / ai_prime, -SQRT_3);
        // The same in `f32`.
        let (ai, ai_prime, bi, bi_prime) = combine(1.0_f32, 0.0, 0.0, 1.0);
        assert_eq!(ai, f32::from_f64(AI_0));
        assert_eq!(ai_prime, -f32::from_f64(NEG_AI_PRIME_0));
        assert!((bi / ai - f32::from_f64(SQRT_3)).abs() < 1e-6);
        assert!((bi_prime / ai_prime + f32::from_f64(SQRT_3)).abs() < 1e-6);
    }

    #[test]
    fn the_leading_quadruple_is_the_asymptotic_envelope() {
        // `x = 1e4` makes `x^{1/4}` exactly `10`.
        let (ai, _, bi, _) = leading_scaled(1e4_f64);
        assert_eq!(1e4_f64.sqrt().sqrt(), 10.0);
        assert!((ai - 0.028_209_479_177_387_815).abs() < 1e-15, "{ai}");
        assert!((bi - 0.056_418_958_354_775_63).abs() < 1e-15, "{bi}");
        // The four are not independent: `Biᵉ = 2Aiᵉ` and `Biᵉ′ = −2Aiᵉ′` exactly, since each pair
        // differs only by the factor of two in the denominator. That pins both prefactors
        // against each other with no reference at all.
        for x in [1e4_f64, 1e5, 1e6, 1e20, 1e300] {
            let (ai, ai_prime, bi, bi_prime) = leading_scaled(x);
            assert_eq!(bi / ai, 2.0, "at {x:e}");
            assert_eq!(bi_prime / ai_prime, -2.0, "at {x:e}");
            // `Aiᵉ` decays like `x^{-1/4}` and its derivative grows like `x^{1/4}`, so their
            // product is the constant `−1/(4π)`.
            let product = ai * ai_prime;
            assert!(
                (product + 1.0 / (4.0 * core::f64::consts::PI)).abs() < 1e-15,
                "at {x:e}: {product}"
            );
            assert!(ai > 0.0 && bi > 0.0 && ai_prime < 0.0 && bi_prime > 0.0);
        }
        // Monotone: the values fall and the derivatives rise as the argument grows.
        assert!(leading_scaled(1e4_f64).0 > leading_scaled(1e6_f64).0);
        assert!(leading_scaled(1e4_f64).3 < leading_scaled(1e6_f64).3);
        // And the same structure in `f32`, whose own `LEADING_MIN` is `1e4`.
        for x in [1e4_f32, 1e6, 1e20] {
            let (ai, ai_prime, bi, bi_prime) = leading_scaled(x);
            assert_eq!(bi / ai, 2.0, "at {x:e}");
            assert_eq!(bi_prime / ai_prime, -2.0, "at {x:e}");
            assert!((ai * ai_prime + 1.0 / (4.0 * core::f32::consts::PI)).abs() < 1e-7);
        }
    }
}
