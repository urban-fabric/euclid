//! Recovering an unscaled value from a scaled one, where the exponential does not fit.
//!
//! Several families in this crate compute a *scaled* form as the primitive — `Aiᵉ`, `Biᵉ`, `Iᵉ`,
//! `Kᵉ`, `erfcx` — because the unscaled value leaves the format long before the argument stops
//! being useful. Recovering the unscaled value is then one multiplication by an exponential that
//! is *allowed* to overflow, as the true value does. The hazard is the band in between, where the
//! exponential has left the format and the product has not.
//!
//! Centralised because both precisions of [`crate::kernels::airy`] carried a character-identical
//! copy. The split has no format-dependent constant in it — it keys off the range of a value the
//! caller already computed — so there was nothing per-precision to rederive.

use super::pair::Slope;
use crate::HostReal;

/// Whether a growth factor fits, or has to be applied as two halves.
///
/// # Mutation testing is what asked for this
///
/// Eight mutants survived across the two precisions' copies, the pair's **derivative** among
/// them. The band that selects the split arm is `x ∈ (104.27, 104.44)` in `f64` and `(26.07,
/// 26.34)` in `f32`, and the sweep that reaches it sweeps `airy_bi` without an `airy_bi_prime`
/// row — so the derivative crossed it unchecked in both. Named and generic, the arm is reachable
/// at any exponent and both components of the pair can be asserted, once.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PairGrowth {
    /// `e^ζ` fits: scale both components by it directly.
    Direct,
    /// `e^ζ` has left the format while the product has not — `Biᵉ` is about `0.25` where this
    /// happens, so the product has a quarter of the range in hand. Applying `e^{ζ/2}` twice keeps
    /// both factors in range, and unlike recovering through a logarithm it costs no `ζ·ε`.
    Split,
}

impl PairGrowth {
    /// The growth factor is formed once by [`unscale_pair`], and its range *is* the branch.
    pub(crate) fn for_growth<T: HostReal>(growth: T) -> Self {
        if HostReal::is_finite(growth) {
            Self::Direct
        } else {
            Self::Split
        }
    }
}

/// Multiply a value and its derivative by `e^ζ`, splitting the exponential where it would
/// overflow.
///
/// Both components take the same factor — the scaling is in the dependent variable — which is
/// why [`Slope::scaled_by`] applies and why the two can never be transposed here.
pub(crate) fn unscale_pair<T: HostReal>(scaled: Slope<T>, z: T) -> Slope<T> {
    let growth = z.exp();
    match PairGrowth::for_growth(growth) {
        PairGrowth::Direct => scaled.scaled_by(growth),
        PairGrowth::Split => {
            let half = (z * T::HALF).exp();
            scaled.scaled_by(half).scaled_by(half)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_growth_split_follows_the_factors_range() {
        use PairGrowth::{Direct, Split};
        assert_eq!(PairGrowth::for_growth(1.0_f64), Direct);
        assert_eq!(PairGrowth::for_growth(f64::MAX), Direct);
        assert_eq!(PairGrowth::for_growth(0.0_f64), Direct);
        assert_eq!(PairGrowth::for_growth(f64::INFINITY), Split);
        assert_eq!(PairGrowth::for_growth(f64::NAN), Split);
        assert_eq!(PairGrowth::for_growth(f32::MAX), Direct);
        assert_eq!(PairGrowth::for_growth(f32::INFINITY), Split);
    }

    #[test]
    fn the_split_applies_the_root_twice_to_both_components() {
        // Where `e^ζ` fits, both components are scaled by it directly.
        for z in [1.0_f64, 10.0, 100.0] {
            let growth = z.exp();
            assert_eq!(
                unscale_pair(Slope::new(0.25, 3.0), z).into_parts(),
                (0.25 * growth, 3.0 * growth)
            );
        }

        // Past `ln(MAX) = 709.78` the growth factor is gone. Passing `ζ` directly reaches that
        // arm with a pair small enough that both products are ordinary numbers — which no Airy
        // argument can produce, and which is the whole reason this is a named component.
        let z = 710.0_f64;
        assert!(!z.exp().is_finite());
        let half = (z * 0.5).exp();
        let (value, derivative) = unscale_pair(Slope::new(1e-300, 2e-300), z).into_parts();
        assert_eq!(value, (1e-300 * half) * half);
        assert_eq!(derivative, (2e-300 * half) * half);
        // Both components, and the derivative is the one that was unchecked: `2.234e8` and
        // `4.468e8`, exactly twice it, so a mutated association is more than a rounding away.
        assert!((value - 223_399_476.616_171_15).abs() < 1.0, "{value}");
        assert!(
            (derivative - 446_798_953.232_342_3).abs() < 1.0,
            "{derivative}"
        );
        assert_eq!(derivative / value, 2.0);

        // And at the magnitude `Biᵉ` actually carries — about `0.25` — the split keeps a value
        // inside the format where `e^ζ` alone is `+∞`.
        let (value, derivative) = unscale_pair(Slope::new(0.25_f64, 0.25), z).into_parts();
        assert!(value.is_finite() && derivative.is_finite());
        assert!(
            (value / 5.584_986_915_404_278e307 - 1.0).abs() < 1e-12,
            "{value}"
        );
        assert_eq!(value, derivative);

        // The two arms agree either side of the crossing, which is what makes the guard free.
        for z in [709.0_f64, 709.7] {
            assert!(z.exp().is_finite());
            let half = (z * 0.5).exp();
            let (value, _) = unscale_pair(Slope::new(1e-300, 1e-300), z).into_parts();
            let split = (1e-300 * half) * half;
            assert!(
                (value - split).abs() <= 4.0 * f64::EPSILON * split,
                "at {z}"
            );
        }
    }

    #[test]
    fn the_split_reaches_the_top_of_single_precision_too() {
        // `ln(f32::MAX) = 88.7228`, so this format's band begins thirty times lower.
        let z = 90.0_f32;
        assert!(!z.exp().is_finite());
        let half = (z * 0.5).exp();
        let (value, derivative) = unscale_pair(Slope::new(1e-30, 2e-30), z).into_parts();
        assert_eq!(value, (1e-30 * half) * half);
        assert_eq!(derivative, (2e-30 * half) * half);
        assert_eq!(value, 1_220_403_300.0);
        assert_eq!(derivative, 2_440_806_700.0);
        assert_eq!(derivative / value, 2.0);
        // `3.051e38` against an `f32::MAX` of `3.4028e38`: inside the format, where `e^ζ` is not.
        let (value, derivative) = unscale_pair(Slope::new(0.25_f32, 0.25), z).into_parts();
        assert!(value.is_finite() && derivative.is_finite());
        assert!((value / 3.051_008_2e38 - 1.0).abs() < 1e-6, "{value}");
        assert_eq!(value, derivative);
    }
}
