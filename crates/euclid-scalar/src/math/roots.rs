//! Numerical steps shared by bracketed inverse CDFs.

use crate::{HostReal, Interpolate, Real};

/// Whether the density admits a Newton step at all, and where it would land.
///
/// # This exists because the whole function could not be tested finely enough
///
/// Folded into [`bracketed_newton_step`], `density > 0` was an **unkillable mutant**: widening it
/// to `>= 0` computes `x − residual/0`, which is `±∞` for a nonzero residual and `NaN` for a zero
/// one, and every one of those then fails the bracket test and falls through to the same midpoint.
/// Two different decisions, one observable answer.
///
/// Splitting the decision out makes it observable — `Unusable` against `Towards(−∞)` is a
/// difference a test can see, where `3.5` against `3.5` is not. The guard is worth keeping as
/// `> 0` on its own terms, because it says a step is taken only where the density is *usable*;
/// relying on the division to produce an out-of-bracket candidate would be relying on an
/// accident.
#[derive(Clone, Copy, Debug, PartialEq)]
enum NewtonStep<T> {
    /// The density is usable. This is where the step lands, before the bracket is consulted.
    Towards(T),
    /// Zero, negative, or not a number — there is no step to take from here.
    Unusable,
}

impl<T: HostReal> NewtonStep<T> {
    fn from_density(x: T, residual: T, density: T) -> Self {
        if density > T::ZERO {
            Self::Towards(x - residual / density)
        } else {
            Self::Unusable
        }
    }
}

/// Take a Newton step inside the bracket, or bisect.
///
/// Two independent reasons to bisect, and they are kept apart in [`NewtonStep`] and here: the
/// density may be unusable, or the step may be usable and land outside the bracket. Safeguarding
/// against the second is what makes this a *bracketed* Newton rather than a Newton.
pub(crate) fn bracketed_newton_step<T>(x: T, residual: T, density: T, lo: T, hi: T) -> T
where
    T: HostReal + Interpolate<Scalar = T>,
{
    match NewtonStep::from_density(x, residual, density) {
        NewtonStep::Towards(candidate) if candidate > lo && candidate < hi => candidate,
        NewtonStep::Towards(_) | NewtonStep::Unusable => lo.midpoint(hi),
    }
}

/// Compare the bracket's width with the upper endpoint's scale.
pub(crate) fn bracket_is_narrow<T: HostReal>(lo: T, hi: T, relative_tolerance: T) -> bool {
    hi - lo <= relative_tolerance * Real::max(hi.abs(), T::MIN_POSITIVE)
}

/// Transform a standard normal quantile into a gamma starting point.
pub(crate) fn wilson_hilferty_guess<T: HostReal>(z: T, shape: T) -> T {
    let cube_root =
        T::ONE - T::ONE / (T::from_f64(9.0) * shape) + z / (T::from_f64(3.0) * shape.sqrt());
    shape * cube_root * cube_root * cube_root
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The density decision on its own, which the whole function cannot distinguish.
    ///
    /// `Unusable` against `Towards(±∞)` is the difference that `bracketed_newton_step` collapses:
    /// both bisect. Asserted here, where it is still visible, and that is the entire reason
    /// [`NewtonStep`] is a separate type.
    #[test]
    fn an_unusable_density_yields_no_candidate_at_all() {
        use NewtonStep::{Towards, Unusable};
        // A usable density gives the Newton point, bracket or no bracket.
        assert_eq!(NewtonStep::from_density(3.0_f64, 2.0, 2.0), Towards(2.0));
        assert_eq!(NewtonStep::from_density(6.0_f64, 2.0, 1.0), Towards(4.0));
        // Zero is unusable — and this is the case that matters. Widening the guard to `>= 0`
        // would make it `Towards(-inf)` here, which the full function cannot tell from
        // `Unusable` because both end at the midpoint.
        assert_eq!(NewtonStep::from_density(6.0_f64, 2.0, 0.0), Unusable);
        assert_eq!(NewtonStep::from_density(6.0_f64, 2.0, -0.0), Unusable);
        // As are a negative density and a `NaN` one.
        assert_eq!(NewtonStep::from_density(6.0_f64, 2.0, -2.0), Unusable);
        assert_eq!(NewtonStep::from_density(6.0_f64, 2.0, f64::NAN), Unusable);
        // A zero residual with a zero density is the `NaN`-producing case, and still unusable.
        assert_eq!(NewtonStep::from_density(6.0_f64, 0.0, 0.0), Unusable);
        // Both carriers, since the guard is generic.
        assert_eq!(NewtonStep::from_density(6.0_f32, 2.0, 0.0), Unusable);
        assert_eq!(NewtonStep::from_density(3.0_f32, 2.0, 2.0), Towards(2.0));
    }

    #[test]
    fn newton_acceptance_agrees_in_both_precisions() {
        // x, residual, density, lower endpoint, upper endpoint, expected next point.
        for [x, f, d, lo, hi, expected] in [
            [3.0_f64, 2.0, 2.0, 1.0, 3.0, 2.0],
            [6.0, 2.0, 2.0, 1.0, 6.0, 5.0],
            [6.0, 2.0, 0.0, 1.0, 6.0, 3.5],
            [6.0, 2.0, -0.0, 1.0, 6.0, 3.5],
            [6.0, -2.0, -2.0, 1.0, 6.0, 3.5],
            [6.0, 2.0, f64::NAN, 1.0, 6.0, 3.5],
            [6.0, 2.0, f64::INFINITY, 1.0, 6.0, 3.5],
            [6.0, 5.0, 1.0, 1.0, 6.0, 3.5],
            [1.0, -5.0, 1.0, 1.0, 6.0, 3.5],
        ] {
            assert_eq!(bracketed_newton_step(x, f, d, lo, hi), expected);
            let [x, f, d, lo, hi, expected] = [x, f, d, lo, hi, expected].map(|v| v as f32);
            assert_eq!(bracketed_newton_step(x, f, d, lo, hi), expected);
        }
    }

    #[test]
    fn width_check_includes_equality_and_scales_to_small_brackets() {
        let upper = 2.0_f64.powi(-800);
        assert!(bracket_is_narrow(0.875 * upper, upper, 0.25));
        assert!(bracket_is_narrow(0.75 * upper, upper, 0.25));
        assert!(!bracket_is_narrow(0.625 * upper, upper, 0.25));
        assert!(bracket_is_narrow(-0.25 * f32::MIN_POSITIVE, 0.0, 0.25));
        assert!(!bracket_is_narrow(-0.5 * f32::MIN_POSITIVE, 0.0, 0.25));
        assert!(bracket_is_narrow(f64::MAX, f64::MAX, f64::EPSILON));
        for hi in [f32::from_bits(1), f32::MIN_POSITIVE, 1.0, 1e37, f32::MAX] {
            assert!(bracket_is_narrow(hi.next_down(), hi, 4.0 * f32::EPSILON));
        }
    }

    #[test]
    fn wilson_hilferty_transform_matches_rational_inputs() {
        for (z, cube_root) in [(-1.5, 13.0 / 18.0), (0.0, 35.0 / 36.0), (1.5, 11.0 / 9.0)] {
            let expected = 4.0 * f64::powi(cube_root, 3);
            let f64_guess = wilson_hilferty_guess(z, 4.0_f64);
            let f32_guess = wilson_hilferty_guess(z as f32, 4.0_f32);
            assert!((f64_guess - expected).abs() <= 3e-16 * expected);
            assert!((f64::from(f32_guess) - expected).abs() <= 3e-7 * expected);
        }
    }
}
