//! The concrete real number [`Real`] — euclid's `OrderedField` algebra fused with
//! its analysis surface (euclid's own `Real`, imported as `LazyReal` here) into a
//! `Copy` number on the real line, plus the
//! `f64`/`f32` implementations.

use euclid_numeric::Real as LazyReal;

/// A **concrete real number** — [`LazyReal`](euclid_numeric::Real) (the whole
/// elementwise analysis surface) plus the few things that only a concrete number on
/// the real line can offer: the associated constants (`PI`, `TAU`, `EPSILON`,
/// `INFINITY`, …) and the `bool`-returning predicates
/// (`is_finite`/`is_nan`/`is_sign_negative`).
///
/// This is **distinct from [`LazyReal`](euclid_numeric::Real)**, deliberately.
/// `LazyReal` is the cross-backend analysis surface a *write-once* formula needs, so
/// it lives in `euclid` and is implemented by `f64`/`f32` **and** non-`Copy`
/// backends like a differentiable `Tensor`. The transcendentals, rounding, and
/// `min`/`max`/`clamp` selectors all live there, because a tensor compiler can lower
/// every one of them. `Real` keeps only what a tensor *cannot* produce — a single
/// `PI` value, a scalar `bool` from a comparison — so code that genuinely needs the
/// whole real number line (e.g. `hausdorff`'s geometry) bounds on `Real` and runs
/// over `f64`, `f32`, and any future decimal / interval / extended-precision type,
/// while tensor backends remain unaffected.
///
/// It **fuses the two foundations** for the concrete reals: the algebraic structure
/// (`+ - * /`, `ZERO`/`ONE`, `÷`, `conjugate`, the total order) comes from
/// [`crate::algebra::OrderedField`]; the analysis
/// (`scalar`/`exp`/`ln`/`sqrt`/trig/rounding/`min`/`max`/`clamp`) from
/// [`LazyReal`](euclid_numeric::Real); this trait adds only the constants and
/// predicates. `Real: LazyReal`, so a concrete `f64` also satisfies any write-once
/// formula bounded on the lazy base.
pub trait Real:
    crate::algebra::OrderedField
    + LazyReal
    + Copy
    + core::ops::AddAssign
    + core::ops::SubAssign
    + core::ops::MulAssign
    + core::ops::DivAssign
    + core::ops::Rem<Output = Self>
    + core::iter::Sum
{
    /// Lift a real literal into the type (`f64` identity, `x as f32` for `f32`, the
    /// nearest representable for a decimal) — the concrete-tier spelling for
    /// [`LazyReal::scalar`](euclid_numeric::Real::scalar).
    #[inline]
    fn from_f64(x: f64) -> Self {
        Self::scalar(x)
    }

    /// `2`.
    const TWO: Self;
    /// `0.5`.
    const HALF: Self;
    /// Archimedes' constant `π`.
    const PI: Self;
    /// `2π`.
    const TAU: Self;
    /// `π/2`.
    const FRAC_PI_2: Self;
    /// `π/4`.
    const FRAC_PI_4: Self;
    /// `√2`.
    const SQRT_2: Self;
    /// `1/√2`.
    const FRAC_1_SQRT_2: Self;
    /// Euler's number `e`.
    const E: Self;
    /// Machine epsilon.
    const EPSILON: Self;
    /// `+∞`.
    const INFINITY: Self;
    /// `−∞`.
    const NEG_INFINITY: Self;
    /// Not-a-number.
    const NAN: Self;
    /// Largest finite value.
    const MAX: Self;
    /// Smallest (most negative) finite value.
    const MIN: Self;
    /// Smallest positive normal value.
    const MIN_POSITIVE: Self;

    // --- predicates ---
    // The only methods that stay concrete-only: they return `bool`, which the lazy
    // tier cannot produce (a tensor comparison yields a mask, not a scalar bool).
    // The elementwise analysis (trig, rounding, `min`/`max`/`clamp`, …) lives on
    // [`LazyReal`](euclid_numeric::Real) and is inherited here through
    // `Real: LazyReal`.
    fn is_finite(self) -> bool;
    fn is_nan(self) -> bool;
    fn is_sign_negative(self) -> bool;
}

impl Real for f64 {
    const TWO: Self = 2.0;
    const HALF: Self = 0.5;
    const PI: Self = core::f64::consts::PI;
    const TAU: Self = core::f64::consts::TAU;
    const FRAC_PI_2: Self = core::f64::consts::FRAC_PI_2;
    const FRAC_PI_4: Self = core::f64::consts::FRAC_PI_4;
    const SQRT_2: Self = core::f64::consts::SQRT_2;
    const FRAC_1_SQRT_2: Self = core::f64::consts::FRAC_1_SQRT_2;
    const E: Self = core::f64::consts::E;
    const EPSILON: Self = f64::EPSILON;
    const INFINITY: Self = f64::INFINITY;
    const NEG_INFINITY: Self = f64::NEG_INFINITY;
    const NAN: Self = f64::NAN;
    const MAX: Self = f64::MAX;
    const MIN: Self = f64::MIN;
    const MIN_POSITIVE: Self = f64::MIN_POSITIVE;

    #[inline]
    fn is_finite(self) -> bool {
        f64::is_finite(self)
    }
    #[inline]
    fn is_nan(self) -> bool {
        f64::is_nan(self)
    }
    #[inline]
    fn is_sign_negative(self) -> bool {
        f64::is_sign_negative(self)
    }
}

impl Real for f32 {
    const TWO: Self = 2.0;
    const HALF: Self = 0.5;
    const PI: Self = core::f32::consts::PI;
    const TAU: Self = core::f32::consts::TAU;
    const FRAC_PI_2: Self = core::f32::consts::FRAC_PI_2;
    const FRAC_PI_4: Self = core::f32::consts::FRAC_PI_4;
    const SQRT_2: Self = core::f32::consts::SQRT_2;
    const FRAC_1_SQRT_2: Self = core::f32::consts::FRAC_1_SQRT_2;
    const E: Self = core::f32::consts::E;
    const EPSILON: Self = f32::EPSILON;
    const INFINITY: Self = f32::INFINITY;
    const NEG_INFINITY: Self = f32::NEG_INFINITY;
    const NAN: Self = f32::NAN;
    const MAX: Self = f32::MAX;
    const MIN: Self = f32::MIN;
    const MIN_POSITIVE: Self = f32::MIN_POSITIVE;

    #[inline]
    fn is_finite(self) -> bool {
        f32::is_finite(self)
    }
    #[inline]
    fn is_nan(self) -> bool {
        f32::is_nan(self)
    }
    #[inline]
    fn is_sign_negative(self) -> bool {
        f32::is_sign_negative(self)
    }
}

#[cfg(test)]
mod tests {
    use super::Real;
    use euclid_numeric::Real as LazyReal;

    /// A point on the unit circle — written once over the fused [`Real`] trait,
    /// which carries the algebra (operators, `ONE`), the analysis (`LazyReal`), and
    /// the concrete real API (`sin_cos`/`PI`) through a single bound.
    fn unit_circle_point<T: Real>(theta: T) -> (T, T) {
        let (s, c) = theta.sin_cos();
        (c, s)
    }

    #[test]
    fn fused_real_runs_on_f64_and_f32() {
        // The fused trait at f64: trig + the `PI` constant.
        let (x, y) = unit_circle_point(f64::PI);
        assert!((x + 1.0).abs() < 1e-12 && y.abs() < 1e-12);
        // ...and at f32, mixing analysis (`hypot`, inherited from `LazyReal`) with
        // the algebra identity `ONE` and the concrete `Real` constants.
        assert!((3.0_f32.hypot(4.0) - 5.0).abs() < 1e-5);
        assert_eq!(<f32 as crate::algebra::HasOne>::ONE, 1.0);
        assert_eq!(f64::e(), <f64 as Real>::E);
        assert_eq!(<f32 as Real>::TWO, 2.0);
    }

    #[test]
    fn real_constants_and_predicates() {
        assert!(!<f64 as Real>::INFINITY.is_finite());
        assert!(<f64 as Real>::NAN.is_nan());
        assert!((-1.0_f64).is_sign_negative());
        assert!((<f64 as Real>::PI - core::f64::consts::PI).abs() < 1e-15);
        // `from_f64` is the concrete-tier spelling of `LazyReal::scalar`.
        assert_eq!(<f64 as Real>::from_f64(2.5), 2.5);
        assert_eq!(<f32 as Real>::from_f64(2.5), 2.5_f32);
    }
}
