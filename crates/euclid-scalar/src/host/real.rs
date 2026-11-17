//! The concrete real number [`HostReal`] — [`euclid_algebra::OrderedField`] structure fused with
//! [`crate::Real`] analysis into a `Copy` number on the real line, plus the `f64`/`f32`
//! implementations.
//!
//! The backend-neutral [`crate::Real`] and eager [`HostReal`] traits share one crate.
//! The distinct names let a formula state whether it needs a deferred carrier or a host value.

/// A **concrete real number** — [`crate::Real`] (the whole
/// elementwise analysis surface) plus the few things that only a concrete number on
/// the real line can offer: the associated constants (`PI`, `TAU`, `EPSILON`,
/// `INFINITY`, …) and the `bool`-returning predicates
/// (`is_finite`/`is_nan`/`is_sign_negative`).
///
/// This is **distinct from [`crate::Real`]**, deliberately.
/// `crate::Real` is the cross-backend analysis surface a *write-once* formula needs, so
/// it lives in `euclid-scalar` and is implemented by `f64`/`f32` **and** non-`Copy`
/// backends like a differentiable `Tensor`. The transcendentals, rounding, and
/// `min`/`max`/`clamp` selectors all live there, because a tensor compiler can lower
/// every one of them. `HostReal` keeps only what a tensor *cannot* produce — a single
/// `PI` value, a scalar `bool` from a comparison — so code that genuinely needs the
/// whole real number line (e.g. `hausdorff`'s geometry) bounds on `HostReal` and runs
/// over `f64`, `f32`, and any future decimal / interval / extended-precision type,
/// while tensor backends remain unaffected.
///
/// It **fuses the two foundations** for the concrete reals: the algebraic structure
/// (`+ - * /`, `ZERO`/`ONE`, `÷`, `conjugate`, the total order) comes from
/// [`euclid_algebra::OrderedField`]; the analysis
/// (`scalar`/`exp`/`ln`/`sqrt`/trig/rounding/`min`/`max`/`clamp`) from
/// [`crate::Real`]; this trait adds only the constants and
/// predicates. `HostReal: crate::Real`, so a concrete `f64` also satisfies any write-once
/// formula bounded on the contract tier.
pub trait HostReal:
    euclid_algebra::OrderedField
    + crate::Real
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
    /// [`crate::Real::scalar`].
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
    // The only methods that stay concrete-only: they return `bool`, which the contract
    // tier cannot produce (a tensor comparison yields a mask, not a scalar bool).
    // The elementwise analysis (trig, rounding, `min`/`max`/`clamp`, …) lives on
    // [`crate::Real`] and is inherited here through
    // `HostReal: crate::Real`.
    /// Neither infinite nor `NaN`.
    fn is_finite(self) -> bool;
    /// Not a number. A `NaN` is not equal to itself, so this is the only way to test for one.
    fn is_nan(self) -> bool;
    /// The sign **bit** is set, which is a question about the representation rather than the
    /// value — so `(-0.0).is_sign_negative()` is `true` while `-0.0 < 0.0` is `false`. Both are
    /// right, and a predicate written as `x < ZERO` is a different test.
    fn is_sign_negative(self) -> bool;
}

impl HostReal for f64 {
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

impl HostReal for f32 {
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
    use super::HostReal;

    /// A point on the unit circle — written once over the fused [`HostReal`] trait,
    /// which carries the algebra (operators, `ONE`), the analysis (`crate::Real`), and
    /// the concrete real API (`sin_cos`/`PI`) through a single bound.
    fn unit_circle_point<T: HostReal>(theta: T) -> (T, T) {
        let (s, c) = theta.sin_cos();
        (c, s)
    }

    #[test]
    fn fused_real_runs_on_f64_and_f32() {
        // The fused trait at f64: trig + the `PI` constant.
        let (x, y) = unit_circle_point(f64::PI);
        assert!((x + 1.0).abs() < 1e-12 && y.abs() < 1e-12);
        // ...and at f32, mixing analysis (`hypot`, inherited from `crate::Real`) with
        // the algebra identity `ONE` and the concrete `HostReal` constants.
        assert!((3.0_f32.hypot(4.0) - 5.0).abs() < 1e-5);
        assert_eq!(<f32 as euclid_algebra::HasOne>::ONE, 1.0);
        // Both tiers at once: the analysis `e()` and the host constant `E`.
        assert_eq!(<f64 as crate::Real>::e(), <f64 as HostReal>::E);
        assert_eq!(<f32 as HostReal>::TWO, 2.0);
    }

    #[test]
    fn real_constants_and_predicates() {
        assert!((<f64 as HostReal>::PI - core::f64::consts::PI).abs() < 1e-15);
        // `from_f64` is the concrete-tier spelling of `crate::Real::scalar`.
        assert_eq!(<f64 as HostReal>::from_f64(2.5), 2.5);
        assert_eq!(<f32 as HostReal>::from_f64(2.5), 2.5_f32);
    }

    #[test]
    fn host_predicates_distinguish_signed_zero_and_nonfinite_values() {
        fn check<T: HostReal>() {
            for (value, finite, nan) in [
                (T::ZERO, true, false),
                (T::MIN_POSITIVE, true, false),
                (T::INFINITY, false, false),
                (T::NEG_INFINITY, false, false),
                (T::NAN, false, true),
            ] {
                assert_eq!(HostReal::is_finite(value), finite);
                assert_eq!(HostReal::is_nan(value), nan);
            }
            assert!(!HostReal::is_sign_negative(T::ZERO));
            assert!(HostReal::is_sign_negative(-T::ZERO));
            assert!(!HostReal::is_sign_negative(T::INFINITY));
            assert!(HostReal::is_sign_negative(T::NEG_INFINITY));
        }

        check::<f32>();
        check::<f64>();
    }
}
