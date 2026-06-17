//! The scalar traits, in two tiers.
//!
//! **Lazy tier** ([`Scalar`] → [`LazyReal`]/[`LazyComplex`]) — the minimal
//! *analysis* surface (`exp`/`ln`/`sqrt`, `conj`, magnitude) over the four
//! operators, `Clone` not `Copy`, and crucially **construction-free**: no way to
//! lift a scalar literal, so a lazy backend (a differentiable `Tensor`, a columnar
//! `Expr`) implements it without pretending a single `f64` is a tensor. A formula
//! written purely in analysis + operators runs on every backend.
//!
//! **Concrete tier** ([`Real`]/[`Complex`]) — a `Copy` real / complex *number* with
//! the full API, formed by **fusing** the lazy analysis with the algebraic field
//! ([`OrderedField`](crate::algebra::OrderedField) /
//! [`AlgebraicallyClosedField`](crate::algebra::AlgebraicallyClosedField)): one
//! bound gives operators, `ZERO`/`ONE`, `÷`, `conjugate`, ordering, *and*
//! `from_f64` (lift a literal) plus trig/`hypot`/rounding/constants. `Real: LazyReal`
//! and `Complex: LazyComplex`, so
//! a concrete number also satisfies a lazy formula — but a `Tensor` (no `Copy`, no
//! `PI`) is *only* lazy. Implemented by `f64`/`f32` (and `jacobi::Complex{32,64}`);
//! the geometry / matrix / physics kernels bound on the concrete tier.
//!
//! The word *field* stays reserved for the algebraic
//! [`Field`](crate::algebra::Field).

use core::ops::{Add, Div, Mul, Neg, Sub};

/// The arithmetic substrate: `Clone` plus the four operations and negation — the
/// bound behind `+ - * /` in a generic formula. Pure operators, no construction:
/// implemented by `f64`/`f32`, complex scalars, `Tensor`, and `Expr` alike.
/// Lifting a scalar *constant* into the type ([`Real::from_f64`]/[`Complex::from_f64`])
/// is a concrete-tier concern, not part of the lazy analysis surface.
pub trait Scalar:
    Clone
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
}

/// A real-valued numeric scalar: a [`Scalar`] plus the real transcendentals that
/// inherently-real formulas (probability densities, geometry kernels) are written
/// in.
pub trait LazyReal: Scalar {
    /// `e^x`.
    fn exp(&self) -> Self;
    /// Natural logarithm.
    fn ln(&self) -> Self;
    /// Square root.
    fn sqrt(&self) -> Self;
    /// Absolute value.
    fn abs(&self) -> Self;
    /// Raise to a real power.
    fn powf(&self, p: f64) -> Self;
    /// Numerically-stable `ln(1 + e^x)` (softplus / `log1pexp`).
    fn ln_1p_exp(&self) -> Self;
}

/// A complex-valued numeric scalar: a [`Scalar`] plus complex structure,
/// parameterized by its real-component type. Implemented by genuinely-complex
/// types — `Complex{32,64}` and a complex-dtype `Tensor` — **not** by the real
/// scalars (`f64`/`f32` are [`LazyReal`]s).
///
/// Construction from explicit `(re, im)` parts is **not** here; it lives on the
/// concrete complex constructors (e.g. `Complex64::new`,
/// `Tensor::complex_from_parts`).
pub trait LazyComplex: Scalar {
    /// The real-component type — `f64` for `Complex64`, `Self` for a dtype-erased
    /// complex `Tensor`.
    type RealPart: LazyReal;

    /// Complex conjugate.
    fn conj(&self) -> Self;
    /// Real part.
    fn re(&self) -> Self::RealPart;
    /// Imaginary part.
    fn im(&self) -> Self::RealPart;
    /// Squared magnitude `re² + im²`.
    fn norm_sqr(&self) -> Self::RealPart;
    /// The magnitude `|z| = √(re² + im²)`.
    fn magnitude(&self) -> Self::RealPart {
        self.norm_sqr().sqrt()
    }
}

/// A **concrete real number** with the full real API — [`LazyReal`] plus the
/// trigonometry, `hypot`/`atan2`, rounding, `mul_add`, NaN-aware `min`/`max`/`clamp`,
/// and the constants (`PI`, `TAU`, `EPSILON`, `INFINITY`, …) that geometry, physics,
/// and signal kernels are written in.
///
/// This is **distinct from [`LazyReal`]**, deliberately. `LazyReal` is the
/// minimal cross-backend analysis surface a *write-once* formula needs, so it is
/// implemented by `f64`/`f32` **and** non-`Copy` backends like a differentiable
/// `Tensor` (which has no single `PI` and no scalar `atan2`). `Real` is the richer
/// trait the *concrete* reals satisfy — `f64`, `f32`, and any future decimal /
/// interval / extended-precision type — so code that genuinely needs the whole
/// real number line (e.g. `hausdorff`'s geometry) bounds on `Real` and runs over
/// every such representation, while tensor backends remain unaffected.
///
/// It **fuses the two stacks** for the concrete reals: the algebraic structure
/// (`+ - * /`, `ZERO`/`ONE`, `÷`, `conjugate`, the total order) comes from
/// [`crate::algebra::OrderedField`]; `from_f64`/`sqrt`/`abs`/`exp`/`ln`/`powf` from
/// [`LazyReal`]; this trait adds only the remaining real API (trig, `hypot`,
/// rounding, the analytic constants). `Real: LazyReal`, so a concrete `f64` also
/// satisfies any write-once formula bounded on the lazy base.
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
    /// nearest representable for a decimal) — the concrete-tier constructor, needed
    /// wherever a generic formula carries an arbitrary literal or tolerance. (Named
    /// constants — `0`/`1`/`π`/… — use the consts below rather than this.)
    fn from_f64(x: f64) -> Self;

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

    // --- trigonometry ---
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn tan(self) -> Self;
    fn asin(self) -> Self;
    fn acos(self) -> Self;
    fn atan(self) -> Self;
    fn atan2(self, x: Self) -> Self;
    fn sinh(self) -> Self;
    fn cosh(self) -> Self;
    fn tanh(self) -> Self;
    /// `(sin, cos)` in one call.
    #[inline]
    fn sin_cos(self) -> (Self, Self) {
        (self.sin(), self.cos())
    }

    // --- powers / roots ---
    /// `√(self² + other²)` without intermediate overflow.
    fn hypot(self, other: Self) -> Self;
    fn cbrt(self) -> Self;
    /// Integer power.
    fn powi(self, n: i32) -> Self;
    /// `self * a + b` (fused where the target supports it).
    fn mul_add(self, a: Self, b: Self) -> Self;
    fn recip(self) -> Self;
    fn log2(self) -> Self;

    // --- rounding / sign ---
    fn floor(self) -> Self;
    fn ceil(self) -> Self;
    fn round(self) -> Self;
    fn trunc(self) -> Self;
    fn fract(self) -> Self;
    fn signum(self) -> Self;
    fn copysign(self, sign: Self) -> Self;
    /// Least non-negative residue of `self mod m`.
    fn rem_euclid(self, m: Self) -> Self;

    // --- angles ---
    fn to_radians(self) -> Self;
    fn to_degrees(self) -> Self;

    // --- NaN-aware ordering ---
    fn min(self, other: Self) -> Self;
    fn max(self, other: Self) -> Self;
    fn clamp(self, lo: Self, hi: Self) -> Self;

    // --- predicates ---
    fn is_finite(self) -> bool;
    fn is_nan(self) -> bool;
    fn is_sign_negative(self) -> bool;
}

/// A **concrete complex number** — the complex peer of [`Real`]. It fuses the
/// algebraically-closed field structure ([`crate::algebra::AlgebraicallyClosedField`])
/// with [`LazyComplex`] analysis (`conj`/`re`/`im`/`magnitude`), `Copy`, a concrete
/// [`Real`] component, and the complex transcendentals. Implemented by
/// `jacobi::Complex{32,64}`.
pub trait Complex:
    crate::algebra::AlgebraicallyClosedField + LazyComplex<RealPart: Real> + Copy
{
    /// Lift a real literal into the complex plane (`x + 0i`).
    fn from_f64(x: f64) -> Self;

    /// The phase angle `arg(z) ∈ (−π, π]`.
    fn argument(self) -> Self::RealPart;
    /// The principal square root.
    fn sqrt(self) -> Self;
    /// `e^z`.
    fn exp(self) -> Self;
    /// The principal natural logarithm.
    fn ln(self) -> Self;
    /// `z^p` for a real exponent.
    fn powf(self, p: f64) -> Self;
    /// The reciprocal `1/z`.
    fn recip(self) -> Self;
}

// --- f64 / f32: the real reference (also degenerate complex) ---------------

impl Scalar for f64 {}

impl LazyReal for f64 {
    #[inline]
    fn exp(&self) -> Self {
        f64::exp(*self)
    }
    #[inline]
    fn ln(&self) -> Self {
        f64::ln(*self)
    }
    #[inline]
    fn sqrt(&self) -> Self {
        f64::sqrt(*self)
    }
    #[inline]
    fn abs(&self) -> Self {
        f64::abs(*self)
    }
    #[inline]
    fn powf(&self, p: f64) -> Self {
        f64::powf(*self, p)
    }
    #[inline]
    fn ln_1p_exp(&self) -> Self {
        // Stable softplus: for x > 0 factor out x to avoid overflow in e^x; for
        // x <= 0 evaluate ln(1 + e^x) directly.
        let x = *self;
        if x > 0.0 {
            x + f64::ln_1p((-x).exp())
        } else {
            f64::ln_1p(x.exp())
        }
    }
}

impl Scalar for f32 {}

impl LazyReal for f32 {
    #[inline]
    fn exp(&self) -> Self {
        f32::exp(*self)
    }
    #[inline]
    fn ln(&self) -> Self {
        f32::ln(*self)
    }
    #[inline]
    fn sqrt(&self) -> Self {
        f32::sqrt(*self)
    }
    #[inline]
    fn abs(&self) -> Self {
        f32::abs(*self)
    }
    #[inline]
    fn powf(&self, p: f64) -> Self {
        f32::powf(*self, p as f32)
    }
    #[inline]
    fn ln_1p_exp(&self) -> Self {
        let x = *self;
        if x > 0.0 {
            x + f32::ln_1p((-x).exp())
        } else {
            f32::ln_1p(x.exp())
        }
    }
}

// The concrete `Real` impls. `ZERO`/`ONE` come from `algebra::{HasZero, HasOne}`,
// `÷`/`conjugate`/ordering from `algebra::OrderedField`, and `from_f64`/`sqrt`/
// `exp`/`ln`/`powf` from `LazyReal`; only the remaining real API is written here.

impl Real for f64 {
    #[inline]
    fn from_f64(x: f64) -> Self {
        x
    }
    const TWO: Self = 2.0;
    const HALF: Self = 0.5;
    const PI: Self = core::f64::consts::PI;
    const TAU: Self = core::f64::consts::TAU;
    const FRAC_PI_2: Self = core::f64::consts::FRAC_PI_2;
    const FRAC_PI_4: Self = core::f64::consts::FRAC_PI_4;
    const SQRT_2: Self = core::f64::consts::SQRT_2;
    const FRAC_1_SQRT_2: Self = core::f64::consts::FRAC_1_SQRT_2;
    const EPSILON: Self = f64::EPSILON;
    const INFINITY: Self = f64::INFINITY;
    const NEG_INFINITY: Self = f64::NEG_INFINITY;
    const NAN: Self = f64::NAN;
    const MAX: Self = f64::MAX;
    const MIN: Self = f64::MIN;
    const MIN_POSITIVE: Self = f64::MIN_POSITIVE;

    #[inline]
    fn sin(self) -> Self {
        f64::sin(self)
    }
    #[inline]
    fn cos(self) -> Self {
        f64::cos(self)
    }
    #[inline]
    fn tan(self) -> Self {
        f64::tan(self)
    }
    #[inline]
    fn asin(self) -> Self {
        f64::asin(self)
    }
    #[inline]
    fn acos(self) -> Self {
        f64::acos(self)
    }
    #[inline]
    fn atan(self) -> Self {
        f64::atan(self)
    }
    #[inline]
    fn atan2(self, x: Self) -> Self {
        f64::atan2(self, x)
    }
    #[inline]
    fn sinh(self) -> Self {
        f64::sinh(self)
    }
    #[inline]
    fn cosh(self) -> Self {
        f64::cosh(self)
    }
    #[inline]
    fn tanh(self) -> Self {
        f64::tanh(self)
    }
    #[inline]
    fn hypot(self, other: Self) -> Self {
        f64::hypot(self, other)
    }
    #[inline]
    fn cbrt(self) -> Self {
        f64::cbrt(self)
    }
    #[inline]
    fn powi(self, n: i32) -> Self {
        f64::powi(self, n)
    }
    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        f64::mul_add(self, a, b)
    }
    #[inline]
    fn recip(self) -> Self {
        f64::recip(self)
    }
    #[inline]
    fn log2(self) -> Self {
        f64::log2(self)
    }
    #[inline]
    fn floor(self) -> Self {
        f64::floor(self)
    }
    #[inline]
    fn ceil(self) -> Self {
        f64::ceil(self)
    }
    #[inline]
    fn round(self) -> Self {
        f64::round(self)
    }
    #[inline]
    fn trunc(self) -> Self {
        f64::trunc(self)
    }
    #[inline]
    fn fract(self) -> Self {
        f64::fract(self)
    }
    #[inline]
    fn signum(self) -> Self {
        f64::signum(self)
    }
    #[inline]
    fn copysign(self, sign: Self) -> Self {
        f64::copysign(self, sign)
    }
    #[inline]
    fn rem_euclid(self, m: Self) -> Self {
        f64::rem_euclid(self, m)
    }
    #[inline]
    fn to_radians(self) -> Self {
        f64::to_radians(self)
    }
    #[inline]
    fn to_degrees(self) -> Self {
        f64::to_degrees(self)
    }
    #[inline]
    fn min(self, other: Self) -> Self {
        f64::min(self, other)
    }
    #[inline]
    fn max(self, other: Self) -> Self {
        f64::max(self, other)
    }
    #[inline]
    fn clamp(self, lo: Self, hi: Self) -> Self {
        f64::clamp(self, lo, hi)
    }
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
    #[inline]
    fn from_f64(x: f64) -> Self {
        x as f32
    }
    const TWO: Self = 2.0;
    const HALF: Self = 0.5;
    const PI: Self = core::f32::consts::PI;
    const TAU: Self = core::f32::consts::TAU;
    const FRAC_PI_2: Self = core::f32::consts::FRAC_PI_2;
    const FRAC_PI_4: Self = core::f32::consts::FRAC_PI_4;
    const SQRT_2: Self = core::f32::consts::SQRT_2;
    const FRAC_1_SQRT_2: Self = core::f32::consts::FRAC_1_SQRT_2;
    const EPSILON: Self = f32::EPSILON;
    const INFINITY: Self = f32::INFINITY;
    const NEG_INFINITY: Self = f32::NEG_INFINITY;
    const NAN: Self = f32::NAN;
    const MAX: Self = f32::MAX;
    const MIN: Self = f32::MIN;
    const MIN_POSITIVE: Self = f32::MIN_POSITIVE;

    #[inline]
    fn sin(self) -> Self {
        f32::sin(self)
    }
    #[inline]
    fn cos(self) -> Self {
        f32::cos(self)
    }
    #[inline]
    fn tan(self) -> Self {
        f32::tan(self)
    }
    #[inline]
    fn asin(self) -> Self {
        f32::asin(self)
    }
    #[inline]
    fn acos(self) -> Self {
        f32::acos(self)
    }
    #[inline]
    fn atan(self) -> Self {
        f32::atan(self)
    }
    #[inline]
    fn atan2(self, x: Self) -> Self {
        f32::atan2(self, x)
    }
    #[inline]
    fn sinh(self) -> Self {
        f32::sinh(self)
    }
    #[inline]
    fn cosh(self) -> Self {
        f32::cosh(self)
    }
    #[inline]
    fn tanh(self) -> Self {
        f32::tanh(self)
    }
    #[inline]
    fn hypot(self, other: Self) -> Self {
        f32::hypot(self, other)
    }
    #[inline]
    fn cbrt(self) -> Self {
        f32::cbrt(self)
    }
    #[inline]
    fn powi(self, n: i32) -> Self {
        f32::powi(self, n)
    }
    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        f32::mul_add(self, a, b)
    }
    #[inline]
    fn recip(self) -> Self {
        f32::recip(self)
    }
    #[inline]
    fn log2(self) -> Self {
        f32::log2(self)
    }
    #[inline]
    fn floor(self) -> Self {
        f32::floor(self)
    }
    #[inline]
    fn ceil(self) -> Self {
        f32::ceil(self)
    }
    #[inline]
    fn round(self) -> Self {
        f32::round(self)
    }
    #[inline]
    fn trunc(self) -> Self {
        f32::trunc(self)
    }
    #[inline]
    fn fract(self) -> Self {
        f32::fract(self)
    }
    #[inline]
    fn signum(self) -> Self {
        f32::signum(self)
    }
    #[inline]
    fn copysign(self, sign: Self) -> Self {
        f32::copysign(self, sign)
    }
    #[inline]
    fn rem_euclid(self, m: Self) -> Self {
        f32::rem_euclid(self, m)
    }
    #[inline]
    fn to_radians(self) -> Self {
        f32::to_radians(self)
    }
    #[inline]
    fn to_degrees(self) -> Self {
        f32::to_degrees(self)
    }
    #[inline]
    fn min(self, other: Self) -> Self {
        f32::min(self, other)
    }
    #[inline]
    fn max(self, other: Self) -> Self {
        f32::max(self, other)
    }
    #[inline]
    fn clamp(self, lo: Self, hi: Self) -> Self {
        f32::clamp(self, lo, hi)
    }
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
    use super::*;

    /// A formula written once over the lazy analysis base — pure analysis, no
    /// scalar construction (that lives on the concrete tier's `from_f64`).
    fn quadratic<T: LazyReal>(x: T) -> T {
        x.clone() * x.clone() + x
    }

    #[test]
    fn real_formula_runs_on_f64_and_f32() {
        assert!((quadratic(2.5_f64) - 8.75).abs() < 1e-12);
        assert!((quadratic(2.5_f32) - 8.75).abs() < 1e-5);
    }

    #[test]
    fn softplus_is_stable_at_extremes() {
        assert!((f64::ln_1p_exp(&0.0) - core::f64::consts::LN_2).abs() < 1e-12);
        let big = 50.0_f64;
        assert!((big.ln_1p_exp() - big).abs() < 1e-10);
    }

    /// A point on the unit circle — written once over the fused [`Real`] trait,
    /// which carries algebra (operators), analysis (`LazyReal`), and the real API
    /// (`sin_cos`/`PI`) through a single bound.
    fn unit_circle_point<T: Real>(theta: T) -> (T, T) {
        let (s, c) = theta.sin_cos();
        (c, s)
    }

    #[test]
    fn fused_real_runs_on_f64_and_f32() {
        // The fused trait at f64: trig + constants.
        let (x, y) = unit_circle_point(f64::PI);
        assert!((x + 1.0).abs() < 1e-12 && y.abs() < 1e-12);
        // ...and at f32, mixing analysis (hypot) with the algebra identity ONE.
        assert!((3.0_f32.hypot(4.0) - 5.0).abs() < 1e-5);
        assert_eq!(<f32 as crate::algebra::HasOne>::ONE, 1.0);
        // A concrete `Real` also satisfies a write-once `LazyReal` formula.
        assert!((quadratic(2.0_f64) - 6.0).abs() < 1e-12);
    }
}
