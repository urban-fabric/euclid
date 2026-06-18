//! The scalar traits, in two tiers.
//!
//! **Lazy tier** ([`Scalar`] → [`LazyReal`]/[`LazyComplex`]) — the *elementwise
//! analysis* surface over the four operators, `Clone` not `Copy`: the
//! transcendentals (`exp`/`ln`/`sqrt`, trig, `atan2`/`hypot`), rounding/sign, the
//! comparison-free `min`/`max`/`clamp` selectors, `conj`/magnitude, plus
//! scalar-literal lifting for the constants formulas name. Every operation here is
//! something a lazy backend (a differentiable `Tensor`, a columnar `Expr`) can
//! lower — a tensor compiler maps them to `Sin`/`Cos`/`Max`/`Select`/… — so a pure
//! formula written in analysis + operators + scalar constants runs on every
//! backend. What is *not* here: anything that returns a scalar `bool` (a tensor
//! comparison yields a mask, not a bool) or that needs the concrete number line.
//!
//! **Concrete tier** ([`Real`]/[`Complex`]) — a `Copy` real / complex *number* with
//! the full API, formed by **fusing** the lazy analysis with the algebraic field
//! ([`OrderedField`](crate::algebra::OrderedField) /
//! [`AlgebraicallyClosedField`](crate::algebra::AlgebraicallyClosedField)): one
//! bound gives operators, `ZERO`/`ONE`, `÷`, `conjugate`, ordering, *and*
//! `from_f64` (the concrete-tier spelling for [`LazyReal::scalar`]) plus
//! trig/`hypot`/rounding/constants. `Real: LazyReal` and `Complex: LazyComplex`, so
//! a concrete number also satisfies a lazy formula — but a `Tensor` (no `Copy`, no
//! associated constants) is *only* lazy. Implemented by `f64`/`f32` (and
//! `jacobi::Complex{32,64}`); the geometry / matrix / physics kernels bound on
//! the concrete tier.
//!
//! The word *field* stays reserved for the algebraic
//! [`Field`](crate::algebra::Field).

use core::ops::{Add, Div, Mul, Neg, Sub};

/// The arithmetic substrate: `Clone` plus the four operations and negation — the
/// bound behind `+ - * /` in a generic formula. Pure operators, no scalar or vector
/// construction: implemented by `f64`/`f32`, complex scalars, `Tensor`, and `Expr`
/// alike.
/// Lifting a scalar *constant* into a real lazy substrate lives on [`LazyReal`],
/// because probability/statistics formulas need constants without giving up lazy
/// execution.
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
    /// Lift a host scalar into this real substrate.
    ///
    /// For `f64` this is the identity and for `f32` it is a narrowing cast. Lazy
    /// graph backends should represent it as a scalar constant node, ideally
    /// interned or otherwise cheap to reuse.
    fn scalar(x: f64) -> Self;

    /// Lift a host scalar using `self` as graph/device/shape context.
    ///
    /// The default is a scalar constant. Tensor or expression backends may
    /// override this when constants need to live in the same arena, dtype, device,
    /// or broadcast context as an existing value.
    #[inline]
    fn scalar_like(&self, x: f64) -> Self {
        Self::scalar(x)
    }

    /// `0`.
    #[inline]
    fn zero() -> Self {
        Self::scalar(0.0)
    }

    /// `1`.
    #[inline]
    fn one() -> Self {
        Self::scalar(1.0)
    }

    /// `2`.
    #[inline]
    fn two() -> Self {
        Self::scalar(2.0)
    }

    /// `0.5`.
    #[inline]
    fn half() -> Self {
        Self::scalar(0.5)
    }

    /// Archimedes' constant `π`.
    #[inline]
    fn pi() -> Self {
        Self::scalar(core::f64::consts::PI)
    }

    /// `2π`.
    #[inline]
    fn tau() -> Self {
        Self::scalar(core::f64::consts::TAU)
    }

    /// `π/2`.
    #[inline]
    fn frac_pi_2() -> Self {
        Self::scalar(core::f64::consts::FRAC_PI_2)
    }

    /// `π/4`.
    #[inline]
    fn frac_pi_4() -> Self {
        Self::scalar(core::f64::consts::FRAC_PI_4)
    }

    /// `√2`.
    #[inline]
    fn sqrt_2() -> Self {
        Self::scalar(core::f64::consts::SQRT_2)
    }

    /// `1/√2`.
    #[inline]
    fn frac_1_sqrt_2() -> Self {
        Self::scalar(core::f64::consts::FRAC_1_SQRT_2)
    }

    /// `ln 2`.
    #[inline]
    fn ln_2() -> Self {
        Self::scalar(core::f64::consts::LN_2)
    }

    /// `ln π`.
    #[inline]
    fn ln_pi() -> Self {
        Self::scalar(core::f64::consts::PI.ln())
    }

    /// `ln τ`, where `τ = 2π`.
    #[inline]
    fn ln_tau() -> Self {
        Self::scalar(core::f64::consts::TAU.ln())
    }

    /// `0.5 * ln τ`, the scalar normal-density normalizer.
    #[inline]
    fn half_ln_tau() -> Self {
        Self::scalar(0.5 * core::f64::consts::TAU.ln())
    }

    /// Euler's number `e`.
    ///
    /// The base of the natural exponential — the fixed point the whole analysis
    /// surface (`exp`/`ln`) is written around. Naming it directly lets a lazy
    /// backend intern the constant instead of recovering it as `T::one().exp()`.
    #[inline]
    fn e() -> Self {
        Self::scalar(core::f64::consts::E)
    }

    /// `log₂ e` — the factor turning a natural log into a base-2 log.
    #[inline]
    fn log2_e() -> Self {
        Self::scalar(core::f64::consts::LOG2_E)
    }

    /// `log₁₀ e` — the factor turning a natural log into a base-10 log.
    #[inline]
    fn log10_e() -> Self {
        Self::scalar(core::f64::consts::LOG10_E)
    }

    /// `ln 10`.
    #[inline]
    fn ln_10() -> Self {
        Self::scalar(core::f64::consts::LN_10)
    }

    /// `log₂ 10`.
    #[inline]
    fn log2_10() -> Self {
        Self::scalar(core::f64::consts::LOG2_10)
    }

    /// `log₁₀ 2`.
    #[inline]
    fn log10_2() -> Self {
        Self::scalar(core::f64::consts::LOG10_2)
    }

    /// The Euler–Mascheroni constant `γ ≈ 0.5772…`.
    ///
    /// The limit of `Σ_{k≤n} 1/k − ln n`; it is `−ψ(1)` (the additive constant of
    /// the digamma function) and the additive term in the closed-form entropies of
    /// the Gumbel, Weibull, and Gamma families. It is not a `core` float constant,
    /// so it is lifted from its decimal expansion.
    #[inline]
    fn euler_gamma() -> Self {
        Self::scalar(0.577_215_664_901_532_9)
    }

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

    // --- trigonometry ---
    // Elementwise transcendentals. Every backend that can run a formula can run
    // these too — a tensor compiler lowers them to `Sin`/`Cos`/`Atan2`/… — so they
    // belong on the write-once surface, not only on the concrete `Real` tier.
    // (These take `self` by value to match the concrete `Real` signatures they were
    // promoted from; the `Clone` bound lets a caller keep the input when needed.)
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
        (self.clone().sin(), self.cos())
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

    // --- elementwise select (NaN-aware on the concrete reals) ---
    // The comparison-free `select` primitives: a tensor backend lowers these to
    // `Max`/`Min`/`Clamp`, so piecewise / support-indicator kernels stay portable
    // without a `bool`-returning comparison the lazy tier cannot express.
    fn min(self, other: Self) -> Self;
    fn max(self, other: Self) -> Self;
    fn clamp(self, lo: Self, hi: Self) -> Self;

    // --- special functions ---
    // The elementwise analytic functions a probability/statistics formula reaches
    // for once it leaves pure algebra: the error function, the gamma family, and the
    // regularized incomplete gamma/beta CDF kernels. Each has a direct tensor-op
    // analogue (`Erf`/`Lgamma`/`Digamma`/`Igamma`/`RegularizedIncompleteBeta`, …), so
    // a backend lowers them the same way it lowers `exp`/`sin`. The concrete `f64`/
    // `f32` reference implementations live in [`crate::special`]. Where a function
    // takes auxiliary parameters, `self` is the *evaluation point* (the lazy
    // variable) and the parameters follow.

    /// The error function `erf(x)`.
    fn erf(self) -> Self;
    /// The complementary error function `erfc(x) = 1 − erf(x)`.
    fn erfc(self) -> Self;
    /// The inverse error function, `erf_inv(erf(x)) = x`.
    fn erf_inv(self) -> Self;
    /// `ln Γ(x)`.
    fn ln_gamma(self) -> Self;
    /// The digamma function `ψ(x) = d/dx ln Γ(x)`.
    fn digamma(self) -> Self;
    /// The trigamma function `ψ₁(x) = d²/dx² ln Γ(x)`.
    fn trigamma(self) -> Self;
    /// Regularized **lower** incomplete gamma `P(shape, self)` — the Gamma /
    /// chi-squared / Poisson CDF kernel, with `self` the (already-scaled) point.
    fn gamma_lr(self, shape: Self) -> Self;
    /// Regularized **upper** incomplete gamma `Q(shape, self) = 1 − P(shape, self)`.
    fn gamma_ur(self, shape: Self) -> Self;
    /// Regularized incomplete beta `I_self(a, b)` — the Beta / Student-t / F CDF
    /// kernel, with `self ∈ [0, 1]` the point and `(a, b)` the shape parameters.
    fn beta_reg(self, a: Self, b: Self) -> Self;

    /// Numerically-stable `ln(1 − e^x)` (`log1mexp`), the complement of
    /// [`ln_1p_exp`](Self::ln_1p_exp); defined for `x < 0`. The stable Gamma /
    /// exponential / Normal log-CDF complements are written through it.
    fn ln_1m_exp(&self) -> Self;
    /// The scaled complementary error function `erfcx(x) = e^{x²}·erfc(x)` — the
    /// stable kernel of a Gaussian / Wald log-CDF tail.
    fn erfcx(self) -> Self;
    /// The exponentially-scaled modified Bessel `I₀ᵉ(x) = e^{−|x|} I₀(x)`. From it
    /// `ln I₀(x) = ln I₀ᵉ(x) + |x|` and `I₁/I₀ = I₁ᵉ/I₀ᵉ`, the directional
    /// normalizers, follow without overflow.
    fn bessel_i0e(self) -> Self;
    /// The exponentially-scaled modified Bessel `I₁ᵉ(x) = e^{−|x|} I₁(x)`.
    fn bessel_i1e(self) -> Self;
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

/// A **concrete real number** — [`LazyReal`] (the whole elementwise analysis
/// surface) plus the few things that only a concrete number on the real line can
/// offer: the associated constants (`PI`, `TAU`, `EPSILON`, `INFINITY`, …) and the
/// `bool`-returning predicates (`is_finite`/`is_nan`/`is_sign_negative`).
///
/// This is **distinct from [`LazyReal`]**, deliberately. `LazyReal` is the
/// cross-backend analysis surface a *write-once* formula needs, so it is
/// implemented by `f64`/`f32` **and** non-`Copy` backends like a differentiable
/// `Tensor`. The transcendentals, rounding, and `min`/`max`/`clamp` selectors all
/// live there, because a tensor compiler can lower every one of them. `Real` keeps
/// only what a tensor *cannot* produce — a single `PI` value, a scalar `bool` from
/// a comparison — so code that genuinely needs the whole real number line (e.g.
/// `hausdorff`'s geometry) bounds on `Real` and runs over `f64`, `f32`, and any
/// future decimal / interval / extended-precision type, while tensor backends
/// remain unaffected.
///
/// It **fuses the two stacks** for the concrete reals: the algebraic structure
/// (`+ - * /`, `ZERO`/`ONE`, `÷`, `conjugate`, the total order) comes from
/// [`crate::algebra::OrderedField`]; the analysis (`scalar`/`exp`/`ln`/`sqrt`/trig/
/// rounding/`min`/`max`/`clamp`) from [`LazyReal`]; this trait adds only the
/// constants and predicates. `Real: LazyReal`, so a concrete `f64` also satisfies
/// any write-once formula bounded on the lazy base.
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
    /// [`LazyReal::scalar`].
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
    // The elementwise analysis (trig, rounding, `min`/`max`/`clamp`, …) now lives on
    // [`LazyReal`] and is inherited here through `Real: LazyReal`.
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
    fn scalar(x: f64) -> Self {
        x
    }
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
    fn erf(self) -> Self {
        crate::special::erf::erf(self)
    }
    #[inline]
    fn erfc(self) -> Self {
        crate::special::erf::erfc(self)
    }
    #[inline]
    fn erf_inv(self) -> Self {
        crate::special::erf::erf_inv(self)
    }
    #[inline]
    fn ln_gamma(self) -> Self {
        crate::special::gamma::ln_gamma(self)
    }
    #[inline]
    fn digamma(self) -> Self {
        crate::special::gamma::digamma(self)
    }
    #[inline]
    fn trigamma(self) -> Self {
        crate::special::gamma::trigamma(self)
    }
    #[inline]
    fn gamma_lr(self, shape: Self) -> Self {
        crate::special::gamma::gamma_lr(shape, self)
    }
    #[inline]
    fn gamma_ur(self, shape: Self) -> Self {
        crate::special::gamma::gamma_ur(shape, self)
    }
    #[inline]
    fn beta_reg(self, a: Self, b: Self) -> Self {
        crate::special::beta::beta_reg(a, b, self)
    }
    #[inline]
    fn ln_1m_exp(&self) -> Self {
        // Stable log1mexp: ln(-expm1(x)) near 0, ln1p(-exp(x)) in the tail.
        let x = *self;
        if x >= 0.0 {
            f64::NAN
        } else if x > -core::f64::consts::LN_2 {
            (-x.exp_m1()).ln()
        } else {
            (-x.exp()).ln_1p()
        }
    }
    #[inline]
    fn erfcx(self) -> Self {
        crate::special::erf::erfcx(self)
    }
    #[inline]
    fn bessel_i0e(self) -> Self {
        crate::special::bessel::i0e(self)
    }
    #[inline]
    fn bessel_i1e(self) -> Self {
        crate::special::bessel::i1e(self)
    }
}

impl Scalar for f32 {}

impl LazyReal for f32 {
    #[inline]
    fn scalar(x: f64) -> Self {
        x as f32
    }
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
    // `f32` has no native special functions: evaluate in `f64` and narrow, the same
    // way `scalar`/`powf` already do.
    #[inline]
    fn erf(self) -> Self {
        crate::special::erf::erf(self as f64) as f32
    }
    #[inline]
    fn erfc(self) -> Self {
        crate::special::erf::erfc(self as f64) as f32
    }
    #[inline]
    fn erf_inv(self) -> Self {
        crate::special::erf::erf_inv(self as f64) as f32
    }
    #[inline]
    fn ln_gamma(self) -> Self {
        crate::special::gamma::ln_gamma(self as f64) as f32
    }
    #[inline]
    fn digamma(self) -> Self {
        crate::special::gamma::digamma(self as f64) as f32
    }
    #[inline]
    fn trigamma(self) -> Self {
        crate::special::gamma::trigamma(self as f64) as f32
    }
    #[inline]
    fn gamma_lr(self, shape: Self) -> Self {
        crate::special::gamma::gamma_lr(shape as f64, self as f64) as f32
    }
    #[inline]
    fn gamma_ur(self, shape: Self) -> Self {
        crate::special::gamma::gamma_ur(shape as f64, self as f64) as f32
    }
    #[inline]
    fn beta_reg(self, a: Self, b: Self) -> Self {
        crate::special::beta::beta_reg(a as f64, b as f64, self as f64) as f32
    }
    #[inline]
    fn ln_1m_exp(&self) -> Self {
        let x = *self;
        if x >= 0.0 {
            f32::NAN
        } else if x > -core::f32::consts::LN_2 {
            (-x.exp_m1()).ln()
        } else {
            (-x.exp()).ln_1p()
        }
    }
    #[inline]
    fn erfcx(self) -> Self {
        crate::special::erf::erfcx(self as f64) as f32
    }
    #[inline]
    fn bessel_i0e(self) -> Self {
        crate::special::bessel::i0e(self as f64) as f32
    }
    #[inline]
    fn bessel_i1e(self) -> Self {
        crate::special::bessel::i1e(self as f64) as f32
    }
}

// The concrete `Real` impls. `ZERO`/`ONE` come from `algebra::{HasZero, HasOne}`,
// `÷`/`conjugate`/ordering from `algebra::OrderedField`, and the whole elementwise
// analysis (`scalar`/`exp`/`ln`/`sqrt`/trig/rounding/`min`/`max`/`clamp`) from
// `LazyReal`; only the constants and the `bool` predicates are written here.

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
    use super::*;

    /// A formula written once over the lazy analysis base — pure analysis plus a
    /// scalar constant, with no vector/tensor construction.
    fn shifted_quadratic<T: LazyReal>(x: T) -> T {
        let half = T::half();
        x.clone() * x.clone() + half * x
    }

    #[test]
    fn real_formula_runs_on_f64_and_f32() {
        assert!((shifted_quadratic(2.5_f64) - 7.5).abs() < 1e-12);
        assert!((shifted_quadratic(2.5_f32) - 7.5).abs() < 1e-5);
    }

    #[test]
    fn lazy_real_constants_lift_scalars() {
        assert_eq!(f64::half(), 0.5);
        assert_eq!(f32::half(), 0.5);
        assert!((f64::pi() - core::f64::consts::PI).abs() < 1e-12);
        assert!((f32::tau() - core::f32::consts::TAU).abs() < 1e-6);
        assert!((f64::half_ln_tau() - 0.5 * core::f64::consts::TAU.ln()).abs() < 1e-12);
    }

    #[test]
    fn euler_constants_lift_on_every_real_tier() {
        // Euler's number names the base of the analysis surface directly: lifting
        // it must agree with recovering it as `exp(1)`.
        assert!((f64::e() - 1.0_f64.exp()).abs() < 1e-12);
        assert!((f32::e() - 1.0_f32.exp()).abs() < 1e-6);
        assert_eq!(f64::e(), <f64 as Real>::E);
        assert_eq!(f32::e(), <f32 as Real>::E);

        // The log-base conversions are mutual inverses through `e` / `10`.
        assert!((f64::log2_e() * f64::ln_2() - 1.0).abs() < 1e-12);
        assert!((f64::log10_e() * f64::ln_10() - 1.0).abs() < 1e-12);
        assert!((f64::log2_10() * f64::log10_2() - 1.0).abs() < 1e-12);

        // Euler–Mascheroni γ, to its leading digits.
        assert!((f64::euler_gamma() - 0.577_215_664_901_532_9).abs() < 1e-15);
        assert!((f32::euler_gamma() - 0.577_215_7).abs() < 1e-6);
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
        assert!((shifted_quadratic(2.0_f64) - 5.0).abs() < 1e-12);
    }

    /// A formula bounded purely on [`LazyReal`] that uses what used to be the
    /// concrete-only API — `atan2`, trig, and the `clamp` selector. It now compiles
    /// for every lazy backend, not only the concrete reals.
    fn wrapped_angle_unit<T: LazyReal>(x: T, y: T) -> T {
        let angle = y.atan2(x);
        let pythagorean =
            angle.clone().sin() * angle.clone().sin() + angle.clone().cos() * angle.cos();
        pythagorean.clamp(T::zero(), T::one())
    }

    #[test]
    fn lazy_real_carries_trig_and_select_on_f64_and_f32() {
        // sin²+cos² = 1, clamped to [0, 1] — trig + clamp through the LazyReal bound
        // alone, with no `Real` in sight.
        assert!((wrapped_angle_unit(0.7_f64, 1.3_f64) - 1.0).abs() < 1e-12);
        assert!((wrapped_angle_unit(0.7_f32, 1.3_f32) - 1.0).abs() < 1e-6);
        // The `min`/`max`/`clamp` selectors are on the lazy tier too.
        assert_eq!(LazyReal::max(LazyReal::min(3.0_f64, 5.0), 1.0), 3.0);
        assert_eq!(LazyReal::clamp(2.0_f32, 0.0, 1.0), 1.0);
    }

    /// The Normal CDF `Φ(z) = ½·erfc(−z/√2)`, written once over [`LazyReal`]: it
    /// needs `erfc`, which now lives on the lazy surface, so the same source lowers
    /// to a tensor backend's `Erfc` op.
    fn normal_cdf<T: LazyReal>(z: T) -> T {
        T::half() * (-z / T::sqrt_2()).erfc()
    }

    #[test]
    fn lazy_real_special_functions_run_on_f64_and_f32() {
        // Φ(0) = ½ through the write-once CDF, on both concrete reals.
        assert!((normal_cdf(0.0_f64) - 0.5).abs() < 1e-15);
        assert!((normal_cdf(0.0_f32) - 0.5).abs() < 1e-6);

        // The methods delegate to `crate::special`; spot-check the defining
        // identities directly through the trait.
        assert!((LazyReal::ln_gamma(5.0_f64) - 24.0_f64.ln()).abs() < 1e-9);
        assert!((LazyReal::digamma(1.0_f64) + f64::euler_gamma()).abs() < 1e-9);
        // P(a, x) + Q(a, x) = 1, with `self` the point `x`.
        let p = LazyReal::gamma_lr(3.0_f64, 2.5); // P(2.5, 3.0)
        let q = LazyReal::gamma_ur(3.0_f64, 2.5); // Q(2.5, 3.0)
        assert!((p + q - 1.0).abs() < 1e-10);
        // I_x(1, 1) = x.
        assert!((LazyReal::beta_reg(0.37_f64, 1.0, 1.0) - 0.37).abs() < 1e-12);
        assert!((LazyReal::erf(1.0_f32) - 0.842_700_8).abs() < 1e-6);

        // erfcx(x)·e^{-x²} = erfc(x); ln(1−e^x) complements ln(1+e^x); scaled
        // Bessel recovers ln I₀ and the I₁/I₀ ratio.
        let x = 0.6_f64;
        assert!((x.erfcx() * (-x * x).exp() - LazyReal::erfc(x)).abs() < 1e-12);
        assert!((LazyReal::ln_1m_exp(&(-0.5_f64)).exp() - (1.0 - (-0.5_f64).exp())).abs() < 1e-12);
        let k = 2.5_f64;
        assert!((k.bessel_i0e().ln() + k - crate::special::bessel::ln_i0(k)).abs() < 1e-9);
        assert!(
            (k.bessel_i1e() / k.bessel_i0e() - crate::special::bessel::i1_over_i0(k)).abs() < 1e-9
        );
        // f32 paths narrow through f64.
        assert!((0.5_f32.erfcx() - 0.5_f64.erfcx() as f32).abs() < 1e-6);
    }
}
