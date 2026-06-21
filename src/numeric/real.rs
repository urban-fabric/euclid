//! The real-number lazy analysis surface [`LazyReal`]. The concrete `Real` number (the
//! `Copy`, ordered tier) lives in the `napier` crate.

use super::{LazyInteger, NormedDivisionAlgebra};

/// A real-valued numeric scalar: an [`Arithmetic`](super::Arithmetic) plus the real
/// transcendentals inherently-real formulas (densities, geometry kernels) are written in.
///
/// `LazyReal: NormedDivisionAlgebra<RealPart = Self>` — a real scalar's real field is
/// itself, so its `conjugate` is the identity and a [`NormedDivisionAlgebra`] kernel
/// collapses to the plain real case here.
pub trait LazyReal: NormedDivisionAlgebra<RealPart = Self> {
    /// The integer substrate indexing this real's integer powers — `i32` for `f32`, `i64`
    /// for `f64`, an integer tensor for a tensor backend; the exponent type of
    /// [`powi`](Self::powi). The bound is the bare [`LazyInteger`], not
    /// `LazyInteger<Real = Self>`: a differentiable real has no integer differentiating
    /// back to it, so requiring the round-trip would make it un-implementable.
    type Integer: LazyInteger;

    /// Lift a host scalar into this real substrate (identity for `f64`, a narrowing cast
    /// for `f32`). Lazy backends should represent it as a constant node, ideally interned.
    fn scalar(x: f64) -> Self;

    /// Lift a host scalar using `self` as graph/device/shape context. Defaults to
    /// [`scalar`](Self::scalar); backends override when constants must share an existing
    /// value's arena, dtype, device, or broadcast context.
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

    /// Euler's number `e`. Named directly so a lazy backend can intern it rather than
    /// recover it as `T::one().exp()`.
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

    /// The Euler–Mascheroni constant `γ ≈ 0.5772…` (`= −ψ(1)`); the additive term in the
    /// Gumbel/Weibull/Gamma entropies. Not a `core` constant, so lifted from its digits.
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
    /// Raise to a real power, `self^p`. The exponent is itself a real of this substrate, so
    /// a lazy backend can pow by a graph-computed value; lift a literal with
    /// `x.powf(x.scalar_like(2.5))`. Integer exponents have their own [`powi`](Self::powi).
    fn powf(&self, p: Self) -> Self;
    /// Numerically-stable `ln(1 + e^x)` (softplus / `log1pexp`).
    fn ln_1p_exp(&self) -> Self;

    // --- trigonometry ---
    // Elementwise transcendentals a tensor compiler lowers to `Sin`/`Cos`/`Atan2`/…; taken
    // by value to match the concrete `Real` signatures (the `Clone` bound keeps the input).
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
    /// Inverse hyperbolic sine, `asinh(x) = ln(x + √(x² + 1))`. Defaulted through
    /// `ln`/`sqrt` so every backend gets it; `f64`/`f32` override with the libm form.
    #[inline]
    fn asinh(self) -> Self {
        (self.clone() + (self.clone() * self + Self::one()).sqrt()).ln()
    }
    /// Inverse hyperbolic cosine, `acosh(x) = ln(x + √(x² − 1))` (for `x ≥ 1`).
    #[inline]
    fn acosh(self) -> Self {
        (self.clone() + (self.clone() * self - Self::one()).sqrt()).ln()
    }
    /// Inverse hyperbolic tangent, `atanh(x) = ½·ln((1 + x)/(1 − x))` (for `|x| < 1`).
    #[inline]
    fn atanh(self) -> Self {
        let one = Self::one();
        ((one.clone() + self.clone()) / (one - self)).ln() * Self::half()
    }
    /// `(sin, cos)` in one call.
    #[inline]
    fn sin_cos(self) -> (Self, Self) {
        (self.clone().sin(), self.cos())
    }

    // --- powers / roots ---
    /// `√(self² + other²)` without intermediate overflow.
    fn hypot(self, other: Self) -> Self;
    fn cbrt(self) -> Self;
    /// Integer power, `self^n` — exact repeated multiplication, defined for negative bases
    /// (unlike [`powf`](Self::powf)). The exponent is a value of the [`Integer`](Self::Integer)
    /// substrate (`i32` for `f32`, `i64` for `f64`), so a backend can raise to a graph-computed
    /// count.
    fn powi(self, n: Self::Integer) -> Self;
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
    // Comparison-free selectors lowering to `Max`/`Min`/`Clamp`, so piecewise kernels stay
    // portable without a `bool`-returning comparison.
    fn min(self, other: Self) -> Self;
    fn max(self, other: Self) -> Self;
    fn clamp(self, lo: Self, hi: Self) -> Self;

    // --- special functions ---
    // The analytic functions a probability formula reaches for past pure algebra. Each has
    // a direct tensor-op analogue (`Erf`/`Lgamma`/`Igamma`/…); the `f64`/`f32` reference
    // impls live in [`crate::numeric::special`]. `self` is always the evaluation point, with
    // any parameters following.

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
    /// [`ln_1p_exp`](Self::ln_1p_exp); defined for `x < 0`.
    fn ln_1m_exp(&self) -> Self;
    /// The scaled complementary error function `erfcx(x) = e^{x²}·erfc(x)` — the stable
    /// kernel of a Gaussian / Wald log-CDF tail.
    fn erfcx(self) -> Self;
    /// The exponentially-scaled modified Bessel `I₀ᵉ(x) = e^{−|x|} I₀(x)`, the overflow-free
    /// route to `ln I₀(x) = ln I₀ᵉ(x) + |x|` and `I₁/I₀ = I₁ᵉ/I₀ᵉ`.
    fn bessel_i0e(self) -> Self;
    /// The exponentially-scaled modified Bessel `I₁ᵉ(x) = e^{−|x|} I₁(x)`.
    fn bessel_i1e(self) -> Self;
}
