//! Real-number analysis over any ordered real carrier.

use super::{Compare, Integer, NormedDivisionAlgebra};

/// A real-valued numeric scalar: an [`Arithmetic`](super::Arithmetic) plus the real
/// transcendentals inherently-real formulas (densities, geometry kernels) are written in.
///
/// Two supertraits come for free. `NormedDivisionAlgebra<RealPart = Self>` means a real
/// scalar's real field is itself, so its `conjugate` is the identity. [`Compare`] means a real
/// scalar is ordered, so any `if`/`then`/`else` over real numbers needs the `Real` bound
/// alone — and the comparison-free [`min`](Self::min)/[`max`](Self::max)/[`clamp`](Self::clamp)
/// selectors cover the common shapes without an explicit mask.
pub trait Real: NormedDivisionAlgebra<RealPart = Self> + Compare {
    /// The integer substrate indexing this real's integer powers — `i32` for `f32`, `i64` for
    /// `f64`, or an integer peer for a tensor/expression carrier; the exponent type of
    /// [`powi`](Self::powi).
    ///
    /// The bound is the bare [`Integer`], not `Integer<Real = Self>`: a differentiable real
    /// has no integer differentiating back to it, so requiring the round-trip would make it
    /// un-implementable.
    type Integer: Integer;

    /// Lift a host scalar into this real substrate (identity for `f64`, a narrowing cast
    /// for `f32`). Non-scalar carriers should represent it as the carrier's own constant.
    fn scalar(x: f64) -> Self;

    /// Lift a host scalar using `self` as graph/device/shape context.
    fn scalar_like(&self, x: f64) -> Self;

    // `0` and `1` come from the [`Element`](super::Element) supertrait (via
    // [`NormedDivisionAlgebra`]), defined directly by each carrier rather than via `scalar`.

    /// `2`.
    fn two() -> Self;

    /// `0.5`.
    fn half() -> Self;

    /// Archimedes' constant `π`.
    fn pi() -> Self;

    /// `2π`.
    fn tau() -> Self;

    /// `π/2`.
    fn frac_pi_2() -> Self;

    /// `π/4`.
    fn frac_pi_4() -> Self;

    /// `√2`.
    fn sqrt_2() -> Self;

    /// `1/√2`.
    fn frac_1_sqrt_2() -> Self;

    /// `ln 2`.
    fn ln_2() -> Self;

    /// `ln π`.
    fn ln_pi() -> Self;

    /// `ln τ`, where `τ = 2π`.
    fn ln_tau() -> Self;

    /// `0.5 * ln τ`, the scalar normal-density normalizer.
    fn half_ln_tau() -> Self;

    /// Euler's number `e`. Named directly so a non-scalar carrier can intern it rather than
    /// recover it as `T::one().exp()`.
    fn e() -> Self;

    /// `log₂ e` — the factor turning a natural log into a base-2 log.
    fn log2_e() -> Self;

    /// `log₁₀ e` — the factor turning a natural log into a base-10 log.
    fn log10_e() -> Self;

    /// `ln 10`.
    fn ln_10() -> Self;

    /// `log₂ 10`.
    fn log2_10() -> Self;

    /// `log₁₀ 2`.
    fn log10_2() -> Self;

    /// The Euler–Mascheroni constant `γ ≈ 0.5772…` (`= −ψ(1)`); the additive term in the
    /// Gumbel/Weibull/Gamma entropies. Not a `core` constant, so lifted from its digits.
    fn euler_gamma() -> Self;

    /// `-1`.
    #[inline]
    fn neg_one() -> Self {
        Self::zero() - Self::one()
    }

    /// Degrees in one full turn, `360`.
    #[inline]
    fn full_turn_degrees() -> Self {
        Self::scalar(360.0)
    }

    /// Degrees in one half turn, `180`.
    #[inline]
    fn half_turn_degrees() -> Self {
        Self::scalar(180.0)
    }

    /// Degrees in one quarter turn, `90`.
    #[inline]
    fn quarter_turn_degrees() -> Self {
        Self::scalar(90.0)
    }

    /// Degrees in one and a half turns, `540`.
    #[inline]
    fn three_half_turn_degrees() -> Self {
        Self::scalar(540.0)
    }

    /// Radians per degree, `pi / 180`.
    #[inline]
    fn radians_per_degree() -> Self {
        Self::pi() / Self::half_turn_degrees()
    }

    /// Degrees per radian, `180 / pi`.
    #[inline]
    fn degrees_per_radian() -> Self {
        Self::half_turn_degrees() / Self::pi()
    }

    // --- non-finite IEEE sentinels ---
    // These bit patterns lift as ordinary constants through `scalar`. They are useful as
    // reduction/clamp bounds (`±∞`), log-domain masks
    // (`select(keep, logit, neg_infinity())` ahead of a softmax), and the open ends of a
    // distribution's support. They poison arithmetic, so detect them with the finiteness
    // predicates below rather than computing through them.

    /// Positive infinity, `+∞`.
    fn infinity() -> Self;

    /// Negative infinity, `−∞`.
    fn neg_infinity() -> Self;

    /// A quiet `NaN` ("not a number") — the IEEE result of an undefined operation
    /// (`0/0`, `∞ − ∞`). It is unequal to everything including itself, so test it with
    /// [`is_nan`](Self::is_nan); never compare it with `eq`.
    fn nan() -> Self;

    // --- finiteness predicates ---

    /// `true` where `self` is `NaN`.
    fn is_nan(&self) -> Self::Mask;

    /// `true` where `self` is `±∞`.
    fn is_infinite(&self) -> Self::Mask;

    /// `true` where `self` is finite — neither `±∞` nor `NaN`.
    fn is_finite(&self) -> Self::Mask;

    // --- non-finite recovery helpers ---
    // Convenience composites that turn the classifier predicates above into value-level
    // recovery — the shape an out-of-support guard takes.

    /// Replace `NaN` lanes with `fill`, leaving `±∞` and finite values untouched — the
    /// portable `nan_to_num` for the `NaN` case. A `NaN` log-density (a bug, or `0·∞`) becomes
    /// a chosen sentinel (often `neg_infinity()`, i.e. "impossible") instead of poisoning a
    /// later sum.
    #[inline]
    fn nan_to_num(self, fill: Self) -> Self {
        let is_nan = self.is_nan();
        Self::select(&is_nan, fill, self)
    }

    /// Replace every non-finite lane (`NaN` **or** `±∞`) with `fill`, keeping the finite ones —
    /// the "force finite" recovery. Use [`nan_to_num`](Self::nan_to_num) instead when an
    /// intended `±∞`, such as a log-domain `−∞`, must survive.
    #[inline]
    fn finite_or(self, fill: Self) -> Self {
        let is_finite = self.is_finite();
        Self::select(&is_finite, self, fill)
    }

    /// Out-of-support guard: keep `self` where `valid`, else `fill`. This is the *output* half
    /// of the "double `where`" — pair it with a sanitized *input* so an eager elementwise
    /// implementation never forms a `NaN` in an invalid lane to begin with:
    ///
    /// ```ignore
    /// let safe_x = T::select(&in_support, x.clone(), x.scalar_like(1.0)); // any in-domain dummy
    /// let logp   = density(safe_x);                                       // no NaN in masked lanes
    /// let out    = logp.mask_invalid(in_support, T::neg_infinity());
    /// ```
    #[inline]
    fn mask_invalid(self, valid: Self::Mask, fill: Self) -> Self {
        Self::select(&valid, self, fill)
    }

    /// A debugging hook that returns `self` unchanged but lets an implementation trap non-finite values.
    /// The default is the identity (no check); the eager reals override it to `debug_assert!`
    /// finiteness, while other carriers may lower it to a runtime check behind a debug flag.
    /// It carries **no** semantic meaning — release builds and unchecked implementations may
    /// erase it — so never rely on it to change a value.
    #[inline]
    fn debug_assert_finite(self) -> Self {
        self
    }

    /// `e^x`.
    fn exp(&self) -> Self;
    /// Natural logarithm.
    fn ln(&self) -> Self;
    /// Natural logarithm of `1 + x`, accurate when `x` is near zero.
    fn ln_1p(&self) -> Self;
    /// `e^x − 1`, accurate when `x` is near zero.
    fn exp_m1(&self) -> Self;
    /// Square root.
    fn sqrt(&self) -> Self;
    /// Absolute value.
    fn abs(&self) -> Self;
    /// Square, `self * self`.
    fn square(&self) -> Self;
    /// Raise to a real power, `self^p`. The exponent is itself a real of this substrate, so
    /// a carrier can raise to a computed value; lift a literal with
    /// `x.powf(x.scalar_like(2.5))`. Integer exponents have their own [`powi`](Self::powi).
    fn powf(&self, p: Self) -> Self;
    /// Numerically-stable `ln(1 + e^x)` (softplus / `log1pexp`).
    fn ln_1p_exp(&self) -> Self;

    // --- trigonometry ---
    // Elementwise transcendentals; taken by value so carrier handles can move cheaply while
    // `Clone` remains available when a formula needs to keep the input.
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
    /// Inverse hyperbolic sine.
    fn asinh(self) -> Self;
    /// Inverse hyperbolic cosine, `acosh(x) = ln(x + √(x² − 1))` (for `x ≥ 1`).
    fn acosh(self) -> Self;
    /// Inverse hyperbolic tangent, `atanh(x) = ½·ln((1 + x)/(1 − x))` (for `|x| < 1`).
    fn atanh(self) -> Self;
    /// `(sin, cos)` in one call.
    fn sin_cos(self) -> (Self, Self);

    // --- powers / roots ---
    /// `√(self² + other²)` without intermediate overflow.
    fn hypot(self, other: Self) -> Self;
    fn cbrt(self) -> Self;
    /// Integer power, `self^n` — exact repeated multiplication, defined for negative bases
    /// (unlike [`powf`](Self::powf)). The exponent is a value of the [`Integer`](Self::Integer)
    /// substrate (`i32` for `f32`, `i64` for `f64`), so a carrier can raise to a computed
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
    // Comparison-free selectors for min/max/clamp-shaped piecewise kernels. NaN-awareness
    // here is about the *value* `self`: the concrete `f64`/`f32` `min`/`max` return the
    // non-NaN operand, and `clamp` preserves a NaN value — not about the `clamp` bounds.
    fn min(self, other: Self) -> Self;
    fn max(self, other: Self) -> Self;
    /// Clamp `self` into `[lo, hi]`.
    ///
    /// # Panics
    /// The concrete `f64`/`f32` impls delegate to `f{64,32}::clamp`, which **panics** if
    /// `lo > hi` or if either bound is NaN. Callers must supply ordered, non-NaN bounds.
    fn clamp(self, lo: Self, hi: Self) -> Self;

    // --- special functions ---
    // The analytic functions a probability formula reaches for past pure algebra. Each has
    // a direct tensor-op analogue (`Erf`/`Lgamma`/`Igamma`/…); the `f64`/`f32` reference
    // impls live in the crate-internal `numeric::special` module. `self` is always the
    // evaluation point, with any parameters following.

    /// The error function `erf(x)`.
    fn erf(self) -> Self;
    /// The complementary error function `erfc(x) = 1 − erf(x)`. Formed by complementation, so
    /// it loses relative precision — and eventually underflows to `0` — in the right tail
    /// (`x ≳ 5`); for a stable large-`x` value, reach for the scaled
    /// [`erfcx`](Self::erfcx) (`e^{x²} erfc(x)`) instead.
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
    ///
    /// # Precision
    /// The `f64`/`f32` reference implementations rest on the Abramowitz & Stegun polynomial
    /// approximations, which are accurate to only about **1e-7 relative** — roughly `f32`
    /// resolution even when evaluated in `f64`, and far weaker than the gamma/beta family on
    /// this trait. Measured worst case grows with `x` and plateaus near 4.7e-7 from `x ≈ 100`.
    /// Do not rely on this for anything needing more than ~7 significant digits.
    fn bessel_i0e(self) -> Self;
    /// The exponentially-scaled modified Bessel `I₁ᵉ(x) = e^{−|x|} I₁(x)`.
    ///
    /// # Precision
    /// About **1e-7 relative**, for the reason given on
    /// [`bessel_i0e`](Self::bessel_i0e); measured worst case 1.1e-7.
    fn bessel_i1e(self) -> Self;

    /// `x · ln y` with the convention `0 · ln 0 = 0` (and `0 · ln y = 0` for any `y`) — the
    /// stable form that keeps entropy / cross-entropy / KL sums finite where a plain
    /// `x * y.ln()` would evaluate `0 · (−∞) = NaN`. `self` is `x`, the multiplier whose zero
    /// is absorbing.
    ///
    /// Required rather than defaulted: the `x = 0` case needs a value-dependent select the
    /// comparison-free real surface cannot express. Host reals branch on `x == 0`; elementwise
    /// carriers can use a `select`-guarded form.
    fn xlogy(self, y: Self) -> Self;

    /// `x · ln(1 + y)` with the same absorbing-zero convention as [`xlogy`](Self::xlogy), and
    /// required for the same reason: `0 · ln(0) = 0` when `y = -1`, and `0 · ln(1 + y) = 0`
    /// for any `y`.
    fn xlog1py(self, y: Self) -> Self;

    // --- stable scalar functions ---

    /// Stable `ln(1 − x)`, accurate when `x` is near zero; defined for `x ≤ 1`.
    fn ln_1m(&self) -> Self;

    /// Stable `ln(e^x − 1)`, the inverse of [`ln_1p_exp`](Self::ln_1p_exp) on positive
    /// inputs. Defined for `x > 0`.
    fn ln_exp_m1(&self) -> Self;

    /// Stable logistic sigmoid `1 / (1 + e^{-x})`, evaluated without overflowing the
    /// intermediate exponential.
    fn sigmoid(&self) -> Self;

    /// Stable `ln(sigmoid(x)) = -ln(1 + e^{-x})`.
    fn ln_sigmoid(&self) -> Self;

    /// Stable `ln(1 − sigmoid(x)) = -ln(1 + e^x)`.
    fn ln_1m_sigmoid(&self) -> Self;

    /// Stable logit, `ln(p / (1 − p))`, using [`ln_1m`](Self::ln_1m) for the complement.
    fn logit(&self) -> Self;

    /// `x · ln(1 − y)` with the same absorbing-zero convention as [`xlogy`](Self::xlogy).
    fn xlog1my(self, y: Self) -> Self;

    /// Stable pairwise `ln(eᵃ + eᵇ) = max(a, b) + ln(1 + e^{−|a−b|})`.
    fn ln_add_exp(self, other: Self) -> Self;

    /// Stable pairwise `ln(eᵃ − eᵇ) = a + ln(1 − e^{b−a})`, valid for `self >= other`.
    fn ln_sub_exp(self, other: Self) -> Self;

    // --- composite special functions ---
    // These are part of the contract surface. Implementors choose the numerics appropriate
    // for their carrier rather than inheriting a generic stability policy.

    /// `Γ(x)`.
    fn gamma(self) -> Self;

    /// `ln B(a, b) = ln Γ(a) + ln Γ(b) − ln Γ(a + b)`, with `self = a`.
    fn ln_beta(self, b: Self) -> Self;

    /// `B(a, b) = exp(ln B(a, b))`, with `self = a`.
    fn beta(self, b: Self) -> Self;

    /// `ln I₀(x) = ln I₀ᵉ(x) + |x|`, the overflow-free log of the modified Bessel `I₀`
    /// (the von Mises / directional normalizer).
    ///
    /// # Precision
    /// Inherits the ~1e-7 relative error of [`bessel_i0e`](Self::bessel_i0e), which in the log
    /// domain shows up as an *absolute* error of that order. The implementation may also take
    /// a different internal path than `|x| + ln(i0e(x))`, so the two agree only to that level.
    fn ln_bessel_i0(self) -> Self;

    /// `I₁(x) / I₀(x) = I₁ᵉ(x) / I₀ᵉ(x)` — the von Mises mean resultant length;
    /// overflow-free, since the shared `e^{|x|}` growth factor cancels.
    ///
    /// # Precision
    /// The ratio compounds both Bessel errors: measured worst case **2e-7 relative**. See
    /// [`bessel_i0e`](Self::bessel_i0e).
    fn bessel_i1_over_i0(self) -> Self;
}
