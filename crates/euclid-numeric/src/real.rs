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
    /// Gumbel/Weibull/Gamma entropies.
    fn euler_gamma() -> Self;

    /// The golden ratio `φ ≈ 1.6180…`, the positive root of `x² = x + 1`.
    fn golden_ratio() -> Self;

    /// `√π ≈ 1.7725…`, equal to `Γ(½)` — the Gaussian normalizer.
    fn sqrt_pi() -> Self;

    /// `√τ = √(2π) ≈ 2.5066…`, the normal density's normalizing divisor.
    fn sqrt_tau() -> Self;

    /// Apéry's constant `ζ(3) ≈ 1.2021…`, equal to `−ψ''(1)/2`.
    fn apery() -> Self;

    /// Catalan's constant `G ≈ 0.9160…`, the value of `Σ (−1)ⁿ/(2n+1)²`.
    fn catalan() -> Self;

    /// The omega constant `Ω ≈ 0.5671…`, the unique real solution of `Ω e^Ω = 1`.
    fn omega() -> Self;

    // The wider catalogue — the lemniscate and plastic numbers, Gauss's constant, the Dottie
    // number, `ζ(2)`, `ζ(4)` — lives in [`constants`](crate::constants) rather than here.
    // A constant earns a trait method when a carrier-generic formula needs to lift it; the
    // rest are reachable directly by code that is already concrete.

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
    // impls live in the crate-internal `kernels` module. `self` is always the
    // evaluation point, with any parameters following.

    /// The error function `erf(x)`.
    fn erf(self) -> Self;
    /// The complementary error function `erfc(x) = 1 − erf(x)`.
    ///
    /// Underflows to `0` in the right tail, around `x ≳ 27`, exactly as the true value does;
    /// for a value that stays in range, reach for the scaled [`erfcx`](Self::erfcx)
    /// (`e^{x²} erfc(x)`) instead.
    ///
    /// # Precision
    /// Machine precision: measured below `1e-14` relative (worst `9.4e-15`, near `x = 1.2`).
    /// Only `|x| ≲ 2` is formed by complementation; above that a continued fraction computes it
    /// directly, so the relative loss the name suggests is confined to the region where `erfc`
    /// is still `O(1)` and the loss is therefore harmless. The handover used to sit at `x = 4`,
    /// which left a band peaking at `1.2e-9`.
    fn erfc(self) -> Self;
    /// The inverse error function, `erf_inv(erf(x)) = x`.
    fn erf_inv(self) -> Self;
    /// `ln Γ(x)`.
    fn ln_gamma(self) -> Self;
    /// The digamma function `ψ(x) = d/dx ln Γ(x)`.
    ///
    /// # Precision
    /// Machine precision: measured below `1e-15` relative against the exact
    /// `ψ(n) = −γ + H_{n−1}`. The recurrence walks the argument up to `20` before the
    /// asymptotic series, which is what puts the series' first omitted term below `ε`; at the
    /// `10` used previously the floor was `7.4e-13`.
    ///
    /// `ψ` has a zero near `1.4616`, where only *absolute* accuracy is meaningful; it is
    /// `1.2e-15` there.
    fn digamma(self) -> Self;
    /// The trigamma function `ψ₁(x) = d²/dx² ln Γ(x)`.
    ///
    /// # Precision
    /// Measured below `5e-14` relative against the exact `ψ₁(n) = ζ(2) − Σ_{k<n} k^{−2}`;
    /// shares [`digamma`](Self::digamma)'s recurrence shift.
    fn trigamma(self) -> Self;
    /// Regularized **lower** incomplete gamma `P(shape, self)` — the Gamma /
    /// chi-squared / Poisson CDF kernel, with `self` the (already-scaled) point.
    ///
    /// # Precision
    /// Machine precision for ordinary shapes, from a power series below `x = shape + 1` and a
    /// continued fraction above it.
    ///
    /// **Above `shape ≈ 3e7`** (`1e3` on `f32`) neither converges — the series needs `O(√a)`
    /// terms, and the front factor `exp(a ln x − x − lnΓ(a))` subtracts three `O(a ln a)`
    /// quantities — so an Edgeworth-corrected normal approximation takes over, with absolute
    /// error `O(1/a)`. That is `3e-8` at the crossover and shrinks from there, reaching exact
    /// in the limit; it is the more accurate of the two routes at every shape above it.
    ///
    /// `P` and `Q` are probabilities at every shape: both lie in `[0, 1]`, sum to `1`, and are
    /// monotone in `self`.
    fn gamma_lr(self, shape: Self) -> Self;
    /// Regularized **upper** incomplete gamma `Q(shape, self) = 1 − P(shape, self)`.
    ///
    /// # Precision
    /// As [`gamma_lr`](Self::gamma_lr). Formed so a small upper tail keeps its *relative*
    /// accuracy rather than being lost to the complement — which is the reason it is a named
    /// operation and not `1 - p`.
    fn gamma_ur(self, shape: Self) -> Self;
    /// Regularized incomplete beta `I_self(a, b)` — the Beta / Student-t / F CDF
    /// kernel, with `self ∈ [0, 1]` the point and `(a, b)` the shape parameters.
    ///
    /// # Precision
    /// Machine precision for ordinary parameters, from a continued fraction taken on whichever
    /// side of the median converges faster.
    ///
    /// Two limits appear as the parameters grow, and they mirror
    /// [`gamma_lr`](Self::gamma_lr)'s. The fraction converges slowest at exactly `x ≈ a/(a+b)`,
    /// needing `O(a^{1/3})` terms; and its front factor
    /// `exp(a ln x + b ln(1−x) − ln B(a,b))` subtracts three quantities of size `≈1.386(a+b)`,
    /// so it carries `≈1.386(a+b)ε` of relative error however long the fraction runs —
    /// `4.7e-10` at `a = b = 1e6`, which is the worst point of the whole surface.
    ///
    /// **Above `min(a, b) ≈ 1e7`** an Edgeworth-corrected normal takes over, with absolute
    /// error `≈ 0.06/min(a,b)`; it is the more accurate route from there up. The gate is on the
    /// *smaller* parameter because `Beta(a,b)` approaches a normal only as both grow — its
    /// skewness is `≈ 2/√min(a,b)`, so `Beta(1, 1e8)` is nothing like a normal however large
    /// the sum.
    ///
    /// The symmetric case is exact: `I_{1/2}(a, a) = ½` to the last bit at every `a`, since
    /// `Beta(a,a)` has zero skewness.
    fn beta_reg(self, a: Self, b: Self) -> Self;

    /// Numerically-stable `ln(1 − e^x)` (`log1mexp`), the complement of
    /// [`ln_1p_exp`](Self::ln_1p_exp); defined for `x ≤ 0`.
    ///
    /// `x = 0` is the boundary and returns `−∞`, which is the limit from inside the domain and
    /// matches [`ln_1m`](Self::ln_1m) at `1` — the same mathematical point reached the other
    /// way. Above zero there is no real value and the result is `NaN`.
    fn ln_1m_exp(&self) -> Self;
    /// The scaled complementary error function `erfcx(x) = e^{x²}·erfc(x)` — the stable
    /// kernel of a Gaussian / Wald log-CDF tail.
    fn erfcx(self) -> Self;
    /// The modified Bessel function of the first kind `I₀(x)`, even in `x`.
    ///
    /// Grows like `e^{|x|}`, so it overflows for `|x| ≳ 710` exactly as the true value does.
    /// Prefer [`bessel_i0e`](Self::bessel_i0e) or [`ln_bessel_i0`](Self::ln_bessel_i0) when the
    /// argument may be large.
    fn bessel_i0(self) -> Self;

    /// The modified Bessel function of the first kind `I₁(x)`, odd in `x`.
    ///
    /// Overflows for large `|x|` like [`bessel_i0`](Self::bessel_i0); prefer
    /// [`bessel_i1e`](Self::bessel_i1e) there.
    fn bessel_i1(self) -> Self;

    /// The exponentially-scaled modified Bessel `I₀ᵉ(x) = e^{−|x|} I₀(x)`, the overflow-free
    /// route to `ln I₀(x) = ln I₀ᵉ(x) + |x|` and `I₁/I₀ = I₁ᵉ/I₀ᵉ`.
    ///
    /// # Precision
    /// Machine precision (measured worst case 1.7e-14), from a peak-factored power series.
    /// Every term of that series is positive, so there is no cancellation at any argument, and
    /// the scale is carried in the log domain so nothing overflows.
    fn bessel_i0e(self) -> Self;
    /// The exponentially-scaled modified Bessel `I₁ᵉ(x) = e^{−|x|} I₁(x)`.
    ///
    /// # Precision
    /// Machine precision, as for [`bessel_i0e`](Self::bessel_i0e).
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
    ///
    /// # Precision
    /// Machine precision at every pair of arguments, including the extremes:
    /// `ln B(1e300, 1)` is exact. It is *not* evaluated as written — that form cancels
    /// whenever `a + b ≫ min(a, b)`, and past `a = 1/ε` the sum `a + b` rounds back to `a` so
    /// the answer vanishes entirely. The rising-factorial form is used instead.
    fn ln_beta(self, b: Self) -> Self;

    /// `B(a, b) = exp(ln B(a, b))`, with `self = a`.
    fn beta(self, b: Self) -> Self;

    /// `ln I₀(x) = ln I₀ᵉ(x) + |x|`, the overflow-free log of the modified Bessel `I₀`
    /// (the von Mises / directional normalizer).
    ///
    /// # Precision
    /// Machine precision. It is taken straight from the series' log-domain form rather than by
    /// scaling `i0e` back up, so it needs no `e^{|x|}` round trip and stays accurate for every
    /// argument.
    fn ln_bessel_i0(self) -> Self;

    /// `I₁(x) / I₀(x) = I₁ᵉ(x) / I₀ᵉ(x)` — the von Mises mean resultant length;
    /// overflow-free, since the shared `e^{|x|}` growth factor cancels.
    ///
    /// # Precision
    /// Machine precision, and specifically *not* formed as a difference of two large
    /// logarithms. Both series peak at the same index, where their peak terms differ by
    /// exactly `ln(x/2) − ln(k+1)`, so the large factors cancel symbolically and the whole
    /// computation stays `O(1)`.
    fn bessel_i1_over_i0(self) -> Self;

    // --- compensated arithmetic ---
    // The primitives state-of-the-art summation, dot products, and polynomial evaluation are
    // built from. They expose the rounding error that `+` and `*` discard, which is what lets
    // a consumer write a reduction accurate beyond the working precision. A carrier that
    // cannot represent its own rounding error (a symbolic one, say) should return a zero
    // error term; the identity `value + error == exact` still holds trivially.

    /// Exact sum: returns `(value, error)` with `value = self + rhs` rounded, and `error` the
    /// discarded remainder, so `value + error` is the exact sum.
    ///
    /// Knuth's two-sum. The building block of compensated (Kahan–Babuška–Neumaier) summation:
    /// carrying `error` forward lets a long sum stay accurate where naive accumulation drifts
    /// linearly in the term count.
    ///
    /// # Non-finite results
    /// If `value` is not finite the identity cannot hold — the exact sum of `f64::MAX` with
    /// itself is `3.6e308`, which no `f64` represents — so `error` is **zero** rather than
    /// whatever the algebra produces. That keeps `value + error == value`, which makes a
    /// compensated reduction saturate to `±∞`; the alternative is a `NaN` error term that
    /// poisons the correction accumulator permanently, long after the overflowing term has
    /// passed.
    fn two_sum(self, rhs: Self) -> (Self, Self);

    /// Exact product: returns `(value, error)` with `value = self * rhs` rounded, and `error`
    /// the discarded remainder, so `value + error` is the exact product.
    ///
    /// Cheap wherever a fused multiply-add is available, since `error = fma(a, b, −value)`.
    /// The companion to [`two_sum`](Self::two_sum) for accurate dot products.
    ///
    /// # Non-finite results
    /// As for [`two_sum`](Self::two_sum): `error` is zero when `value` is not finite, since
    /// `fma(a, b, −∞)` is `−∞` and that is not a remainder.
    fn two_product(self, rhs: Self) -> (Self, Self);

    // --- cancellation-prone elementary forms ---

    /// `cos(x) − 1`, without the cancellation of forming it from [`cos`](Self::cos).
    ///
    /// Near zero `cos x` is within one ulp of 1, so `cos(x) - 1.0` loses every significant
    /// digit — the same failure [`exp_m1`](Self::exp_m1) exists to avoid. Computed from the
    /// half-angle identity `cos x − 1 = −2 sin²(x/2)`, which is accurate throughout.
    ///
    /// This is the kernel of the haversine formula, where the cancelling form silently
    /// destroys short-distance accuracy.
    fn cos_m1(self) -> Self;

    /// `sin(x) / x`, with the removable singularity at `0` filled in as `1`.
    ///
    /// The unnormalized convention (not `sin(πx)/(πx)`). Direct evaluation is `0/0` at the
    /// origin and loses digits nearby; this is continuous and accurate across it.
    fn sinc(self) -> Self;

    /// `√(x² + y² + z²)` without intermediate overflow or underflow — the three-argument
    /// [`hypot`](Self::hypot).
    ///
    /// Squaring any component of magnitude above ~1e154 overflows `f64` even when the norm is
    /// representable, so the naive form fails on inputs the answer is fine for.
    fn hypot3(self, y: Self, z: Self) -> Self;

    /// `ln Γ(a + n) − ln Γ(a)` with `self = a` — the log of the rising factorial (Pochhammer
    /// symbol) `(a)ₙ = a(a+1)…(a+n−1)`.
    ///
    /// A named operation because the difference of two [`ln_gamma`](Self::ln_gamma) values
    /// cancels badly when `a` is large and `n` is small: both terms are large and nearly
    /// equal, so the result loses the digits that matter. It appears wherever a ratio of
    /// gamma functions does — binomial and beta-binomial weights, Dirichlet updates, the
    /// Pólya urn.
    fn ln_rising_factorial(self, n: Self) -> Self;

    // --- elementary completeness ---

    /// Base-10 logarithm. Named alongside [`log2`](Self::log2) rather than left to
    /// `ln(x) * log10_e`, so a carrier can use a dedicated kernel and so decade arithmetic
    /// (decibels, pH, Richter, order-of-magnitude work) does not go through two roundings.
    fn log10(self) -> Self;

    /// `2^x`. The exponential peer of [`log2`](Self::log2); exact at integral `x` where
    /// `exp(x * ln 2)` is not.
    fn exp2(self) -> Self;

    /// The inverse complementary error function: the `x` with `erfc(x) = y`, for `y` in
    /// `[0, 2]`.
    ///
    /// Not `erf_inv(1 - y)`: that subtraction cancels for small `y`, which is exactly the
    /// regime an upper-tail threshold lives in. This stays accurate as far as
    /// [`ndtri`](Self::ndtri) does.
    fn erfc_inv(self) -> Self;

    // --- the standard normal distribution ---
    //
    // The CDF, its log, and its inverse. Every probabilistic consumer needs these, and
    // composing them correctly from the error function is easy to get wrong, so they are named
    // here rather than left to the caller.

    /// The standard normal CDF `Φ(x) = P(Z ≤ x)`, computed as `erfc(−x/√2)/2` so the left
    /// tail does not cancel.
    fn ndtr(self) -> Self;

    /// `ln Φ(x)`, finite as far into the left tail as the exponent range allows.
    ///
    /// The reason this exists: `Φ(x)` underflows to zero around `x = −38`, so `ndtr(x).ln()`
    /// returns `−∞` while the true value is a representable `−724`. Routing through
    /// [`erfcx`](Self::erfcx) moves the vanishing factor inside the logarithm. A log-likelihood
    /// over a censored or truncated normal needs this; built the naive way it silently becomes
    /// `−∞` for any observation far enough into the tail.
    fn log_ndtr(self) -> Self;

    /// The gamma quantile: the `x` with [`gamma_lr`](Self::gamma_lr)`(x, shape) == self`,
    /// where `self` is the probability.
    ///
    /// Solved rather than fitted — a bracketed Newton/bisection root-find on the forward CDF.
    /// That makes it unconditionally convergent for every shape, and ties its accuracy to
    /// `gamma_lr`, so the quantile cannot drift away from the CDF it inverts.
    fn gamma_lr_inv(self, shape: Self) -> Self;

    /// The beta quantile: the `x` in `[0, 1]` with
    /// [`beta_reg`](Self::beta_reg)`(x, a, b) == self`, where `self` is the probability.
    ///
    /// Solved on a known bracket (`[0, 1]`), so convergence is guaranteed for any `a`, `b`.
    fn beta_reg_inv(self, a: Self, b: Self) -> Self;

    // --- floating-point representation ---
    //
    // How a value decomposes into mantissa and exponent, and how far apart representable
    // values are. These are the primitives overflow-safe scaling and tolerance reasoning are
    // built from — a norm that rescales before squaring, a bisection that stops when the
    // bracket is one representable step wide, a comparison expressed in ulp rather than a
    // guessed epsilon. Without them a consumer has to reach for bit casts.
    //
    // Exponents are carried as `Self`, holding an integral value, rather than as
    // [`Integer`](Self::Integer). That is deliberate: `Self::Integer` is a *scalar* type (it is
    // the exponent of [`powi`](Self::powi), one power applied to a whole array), whereas the
    // exponent of a decomposition is per-element. An array carrier would have no way to return
    // a lane-wise exponent through a scalar type.

    /// Decompose into a mantissa and a power of two: returns `(m, e)` with
    /// `self == m · 2^e` and `|m| ∈ [0.5, 1)`.
    ///
    /// Zero, infinities, and `NaN` are returned unchanged with `e = 0`. Subnormals are
    /// normalized, so the mantissa is in range for them too.
    fn frexp(self) -> (Self, Self);

    /// `self · 2^exp`, the inverse of [`frexp`](Self::frexp).
    ///
    /// Computed without forming `2^exp` as a separate value, so it is exact and cannot
    /// overflow for an `exp` whose *result* is representable — `1e-300.ldexp(1000)` is fine
    /// even though `2^1000` is not.
    fn ldexp(self, exp: Self) -> Self;

    /// The next representable value after `self` in the direction of `toward`.
    ///
    /// Returns `toward` when the two are equal. The primitive for walking the floating-point
    /// grid: exhausting a bisection bracket, nudging off a boundary, or building an exclusive
    /// bound from an inclusive one.
    fn next_after(self, toward: Self) -> Self;

    /// The distance from `self` to the next representable value away from zero — one unit in
    /// the last place.
    ///
    /// The honest unit for a tolerance. `ulp(1.0)` is [`EPSILON`](f64::EPSILON), but `ulp`
    /// scales with the value, which a fixed epsilon does not, so `n * x.ulp()` expresses "n
    /// representable steps" at any magnitude.
    fn ulp(self) -> Self;

    /// `true` where the sign bit is set.
    ///
    /// Distinct from a comparison against zero, and from [`signum`](Self::signum): this is the
    /// only way to observe the sign of a negative zero, and it reports the sign of a `NaN`
    /// rather than propagating it.
    fn signbit(&self) -> Self::Mask;

    /// Split into fractional and integral parts: returns `(fract, trunc)`, both carrying the
    /// sign of `self`, with `fract + trunc == self`.
    ///
    /// Ordered fractional-first to match the usual `modf`. The two halves are available
    /// separately as [`fract`](Self::fract) and [`trunc`](Self::trunc); this returns them as
    /// a pair so a carrier can compute them in one pass.
    fn modf(self) -> (Self, Self);

    // --- further analysis ---

    /// `(eˣ − 1) / x`, with the removable singularity at `0` filled in as `1`.
    ///
    /// The relative exponential. Direct evaluation is `0/0` at the origin and cancels nearby,
    /// the same failure [`exp_m1`](Self::exp_m1) addresses — this is that fix in ratio form,
    /// which is how it usually appears: a rate over an interval, a divided difference of an
    /// exponential, the `(eˣ−1)/x` factor in a discretized growth model.
    fn exprel(self) -> Self;

    /// The reciprocal gamma function `1 / Γ(x)`.
    ///
    /// Entire — unlike `Γ` it has no poles, taking the value `0` at every non-positive
    /// integer, which is what makes it the right form for a series coefficient. It is also
    /// defined where `Γ` overflows: `1/Γ(175)` is a representable `1e-316`, while
    /// `Γ(175).recip()` gives `0` because the intermediate overflowed.
    fn rgamma(self) -> Self;

    /// The polygamma function `ψ⁽ⁿ⁾(x)` — the `n`-th derivative of
    /// [`digamma`](Self::digamma).
    ///
    /// `order = 0` is `digamma` and `order = 1` is [`trigamma`](Self::trigamma); this
    /// generalizes both, and the two remain available as named methods because they are by far
    /// the most used.
    ///
    /// The order is a host `u32` rather than a carrier value, for the same reason a tensor
    /// shape is host-known: it selects *which function* to evaluate rather than being an
    /// argument to it.
    fn polygamma(self, order: u32) -> Self;

    /// The standard normal quantile (probit) `Φ⁻¹(p)`, for `p` in `[0, 1]`.
    ///
    /// # Accuracy
    /// Full precision for `p ∈ [1e-300, 0.5]`, and the same by symmetry above `0.5` *as far as
    /// `1 − p` is representable*. For an extreme upper quantile, ask for `-q.ndtri()` with the
    /// small `q` directly rather than `(1 - q).ndtri()` — the subtraction is what loses the
    /// digits.
    fn ndtri(self) -> Self;

    // --- classical special functions ---
    // The named transcendentals of classical analysis: the rest of the Bessel family, the
    // elementary integrals that have no elementary antiderivative, the Lambert `W`, and the
    // Riemann zeta. Elementwise unary maps like the rest of this surface, with the `f64`/`f32`
    // reference implementations in the crate-internal `kernels` module.
    //
    // A note on what "precision" means for the oscillatory members. `J`, `Y`, `Si` and `Ci`
    // all cross zero infinitely often, and at a zero no implementation can be accurate in the
    // *relative* sense: the zeros are irrational, so a representable argument near one already
    // differs from the argument the caller meant by up to `ε·x`, and a non-vanishing
    // derivative there turns that input uncertainty into output error. Absolute accuracy is
    // the meaningful contract, and it is the one quoted below.

    /// The Bessel function of the first kind `J₀(x)`, even in `x`.
    ///
    /// # Precision
    /// Machine precision in absolute terms at every argument (measured below `1e-15`). Below
    /// `|x| = 18` this comes from an *aliased trapezoidal rule* — `J₀`'s integral
    /// representation is periodic and entire, so an equally-spaced rule is exact except for
    /// the Fourier modes it aliases, and those are `J_N(x)` for `N` above `x`, already below
    /// rounding. Above `18`, from the Hankel expansion. See the note above regarding the zeros.
    fn bessel_j0(self) -> Self;

    /// The Bessel function of the first kind `J₁(x)`, odd in `x`.
    ///
    /// # Precision
    /// As [`bessel_j0`](Self::bessel_j0).
    fn bessel_j1(self) -> Self;

    /// The Bessel function of the second kind `Y₀(x)`, for `x > 0`; `NaN` below zero, where it
    /// is not real, and `−∞` at the origin, where it has a logarithmic singularity.
    ///
    /// # Precision
    /// Machine precision in absolute terms at every argument (measured below `1e-15`). `Y` has
    /// no quadrature as clean as `J`'s, so the middle range is reached by **marching Bessel's
    /// equation outward** from where the power series is still exact — stable here precisely
    /// because `J` and `Y` both decay like `x^{−1/2}`, so neither dominates the other and
    /// error introduced at one step is not amplified at the next.
    fn bessel_y0(self) -> Self;

    /// The Bessel function of the second kind `Y₁(x)`, for `x > 0`.
    ///
    /// # Precision
    /// As [`bessel_y0`](Self::bessel_y0); the two are produced by the same march, since
    /// `Y₀′ = −Y₁` makes them mutually consistent by construction.
    fn bessel_y1(self) -> Self;

    /// The modified Bessel function of the second kind `K₀(x)`, for `x > 0`.
    ///
    /// Decays like `e^{−x}`, so it **underflows to zero for `x ≳ 746`** exactly as the true
    /// value does; reach for [`bessel_k0e`](Self::bessel_k0e) when the argument may be large.
    ///
    /// # Precision
    /// Machine precision at every argument (measured below `1e-14` relative, and below `1e-15`
    /// away from the series/quadrature join). Unlike the oscillatory members this is a genuine
    /// *relative* guarantee: `K` has no zeros, and the method — the trapezoidal rule on
    /// `K₀(x) = ∫₀^∞ e^{−x cosh t}dt` — is a sum of positive terms, so nothing cancels at any
    /// argument.
    fn bessel_k0(self) -> Self;

    /// The modified Bessel function of the second kind `K₁(x)`, for `x > 0`.
    ///
    /// Has a simple pole at the origin (`K₁ ~ 1/x`) and underflows for large `x` like
    /// [`bessel_k0`](Self::bessel_k0).
    ///
    /// # Precision
    /// As [`bessel_k0`](Self::bessel_k0).
    fn bessel_k1(self) -> Self;

    /// The exponentially-scaled `K₀ᵉ(x) = e^{x}K₀(x)`, finite for every positive argument.
    ///
    /// The overflow-free companion to [`bessel_k0`](Self::bessel_k0), and the primitive of the
    /// two: `K₀(746)` underflows to zero while `K₀ᵉ(746)` is an ordinary `0.046`, and the
    /// quadrature computes the scaled form directly rather than scaling back down.
    ///
    /// # Precision
    /// Machine precision at every argument, including `1e8` and beyond.
    fn bessel_k0e(self) -> Self;

    /// The exponentially-scaled `K₁ᵉ(x) = e^{x}K₁(x)`.
    ///
    /// # Precision
    /// As [`bessel_k0e`](Self::bessel_k0e).
    fn bessel_k1e(self) -> Self;

    /// The sine integral `Si(x) = ∫₀ˣ sin(t)/t dt`. Odd in `x`, with `Si(±∞) = ±π/2`.
    ///
    /// # Precision
    /// Machine precision at every argument (measured below `1e-15`). Reached through the
    /// *same continued fraction* as [`exp_int_e1`](Self::exp_int_e1), evaluated on the
    /// imaginary axis where `E₁(ix) = −Ci(x) + i(Si(x) − π/2)`, so `Si` and `Ci` come from one
    /// pass and stay mutually consistent. See the note above regarding the zeros.
    fn sin_int(self) -> Self;

    /// The cosine integral `Ci(x) = γ + ln x + ∫₀ˣ (cos t − 1)/t dt`, for `x > 0`; `NaN` below
    /// zero and `−∞` at the origin.
    ///
    /// # Precision
    /// As [`sin_int`](Self::sin_int).
    fn cos_int(self) -> Self;

    /// The exponential integral `E₁(x) = ∫ₓ^∞ e^{−t}/t dt`, for `x > 0`; `+∞` at the origin
    /// and `NaN` below it.
    ///
    /// # Precision
    /// Machine precision at every argument (measured below `1e-15`), from the power series
    /// below `x = 1` and a continued fraction above it.
    fn exp_int_e1(self) -> Self;

    /// The exponential integral `Ei(x)`, the principal value of `−∫_{−x}^∞ e^{−t}/t dt`.
    ///
    /// Related to [`exp_int_e1`](Self::exp_int_e1) by `Ei(−x) = −E₁(x)`, which is how negative
    /// arguments are evaluated. Grows like `e^x/x`, so it overflows for `x ≳ 710`.
    ///
    /// # Precision
    /// Machine precision at every argument (measured below `1e-15`).
    fn exp_int_ei(self) -> Self;

    /// The principal branch of the Lambert `W` function: the `w ≥ −1` satisfying `w·e^w = x`.
    ///
    /// Defined for `x ≥ −1/e`, and `NaN` below that, where the real branches end. This is the
    /// function to reach for whenever an unknown appears both inside and outside an
    /// exponential.
    ///
    /// # Precision
    /// Machine precision away from the branch point, from Halley's method (which triples the
    /// correct digits per step) started from a regime-appropriate guess.
    ///
    /// **At the branch point accuracy is inherently halved.** `W` behaves like
    /// `−1 + √(2(1 + ex))` there, so a relative perturbation `ε` in `x` moves `w + 1` by
    /// `√ε` — about `1e-8` rather than `1e-16`. That is a property of the function, not of the
    /// method: the derivative of `w·e^w` vanishes at `w = −1`, so the inverse is not
    /// Lipschitz there.
    fn lambert_w(self) -> Self;

    /// The Riemann zeta function `ζ(s)` for real `s`, with `+∞` at the pole `s = 1`.
    ///
    /// For `s > 1` this is the Euler–Maclaurin summation of the defining series; for `s < 1`
    /// the functional equation continues it, with the trivial zeros at the negative even
    /// integers returned as exact zeros rather than left to cancellation.
    ///
    /// # Precision
    /// Measured below `1e-13` relative across `s ∈ [−10, 200]`.
    fn zeta(self) -> Self;

    // --- the Airy family ---
    // The two solutions of `y″ = xy`, the equation of a linear turning point. They are the one
    // family on this surface that changes character *within* its domain: exponential for
    // `x > 0`, oscillatory for `x < 0`. So both contracts above apply, each on its own side —
    // a relative guarantee on the right, and the absolute one the oscillatory note describes
    // on the left.

    /// The Airy function of the first kind `Ai(x)`, the solution of `y″ = xy` that decays.
    ///
    /// Decays like `e^{−ζ}` with `ζ = ⅔x^{3/2}`, so it **underflows to zero for `x ≳ 104`**
    /// exactly as the true value does; reach for [`airy_ai_e`](Self::airy_ai_e) there.
    ///
    /// # Precision
    /// Machine precision, delivered by three methods meeting: a Maclaurin series near the
    /// origin, the `K_{1/3}` quadrature on the positive axis, and an outward march of the
    /// equation itself through the band on the negative axis where neither series nor
    /// asymptotic reaches it.
    ///
    /// - `x < 0`: **absolute**, and degrading as `ζ·ε` with `ζ = ⅔|x|^{3/2}` — below `1e-15`
    ///   out to `|x| ≈ 8`, and below `1e-13` out to `|x| = 30` (measured worst `3.7e-14`, in
    ///   the derivatives, which carry the larger amplitude). `ζ` is the *phase* on this side,
    ///   so its own rounding transfers straight into the value: one ulp of `ζ ≈ 98` is already
    ///   `1.4e-14`. `J` and `Y` carry the same mechanism, but in their argument directly rather
    ///   than through a `3/2` power, so it bites them far later.
    ///
    ///   Relative accuracy is not promised here, and cannot be — `Ai` has infinitely many zeros
    ///   on this axis; see the note above.
    /// - `0 ≤ x ≲ 50`: below `1e-14` relative (measured `5e-15`).
    /// - `50 ≲ x ≲ 104`: the relative error grows like `ζ·ε` with `ζ = ⅔x^{3/2}`, reaching
    ///   `1.5e-13` at `x = 100` just before the value underflows away. That term is the price
    ///   of the `e^{−ζ}` that recovers this function from the scaled one, and `ζ`'s own
    ///   rounding is amplified by it. [`airy_ai_e`](Self::airy_ai_e) never forms it and stays
    ///   below `1e-15` at every argument; prefer it if this range matters.
    ///
    /// Above `|x| ≈ 1e11` the *phase* `ζ` can no longer be resolved in `f64` — `ulp(ζ)` passes
    /// `2π` — so on the negative axis the values there are bounded by the envelope
    /// `π^{−1/2}|x|^{−1/4}` and on it, but no longer accurate pointwise. This is the same
    /// limit `J` and `Y` carry, and for the same reason.
    fn airy_ai(self) -> Self;

    /// The derivative `Ai′(x)`.
    ///
    /// # Precision
    /// As [`airy_ai`](Self::airy_ai); produced by the same march and the same quadrature, at
    /// order `2/3` rather than `1/3`.
    fn airy_ai_prime(self) -> Self;

    /// The Airy function of the second kind `Bi(x)`, the solution of `y″ = xy` that grows.
    ///
    /// Grows like `e^{+ζ}`, so it **overflows for `x ≳ 104`** exactly as the true value does;
    /// reach for [`airy_bi_e`](Self::airy_bi_e) there.
    ///
    /// # Precision
    /// As [`airy_ai`](Self::airy_ai), including the `ζ·ε` term above `x ≈ 50`. On the positive
    /// axis `Bi` is the easier of the two — it is a *sum* of positive series terms there, with
    /// none of `Ai`'s cancellation — so the series carries it to `x = 9` before the `I`
    /// asymptotic takes over, against `x = 1` for `Ai`.
    fn airy_bi(self) -> Self;

    /// The derivative `Bi′(x)`.
    ///
    /// # Precision
    /// As [`airy_bi`](Self::airy_bi).
    fn airy_bi_prime(self) -> Self;

    /// The scaled `Aiᵉ(x) = e^{ζ}Ai(x)` for `x > 0`, and `Ai(x)` itself for `x ≤ 0`.
    ///
    /// The overflow-free companion to [`airy_ai`](Self::airy_ai), and the primitive from which
    /// it is recovered: finite at every finite argument, tending to zero like
    /// `1/(2√π·x^{1/4})`.
    ///
    /// **On the negative axis this returns the unscaled value**, because there `ζ` is a phase
    /// rather than a magnitude and the function already sits inside a decaying envelope.
    /// Scaling it would mean multiplying by `e^{iζ}`'s modulus, which is `1`. The mathematics
    /// does not settle this; the convention is chosen so that the scaled form always does what
    /// scaled forms are *for* — keep a value in range — and never changes one that is already
    /// in range.
    ///
    /// # Precision
    /// Below `1e-15` relative at **every** positive argument — measured worst `2.7e-16` across
    /// `x ∈ [20, 1e100]` — because the exponential is never formed. This is the accurate form;
    /// [`airy_ai`](Self::airy_ai) is the convenient one. On the negative axis, as
    /// [`airy_ai`](Self::airy_ai), the two being the same function there.
    fn airy_ai_e(self) -> Self;

    /// The scaled `Aiᵉ′(x) = e^{ζ}Ai′(x)` for `x > 0`, and `Ai′(x)` itself for `x ≤ 0`.
    ///
    /// # Precision
    /// As [`airy_ai_e`](Self::airy_ai_e). Grows like `−x^{1/4}/(2√π)` rather than decaying.
    fn airy_ai_e_prime(self) -> Self;

    /// The scaled `Biᵉ(x) = e^{−ζ}Bi(x)` for `x > 0`, and `Bi(x)` itself for `x ≤ 0`.
    ///
    /// The overflow-free companion to [`airy_bi`](Self::airy_bi), tending to zero like
    /// `1/(√π·x^{1/4})`. See [`airy_ai_e`](Self::airy_ai_e) for the negative-axis convention.
    ///
    /// # Precision
    /// As [`airy_ai_e`](Self::airy_ai_e): below `1e-15` relative at every positive argument.
    fn airy_bi_e(self) -> Self;

    /// The scaled `Biᵉ′(x) = e^{−ζ}Bi′(x)` for `x > 0`, and `Bi′(x)` itself for `x ≤ 0`.
    ///
    /// # Precision
    /// As [`airy_bi_e`](Self::airy_bi_e). Grows like `x^{1/4}/√π`.
    fn airy_bi_e_prime(self) -> Self;
}
