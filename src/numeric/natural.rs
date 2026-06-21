//! The natural lazy semiring surface [`LazyNatural`] — the semiring-tier (no subtraction,
//! no negation) peer of [`LazyInteger`](super::LazyInteger).
//!
//! Naturals form a semiring: `+ *` with identities `0`/`1`, and Euclidean
//! [`div_rem`](LazyNatural::div_rem), but no additive inverse, so `LazyNatural` is a
//! [`LazySemiring`](super::LazySemiring) without `Sub`/`Neg` — strictly one rung below the
//! integer ring. It is implemented for `u32`/`u64`/`u128` — XLA/JAX's unsigned
//! `u32`/`u64` convention — each width-matched to the real it casts into (`u32 → f32`,
//! `u64`/`u128 → f64`) and widened into the signed ring it bridges to (`u32 → i64`,
//! `u64`/`u128 → i128`). This is the dtype a backend uses for non-negative count/index/shape
//! data.

use super::{LazyInteger, LazyReal, LazySemiring};

/// A lazy semiring of naturals: a [`LazySemiring`] plus the natural surface — machine-natural
/// lift, the cast into the real analysis surface, the widen into the signed integer ring, and
/// Euclidean division-with-remainder. Implemented by `u32`/`u64`/`u128`, and the seam an
/// unsigned tensor (an XLA `u64` array) slots into.
///
/// Naturals carry no `signum`/`abs` (they are already non-negative) and no `Sub`/`Neg` (the
/// semiring has no additive inverse) — cross into [`Integer`](Self::Integer) via
/// [`to_integer`](Self::to_integer) when a formula needs subtraction.
pub trait LazyNatural: LazySemiring {
    /// The real surface this natural casts into — `f32` for `u32`, `f64` for `u64`/`u128`.
    /// Mirrors [`LazyInteger::Real`](super::LazyInteger::Real).
    type Real: LazyReal;

    /// The signed integer ring this natural widens into — `i64` for `u32`, `i128` for
    /// `u64`/`u128`, the destination of [`to_integer`](Self::to_integer). Wide enough to
    /// hold every value for `u32`/`u64`; `u128` wraps above `i128::MAX`, as the hardware's
    /// convert does.
    type Integer: LazyInteger;

    /// Lift a machine natural into this substrate — the natural analogue of
    /// [`LazyInteger::from_i64`].
    fn from_u64(x: u64) -> Self;

    /// `0`, the additive identity.
    #[inline]
    fn zero() -> Self {
        Self::from_u64(0)
    }

    /// `1`, the multiplicative identity.
    #[inline]
    fn one() -> Self {
        Self::from_u64(1)
    }

    /// Cast into the real analysis surface (XLA `ConvertElementType`) — the boundary a
    /// discrete formula crosses once, after exact natural arithmetic and before `ln` /
    /// `beta_reg` / … . Beyond `2^53` the `f64` cast rounds, as the hardware does.
    fn to_real(&self) -> Self::Real;

    /// Widen into the signed integer ring (XLA `ConvertElementType`, `u → s`) — the bridge a
    /// formula crosses when it needs subtraction or negation the semiring withholds. The
    /// natural↔integer peer of [`to_real`](Self::to_real)'s integer↔real cast; total for
    /// `u32`/`u64`, wrapping for `u128` above `i128::MAX`.
    fn to_integer(&self) -> Self::Integer;

    /// Euclidean division with remainder `(quotient, remainder)` so
    /// `self == quotient * rhs + remainder` (XLA `Div`+`Rem`). With no sign this is the
    /// ordinary floor division — the semiring peer of [`LazyInteger::div_rem`], without the
    /// truncate-toward-zero caveat.
    fn div_rem(self, rhs: Self) -> (Self, Self);

    /// `ln(n!) = ln Γ(n + 1)` in the associated real surface.
    ///
    /// For a lazy unsigned tensor this is the elementwise count-to-real formula. Concrete
    /// host naturals may override with exact small-count loops before falling back to
    /// `ln_gamma`.
    #[inline]
    fn ln_factorial(&self) -> Self::Real {
        (self.to_real() + Self::Real::one()).ln_gamma()
    }

    /// `n!` in the associated real surface.
    #[inline]
    fn factorial(&self) -> Self::Real {
        self.ln_factorial().exp()
    }

    /// `ln C(n, k)`, valid on the combinatorial support `k <= n`.
    ///
    /// The generic lazy form cannot branch on `k > n` without a mask/select surface, so
    /// out-of-support policy is left to the caller or to concrete overrides.
    #[inline]
    fn ln_binomial(&self, k: &Self) -> Self::Real {
        let one = Self::Real::one();
        let n = self.to_real();
        let k = k.to_real();
        let n_minus_k = n.clone() - k.clone();
        (n + one.clone()).ln_gamma() - (k + one.clone()).ln_gamma() - (n_minus_k + one).ln_gamma()
    }

    /// `C(n, k)` in the associated real surface, valid on the combinatorial support
    /// `k <= n`.
    #[inline]
    fn binomial(&self, k: &Self) -> Self::Real {
        self.ln_binomial(k).exp()
    }
}

impl LazyNatural for u32 {
    type Real = f32;
    type Integer = i64;
    #[inline]
    fn from_u64(x: u64) -> Self {
        x as u32
    }
    #[inline]
    fn to_real(&self) -> f32 {
        *self as f32
    }
    #[inline]
    fn to_integer(&self) -> i64 {
        *self as i64
    }
    #[inline]
    fn div_rem(self, rhs: Self) -> (Self, Self) {
        (self / rhs, self % rhs)
    }
    #[inline]
    fn ln_factorial(&self) -> f32 {
        crate::numeric::special::host::factorial::ln_factorial_u32(*self)
    }
    #[inline]
    fn factorial(&self) -> f32 {
        crate::numeric::special::host::factorial::factorial_u32(*self)
    }
    #[inline]
    fn ln_binomial(&self, k: &Self) -> f32 {
        crate::numeric::special::host::factorial::ln_binomial_u32(*self, *k)
    }
    #[inline]
    fn binomial(&self, k: &Self) -> f32 {
        crate::numeric::special::host::factorial::binomial_u32(*self, *k)
    }
}

impl LazyNatural for u64 {
    type Real = f64;
    type Integer = i128;
    #[inline]
    fn from_u64(x: u64) -> Self {
        x
    }
    #[inline]
    fn to_real(&self) -> f64 {
        *self as f64
    }
    #[inline]
    fn to_integer(&self) -> i128 {
        *self as i128
    }
    #[inline]
    fn div_rem(self, rhs: Self) -> (Self, Self) {
        (self / rhs, self % rhs)
    }
    #[inline]
    fn ln_factorial(&self) -> f64 {
        crate::numeric::special::host::factorial::ln_factorial(*self)
    }
    #[inline]
    fn factorial(&self) -> f64 {
        crate::numeric::special::host::factorial::factorial(*self)
    }
    #[inline]
    fn ln_binomial(&self, k: &Self) -> f64 {
        crate::numeric::special::host::factorial::ln_binomial(*self, *k)
    }
    #[inline]
    fn binomial(&self, k: &Self) -> f64 {
        crate::numeric::special::host::factorial::binomial(*self, *k)
    }
}

impl LazyNatural for u128 {
    type Real = f64;
    type Integer = i128;
    #[inline]
    fn from_u64(x: u64) -> Self {
        x as u128
    }
    #[inline]
    fn to_real(&self) -> f64 {
        *self as f64
    }
    #[inline]
    fn to_integer(&self) -> i128 {
        *self as i128
    }
    #[inline]
    fn div_rem(self, rhs: Self) -> (Self, Self) {
        (self / rhs, self % rhs)
    }
}
