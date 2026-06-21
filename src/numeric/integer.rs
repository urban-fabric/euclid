//! The integer lazy ring surface [`LazyInteger`] — the ring-tier (no division) peer of
//! [`LazyReal`](super::LazyReal).
//!
//! Integers form a ring: `+ - *` and Euclidean [`div_rem`](LazyInteger::div_rem), but no
//! inverse, so `LazyInteger` is a [`LazyRing`](super::LazyRing) without `Div`. Only signed
//! integers qualify (the naturals are a semiring), so it is implemented for
//! `i32`/`i64`/`i128` — XLA/JAX's signed `s32`/`s64` convention — each width-matched to the
//! real it casts into (`i32 → f32`, `i64`/`i128 → f64`). This is the dtype a backend uses
//! for count/index/shape data.

use super::{LazyReal, LazyRing};

/// A lazy ring of integers: a [`LazyRing`] plus the integer surface — machine-integer lift,
/// the cast into the real analysis surface, sign/abs, and Euclidean division-with-remainder.
/// Implemented by `i32`/`i64`/`i128`, and the seam an integer tensor (an XLA `s64` array)
/// slots into.
pub trait LazyInteger: LazyRing {
    /// The real surface this integer casts into — `f32` for `i32`, `f64` for `i64`/`i128`.
    /// Mirrors [`NormedDivisionAlgebra::RealPart`](super::NormedDivisionAlgebra::RealPart).
    type Real: LazyReal;

    /// Lift a machine integer into this substrate — the integer analogue of
    /// [`LazyReal::scalar`](super::LazyReal::scalar).
    fn from_i64(x: i64) -> Self;

    /// `0`, the additive identity.
    #[inline]
    fn zero() -> Self {
        Self::from_i64(0)
    }

    /// `1`, the multiplicative identity.
    #[inline]
    fn one() -> Self {
        Self::from_i64(1)
    }

    /// Cast into the real analysis surface (XLA `ConvertElementType`) — the boundary a
    /// discrete formula crosses once, after exact integer arithmetic and before `ln` /
    /// `beta_reg` / … . Beyond `2^53` the `f64` cast rounds, as the hardware does.
    fn to_real(&self) -> Self::Real;

    /// The sign as one of `-1`, `0`, `1` (XLA `Sign`).
    fn signum(self) -> Self;

    /// The absolute value (XLA `Abs`).
    fn abs(self) -> Self;

    /// Euclidean division with remainder `(quotient, remainder)`, truncating toward zero so
    /// `self == quotient * rhs + remainder` (XLA `Div`+`Rem`) — not the field `÷`.
    fn div_rem(self, rhs: Self) -> (Self, Self);
}

impl LazyInteger for i32 {
    type Real = f32;
    #[inline]
    fn from_i64(x: i64) -> Self {
        x as i32
    }
    #[inline]
    fn to_real(&self) -> f32 {
        *self as f32
    }
    #[inline]
    fn signum(self) -> Self {
        i32::signum(self)
    }
    #[inline]
    fn abs(self) -> Self {
        i32::abs(self)
    }
    #[inline]
    fn div_rem(self, rhs: Self) -> (Self, Self) {
        (self / rhs, self % rhs)
    }
}

impl LazyInteger for i64 {
    type Real = f64;
    #[inline]
    fn from_i64(x: i64) -> Self {
        x
    }
    #[inline]
    fn to_real(&self) -> f64 {
        *self as f64
    }
    #[inline]
    fn signum(self) -> Self {
        i64::signum(self)
    }
    #[inline]
    fn abs(self) -> Self {
        i64::abs(self)
    }
    #[inline]
    fn div_rem(self, rhs: Self) -> (Self, Self) {
        (self / rhs, self % rhs)
    }
}

impl LazyInteger for i128 {
    type Real = f64;
    #[inline]
    fn from_i64(x: i64) -> Self {
        x as i128
    }
    #[inline]
    fn to_real(&self) -> f64 {
        *self as f64
    }
    #[inline]
    fn signum(self) -> Self {
        i128::signum(self)
    }
    #[inline]
    fn abs(self) -> Self {
        i128::abs(self)
    }
    #[inline]
    fn div_rem(self, rhs: Self) -> (Self, Self) {
        (self / rhs, self % rhs)
    }
}
