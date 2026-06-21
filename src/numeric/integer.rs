//! The integer lazy ring surface [`LazyInteger`] — the ring-tier (no field
//! division) peer of [`LazyReal`](super::LazyReal).
//!
//! Where [`Arithmetic`](super::Arithmetic) is the *field* substrate (`+ - * /`),
//! integers form a **ring**: `+ - *` and Euclidean [`div_rem`](LazyInteger::div_rem),
//! but no multiplicative inverse. `LazyInteger` is therefore a
//! [`LazyRing`](super::LazyRing) — it drops `Div` — exactly as the concrete
//! `Integer` is a `Ring`, not a
//! `Field`. Only the **signed** integers (a true additive
//! group) qualify; the unsigned naturals are a semiring, so — as with the algebra
//! [`Integer`] — this is implemented for `i32`/`i64`/`i128`, the same convention XLA/JAX
//! follow when they default index/count arrays to the signed `s32`/`s64`. Each is
//! width-matched to the real it casts into: `i32 → f32`, `i64`/`i128 → f64`.
//!
//! It is the dtype a lazy backend uses for count / index / shape data (an XLA `s64`
//! tensor): the operators lower to integer `Add`/`Sub`/`Mul`, `div_rem` to `Div`/`Rem`,
//! and [`to_real`](LazyInteger::to_real) to `ConvertElementType s64→f64` — the single
//! cast a discrete log-density takes before it meets the real analysis surface.

use super::{LazyReal, LazyRing};

/// A lazy ring of integers: a [`LazyRing`] (`+ - *`, no `÷`) with the integer-specific
/// surface — the machine-integer lift, the cast into the real analysis surface, sign /
/// absolute value, and Euclidean division-with-remainder. The lazy peer of the concrete
/// `Integer`.
///
/// Implemented by `i32`/`i64`/`i128`; the seam a lazy integer **tensor** (an XLA `s64`
/// array) slots into, so a discrete-statistics formula written over counts runs on every
/// backend.
pub trait LazyInteger: LazyRing {
    /// The real surface this integer casts into — `f32` for `i32`, `f64` for `i64`/`i128`,
    /// a float-dtype tensor for a tensor backend. Mirrors
    /// [`NormedDivisionAlgebra::RealPart`](super::NormedDivisionAlgebra::RealPart): the
    /// real field associated to the carrier.
    type Real: LazyReal;

    /// Lift a machine integer into this integer substrate — the integer analogue of
    /// [`LazyReal::scalar`](super::LazyReal::scalar). A lazy backend represents it as an
    /// integer constant node.
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

    /// Cast into the real analysis surface (XLA `ConvertElementType`). The boundary a
    /// discrete formula crosses **once**: counts stay exact integers through `n − k` /
    /// `k + 1`, then cross here to meet `ln` / `beta_reg` / `gamma_ur`. Casting an
    /// integer beyond `2^53` to an `f64` real rounds, exactly as the hardware cast does.
    fn to_real(&self) -> Self::Real;

    /// The sign as one of `-1`, `0`, `1` (XLA `Sign`).
    fn signum(self) -> Self;

    /// The absolute value (XLA `Abs`).
    fn abs(self) -> Self;

    /// Euclidean division with remainder `(quotient, remainder)` — the integers'
    /// division, **not** the field `÷` a [`LazyRing`] lacks. Truncates toward zero, so
    /// `self == quotient * rhs + remainder` (XLA `Div` paired with `Rem`).
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
