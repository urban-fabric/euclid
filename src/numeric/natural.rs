//! Unsigned natural-number carrier surface.
//!
//! Naturals form a semiring: `+ *` with identities `0`/`1`, and Euclidean
//! [`div_rem`](Natural::div_rem), but no additive inverse, so `Natural` is a
//! [`Semiring`](super::Semiring) without `Sub`/`Neg` — strictly one rung below the
//! integer ring. It is implemented for `u32`/`u64`/`u128`, each width-matched to the real it
//! casts into (`u32 → f32`, `u64`/`u128 → f64`) and widened into the signed ring it bridges to
//! (`u32 → i64`, `u64`/`u128 → i128`). Use it for non-negative count, index, and shape
//! arithmetic. The concrete `u32`/`u64`/`u128` impls live in
//! [`impls::natural_impl`](super::impls).

use super::{Integer, Real, Semiring};

/// A semiring of naturals: a [`Semiring`] plus the natural surface — machine-natural
/// lift, the cast into the real analysis surface, the widen into the signed integer ring, and
/// Euclidean division-with-remainder. Implemented by `u32`/`u64`/`u128`; unsigned carriers can
/// implement the same surface.
///
/// Naturals carry no `signum`/`abs` (they are already non-negative) and no `Sub`/`Neg` (the
/// semiring has no additive inverse) — cross into [`Integer`](Self::Integer) via
/// [`to_integer`](Self::to_integer) when a formula needs subtraction.
pub trait Natural: Semiring {
    /// The real surface this natural casts into — `f32` for `u32`, `f64` for `u64`/`u128`.
    /// Mirrors [`Integer::Real`](super::Integer::Real).
    type Real: Real;

    /// The signed integer ring this natural widens into — `i64` for `u32`, `i128` for
    /// `u64`/`u128`, the destination of [`to_integer`](Self::to_integer). Wide enough to
    /// hold every value for `u32`/`u64`; `u128` wraps above `i128::MAX`, as the hardware's
    /// convert does.
    type Integer: Integer;

    /// Lift a machine natural into this substrate — the natural analogue of
    /// [`Integer::from_i64`]. (`0`/`1` come from the [`Element`](super::Element)
    /// supertrait, defined directly by each carrier.)
    fn from_u64(x: u64) -> Self;

    /// Cast into the real analysis surface — the boundary a discrete formula crosses once,
    /// after exact natural arithmetic and before `ln` /
    /// `beta_reg` / … . Beyond `2^53` the `f64` cast rounds, as the hardware does.
    fn to_real(&self) -> Self::Real;

    /// Widen into the signed integer ring — the bridge a formula crosses when it needs
    /// subtraction or negation the semiring withholds. The
    /// natural↔integer peer of [`to_real`](Self::to_real)'s integer↔real cast; total for
    /// `u32`/`u64`, wrapping for `u128` above `i128::MAX`.
    fn to_integer(&self) -> Self::Integer;

    /// Euclidean division with remainder `(quotient, remainder)` so
    /// `self == quotient * rhs + remainder`. With no sign this is the
    /// ordinary floor division — the semiring peer of [`Integer::div_rem`], without the
    /// truncate-toward-zero caveat.
    fn div_rem(self, rhs: Self) -> (Self, Self);

    /// `ln(n!)` in the associated real surface.
    fn ln_factorial(&self) -> Self::Real;

    /// `n!` in the associated real surface.
    fn factorial(&self) -> Self::Real;

    /// `ln C(n, k)`, valid on the combinatorial support `k <= n`.
    fn ln_binomial(&self, k: &Self) -> Self::Real;

    /// `C(n, k)` in the associated real surface, valid on the combinatorial support
    /// `k <= n`.
    fn binomial(&self, k: &Self) -> Self::Real;
}
