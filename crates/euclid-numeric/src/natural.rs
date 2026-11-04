//! Unsigned natural-number carrier surface.
//!
//! Use it for non-negative count, index, and shape arithmetic. Implemented for
//! `u32`/`u64`/`u128`, each width-matched to the real it casts into (`u32 → f32`,
//! `u64`/`u128 → f64`) and the signed ring it widens into (`u32 → i64`, `u64`/`u128 → i128`).

use super::{Integer, Real, Semiring};

/// A [`Semiring`] plus the natural surface: machine-natural lift, the cast into real
/// analysis, the widen into the signed integer ring, and Euclidean division-with-remainder.
/// Implemented by `u32`/`u64`/`u128`.
///
/// Naturals carry no `signum`/`abs` (they are already non-negative) and no `Sub`/`Neg` (the
/// semiring has no additive inverse) — cross into [`Integer`](Self::Integer) via
/// [`to_integer`](Self::to_integer) when a formula needs subtraction.
pub trait Natural: Semiring {
    /// The real surface this natural casts into — `f32` for `u32`, `f64` for `u64`/`u128`.
    type Real: Real;

    /// The signed integer ring this natural widens into — `i64` for `u32`, `i128` for
    /// `u64`/`u128`. Wide enough to hold every `u32`/`u64` value; `u128` wraps above
    /// `i128::MAX`, as the hardware's convert does.
    type Integer: Integer;

    /// Lift a machine natural into this carrier. (`0`/`1` come from
    /// [`Element`](super::Element).)
    fn from_u64(x: u64) -> Self;

    /// Cast into real analysis — the boundary a discrete formula crosses once, after exact
    /// natural arithmetic and before `ln`/`beta_reg`/… . Beyond `2^53` the `f64` cast rounds.
    fn to_real(&self) -> Self::Real;

    /// Round a real back into this carrier, the inverse of [`to_real`](Self::to_real).
    /// Negative inputs clamp to zero; out-of-range values saturate.
    fn from_real(r: Self::Real) -> Self;

    /// Widen into the signed integer ring, for when a formula needs the subtraction or
    /// negation the semiring withholds. Total for `u32`/`u64`, wrapping for `u128` above
    /// `i128::MAX`.
    fn to_integer(&self) -> Self::Integer;

    /// Euclidean division with remainder `(quotient, remainder)` so
    /// `self == quotient * rhs + remainder`. With no sign this is ordinary floor division.
    ///
    /// # Panics
    /// Panics when `rhs == 0`, as all integer division does. Unsigned division has no
    /// overflow case.
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

    // --- exact integer arithmetic ---
    //
    // Operations that are *exactly* representable in the natural tier and would lose that
    // guarantee if routed through the real surface. A count has no business taking a detour
    // through floating point.

    /// The greatest common divisor. `gcd(0, 0)` is `0`, and `gcd(n, 0) == n`.
    ///
    /// Exact and total — no overflow case exists, since the result never exceeds either
    /// operand. The basis of exact rational arithmetic, stride and lattice reasoning, and
    /// reducing a ratio of counts to lowest terms.
    fn gcd(self, other: Self) -> Self;

    /// The least common multiple, `0` when either operand is `0`.
    ///
    /// # Panics
    /// The result can exceed the carrier's range even when both operands are small
    /// (`lcm(2^63, 3)` does not fit in a `u64`). Implementations compute
    /// `(self / gcd) * other`, which avoids overflowing on the way to a representable answer,
    /// but a genuinely unrepresentable result overflows as any multiplication would.
    fn lcm(self, other: Self) -> Self;

    /// The integer square root: the largest `r` with `r * r <= self`.
    ///
    /// Exact, unlike `to_real().sqrt()` rounded back — which is off by one near perfect
    /// squares above `2^52`, where the `f64` round trip cannot distinguish `n` from `n+1`.
    fn isqrt(self) -> Self;
}
