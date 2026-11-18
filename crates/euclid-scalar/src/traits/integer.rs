//! Signed integer carrier surface.
//!
//! Use it for signed count, index, and shape arithmetic. Implemented for `i32`/`i64`/`i128`,
//! each width-matched to the real it casts into (`i32 → f32`, `i64`/`i128 → f64`).

use super::{Real, Ring};

/// A [`Ring`] plus the integer surface: machine-integer lift, the cast into real analysis,
/// sign/abs, and Euclidean division-with-remainder. Implemented by `i32`/`i64`/`i128`.
pub trait Integer: Ring {
    /// The real surface this integer casts into — `f32` for `i32`, `f64` for `i64`/`i128`.
    type Real: Real;

    /// Lift a machine integer into this carrier. (`0`/`1` come from
    /// [`Element`](super::Element).)
    fn from_i64(x: i64) -> Self;

    /// Cast into real analysis — the boundary a discrete formula crosses once, after exact
    /// integer arithmetic and before `ln`/`beta_reg`/… . Beyond `2^53` the `f64` cast rounds.
    fn to_real(&self) -> Self::Real;

    /// Round a real back into this carrier, the inverse of [`to_real`](Self::to_real).
    /// Out-of-range values saturate.
    fn from_real(r: Self::Real) -> Self;

    /// The sign as one of `-1`, `0`, `1`.
    fn signum(self) -> Self;

    /// The absolute value.
    fn abs(self) -> Self;

    /// Euclidean division with remainder `(quotient, remainder)`, so
    /// `self == quotient * rhs + remainder` and `0 <= remainder < abs(rhs)` when `rhs != 0`.
    /// This is integer division, not the field operator `÷`.
    ///
    /// # Panics
    /// Panics when `rhs == 0`, and on the overflow case `MIN.div_rem(-1)` whose quotient
    /// `|MIN|` is unrepresentable — matching Rust's `div_euclid`/`rem_euclid`.
    fn div_rem(self, rhs: Self) -> (Self, Self);

    // --- exact integer arithmetic ---

    /// The greatest common divisor, always **non-negative** — `gcd(-12, 18) == 6`.
    ///
    /// Sign is dropped because a common divisor is a magnitude; returning the non-negative
    /// representative is the convention that makes `gcd` usable for reducing a fraction
    /// without a separate sign fix-up.
    ///
    /// # Panics
    /// On `MIN` inputs, whose absolute value is unrepresentable.
    fn gcd(self, other: Self) -> Self;

    /// The least common multiple, always **non-negative**, and `0` when either operand is `0`.
    ///
    /// # Panics
    /// On an unrepresentable result, and on `MIN` inputs. See [`Natural::lcm`](super::Natural::lcm).
    fn lcm(self, other: Self) -> Self;
}
