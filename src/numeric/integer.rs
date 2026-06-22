//! Signed integer carrier surface.
//!
//! Integers form a ring: `+ - *` and Euclidean [`div_rem`](Integer::div_rem), but no
//! inverse, so `Integer` is a [`Ring`](super::Ring) without `Div`. Only signed
//! integers qualify (the naturals are a [semiring](super::Natural), one rung below). It is
//! implemented for `i32`/`i64`/`i128`, each width-matched to the real it casts into
//! (`i32 → f32`, `i64`/`i128 → f64`). Use it for signed count, index, and shape arithmetic.
//! The concrete `i32`/`i64`/`i128` impls live in
//! [`impls::integer_impl`](super::impls).

use super::{Real, Ring};

/// A ring of integers: a [`Ring`] plus the integer surface — machine-integer lift,
/// the cast into the real analysis surface, sign/abs, and Euclidean division-with-remainder.
/// Implemented by `i32`/`i64`/`i128`; signed integer carriers can implement the same surface.
pub trait Integer: Ring {
    /// The real surface this integer casts into — `f32` for `i32`, `f64` for `i64`/`i128`.
    /// Mirrors [`NormedDivisionAlgebra::RealPart`](super::NormedDivisionAlgebra::RealPart).
    type Real: Real;

    /// Lift a machine integer into this substrate — the integer analogue of
    /// [`Real::scalar`](super::Real::scalar). (`0`/`1` come from the
    /// [`Element`](super::Element) supertrait, defined directly by each carrier.)
    fn from_i64(x: i64) -> Self;

    /// Cast into the real analysis surface — the boundary a discrete formula crosses once,
    /// after exact integer arithmetic and before `ln` /
    /// `beta_reg` / … . Beyond `2^53` the `f64` cast rounds, as the hardware does.
    fn to_real(&self) -> Self::Real;

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
}
