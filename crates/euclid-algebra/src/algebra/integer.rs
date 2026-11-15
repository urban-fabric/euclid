//! The [`Integer`] trait — the ring of integers that [`Rational`](super::Rational)
//! is generic over, and that integer matrices (and a fraction-free determinant)
//! operate in.
//!
//! `i64`/`i128` get their algebra trunk (`Semiring → Ring → … → EuclideanDomain`)
//! and this [`Integer`] gcd/exact-division extension from **euclid** (which owns the
//! traits *and* the primitive impls). The matrix-kernel layer (`MatrixRing`, the
//! portable fallback — integers have no packed path) is supplied by **jacobi**, the
//! crate that owns that trait. Integers are rings, not fields (no conjugate, no field
//! division).

use crate::algebra::EuclideanDomain;

/// A ring of integers: a totally-ordered [`EuclideanDomain`] with gcd and the *exact*
/// division the fraction-free algorithms rely on. Implemented for `i64` and `i128`.
///
/// Being a trait (rather than hard-coding `i64`) is the seam an arbitrary-precision
/// `BigInt` slots into later — `Rational<BigInt>` would then drop in without
/// touching any caller, lifting the overflow ceiling of the fixed-width backings.
pub trait Integer: EuclideanDomain + Ord {
    /// Lift a machine integer into this integer type.
    fn from_i64(x: i64) -> Self;

    /// The greatest common divisor `gcd(self, other) ≥ 0` (with `gcd(0, 0) = 0`).
    fn gcd(self, other: Self) -> Self;

    /// The sign as one of `-1`, `0`, `1`.
    fn signum(self) -> Self;

    /// The absolute value.
    fn abs(self) -> Self;

    /// Exact integer division `self / other`, valid only where `other` divides
    /// `self` exactly — the sole context the Bareiss / rational-reduction paths use
    /// it, where exactness is guaranteed.
    fn exact_div(self, other: Self) -> Self;
}

/// Binary GCD by the Euclidean algorithm on the absolute values; `gcd(0, 0) = 0`.
#[inline]
fn gcd_abs<I: Copy + PartialEq + core::ops::Rem<Output = I>>(mut a: I, mut b: I, zero: I) -> I {
    while b != zero {
        let t = b;
        b = a % t;
        a = t;
    }
    a
}

impl Integer for i64 {
    fn from_i64(x: i64) -> Self {
        x
    }
    fn gcd(self, other: Self) -> Self {
        gcd_abs(self.abs(), other.abs(), 0)
    }
    fn signum(self) -> Self {
        i64::signum(self)
    }
    fn abs(self) -> Self {
        i64::abs(self)
    }
    fn exact_div(self, other: Self) -> Self {
        self / other
    }
}

impl Integer for i128 {
    fn from_i64(x: i64) -> Self {
        x as i128
    }
    fn gcd(self, other: Self) -> Self {
        gcd_abs(self.abs(), other.abs(), 0)
    }
    fn signum(self) -> Self {
        i128::signum(self)
    }
    fn abs(self) -> Self {
        i128::abs(self)
    }
    fn exact_div(self, other: Self) -> Self {
        self / other
    }
}
