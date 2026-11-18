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

use crate::EuclideanDomain;

#[path = "integer/wide.rs"]
mod wide;

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
    ///
    /// # Panics
    /// If the nonnegative result exceeds the backing range.
    fn gcd(self, other: Self) -> Self;

    /// The sign as one of `-1`, `0`, `1`.
    fn signum(self) -> Self;

    /// The absolute value.
    ///
    /// # Panics
    /// If the nonnegative result exceeds the backing range.
    fn abs(self) -> Self;

    /// Exact integer division `self / other`, valid only where `other` divides
    /// `self` exactly — the sole context the Bareiss / rational-reduction paths use
    /// it, where exactness is guaranteed.
    fn exact_div(self, other: Self) -> Self;

    /// Multiply, returning `None` when the exact product is outside the backing range.
    fn checked_product(self, other: Self) -> Option<Self>;

    /// Negate, returning `None` when the exact result is outside the backing range.
    fn checked_negation(self) -> Option<Self>;

    /// Add or subtract two reduced fractions with positive denominators.
    ///
    /// Return the reduced result with a positive denominator, or `None` if its components
    /// do not fit. Intermediate overflow must be resolved before deciding whether the result fits.
    fn checked_fraction_sum(
        self,
        den: Self,
        rhs: Self,
        rhs_den: Self,
        subtract: bool,
    ) -> Option<(Self, Self)>;
}

/// Euclidean algorithm on unsigned magnitudes.
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
        let gcd = gcd_abs(self.unsigned_abs(), other.unsigned_abs(), 0);
        assert!(i64::try_from(gcd).is_ok(), "integer gcd overflow");
        gcd as i64
    }
    fn signum(self) -> Self {
        i64::signum(self)
    }
    fn abs(self) -> Self {
        assert!(self != i64::MIN, "integer absolute value overflow");
        i64::abs(self)
    }
    fn exact_div(self, other: Self) -> Self {
        self / other
    }
    fn checked_product(self, other: Self) -> Option<Self> {
        self.checked_mul(other)
    }
    fn checked_negation(self) -> Option<Self> {
        self.checked_neg()
    }
    fn checked_fraction_sum(
        self,
        den: Self,
        rhs: Self,
        rhs_den: Self,
        subtract: bool,
    ) -> Option<(Self, Self)> {
        let (num, den) = wide::fraction_sum(
            self.into(),
            den.into(),
            rhs.into(),
            rhs_den.into(),
            subtract,
        )?;
        Some((i64::try_from(num).ok()?, i64::try_from(den).ok()?))
    }
}

impl Integer for i128 {
    fn from_i64(x: i64) -> Self {
        x as i128
    }
    fn gcd(self, other: Self) -> Self {
        let gcd = gcd_abs(self.unsigned_abs(), other.unsigned_abs(), 0);
        assert!(i128::try_from(gcd).is_ok(), "integer gcd overflow");
        gcd as i128
    }
    fn signum(self) -> Self {
        i128::signum(self)
    }
    fn abs(self) -> Self {
        assert!(self != i128::MIN, "integer absolute value overflow");
        i128::abs(self)
    }
    fn exact_div(self, other: Self) -> Self {
        self / other
    }
    fn checked_product(self, other: Self) -> Option<Self> {
        self.checked_mul(other)
    }
    fn checked_negation(self) -> Option<Self> {
        self.checked_neg()
    }
    fn checked_fraction_sum(
        self,
        den: Self,
        rhs: Self,
        rhs_den: Self,
        subtract: bool,
    ) -> Option<(Self, Self)> {
        wide::fraction_sum(self, den, rhs, rhs_den, subtract)
    }
}
