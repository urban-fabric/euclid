//! Exact rational arithmetic — [`Rational<I>`] over an [`Integer`] backing.
//!
//! A rational is kept in lowest terms with a positive denominator, so its arithmetic
//! is *exact* (no rounding) and structural equality is mathematical equality. As a
//! [`Field`] it powers exact linear algebra (Gaussian elimination over ℚ). The `i128`
//! backing ([`Rational128`]) can still overflow on long elimination chains; the
//! [`Integer`] seam is the upgrade path to an arbitrary-precision backing.

use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::algebra::{
    AdditiveMagma, AssociativeAddition, AssociativeMultiplication, CommutativeAddition,
    CommutativeMultiplication, Distributive, EuclideanDomain, Field, HasOne, HasZero, Integer,
    IntegralDomain, MultiplicativeMagma, Pid, RefAdditiveMagma, RefMultiplicativeMagma, Ufd,
    ZeroAnnihilatesMultiplication,
};

/// A rational number `num/den` over an integer backing `I`, kept in lowest terms with
/// `den > 0`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rational<I> {
    num: I,
    den: I,
}

/// A rational backed by `i64`.
pub type Rational64 = Rational<i64>;
/// A rational backed by `i128` (wider headroom before overflow).
pub type Rational128 = Rational<i128>;

impl<I: Integer> Rational<I> {
    /// `num / den` reduced to lowest terms (the sign is carried on the numerator).
    ///
    /// # Panics
    /// If `den` is zero.
    pub fn new(num: I, den: I) -> Self {
        assert!(den != I::ZERO, "rational denominator is zero");
        Self::reduce(num, den)
    }

    /// The integer `n` as the rational `n/1`.
    pub fn integer(n: I) -> Self {
        Self {
            num: n,
            den: I::ONE,
        }
    }

    /// The numerator (in lowest terms).
    pub fn numerator(self) -> I {
        self.num
    }

    /// The denominator (in lowest terms, always positive).
    pub fn denominator(self) -> I {
        self.den
    }

    /// Reduce `num/den` to lowest terms with a positive denominator.
    fn reduce(num: I, den: I) -> Self {
        let (num, den) = if den.signum() == -I::ONE {
            (-num, -den)
        } else {
            (num, den)
        };
        if num == I::ZERO {
            return Self {
                num: I::ZERO,
                den: I::ONE,
            };
        }
        let g = num.gcd(den);
        Self {
            num: num.exact_div(g),
            den: den.exact_div(g),
        }
    }
}

// The fused algebra trunk: ℚ is a numeric carrier, so its magma `add`/`mul` ARE
// `+`/`*`, and `Semiring`/`Ring`/`CommutativeRing`/`OrderedField` are blanket-derived
// from the magma layer, identities, law markers, and (for the ordering) the
// `PartialOrd` below — never written directly. ℚ is an ordered field but not
// algebraically closed.
impl<I: Integer> AdditiveMagma for Rational<I> {
    #[inline]
    fn add(self, rhs: Self) -> Self {
        self + rhs
    }
}
impl<I: Integer> MultiplicativeMagma for Rational<I> {
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        self * rhs
    }
}
impl<I: Integer> RefAdditiveMagma for Rational<I> {
    #[inline]
    fn add_ref(&self, rhs: &Self) -> Self {
        *self + *rhs
    }
}
impl<I: Integer> RefMultiplicativeMagma for Rational<I> {
    #[inline]
    fn mul_ref(&self, rhs: &Self) -> Self {
        *self * *rhs
    }
}
impl<I: Integer> HasZero for Rational<I> {
    const ZERO: Self = Self {
        num: I::ZERO,
        den: I::ONE,
    };
}
impl<I: Integer> HasOne for Rational<I> {
    const ONE: Self = Self {
        num: I::ONE,
        den: I::ONE,
    };
}
impl<I: Integer> AssociativeAddition for Rational<I> {}
impl<I: Integer> CommutativeAddition for Rational<I> {}
impl<I: Integer> AssociativeMultiplication for Rational<I> {}
impl<I: Integer> CommutativeMultiplication for Rational<I> {}
impl<I: Integer> Distributive for Rational<I> {}
impl<I: Integer> ZeroAnnihilatesMultiplication for Rational<I> {}
impl<I: Integer> IntegralDomain for Rational<I> {}
impl<I: Integer> Ufd for Rational<I> {}
impl<I: Integer> Pid for Rational<I> {}
impl<I: Integer> EuclideanDomain for Rational<I> {
    fn div_rem(self, rhs: Self) -> (Self, Self) {
        (self / rhs, Self::ZERO)
    }
}
impl<I: Integer> Field for Rational<I> {
    /// Conjugation is the identity — the rationals are a real field.
    fn conjugate(self) -> Self {
        self
    }
}

impl<I: Integer> Add for Rational<I> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::reduce(self.num * rhs.den + rhs.num * self.den, self.den * rhs.den)
    }
}

impl<I: Integer> Sub for Rational<I> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::reduce(self.num * rhs.den - rhs.num * self.den, self.den * rhs.den)
    }
}

impl<I: Integer> Mul for Rational<I> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self::reduce(self.num * rhs.num, self.den * rhs.den)
    }
}

impl<I: Integer> Div for Rational<I> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        Self::reduce(self.num * rhs.den, self.den * rhs.num)
    }
}

impl<I: Integer> Neg for Rational<I> {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            num: -self.num,
            den: self.den,
        }
    }
}

impl<I: Integer> AddAssign for Rational<I> {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<I: Integer> SubAssign for Rational<I> {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<I: Integer> MulAssign for Rational<I> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<I: Integer> DivAssign for Rational<I> {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl<I: Integer> PartialOrd for Rational<I> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<I: Integer> Ord for Rational<I> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Denominators are positive, so cross-multiplication preserves order.
        (self.num * other.den).cmp(&(other.num * self.den))
    }
}

impl<I: Integer + fmt::Display> fmt::Display for Rational<I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.den == I::ONE {
            write!(f, "{}", self.num)
        } else {
            write!(f, "{}/{}", self.num, self.den)
        }
    }
}
