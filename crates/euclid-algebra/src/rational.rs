//! Exact rational arithmetic — [`Rational<I>`] over an [`Integer`] backing.
//!
//! A rational is kept in lowest terms with a positive denominator, so its arithmetic
//! is *exact* (no rounding) and structural equality is mathematical equality. As a
//! [`Field`] it powers exact linear algebra (Gaussian elimination over ℚ). The `i128`
//! backing ([`Rational128`]) can still overflow on long elimination chains. Arithmetic
//! operators panic if the reduced result cannot be represented. Use the checked methods
//! to handle that case as a value.

use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::{
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
    /// If `den` is zero or the reduced components do not fit.
    pub fn new(num: I, den: I) -> Self {
        assert!(den != I::ZERO, "rational denominator is zero");
        Self::exact(Self::checked_new(num, den))
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

    /// Construct a reduced fraction, returning `None` for zero denominator or overflow.
    pub fn checked_new(num: I, den: I) -> Option<Self> {
        if den == I::ZERO {
            return None;
        }
        if num == I::ZERO {
            return Some(Self::ZERO);
        }
        // The gcd of two signed minima does not fit in their backing type.
        if num == den {
            return Some(Self::ONE);
        }
        let g = num.gcd(den);
        let num = num.exact_div(g);
        let den = den.exact_div(g);
        if den.signum() == -I::ONE {
            return Some(Self {
                num: num.checked_negation()?,
                den: den.checked_negation()?,
            });
        }
        Some(Self { num, den })
    }

    /// Add two fractions, returning `None` if the reduced result overflows.
    pub fn checked_add(self, rhs: Self) -> Option<Self> {
        let (num, den) = self
            .num
            .checked_fraction_sum(self.den, rhs.num, rhs.den, false)?;
        Some(Self { num, den })
    }

    /// Subtract two fractions, returning `None` if the reduced result overflows.
    pub fn checked_sub(self, rhs: Self) -> Option<Self> {
        let (num, den) = self
            .num
            .checked_fraction_sum(self.den, rhs.num, rhs.den, true)?;
        Some(Self { num, den })
    }

    /// Multiply two fractions, returning `None` if the reduced result overflows.
    pub fn checked_mul(self, rhs: Self) -> Option<Self> {
        let left_cancel = self.num.gcd(rhs.den);
        let right_cancel = rhs.num.gcd(self.den);
        let num = self
            .num
            .exact_div(left_cancel)
            .checked_product(rhs.num.exact_div(right_cancel))?;
        let den = self
            .den
            .exact_div(right_cancel)
            .checked_product(rhs.den.exact_div(left_cancel))?;
        Some(Self { num, den })
    }

    /// Divide two fractions, returning `None` for a zero divisor or overflow.
    pub fn checked_div(self, rhs: Self) -> Option<Self> {
        if rhs.num == I::ZERO {
            return None;
        }
        if self.num == I::ZERO {
            return Some(Self::ZERO);
        }
        if self.num == rhs.num {
            return Self::checked_new(rhs.den, self.den);
        }
        let num_cancel = self.num.gcd(rhs.num);
        let den_cancel = self.den.gcd(rhs.den);
        let left = self.num.exact_div(num_cancel);
        let right = rhs.num.exact_div(num_cancel);
        let (left, right) = if right.signum() == -I::ONE {
            (left.checked_negation()?, right.checked_negation()?)
        } else {
            (left, right)
        };
        let num = left.checked_product(rhs.den.exact_div(den_cancel))?;
        let den = self.den.exact_div(den_cancel).checked_product(right)?;
        Some(Self { num, den })
    }

    /// Negate a fraction, returning `None` if its numerator cannot be negated.
    pub fn checked_neg(self) -> Option<Self> {
        Some(Self {
            num: self.num.checked_negation()?,
            den: self.den,
        })
    }

    #[track_caller]
    #[allow(clippy::panic)]
    fn exact(result: Option<Self>) -> Self {
        match result {
            Some(value) => value,
            None => panic!("rational overflow"),
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
        Self::exact(self.checked_add(rhs))
    }
}

impl<I: Integer> Sub for Rational<I> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::exact(self.checked_sub(rhs))
    }
}

impl<I: Integer> Mul for Rational<I> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self::exact(self.checked_mul(rhs))
    }
}

impl<I: Integer> Div for Rational<I> {
    type Output = Self;
    /// Divide by a nonzero rational.
    ///
    /// # Panics
    /// Panics if `rhs` is zero or the reduced result overflows.
    fn div(self, rhs: Self) -> Self {
        assert!(rhs.num != I::ZERO, "rational division by zero");
        Self::exact(self.checked_div(rhs))
    }
}

impl<I: Integer> Neg for Rational<I> {
    type Output = Self;
    fn neg(self) -> Self {
        Self::exact(self.checked_neg())
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
        // Every Integer can represent an i64. Restrict cross-products to a range
        // where they fit even in that backing, and use division for wider values.
        let small = I::from_i64(i64::from(i32::MIN))..=I::from_i64(i64::from(i32::MAX));
        if [self.num, self.den, other.num, other.den]
            .iter()
            .all(|part| small.contains(part))
        {
            return (self.num * other.den).cmp(&(other.num * self.den));
        }
        let (mut left_num, mut left_den) = (self.num, self.den);
        let (mut right_num, mut right_den) = (other.num, other.den);
        loop {
            let (left_whole, left_rem) = div_floor_rem(left_num, left_den);
            let (right_whole, right_rem) = div_floor_rem(right_num, right_den);
            let whole_order = left_whole.cmp(&right_whole);
            if whole_order != Ordering::Equal {
                return whole_order;
            }
            if left_rem == I::ZERO || right_rem == I::ZERO {
                return left_rem.cmp(&right_rem);
            }
            // Equal whole parts leave only the two proper fractions. Reciprocate
            // them and swap sides to preserve the ordering as the denominators shrink.
            (left_num, left_den, right_num, right_den) = (right_den, right_rem, left_den, left_rem);
        }
    }
}

/// Split a fraction with a positive denominator into a whole part and a nonnegative remainder.
fn div_floor_rem<I: Integer>(num: I, den: I) -> (I, I) {
    let (whole, rem) = num.div_rem(den);
    let (whole, rem) = if rem < I::ZERO {
        (whole - I::ONE, rem + den)
    } else {
        (whole, rem)
    };
    // Reciprocal steps in Rational::cmp depend on strictly smaller denominators.
    debug_assert!((I::ZERO..den).contains(&rem), "invalid floor remainder");
    (whole, rem)
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
