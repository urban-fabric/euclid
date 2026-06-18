//! Gaussian rationals **ℚ(i)** — exact complex numbers with rational parts.

use std::fmt;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::algebra::{
    AdditiveMagma, AssociativeAddition, AssociativeMultiplication, CommutativeAddition,
    CommutativeMultiplication, Distributive, EuclideanDomain, Field, HasOne, HasZero,
    IntegralDomain, MultiplicativeMagma, Pid, Rational128, RefAdditiveMagma, RefMultiplicativeMagma,
    Ufd, ZeroAnnihilatesMultiplication,
};

/// A Gaussian rational `a + b·i` with `a, b ∈ ℚ` — the field ℚ(i). Exact complex
/// arithmetic, no rounding. Unlike the real-like fields its
/// [`conjugate`](Field::conjugate) is genuinely `a − b·i`, so a Gaussian-rational
/// matrix exercises the Hermitian (not merely symmetric) paths exactly. It is a
/// [`Field`] but neither ordered nor algebraically closed (e.g. `x² − 2` has no root in
/// ℚ(i)), and not an analytic matrix scalar (no `f64` magnitude / `sqrt`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GaussianRational {
    re: Rational128,
    im: Rational128,
}

impl GaussianRational {
    /// `re + im·i`.
    pub fn new(re: Rational128, im: Rational128) -> Self {
        Self { re, im }
    }

    /// The integer `n` as `n + 0·i`.
    pub fn integer(n: i128) -> Self {
        Self {
            re: Rational128::integer(n),
            im: Rational128::ZERO,
        }
    }

    /// The real part.
    pub fn re(self) -> Rational128 {
        self.re
    }

    /// The imaginary part.
    pub fn im(self) -> Rational128 {
        self.im
    }

    /// The squared modulus `a² + b²` — a (nonnegative) rational.
    pub fn norm(self) -> Rational128 {
        self.re * self.re + self.im * self.im
    }
}

// The fused algebra trunk: ℚ(i) is a numeric carrier, so its magma `add`/`mul` ARE
// `+`/`*`, and `Semiring`/`Ring`/`CommutativeRing` are blanket-derived from the
// magma layer, identities, and law markers — never written directly. ℚ(i) is a
// field but neither ordered nor algebraically closed.
impl AdditiveMagma for GaussianRational {
    #[inline]
    fn add(self, rhs: Self) -> Self {
        self + rhs
    }
}
impl MultiplicativeMagma for GaussianRational {
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        self * rhs
    }
}
impl RefAdditiveMagma for GaussianRational {
    #[inline]
    fn add_ref(&self, rhs: &Self) -> Self {
        *self + *rhs
    }
}
impl RefMultiplicativeMagma for GaussianRational {
    #[inline]
    fn mul_ref(&self, rhs: &Self) -> Self {
        *self * *rhs
    }
}
impl HasZero for GaussianRational {
    const ZERO: Self = Self {
        re: Rational128::ZERO,
        im: Rational128::ZERO,
    };
}
impl HasOne for GaussianRational {
    const ONE: Self = Self {
        re: Rational128::ONE,
        im: Rational128::ZERO,
    };
}
impl AssociativeAddition for GaussianRational {}
impl CommutativeAddition for GaussianRational {}
impl AssociativeMultiplication for GaussianRational {}
impl CommutativeMultiplication for GaussianRational {}
impl Distributive for GaussianRational {}
impl ZeroAnnihilatesMultiplication for GaussianRational {}
impl IntegralDomain for GaussianRational {}
impl Ufd for GaussianRational {}
impl Pid for GaussianRational {}
impl EuclideanDomain for GaussianRational {
    fn div_rem(self, rhs: Self) -> (Self, Self) {
        (self / rhs, Self::ZERO)
    }
}
impl Field for GaussianRational {
    /// `a + b·i ↦ a − b·i` — a genuine conjugation (not the identity).
    fn conjugate(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }
}

impl Add for GaussianRational {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

impl Sub for GaussianRational {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
}

impl Mul for GaussianRational {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

impl Div for GaussianRational {
    type Output = Self;
    // (a+bi)/(c+di) = (a+bi)(c−di) / (c²+d²) — exact.
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Self) -> Self {
        let denom = rhs.norm();
        let num = self * rhs.conjugate();
        Self {
            re: num.re / denom,
            im: num.im / denom,
        }
    }
}

impl Neg for GaussianRational {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            re: -self.re,
            im: -self.im,
        }
    }
}

impl AddAssign for GaussianRational {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl SubAssign for GaussianRational {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl MulAssign for GaussianRational {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl DivAssign for GaussianRational {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl fmt::Display for GaussianRational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} + {}i", self.re, self.im)
    }
}
