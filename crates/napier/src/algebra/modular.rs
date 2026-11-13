//! Modular arithmetic — the finite field [`Zp<P>`] (prime modulus) and the ring
//! [`Zn<N>`] (any modulus).
//!
//! Const-generic residue types: the modulus is baked into the type, so `Zp<7>` and
//! `Zp<2>` are distinct types with no per-value overhead. As a [`Field`], `Zp<P>`
//! powers the exact linear algebra in [`crate::exact`] over a finite field — the
//! setting of coding theory, cryptography, and combinatorics.
//!
//! [`Zn<N>`] is the **ring** ℤ/Nℤ for *any* `N` — only a [`CommutativeRing`] (no
//! division), since for composite `N` it has zero divisors. It is exactly where the
//! Howell normal form lives (the ℤ/Nℤ analog of Hermite form, since ℤ/Nℤ is not a PID).
//!
//! `P`/`N` must be `< 2⁶³` so the additive operations stay within `u64` (multiplication
//! widens through `u128`); `Zp` additionally requires `P` **prime** for inverses.

use std::fmt;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::algebra::{
    AdditiveMagma, AssociativeAddition, AssociativeMultiplication, CommutativeAddition,
    CommutativeMultiplication, Distributive, EuclideanDomain, Field, HasOne, HasZero,
    IntegralDomain, MultiplicativeMagma, Pid, RefAdditiveMagma, RefMultiplicativeMagma, Ufd,
    ZeroAnnihilatesMultiplication,
};

/// An element of GF(`P`) — a residue in `[0, P)`. `P` must be prime (see the module
/// docs); inverses use Fermat's little theorem.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Zp<const P: u64>(u64);

impl<const P: u64> Zp<P> {
    /// The residue of `x` modulo `P`.
    pub fn new(x: u64) -> Self {
        Self(x % P)
    }

    /// The residue of a (possibly negative) integer modulo `P`.
    pub fn from_i64(x: i64) -> Self {
        Self(x.rem_euclid(P as i64) as u64)
    }

    /// The underlying residue in `[0, P)`.
    pub fn value(self) -> u64 {
        self.0
    }

    /// The multiplicative inverse `a⁻¹` by Fermat's little theorem (`a^(P−2) mod P`).
    ///
    /// # Panics (debug)
    /// If `self` is zero (zero has no inverse).
    pub fn inverse(self) -> Self {
        debug_assert!(self.0 != 0, "Zp: inverse of zero");
        self.pow(P - 2)
    }

    /// `self^e mod P` by square-and-multiply.
    fn pow(self, mut e: u64) -> Self {
        let mut base = self;
        let mut acc = Self(1 % P);
        while e > 0 {
            if e & 1 == 1 {
                acc *= base;
            }
            base *= base;
            e >>= 1;
        }
        acc
    }
}

// The fused algebra trunk: GF(P) is a numeric carrier, so its magma `add`/`mul`
// ARE `+`/`*`, and `Semiring`/`Ring`/`CommutativeRing` are blanket-derived from the
// magma layer, identities, and law markers — never written directly. GF(P) is a
// field but, being finite, neither ordered nor algebraically closed.
impl<const P: u64> AdditiveMagma for Zp<P> {
    #[inline]
    fn add(self, rhs: Self) -> Self {
        self + rhs
    }
}
impl<const P: u64> MultiplicativeMagma for Zp<P> {
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        self * rhs
    }
}
impl<const P: u64> RefAdditiveMagma for Zp<P> {
    #[inline]
    fn add_ref(&self, rhs: &Self) -> Self {
        *self + *rhs
    }
}
impl<const P: u64> RefMultiplicativeMagma for Zp<P> {
    #[inline]
    fn mul_ref(&self, rhs: &Self) -> Self {
        *self * *rhs
    }
}
impl<const P: u64> HasZero for Zp<P> {
    const ZERO: Self = Zp(0);
}
impl<const P: u64> HasOne for Zp<P> {
    const ONE: Self = Zp(1 % P);
}
impl<const P: u64> AssociativeAddition for Zp<P> {}
impl<const P: u64> CommutativeAddition for Zp<P> {}
impl<const P: u64> AssociativeMultiplication for Zp<P> {}
impl<const P: u64> CommutativeMultiplication for Zp<P> {}
impl<const P: u64> Distributive for Zp<P> {}
impl<const P: u64> ZeroAnnihilatesMultiplication for Zp<P> {}
impl<const P: u64> IntegralDomain for Zp<P> {}
impl<const P: u64> Ufd for Zp<P> {}
impl<const P: u64> Pid for Zp<P> {}
impl<const P: u64> EuclideanDomain for Zp<P> {
    fn div_rem(self, rhs: Self) -> (Self, Self) {
        (self / rhs, Zp(0))
    }
}
impl<const P: u64> Field for Zp<P> {
    /// Conjugation is the identity — a finite field is its own "real" field.
    fn conjugate(self) -> Self {
        self
    }
}

impl<const P: u64> Add for Zp<P> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Zp((self.0 + rhs.0) % P)
    }
}

impl<const P: u64> Sub for Zp<P> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Zp((self.0 + P - rhs.0) % P)
    }
}

impl<const P: u64> Mul for Zp<P> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Zp(((self.0 as u128 * rhs.0 as u128) % P as u128) as u64)
    }
}

impl<const P: u64> Div for Zp<P> {
    type Output = Self;
    // Field division is multiplication by the inverse — `*` here is intentional.
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Self) -> Self {
        self * rhs.inverse()
    }
}

impl<const P: u64> Neg for Zp<P> {
    type Output = Self;
    fn neg(self) -> Self {
        Zp((P - self.0) % P)
    }
}

impl<const P: u64> AddAssign for Zp<P> {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<const P: u64> SubAssign for Zp<P> {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<const P: u64> MulAssign for Zp<P> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<const P: u64> DivAssign for Zp<P> {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl<const P: u64> fmt::Display for Zp<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} (mod {})", self.0, P)
    }
}

/// An element of the ring **ℤ/Nℤ** for any modulus `N` — a residue in `[0, N)`. A
/// [`CommutativeRing`] but **not** a field (composite `N` has zero divisors and
/// non-invertible elements), so it has no division. Use [`Zp`] when `N` is prime and
/// you need a field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Zn<const N: u64>(u64);

impl<const N: u64> Zn<N> {
    /// The residue of `x` modulo `N`.
    pub fn new(x: u64) -> Self {
        Self(x % N)
    }

    /// The residue of a (possibly negative) integer modulo `N`.
    pub fn from_i64(x: i64) -> Self {
        Self(x.rem_euclid(N as i64) as u64)
    }

    /// The underlying residue in `[0, N)`.
    pub fn value(self) -> u64 {
        self.0
    }
}

// The fused algebra trunk: ℤ/Nℤ is a numeric carrier (its magma `add`/`mul` ARE
// `+`/`*`), so `Semiring`/`Ring`/`CommutativeRing` are blanket-derived from the
// magma layer, identities, and law markers — never written directly. For composite
// `N` it has zero divisors, so it is NOT an integral domain / field: it stops at
// `CommutativeRing` (no `IntegralDomain` marker, no division).
impl<const N: u64> AdditiveMagma for Zn<N> {
    #[inline]
    fn add(self, rhs: Self) -> Self {
        self + rhs
    }
}
impl<const N: u64> MultiplicativeMagma for Zn<N> {
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        self * rhs
    }
}
impl<const N: u64> RefAdditiveMagma for Zn<N> {
    #[inline]
    fn add_ref(&self, rhs: &Self) -> Self {
        *self + *rhs
    }
}
impl<const N: u64> RefMultiplicativeMagma for Zn<N> {
    #[inline]
    fn mul_ref(&self, rhs: &Self) -> Self {
        *self * *rhs
    }
}
impl<const N: u64> HasZero for Zn<N> {
    const ZERO: Self = Zn(0);
}
impl<const N: u64> HasOne for Zn<N> {
    const ONE: Self = Zn(1 % N);
}
impl<const N: u64> AssociativeAddition for Zn<N> {}
impl<const N: u64> CommutativeAddition for Zn<N> {}
impl<const N: u64> AssociativeMultiplication for Zn<N> {}
impl<const N: u64> CommutativeMultiplication for Zn<N> {}
impl<const N: u64> Distributive for Zn<N> {}
impl<const N: u64> ZeroAnnihilatesMultiplication for Zn<N> {}

impl<const N: u64> Add for Zn<N> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Zn((self.0 + rhs.0) % N)
    }
}

impl<const N: u64> Sub for Zn<N> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Zn((self.0 + N - rhs.0) % N)
    }
}

impl<const N: u64> Mul for Zn<N> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Zn(((self.0 as u128 * rhs.0 as u128) % N as u128) as u64)
    }
}

impl<const N: u64> Neg for Zn<N> {
    type Output = Self;
    fn neg(self) -> Self {
        Zn((N - self.0) % N)
    }
}

impl<const N: u64> AddAssign for Zn<N> {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<const N: u64> SubAssign for Zn<N> {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<const N: u64> MulAssign for Zn<N> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<const N: u64> fmt::Display for Zn<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} (mod {})", self.0, N)
    }
}
