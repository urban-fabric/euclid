//! Concrete implementations of the algebra tower for the primitive carriers
//! (`f64`/`f32` fields, `i64`/`i128` rings).

use super::*;

// ===== f64 / f32: numeric carriers climbing Semiring -> Ring -> Field ======
//
// The magma `add`/`mul` ARE `+`/`*` (the bridge between trunk and numeric branch).
// Reals are associative/commutative/distributive but not additively idempotent,
// so they are Semirings/Rings/Fields -- never dioids.

// Ring / CommutativeRing / OrderedField are blanket-derived for both; only the
// magma bridge, identities, law markers, and the two method-bearing traits
// (EuclideanDomain::div_rem, Field::conjugate) are written per type.

impl AdditiveMagma for f64 {
    #[inline]
    fn add(self, rhs: Self) -> Self {
        self + rhs
    }
}
impl MultiplicativeMagma for f64 {
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        self * rhs
    }
}
impl RefAdditiveMagma for f64 {
    #[inline]
    fn add_ref(&self, rhs: &Self) -> Self {
        *self + *rhs
    }
}
impl RefMultiplicativeMagma for f64 {
    #[inline]
    fn mul_ref(&self, rhs: &Self) -> Self {
        *self * *rhs
    }
}
impl HasZero for f64 {
    const ZERO: Self = 0.0;
}
impl HasOne for f64 {
    const ONE: Self = 1.0;
}
impl AssociativeAddition for f64 {}
impl CommutativeAddition for f64 {}
impl AssociativeMultiplication for f64 {}
impl CommutativeMultiplication for f64 {}
impl Distributive for f64 {}
impl ZeroAnnihilatesMultiplication for f64 {}
impl IntegralDomain for f64 {}
impl Ufd for f64 {}
impl Pid for f64 {}
impl EuclideanDomain for f64 {
    #[inline]
    fn div_rem(self, rhs: Self) -> (Self, Self) {
        (self / rhs, 0.0)
    }
}
impl Field for f64 {
    #[inline]
    fn conjugate(self) -> Self {
        self
    }
}

impl AdditiveMagma for f32 {
    #[inline]
    fn add(self, rhs: Self) -> Self {
        self + rhs
    }
}
impl MultiplicativeMagma for f32 {
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        self * rhs
    }
}
impl RefAdditiveMagma for f32 {
    #[inline]
    fn add_ref(&self, rhs: &Self) -> Self {
        *self + *rhs
    }
}
impl RefMultiplicativeMagma for f32 {
    #[inline]
    fn mul_ref(&self, rhs: &Self) -> Self {
        *self * *rhs
    }
}
impl HasZero for f32 {
    const ZERO: Self = 0.0;
}
impl HasOne for f32 {
    const ONE: Self = 1.0;
}
impl AssociativeAddition for f32 {}
impl CommutativeAddition for f32 {}
impl AssociativeMultiplication for f32 {}
impl CommutativeMultiplication for f32 {}
impl Distributive for f32 {}
impl ZeroAnnihilatesMultiplication for f32 {}
impl IntegralDomain for f32 {}
impl Ufd for f32 {}
impl Pid for f32 {}
impl EuclideanDomain for f32 {
    #[inline]
    fn div_rem(self, rhs: Self) -> (Self, Self) {
        (self / rhs, 0.0)
    }
}
impl Field for f32 {
    #[inline]
    fn conjugate(self) -> Self {
        self
    }
}

// ===== i64 / i128: the canonical non-field rings (ℤ) =======================
//
// Integers climb Semiring -> Ring -> CommutativeRing -> ... -> EuclideanDomain
// but stop short of Field: there is no multiplicative inverse. `div_rem` is the
// usual integer division and modulus (the exact-division primitive the
// fraction-free algorithms build on). The magma `add`/`mul` ARE `+`/`*` (the
// trunk<->numeric-branch bridge). They are ordered, but euclid's OrderedField
// marker is field-only, so order lives on the consumer (`Ord`) rather than here.

impl AdditiveMagma for i64 {
    #[inline]
    fn add(self, rhs: Self) -> Self {
        self + rhs
    }
}
impl MultiplicativeMagma for i64 {
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        self * rhs
    }
}
impl RefAdditiveMagma for i64 {
    #[inline]
    fn add_ref(&self, rhs: &Self) -> Self {
        *self + *rhs
    }
}
impl RefMultiplicativeMagma for i64 {
    #[inline]
    fn mul_ref(&self, rhs: &Self) -> Self {
        *self * *rhs
    }
}
impl HasZero for i64 {
    const ZERO: Self = 0;
}
impl HasOne for i64 {
    const ONE: Self = 1;
}
impl AssociativeAddition for i64 {}
impl CommutativeAddition for i64 {}
impl AssociativeMultiplication for i64 {}
impl CommutativeMultiplication for i64 {}
impl Distributive for i64 {}
impl ZeroAnnihilatesMultiplication for i64 {}
impl IntegralDomain for i64 {}
impl Ufd for i64 {}
impl Pid for i64 {}
impl EuclideanDomain for i64 {
    #[inline]
    fn div_rem(self, rhs: Self) -> (Self, Self) {
        (self / rhs, self % rhs)
    }
}

impl AdditiveMagma for i128 {
    #[inline]
    fn add(self, rhs: Self) -> Self {
        self + rhs
    }
}
impl MultiplicativeMagma for i128 {
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        self * rhs
    }
}
impl RefAdditiveMagma for i128 {
    #[inline]
    fn add_ref(&self, rhs: &Self) -> Self {
        *self + *rhs
    }
}
impl RefMultiplicativeMagma for i128 {
    #[inline]
    fn mul_ref(&self, rhs: &Self) -> Self {
        *self * *rhs
    }
}
impl HasZero for i128 {
    const ZERO: Self = 0;
}
impl HasOne for i128 {
    const ONE: Self = 1;
}
impl AssociativeAddition for i128 {}
impl CommutativeAddition for i128 {}
impl AssociativeMultiplication for i128 {}
impl CommutativeMultiplication for i128 {}
impl Distributive for i128 {}
impl ZeroAnnihilatesMultiplication for i128 {}
impl IntegralDomain for i128 {}
impl Ufd for i128 {}
impl Pid for i128 {}
impl EuclideanDomain for i128 {
    #[inline]
    fn div_rem(self, rhs: Self) -> (Self, Self) {
        (self / rhs, self % rhs)
    }
}
