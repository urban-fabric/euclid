//! The operator and comparison surface of the faulty carrier.
//!
//! A child module so the file stays inside the crate's 400-line budget. These impls are
//! *faithful* — they forward to `f64` without perturbing anything — which is the point: a law
//! that fails has to fail because of the one perturbed method, not because arithmetic on the
//! wrapper is broken. The perturbation itself lives in [`super::perturb`].

use super::*;

impl<const WHICH: u32> Add for Faulty<WHICH> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}
impl<const WHICH: u32> Sub for Faulty<WHICH> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}
impl<const WHICH: u32> Mul for Faulty<WHICH> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self(self.0 * rhs.0)
    }
}
impl<const WHICH: u32> Div for Faulty<WHICH> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        Self(self.0 / rhs.0)
    }
}
impl<const WHICH: u32> Neg for Faulty<WHICH> {
    type Output = Self;
    fn neg(self) -> Self {
        Self(-self.0)
    }
}
impl<const WHICH: u32> AddAssign for Faulty<WHICH> {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}
impl<const WHICH: u32> SubAssign for Faulty<WHICH> {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}
impl<const WHICH: u32> MulAssign for Faulty<WHICH> {
    fn mul_assign(&mut self, rhs: Self) {
        self.0 *= rhs.0;
    }
}

// `f64` reaches `Arithmetic` through an explicit empty impl, and so does this.
impl<const WHICH: u32> crate::traits::Arithmetic for Faulty<WHICH> {}

impl<const WHICH: u32> Compare for Faulty<WHICH> {
    type Mask = bool;
    fn lt(&self, other: &Self) -> Self::Mask {
        perturb_mask::<WHICH>(1, <f64 as Compare>::lt(&self.0, &other.0))
    }
    fn le(&self, other: &Self) -> Self::Mask {
        perturb_mask::<WHICH>(2, <f64 as Compare>::le(&self.0, &other.0))
    }
    fn gt(&self, other: &Self) -> Self::Mask {
        perturb_mask::<WHICH>(3, <f64 as Compare>::gt(&self.0, &other.0))
    }
    fn ge(&self, other: &Self) -> Self::Mask {
        perturb_mask::<WHICH>(4, <f64 as Compare>::ge(&self.0, &other.0))
    }
    fn eq(&self, other: &Self) -> Self::Mask {
        perturb_mask::<WHICH>(5, <f64 as Compare>::eq(&self.0, &other.0))
    }
    fn ne(&self, other: &Self) -> Self::Mask {
        perturb_mask::<WHICH>(6, <f64 as Compare>::ne(&self.0, &other.0))
    }
    fn select(mask: &Self::Mask, on_true: Self, on_false: Self) -> Self {
        Self(perturb::<WHICH>(
            7,
            <f64 as Compare>::select(mask, on_true.0, on_false.0),
        ))
    }
    fn while_loop<C, P, B>(init: C, cond: P, body: B) -> C
    where
        P: Fn(&C) -> Self::Mask,
        B: Fn(C) -> C,
    {
        // Generic over the carried state, so there is nothing to unwrap: the host form is a
        // plain `while`, which is what `f64` does too.
        let mut state = init;
        while cond(&state) {
            state = body(state);
        }
        state
    }
}

impl<const WHICH: u32> NormedDivisionAlgebra for Faulty<WHICH> {
    type RealPart = Self;
    fn conjugate(&self) -> Self {
        Self(perturb::<WHICH>(
            8,
            <f64 as NormedDivisionAlgebra>::conjugate(&self.0),
        ))
    }
    fn norm_sqr(&self) -> Self::RealPart {
        Self(perturb::<WHICH>(
            9,
            <f64 as NormedDivisionAlgebra>::norm_sqr(&self.0),
        ))
    }
    fn real_part(&self) -> Self::RealPart {
        Self(perturb::<WHICH>(
            10,
            <f64 as NormedDivisionAlgebra>::real_part(&self.0),
        ))
    }
    fn from_real(re: Self::RealPart) -> Self {
        Self(perturb::<WHICH>(
            11,
            <f64 as NormedDivisionAlgebra>::from_real(re.0),
        ))
    }
    fn magnitude(&self) -> Self::RealPart {
        Self(perturb::<WHICH>(
            12,
            <f64 as NormedDivisionAlgebra>::magnitude(&self.0),
        ))
    }
}

impl<const WHICH: u32> Element for Faulty<WHICH> {
    fn zero() -> Self {
        Self(perturb::<WHICH>(13, <f64 as Element>::zero()))
    }
    fn one() -> Self {
        Self(perturb::<WHICH>(14, <f64 as Element>::one()))
    }
}
