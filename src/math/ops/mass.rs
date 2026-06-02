use crate::{
    MassGrams, MassKilograms, MassMilligrams, scalar_add, scalar_div, scalar_mul, scalar_neg,
    scalar_sub, scalar_sum,
};

impl ::core::ops::Add for MassKilograms {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        scalar_add(self, rhs)
    }
}
impl ::core::ops::Sub for MassKilograms {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        scalar_sub(self, rhs)
    }
}
impl ::core::ops::AddAssign for MassKilograms {
    fn add_assign(&mut self, rhs: Self) {
        *self = scalar_add(*self, rhs);
    }
}
impl ::core::ops::SubAssign for MassKilograms {
    fn sub_assign(&mut self, rhs: Self) {
        *self = scalar_sub(*self, rhs);
    }
}
impl ::core::ops::Neg for MassKilograms {
    type Output = Self;
    fn neg(self) -> Self {
        scalar_neg(self)
    }
}
impl ::core::ops::Mul<f64> for MassKilograms {
    type Output = Self;
    fn mul(self, k: f64) -> Self {
        scalar_mul(self, k)
    }
}
impl ::core::ops::Mul<MassKilograms> for f64 {
    type Output = MassKilograms;
    fn mul(self, x: MassKilograms) -> MassKilograms {
        scalar_mul(x, self)
    }
}
impl ::core::ops::Div<f64> for MassKilograms {
    type Output = Self;
    fn div(self, k: f64) -> Self {
        scalar_div(self, k)
    }
}
impl ::core::iter::Sum for MassKilograms {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        scalar_sum(iter)
    }
}

impl ::core::ops::Add for MassGrams {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        scalar_add(self, rhs)
    }
}
impl ::core::ops::Sub for MassGrams {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        scalar_sub(self, rhs)
    }
}
impl ::core::ops::AddAssign for MassGrams {
    fn add_assign(&mut self, rhs: Self) {
        *self = scalar_add(*self, rhs);
    }
}
impl ::core::ops::SubAssign for MassGrams {
    fn sub_assign(&mut self, rhs: Self) {
        *self = scalar_sub(*self, rhs);
    }
}
impl ::core::ops::Neg for MassGrams {
    type Output = Self;
    fn neg(self) -> Self {
        scalar_neg(self)
    }
}
impl ::core::ops::Mul<f64> for MassGrams {
    type Output = Self;
    fn mul(self, k: f64) -> Self {
        scalar_mul(self, k)
    }
}
impl ::core::ops::Mul<MassGrams> for f64 {
    type Output = MassGrams;
    fn mul(self, x: MassGrams) -> MassGrams {
        scalar_mul(x, self)
    }
}
impl ::core::ops::Div<f64> for MassGrams {
    type Output = Self;
    fn div(self, k: f64) -> Self {
        scalar_div(self, k)
    }
}
impl ::core::iter::Sum for MassGrams {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        scalar_sum(iter)
    }
}

impl ::core::ops::Add for MassMilligrams {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        scalar_add(self, rhs)
    }
}
impl ::core::ops::Sub for MassMilligrams {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        scalar_sub(self, rhs)
    }
}
impl ::core::ops::AddAssign for MassMilligrams {
    fn add_assign(&mut self, rhs: Self) {
        *self = scalar_add(*self, rhs);
    }
}
impl ::core::ops::SubAssign for MassMilligrams {
    fn sub_assign(&mut self, rhs: Self) {
        *self = scalar_sub(*self, rhs);
    }
}
impl ::core::ops::Neg for MassMilligrams {
    type Output = Self;
    fn neg(self) -> Self {
        scalar_neg(self)
    }
}
impl ::core::ops::Mul<f64> for MassMilligrams {
    type Output = Self;
    fn mul(self, k: f64) -> Self {
        scalar_mul(self, k)
    }
}
impl ::core::ops::Mul<MassMilligrams> for f64 {
    type Output = MassMilligrams;
    fn mul(self, x: MassMilligrams) -> MassMilligrams {
        scalar_mul(x, self)
    }
}
impl ::core::ops::Div<f64> for MassMilligrams {
    type Output = Self;
    fn div(self, k: f64) -> Self {
        scalar_div(self, k)
    }
}
impl ::core::iter::Sum for MassMilligrams {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        scalar_sum(iter)
    }
}
