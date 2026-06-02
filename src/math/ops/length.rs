use crate::{
    LengthFeet, LengthKm, LengthMeters, scalar_add, scalar_div, scalar_mul, scalar_neg, scalar_sub,
    scalar_sum,
};

impl ::core::ops::Add for LengthMeters {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        scalar_add(self, rhs)
    }
}
impl ::core::ops::Sub for LengthMeters {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        scalar_sub(self, rhs)
    }
}
impl ::core::ops::AddAssign for LengthMeters {
    fn add_assign(&mut self, rhs: Self) {
        *self = scalar_add(*self, rhs);
    }
}
impl ::core::ops::SubAssign for LengthMeters {
    fn sub_assign(&mut self, rhs: Self) {
        *self = scalar_sub(*self, rhs);
    }
}
impl ::core::ops::Neg for LengthMeters {
    type Output = Self;
    fn neg(self) -> Self {
        scalar_neg(self)
    }
}
impl ::core::ops::Mul<f64> for LengthMeters {
    type Output = Self;
    fn mul(self, k: f64) -> Self {
        scalar_mul(self, k)
    }
}
impl ::core::ops::Mul<LengthMeters> for f64 {
    type Output = LengthMeters;
    fn mul(self, x: LengthMeters) -> LengthMeters {
        scalar_mul(x, self)
    }
}
impl ::core::ops::Div<f64> for LengthMeters {
    type Output = Self;
    fn div(self, k: f64) -> Self {
        scalar_div(self, k)
    }
}
impl ::core::iter::Sum for LengthMeters {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        scalar_sum(iter)
    }
}

impl ::core::ops::Add for LengthKm {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        scalar_add(self, rhs)
    }
}
impl ::core::ops::Sub for LengthKm {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        scalar_sub(self, rhs)
    }
}
impl ::core::ops::AddAssign for LengthKm {
    fn add_assign(&mut self, rhs: Self) {
        *self = scalar_add(*self, rhs);
    }
}
impl ::core::ops::SubAssign for LengthKm {
    fn sub_assign(&mut self, rhs: Self) {
        *self = scalar_sub(*self, rhs);
    }
}
impl ::core::ops::Neg for LengthKm {
    type Output = Self;
    fn neg(self) -> Self {
        scalar_neg(self)
    }
}
impl ::core::ops::Mul<f64> for LengthKm {
    type Output = Self;
    fn mul(self, k: f64) -> Self {
        scalar_mul(self, k)
    }
}
impl ::core::ops::Mul<LengthKm> for f64 {
    type Output = LengthKm;
    fn mul(self, x: LengthKm) -> LengthKm {
        scalar_mul(x, self)
    }
}
impl ::core::ops::Div<f64> for LengthKm {
    type Output = Self;
    fn div(self, k: f64) -> Self {
        scalar_div(self, k)
    }
}
impl ::core::iter::Sum for LengthKm {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        scalar_sum(iter)
    }
}

impl ::core::ops::Add for LengthFeet {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        scalar_add(self, rhs)
    }
}
impl ::core::ops::Sub for LengthFeet {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        scalar_sub(self, rhs)
    }
}
impl ::core::ops::AddAssign for LengthFeet {
    fn add_assign(&mut self, rhs: Self) {
        *self = scalar_add(*self, rhs);
    }
}
impl ::core::ops::SubAssign for LengthFeet {
    fn sub_assign(&mut self, rhs: Self) {
        *self = scalar_sub(*self, rhs);
    }
}
impl ::core::ops::Neg for LengthFeet {
    type Output = Self;
    fn neg(self) -> Self {
        scalar_neg(self)
    }
}
impl ::core::ops::Mul<f64> for LengthFeet {
    type Output = Self;
    fn mul(self, k: f64) -> Self {
        scalar_mul(self, k)
    }
}
impl ::core::ops::Mul<LengthFeet> for f64 {
    type Output = LengthFeet;
    fn mul(self, x: LengthFeet) -> LengthFeet {
        scalar_mul(x, self)
    }
}
impl ::core::ops::Div<f64> for LengthFeet {
    type Output = Self;
    fn div(self, k: f64) -> Self {
        scalar_div(self, k)
    }
}
impl ::core::iter::Sum for LengthFeet {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        scalar_sum(iter)
    }
}
