use crate::{
    AngleDegrees, AngleRadians, scalar_add, scalar_div, scalar_mul, scalar_neg, scalar_sub,
    scalar_sum,
};

impl ::core::ops::Add for AngleRadians {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        scalar_add(self, rhs)
    }
}
impl ::core::ops::Sub for AngleRadians {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        scalar_sub(self, rhs)
    }
}
impl ::core::ops::AddAssign for AngleRadians {
    fn add_assign(&mut self, rhs: Self) {
        *self = scalar_add(*self, rhs);
    }
}
impl ::core::ops::SubAssign for AngleRadians {
    fn sub_assign(&mut self, rhs: Self) {
        *self = scalar_sub(*self, rhs);
    }
}
impl ::core::ops::Neg for AngleRadians {
    type Output = Self;
    fn neg(self) -> Self {
        scalar_neg(self)
    }
}
impl ::core::ops::Mul<f64> for AngleRadians {
    type Output = Self;
    fn mul(self, k: f64) -> Self {
        scalar_mul(self, k)
    }
}
impl ::core::ops::Mul<AngleRadians> for f64 {
    type Output = AngleRadians;
    fn mul(self, x: AngleRadians) -> AngleRadians {
        scalar_mul(x, self)
    }
}
impl ::core::ops::Div<f64> for AngleRadians {
    type Output = Self;
    fn div(self, k: f64) -> Self {
        scalar_div(self, k)
    }
}
impl ::core::iter::Sum for AngleRadians {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        scalar_sum(iter)
    }
}

impl ::core::ops::Add for AngleDegrees {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        scalar_add(self, rhs)
    }
}
impl ::core::ops::Sub for AngleDegrees {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        scalar_sub(self, rhs)
    }
}
impl ::core::ops::AddAssign for AngleDegrees {
    fn add_assign(&mut self, rhs: Self) {
        *self = scalar_add(*self, rhs);
    }
}
impl ::core::ops::SubAssign for AngleDegrees {
    fn sub_assign(&mut self, rhs: Self) {
        *self = scalar_sub(*self, rhs);
    }
}
impl ::core::ops::Neg for AngleDegrees {
    type Output = Self;
    fn neg(self) -> Self {
        scalar_neg(self)
    }
}
impl ::core::ops::Mul<f64> for AngleDegrees {
    type Output = Self;
    fn mul(self, k: f64) -> Self {
        scalar_mul(self, k)
    }
}
impl ::core::ops::Mul<AngleDegrees> for f64 {
    type Output = AngleDegrees;
    fn mul(self, x: AngleDegrees) -> AngleDegrees {
        scalar_mul(x, self)
    }
}
impl ::core::ops::Div<f64> for AngleDegrees {
    type Output = Self;
    fn div(self, k: f64) -> Self {
        scalar_div(self, k)
    }
}
impl ::core::iter::Sum for AngleDegrees {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        scalar_sum(iter)
    }
}
