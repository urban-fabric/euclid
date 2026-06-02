use crate::{
    Factor, Percentage, scalar_add, scalar_div, scalar_mul, scalar_neg, scalar_sub, scalar_sum,
};

impl ::core::ops::Add for Factor {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        scalar_add(self, rhs)
    }
}
impl ::core::ops::Sub for Factor {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        scalar_sub(self, rhs)
    }
}
impl ::core::ops::AddAssign for Factor {
    fn add_assign(&mut self, rhs: Self) {
        *self = scalar_add(*self, rhs);
    }
}
impl ::core::ops::SubAssign for Factor {
    fn sub_assign(&mut self, rhs: Self) {
        *self = scalar_sub(*self, rhs);
    }
}
impl ::core::ops::Neg for Factor {
    type Output = Self;
    fn neg(self) -> Self {
        scalar_neg(self)
    }
}
impl ::core::ops::Mul<f64> for Factor {
    type Output = Self;
    fn mul(self, k: f64) -> Self {
        scalar_mul(self, k)
    }
}
impl ::core::ops::Mul<Factor> for f64 {
    type Output = Factor;
    fn mul(self, x: Factor) -> Factor {
        scalar_mul(x, self)
    }
}
impl ::core::ops::Div<f64> for Factor {
    type Output = Self;
    fn div(self, k: f64) -> Self {
        scalar_div(self, k)
    }
}
impl ::core::iter::Sum for Factor {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        scalar_sum(iter)
    }
}

impl ::core::ops::Add for Percentage {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        scalar_add(self, rhs)
    }
}
impl ::core::ops::Sub for Percentage {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        scalar_sub(self, rhs)
    }
}
impl ::core::ops::AddAssign for Percentage {
    fn add_assign(&mut self, rhs: Self) {
        *self = scalar_add(*self, rhs);
    }
}
impl ::core::ops::SubAssign for Percentage {
    fn sub_assign(&mut self, rhs: Self) {
        *self = scalar_sub(*self, rhs);
    }
}
impl ::core::ops::Neg for Percentage {
    type Output = Self;
    fn neg(self) -> Self {
        scalar_neg(self)
    }
}
impl ::core::ops::Mul<f64> for Percentage {
    type Output = Self;
    fn mul(self, k: f64) -> Self {
        scalar_mul(self, k)
    }
}
impl ::core::ops::Mul<Percentage> for f64 {
    type Output = Percentage;
    fn mul(self, x: Percentage) -> Percentage {
        scalar_mul(x, self)
    }
}
impl ::core::ops::Div<f64> for Percentage {
    type Output = Self;
    fn div(self, k: f64) -> Self {
        scalar_div(self, k)
    }
}
impl ::core::iter::Sum for Percentage {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        scalar_sum(iter)
    }
}
