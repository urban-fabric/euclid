use crate::{
    SpeedKph, SpeedMph, SpeedMs, scalar_add, scalar_div, scalar_mul, scalar_neg, scalar_sub,
    scalar_sum,
};

impl ::core::ops::Add for SpeedMs {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        scalar_add(self, rhs)
    }
}
impl ::core::ops::Sub for SpeedMs {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        scalar_sub(self, rhs)
    }
}
impl ::core::ops::AddAssign for SpeedMs {
    fn add_assign(&mut self, rhs: Self) {
        *self = scalar_add(*self, rhs);
    }
}
impl ::core::ops::SubAssign for SpeedMs {
    fn sub_assign(&mut self, rhs: Self) {
        *self = scalar_sub(*self, rhs);
    }
}
impl ::core::ops::Neg for SpeedMs {
    type Output = Self;
    fn neg(self) -> Self {
        scalar_neg(self)
    }
}
impl ::core::ops::Mul<f64> for SpeedMs {
    type Output = Self;
    fn mul(self, k: f64) -> Self {
        scalar_mul(self, k)
    }
}
impl ::core::ops::Mul<SpeedMs> for f64 {
    type Output = SpeedMs;
    fn mul(self, x: SpeedMs) -> SpeedMs {
        scalar_mul(x, self)
    }
}
impl ::core::ops::Div<f64> for SpeedMs {
    type Output = Self;
    fn div(self, k: f64) -> Self {
        scalar_div(self, k)
    }
}
impl ::core::iter::Sum for SpeedMs {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        scalar_sum(iter)
    }
}

impl ::core::ops::Add for SpeedKph {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        scalar_add(self, rhs)
    }
}
impl ::core::ops::Sub for SpeedKph {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        scalar_sub(self, rhs)
    }
}
impl ::core::ops::AddAssign for SpeedKph {
    fn add_assign(&mut self, rhs: Self) {
        *self = scalar_add(*self, rhs);
    }
}
impl ::core::ops::SubAssign for SpeedKph {
    fn sub_assign(&mut self, rhs: Self) {
        *self = scalar_sub(*self, rhs);
    }
}
impl ::core::ops::Neg for SpeedKph {
    type Output = Self;
    fn neg(self) -> Self {
        scalar_neg(self)
    }
}
impl ::core::ops::Mul<f64> for SpeedKph {
    type Output = Self;
    fn mul(self, k: f64) -> Self {
        scalar_mul(self, k)
    }
}
impl ::core::ops::Mul<SpeedKph> for f64 {
    type Output = SpeedKph;
    fn mul(self, x: SpeedKph) -> SpeedKph {
        scalar_mul(x, self)
    }
}
impl ::core::ops::Div<f64> for SpeedKph {
    type Output = Self;
    fn div(self, k: f64) -> Self {
        scalar_div(self, k)
    }
}
impl ::core::iter::Sum for SpeedKph {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        scalar_sum(iter)
    }
}

impl ::core::ops::Add for SpeedMph {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        scalar_add(self, rhs)
    }
}
impl ::core::ops::Sub for SpeedMph {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        scalar_sub(self, rhs)
    }
}
impl ::core::ops::AddAssign for SpeedMph {
    fn add_assign(&mut self, rhs: Self) {
        *self = scalar_add(*self, rhs);
    }
}
impl ::core::ops::SubAssign for SpeedMph {
    fn sub_assign(&mut self, rhs: Self) {
        *self = scalar_sub(*self, rhs);
    }
}
impl ::core::ops::Neg for SpeedMph {
    type Output = Self;
    fn neg(self) -> Self {
        scalar_neg(self)
    }
}
impl ::core::ops::Mul<f64> for SpeedMph {
    type Output = Self;
    fn mul(self, k: f64) -> Self {
        scalar_mul(self, k)
    }
}
impl ::core::ops::Mul<SpeedMph> for f64 {
    type Output = SpeedMph;
    fn mul(self, x: SpeedMph) -> SpeedMph {
        scalar_mul(x, self)
    }
}
impl ::core::ops::Div<f64> for SpeedMph {
    type Output = Self;
    fn div(self, k: f64) -> Self {
        scalar_div(self, k)
    }
}
impl ::core::iter::Sum for SpeedMph {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        scalar_sum(iter)
    }
}
