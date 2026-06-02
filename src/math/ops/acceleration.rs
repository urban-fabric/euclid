use crate::{
    AccelerationCentimetersPerSecondSquared, AccelerationMetersPerSecondSquared, scalar_add,
    scalar_div, scalar_mul, scalar_neg, scalar_sub, scalar_sum,
};

impl ::core::ops::Add for AccelerationCentimetersPerSecondSquared {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        scalar_add(self, rhs)
    }
}
impl ::core::ops::Sub for AccelerationCentimetersPerSecondSquared {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        scalar_sub(self, rhs)
    }
}
impl ::core::ops::AddAssign for AccelerationCentimetersPerSecondSquared {
    fn add_assign(&mut self, rhs: Self) {
        *self = scalar_add(*self, rhs);
    }
}
impl ::core::ops::SubAssign for AccelerationCentimetersPerSecondSquared {
    fn sub_assign(&mut self, rhs: Self) {
        *self = scalar_sub(*self, rhs);
    }
}
impl ::core::ops::Neg for AccelerationCentimetersPerSecondSquared {
    type Output = Self;
    fn neg(self) -> Self {
        scalar_neg(self)
    }
}
impl ::core::ops::Mul<f64> for AccelerationCentimetersPerSecondSquared {
    type Output = Self;
    fn mul(self, k: f64) -> Self {
        scalar_mul(self, k)
    }
}
impl ::core::ops::Mul<AccelerationCentimetersPerSecondSquared> for f64 {
    type Output = AccelerationCentimetersPerSecondSquared;
    fn mul(
        self,
        x: AccelerationCentimetersPerSecondSquared,
    ) -> AccelerationCentimetersPerSecondSquared {
        scalar_mul(x, self)
    }
}
impl ::core::ops::Div<f64> for AccelerationCentimetersPerSecondSquared {
    type Output = Self;
    fn div(self, k: f64) -> Self {
        scalar_div(self, k)
    }
}
impl ::core::iter::Sum for AccelerationCentimetersPerSecondSquared {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        scalar_sum(iter)
    }
}

impl ::core::ops::Add for AccelerationMetersPerSecondSquared {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        scalar_add(self, rhs)
    }
}
impl ::core::ops::Sub for AccelerationMetersPerSecondSquared {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        scalar_sub(self, rhs)
    }
}
impl ::core::ops::AddAssign for AccelerationMetersPerSecondSquared {
    fn add_assign(&mut self, rhs: Self) {
        *self = scalar_add(*self, rhs);
    }
}
impl ::core::ops::SubAssign for AccelerationMetersPerSecondSquared {
    fn sub_assign(&mut self, rhs: Self) {
        *self = scalar_sub(*self, rhs);
    }
}
impl ::core::ops::Neg for AccelerationMetersPerSecondSquared {
    type Output = Self;
    fn neg(self) -> Self {
        scalar_neg(self)
    }
}
impl ::core::ops::Mul<f64> for AccelerationMetersPerSecondSquared {
    type Output = Self;
    fn mul(self, k: f64) -> Self {
        scalar_mul(self, k)
    }
}
impl ::core::ops::Mul<AccelerationMetersPerSecondSquared> for f64 {
    type Output = AccelerationMetersPerSecondSquared;
    fn mul(self, x: AccelerationMetersPerSecondSquared) -> AccelerationMetersPerSecondSquared {
        scalar_mul(x, self)
    }
}
impl ::core::ops::Div<f64> for AccelerationMetersPerSecondSquared {
    type Output = Self;
    fn div(self, k: f64) -> Self {
        scalar_div(self, k)
    }
}
impl ::core::iter::Sum for AccelerationMetersPerSecondSquared {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        scalar_sum(iter)
    }
}
