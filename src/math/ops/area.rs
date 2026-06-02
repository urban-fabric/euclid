use crate::{
    AreaSquareKilometers, AreaSquareMeters, scalar_add, scalar_div, scalar_mul, scalar_neg,
    scalar_sub, scalar_sum,
};

impl ::core::ops::Add for AreaSquareMeters {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        scalar_add(self, rhs)
    }
}
impl ::core::ops::Sub for AreaSquareMeters {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        scalar_sub(self, rhs)
    }
}
impl ::core::ops::AddAssign for AreaSquareMeters {
    fn add_assign(&mut self, rhs: Self) {
        *self = scalar_add(*self, rhs);
    }
}
impl ::core::ops::SubAssign for AreaSquareMeters {
    fn sub_assign(&mut self, rhs: Self) {
        *self = scalar_sub(*self, rhs);
    }
}
impl ::core::ops::Neg for AreaSquareMeters {
    type Output = Self;
    fn neg(self) -> Self {
        scalar_neg(self)
    }
}
impl ::core::ops::Mul<f64> for AreaSquareMeters {
    type Output = Self;
    fn mul(self, k: f64) -> Self {
        scalar_mul(self, k)
    }
}
impl ::core::ops::Mul<AreaSquareMeters> for f64 {
    type Output = AreaSquareMeters;
    fn mul(self, x: AreaSquareMeters) -> AreaSquareMeters {
        scalar_mul(x, self)
    }
}
impl ::core::ops::Div<f64> for AreaSquareMeters {
    type Output = Self;
    fn div(self, k: f64) -> Self {
        scalar_div(self, k)
    }
}
impl ::core::iter::Sum for AreaSquareMeters {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        scalar_sum(iter)
    }
}

impl ::core::ops::Add for AreaSquareKilometers {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        scalar_add(self, rhs)
    }
}
impl ::core::ops::Sub for AreaSquareKilometers {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        scalar_sub(self, rhs)
    }
}
impl ::core::ops::AddAssign for AreaSquareKilometers {
    fn add_assign(&mut self, rhs: Self) {
        *self = scalar_add(*self, rhs);
    }
}
impl ::core::ops::SubAssign for AreaSquareKilometers {
    fn sub_assign(&mut self, rhs: Self) {
        *self = scalar_sub(*self, rhs);
    }
}
impl ::core::ops::Neg for AreaSquareKilometers {
    type Output = Self;
    fn neg(self) -> Self {
        scalar_neg(self)
    }
}
impl ::core::ops::Mul<f64> for AreaSquareKilometers {
    type Output = Self;
    fn mul(self, k: f64) -> Self {
        scalar_mul(self, k)
    }
}
impl ::core::ops::Mul<AreaSquareKilometers> for f64 {
    type Output = AreaSquareKilometers;
    fn mul(self, x: AreaSquareKilometers) -> AreaSquareKilometers {
        scalar_mul(x, self)
    }
}
impl ::core::ops::Div<f64> for AreaSquareKilometers {
    type Output = Self;
    fn div(self, k: f64) -> Self {
        scalar_div(self, k)
    }
}
impl ::core::iter::Sum for AreaSquareKilometers {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        scalar_sum(iter)
    }
}
