use super::{
    AngleDegrees, AngleRadians, Factor, LengthFeet, LengthKm, LengthMeters, Percentage, SpeedKph,
    SpeedMph, SpeedMs, TimeHours, TimeMillis, TimeMinutes, TimeSeconds, scalar_add, scalar_div,
    scalar_mul, scalar_neg, scalar_sub, scalar_sum,
};

macro_rules! impl_scalar_quantity_ops {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl ::core::ops::Add for $ty {
                type Output = Self;

                fn add(self, rhs: Self) -> Self {
                    scalar_add(self, rhs)
                }
            }

            impl ::core::ops::Sub for $ty {
                type Output = Self;

                fn sub(self, rhs: Self) -> Self {
                    scalar_sub(self, rhs)
                }
            }

            impl ::core::ops::AddAssign for $ty {
                fn add_assign(&mut self, rhs: Self) {
                    *self = scalar_add(*self, rhs);
                }
            }

            impl ::core::ops::SubAssign for $ty {
                fn sub_assign(&mut self, rhs: Self) {
                    *self = scalar_sub(*self, rhs);
                }
            }

            impl ::core::ops::Neg for $ty {
                type Output = Self;

                fn neg(self) -> Self {
                    scalar_neg(self)
                }
            }

            impl ::core::ops::Mul<f64> for $ty {
                type Output = Self;

                fn mul(self, k: f64) -> Self {
                    scalar_mul(self, k)
                }
            }

            impl ::core::ops::Mul<$ty> for f64 {
                type Output = $ty;

                fn mul(self, x: $ty) -> $ty {
                    scalar_mul(x, self)
                }
            }

            impl ::core::ops::Div<f64> for $ty {
                type Output = Self;

                fn div(self, k: f64) -> Self {
                    scalar_div(self, k)
                }
            }

            impl ::core::iter::Sum for $ty {
                fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
                    scalar_sum(iter)
                }
            }
        )+
    };
}

impl_scalar_quantity_ops!(
    Factor,
    AngleRadians,
    AngleDegrees,
    LengthMeters,
    LengthKm,
    LengthFeet,
    SpeedMs,
    SpeedKph,
    SpeedMph,
    Percentage,
    TimeSeconds,
    TimeMinutes,
    TimeHours,
    TimeMillis,
);
