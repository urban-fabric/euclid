//! Same-unit scalar operators for every linear quantity.
//!
//! Each [`LinearQuantity`](crate::LinearQuantity) newtype gets the same vector-space
//! operator set (`+`, `-`, unary `-`, `*` / `/` by `f64`, and [`Sum`](core::iter::Sum)),
//! all delegating to the `scalar_*` free functions. The `impl_linear_ops!` macro
//! generates them, so adding a new unit means adding one name to the list below
//! rather than copying nine `impl` blocks.

use crate::{
    AccelerationCentimetersPerSecondSquared, AccelerationMetersPerSecondSquared, AngleDegrees,
    AngleRadians, AreaSquareKilometers, AreaSquareMeters, Factor, LengthFeet, LengthKm,
    LengthMeters, MassGrams, MassKilograms, MassMilligrams, Percentage, SpeedKph, SpeedMph,
    SpeedMs, TimeHours, TimeMillis, TimeMinutes, TimeSeconds,
};

/// Implement the standard linear-quantity operator set for each listed type.
macro_rules! impl_linear_ops {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl ::core::ops::Add for $ty {
                type Output = Self;
                fn add(self, rhs: Self) -> Self {
                    $crate::scalar_add(self, rhs)
                }
            }

            impl ::core::ops::Sub for $ty {
                type Output = Self;
                fn sub(self, rhs: Self) -> Self {
                    $crate::scalar_sub(self, rhs)
                }
            }

            impl ::core::ops::AddAssign for $ty {
                fn add_assign(&mut self, rhs: Self) {
                    *self = $crate::scalar_add(*self, rhs);
                }
            }

            impl ::core::ops::SubAssign for $ty {
                fn sub_assign(&mut self, rhs: Self) {
                    *self = $crate::scalar_sub(*self, rhs);
                }
            }

            impl ::core::ops::Neg for $ty {
                type Output = Self;
                fn neg(self) -> Self {
                    $crate::scalar_neg(self)
                }
            }

            impl ::core::ops::Mul<f64> for $ty {
                type Output = Self;
                fn mul(self, k: f64) -> Self {
                    $crate::scalar_mul(self, k)
                }
            }

            impl ::core::ops::Mul<$ty> for f64 {
                type Output = $ty;
                fn mul(self, x: $ty) -> $ty {
                    $crate::scalar_mul(x, self)
                }
            }

            impl ::core::ops::Div<f64> for $ty {
                type Output = Self;
                fn div(self, k: f64) -> Self {
                    $crate::scalar_div(self, k)
                }
            }

            impl ::core::iter::Sum for $ty {
                fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
                    $crate::scalar_sum(iter)
                }
            }
        )+
    };
}

impl_linear_ops!(
    // measure
    LengthMeters,
    LengthKm,
    LengthFeet,
    TimeSeconds,
    TimeMinutes,
    TimeHours,
    TimeMillis,
    AngleRadians,
    AngleDegrees,
    MassKilograms,
    MassGrams,
    MassMilligrams,
    Factor,
    Percentage,
    // derived
    SpeedMs,
    SpeedKph,
    SpeedMph,
    AreaSquareMeters,
    AreaSquareKilometers,
    AccelerationMetersPerSecondSquared,
    AccelerationCentimetersPerSecondSquared,
);
