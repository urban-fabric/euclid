use core::fmt;

use crate::{
    AccelerationCentimetersPerSecondSquared, AccelerationMetersPerSecondSquared, AngleDegrees,
    AngleRadians, AreaSquareKilometers, AreaSquareMeters, Factor, LengthFeet, LengthKm,
    LengthMeters, MassGrams, MassKilograms, MassMilligrams, Percentage, Quantity, SpeedKph,
    SpeedMph, SpeedMs, TemperatureCelsius, TemperatureFahrenheit, TemperatureKelvin, TimeHours,
    TimeMillis, TimeMinutes, TimeSeconds,
};

use super::{DimMarker, UnitOfMeasure, dim};

fn fmt_quantity<Q: Quantity<Repr = f64>>(value: Q, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    if Q::UNIT_NAME.is_empty() {
        write!(f, "{}", value.raw())
    } else {
        write!(f, "{} {}", value.raw(), Q::UNIT_NAME)
    }
}

/// Implement [`UnitOfMeasure`] and a unit-suffixed [`Display`](fmt::Display) for a
/// scalar unit newtype. The primary unit of each dimension uses `to_primary = 1.0`;
/// pass `offset = ...` for affine units (e.g. Celsius / Fahrenheit temperatures).
macro_rules! impl_unit {
    ($ty:ty, $dim:ty, $to_primary:expr $(, offset = $offset:expr)?) => {
        impl UnitOfMeasure for $ty {
            type Dim = $dim;
            const TO_PRIMARY: f64 = $to_primary;
            $(const PRIMARY_OFFSET: f64 = $offset;)?
        }

        const _: () = assert!(
            <$ty as Quantity>::DIMENSION.same(<$dim as DimMarker>::RUNTIME),
            concat!("dimension mismatch for ", stringify!($ty)),
        );

        impl fmt::Display for $ty {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt_quantity(*self, f)
            }
        }
    };
}

impl_unit!(AccelerationMetersPerSecondSquared, dim::Acceleration, 1.0);
impl_unit!(
    AccelerationCentimetersPerSecondSquared,
    dim::Acceleration,
    crate::units::derived::acceleration::MPS2_PER_CMPS2
);

impl_unit!(AngleRadians, dim::Angle, 1.0);
impl_unit!(AngleDegrees, dim::Angle, std::f64::consts::PI / 180.0);

impl_unit!(AreaSquareMeters, dim::Area, 1.0);
impl_unit!(
    AreaSquareKilometers,
    dim::Area,
    crate::units::derived::area::SQUARE_METERS_PER_SQUARE_KILOMETER
);

impl_unit!(LengthMeters, dim::Length, 1.0);
impl_unit!(
    LengthKm,
    dim::Length,
    crate::units::measure::length::M_PER_KM
);
impl_unit!(
    LengthFeet,
    dim::Length,
    crate::units::measure::length::M_PER_FOOT
);

impl_unit!(MassKilograms, dim::Mass, 1.0);
impl_unit!(
    MassGrams,
    dim::Mass,
    crate::units::measure::mass::KG_PER_GRAM
);
impl_unit!(
    MassMilligrams,
    dim::Mass,
    crate::units::measure::mass::KG_PER_MILLIGRAM
);

impl_unit!(TimeSeconds, dim::Time, 1.0);
impl_unit!(
    TimeMinutes,
    dim::Time,
    crate::units::measure::time::S_PER_MIN
);
impl_unit!(
    TimeHours,
    dim::Time,
    crate::units::measure::time::S_PER_HOUR
);
impl_unit!(
    TimeMillis,
    dim::Time,
    1.0 / crate::units::measure::time::MS_PER_S
);

impl_unit!(SpeedMs, dim::Speed, 1.0);
impl_unit!(
    SpeedKph,
    dim::Speed,
    crate::units::derived::speed::MS_PER_KPH
);
impl_unit!(
    SpeedMph,
    dim::Speed,
    crate::units::derived::speed::MS_PER_MPH
);

impl_unit!(TemperatureKelvin, dim::Temperature, 1.0);
impl_unit!(
    TemperatureCelsius,
    dim::Temperature,
    1.0,
    offset = crate::units::measure::temperature::KELVIN_OFFSET_CELSIUS
);
impl_unit!(
    TemperatureFahrenheit,
    dim::Temperature,
    crate::units::measure::temperature::KELVIN_PER_FAHRENHEIT_DEGREE,
    offset = crate::units::measure::temperature::KELVIN_OFFSET_FAHRENHEIT
);

impl_unit!(Factor, dim::Dimensionless, 1.0);
impl_unit!(Percentage, dim::Dimensionless, 1.0 / 100.0);
