use core::fmt;

use crate::{
    AccelerationCentimetersPerSecondSquared, AccelerationMetersPerSecondSquared, AngleDegrees,
    AngleRadians, AreaSquareKilometers, AreaSquareMeters, Factor, LengthFeet, LengthKm,
    LengthMeters, MassGrams, MassKilograms, MassMilligrams, Percentage, Quantity, SpeedKph,
    SpeedMph, SpeedMs, TemperatureCelsius, TemperatureFahrenheit, TemperatureKelvin, TimeHours,
    TimeMillis, TimeMinutes, TimeSeconds,
};

use super::{UnitOfMeasure, dim};

fn fmt_quantity<Q: Quantity<Repr = f64>>(value: Q, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "{} {}", value.raw(), Q::UNIT_NAME)
}

impl UnitOfMeasure for AccelerationMetersPerSecondSquared {
    type Dim = dim::Acceleration;
    const TO_PRIMARY: f64 = 1.0;
}

impl fmt::Display for AccelerationMetersPerSecondSquared {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_quantity(*self, f)
    }
}

impl UnitOfMeasure for AccelerationCentimetersPerSecondSquared {
    type Dim = dim::Acceleration;
    const TO_PRIMARY: f64 = crate::acceleration::MPS2_PER_CMPS2;
}

impl fmt::Display for AccelerationCentimetersPerSecondSquared {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_quantity(*self, f)
    }
}

impl UnitOfMeasure for AngleRadians {
    type Dim = dim::Angle;
    const TO_PRIMARY: f64 = 1.0;
}

impl fmt::Display for AngleRadians {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_quantity(*self, f)
    }
}

impl UnitOfMeasure for AngleDegrees {
    type Dim = dim::Angle;
    const TO_PRIMARY: f64 = std::f64::consts::PI / 180.0;
}

impl fmt::Display for AngleDegrees {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_quantity(*self, f)
    }
}

impl UnitOfMeasure for AreaSquareMeters {
    type Dim = dim::Area;
    const TO_PRIMARY: f64 = 1.0;
}

impl fmt::Display for AreaSquareMeters {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_quantity(*self, f)
    }
}

impl UnitOfMeasure for AreaSquareKilometers {
    type Dim = dim::Area;
    const TO_PRIMARY: f64 = crate::area::SQUARE_METERS_PER_SQUARE_KILOMETER;
}

impl fmt::Display for AreaSquareKilometers {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_quantity(*self, f)
    }
}

impl UnitOfMeasure for LengthMeters {
    type Dim = dim::Length;
    const TO_PRIMARY: f64 = 1.0;
}

impl fmt::Display for LengthMeters {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_quantity(*self, f)
    }
}

impl UnitOfMeasure for LengthKm {
    type Dim = dim::Length;
    const TO_PRIMARY: f64 = crate::length::M_PER_KM;
}

impl fmt::Display for LengthKm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_quantity(*self, f)
    }
}

impl UnitOfMeasure for LengthFeet {
    type Dim = dim::Length;
    const TO_PRIMARY: f64 = crate::length::M_PER_FOOT;
}

impl fmt::Display for LengthFeet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_quantity(*self, f)
    }
}

impl UnitOfMeasure for MassKilograms {
    type Dim = dim::Mass;
    const TO_PRIMARY: f64 = 1.0;
}

impl fmt::Display for MassKilograms {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_quantity(*self, f)
    }
}

impl UnitOfMeasure for MassGrams {
    type Dim = dim::Mass;
    const TO_PRIMARY: f64 = crate::mass::KG_PER_GRAM;
}

impl fmt::Display for MassGrams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_quantity(*self, f)
    }
}

impl UnitOfMeasure for MassMilligrams {
    type Dim = dim::Mass;
    const TO_PRIMARY: f64 = crate::mass::KG_PER_MILLIGRAM;
}

impl fmt::Display for MassMilligrams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_quantity(*self, f)
    }
}

impl UnitOfMeasure for TimeSeconds {
    type Dim = dim::Time;
    const TO_PRIMARY: f64 = 1.0;
}

impl fmt::Display for TimeSeconds {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_quantity(*self, f)
    }
}

impl UnitOfMeasure for TimeMinutes {
    type Dim = dim::Time;
    const TO_PRIMARY: f64 = crate::time::S_PER_MIN;
}

impl fmt::Display for TimeMinutes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_quantity(*self, f)
    }
}

impl UnitOfMeasure for TimeHours {
    type Dim = dim::Time;
    const TO_PRIMARY: f64 = crate::time::S_PER_HOUR;
}

impl fmt::Display for TimeHours {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_quantity(*self, f)
    }
}

impl UnitOfMeasure for TimeMillis {
    type Dim = dim::Time;
    const TO_PRIMARY: f64 = 1.0 / crate::time::MS_PER_S;
}

impl fmt::Display for TimeMillis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_quantity(*self, f)
    }
}

impl UnitOfMeasure for SpeedMs {
    type Dim = dim::Speed;
    const TO_PRIMARY: f64 = 1.0;
}

impl fmt::Display for SpeedMs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_quantity(*self, f)
    }
}

impl UnitOfMeasure for SpeedKph {
    type Dim = dim::Speed;
    const TO_PRIMARY: f64 = crate::speed::MS_PER_KPH;
}

impl fmt::Display for SpeedKph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_quantity(*self, f)
    }
}

impl UnitOfMeasure for SpeedMph {
    type Dim = dim::Speed;
    const TO_PRIMARY: f64 = crate::speed::MS_PER_MPH;
}

impl fmt::Display for SpeedMph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_quantity(*self, f)
    }
}

impl UnitOfMeasure for TemperatureKelvin {
    type Dim = dim::Temperature;
    const TO_PRIMARY: f64 = 1.0;
    const PRIMARY_OFFSET: f64 = 0.0;
}

impl fmt::Display for TemperatureKelvin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_quantity(*self, f)
    }
}

impl UnitOfMeasure for TemperatureCelsius {
    type Dim = dim::Temperature;
    const TO_PRIMARY: f64 = 1.0;
    const PRIMARY_OFFSET: f64 = crate::temperature::KELVIN_OFFSET_CELSIUS;
}

impl fmt::Display for TemperatureCelsius {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_quantity(*self, f)
    }
}

impl UnitOfMeasure for TemperatureFahrenheit {
    type Dim = dim::Temperature;
    const TO_PRIMARY: f64 = crate::temperature::KELVIN_PER_FAHRENHEIT_DEGREE;
    const PRIMARY_OFFSET: f64 = crate::temperature::KELVIN_OFFSET_FAHRENHEIT;
}

impl fmt::Display for TemperatureFahrenheit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_quantity(*self, f)
    }
}

impl UnitOfMeasure for Factor {
    type Dim = dim::Dimensionless;
    const TO_PRIMARY: f64 = 1.0;
}

impl fmt::Display for Factor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_quantity(*self, f)
    }
}

impl UnitOfMeasure for Percentage {
    type Dim = dim::Dimensionless;
    const TO_PRIMARY: f64 = 1.0 / 100.0;
}

impl fmt::Display for Percentage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_quantity(*self, f)
    }
}
