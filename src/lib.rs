#![doc = include_str!("../README.md")]

pub mod core;
pub mod data;
pub mod derived;
pub mod field;
pub mod math;
pub mod measure;

pub use core::units;
pub use core::{
    Dimension, LinearQuantity, Quantity, QuantityError, ScalarQuantity, UnitOfMeasure, scalar_add,
    scalar_div, scalar_mul, scalar_neg, scalar_sub, scalar_sum,
};

pub use data::{
    ByteRange, Bytes, ConversionError, Count, DistanceDecimeters, DistanceWeight, ThreadCount,
    TimeCentiseconds, TimeWeight, UnixMillis, UnixSeconds,
};
pub use data::{bytes, count, quantize, storage, timestamp, weights};

pub use derived::{
    AccelerationCentimetersPerSecondSquared, AccelerationMetersPerSecondSquared,
    AreaSquareKilometers, AreaSquareMeters, SpeedKph, SpeedMph, SpeedMs,
};
pub use derived::{acceleration, area, speed};

pub use field::{ComplexField, RealField, Scalar};

pub use math::{Interval, Point};
pub use math::{affine, algebra, interval};

pub use measure::{
    AngleDegrees, AngleRadians, Factor, LengthFeet, LengthKm, LengthMeters, MassGrams,
    MassKilograms, MassMilligrams, Percentage, TemperatureCelsius, TemperatureFahrenheit,
    TemperatureKelvin, TimeHours, TimeMicros, TimeMicrosDelta, TimeMillis, TimeMinutes, TimeNanos,
    TimeNanosDelta, TimePicos, TimePicosDelta, TimeSeconds,
};
pub use measure::{angle, factor, length, mass, percentage, temperature, time};
