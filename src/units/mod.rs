//! Unit and quantity types.
//!
//! This module owns the physical-unit side of the crate: quantity contracts,
//! unit conversions, base and derived measurements, storage wrappers, and
//! operations over quantities. The abstract algebra and generic scalar traits
//! live separately at [`crate::algebra`] and [`crate::numeric`].

pub mod conversion;
pub mod data;
pub mod derived;
pub mod measure;
pub mod ops;
pub mod quantity;

pub use conversion::UnitOfMeasure;
pub use quantity::{
    Dimension, LinearQuantity, Quantity, QuantityError, ScalarQuantity, scalar_mul,
};

pub use data::{
    ByteRange, Bytes, Count, DistanceDecimeters, ThreadCount, TimeCentiseconds, UnixMillis,
    UnixSeconds,
};

pub use derived::{
    AccelerationCentimetersPerSecondSquared, AccelerationMetersPerSecondSquared,
    AreaSquareKilometers, AreaSquareMeters, SpeedKph, SpeedMph, SpeedMs,
};

pub use ops::{Interval, Point};

pub use measure::{
    AngleDegrees, AngleRadians, Factor, LengthFeet, LengthKm, LengthMeters, MassGrams,
    MassKilograms, MassMilligrams, Percentage, TemperatureCelsius, TemperatureFahrenheit,
    TemperatureKelvin, TimeHours, TimeMicros, TimeMicrosDelta, TimeMillis, TimeMinutes, TimeNanos,
    TimeNanosDelta, TimePicos, TimePicosDelta, TimeSeconds,
};
