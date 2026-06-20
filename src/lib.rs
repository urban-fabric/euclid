#![doc = include_str!("../README.md")]

pub mod algebra;
pub mod array;
pub mod numeric;
pub mod special;
pub mod units;

pub use algebra::{GaussianRational, Integer, Rational, Rational64, Rational128, Zn, Zp};
pub use array::{LazyMatrix, LazyTensor, LazyVector};
pub use numeric::{
    Arithmetic, Complex, LazyComplex, LazyQuaternion, LazyReal, NormedDivisionAlgebra, Real,
};

pub use units::{
    AngleDegrees, AngleRadians, ByteRange, Bytes, Count, Dimension, DistanceDecimeters, Factor,
    LengthFeet, LengthKm, LengthMeters, LinearQuantity, MassGrams, MassKilograms, MassMilligrams,
    Percentage, Quantity, QuantityError, ScalarQuantity, TemperatureCelsius, TemperatureFahrenheit,
    TemperatureKelvin, ThreadCount, TimeCentiseconds, TimeHours, TimeMicros, TimeMicrosDelta,
    TimeMillis, TimeMinutes, TimeNanos, TimeNanosDelta, TimePicos, TimePicosDelta, TimeSeconds,
    UnitOfMeasure, UnixMillis, UnixSeconds, scalar_mul,
};

pub use units::{
    AccelerationCentimetersPerSecondSquared, AccelerationMetersPerSecondSquared,
    AreaSquareKilometers, AreaSquareMeters, Interval, Point, SpeedKph, SpeedMph, SpeedMs,
};
