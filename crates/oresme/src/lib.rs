//! # oresme — unit and quantity types.
//!
//! The physical-unit primitives: quantity contracts, unit conversions, and physical units
//! grouped by dimension family (lengths, times, angles, mass, temperature, speed, area, and the
//! wider SI web — force, energy, electromagnetism, chemistry, …), storage/count/timestamp
//! wrappers, and the operations over quantities (linear vector-space ops, affine positions,
//! intervals). Lifted out of `euclid`, which owns the carrier-generic numeric, tensor, and
//! unit-contract foundations. Oresme supplies the concrete `f64` unit/storage types and
//! implements euclid's unit contracts for them.

pub mod conversion;
pub mod data;
pub mod ops;
pub mod quantity;
pub mod units;

pub use conversion::UnitOfMeasure;

pub use quantity::{
    Dimension, LinearQuantity, Quantity, QuantityError, ScalarQuantity, scalar_mul,
};

pub use data::{
    ByteRange, Bytes, Count, DistanceDecimeters, ThreadCount, TimeCentiseconds, UnixMillis,
    UnixSeconds,
};

pub use ops::{Interval, Point};

pub use units::*;

/// euclid's dimension contracts, re-exported so the cross-dimensional algebra reads ergonomically
/// from oresme alone: `use oresme::Length;` then `distance.per_time(elapsed)`. The base contracts
/// (`Quantity`, `ScalarQuantity`, `LinearQuantity`) are intentionally *not* re-exported — oresme
/// defines its own same-named traits — so reach for `euclid_units::*` if you need those directly.
pub use euclid_units::{
    Acceleration, AmountOfSubstance, Angle, AngularAcceleration, AngularMomentum, AngularVelocity,
    Area, Capacitance, CatalyticActivity, Concentration, Conductance, CurrentDensity, Density,
    Dimensionless, DynamicViscosity, ElectricCharge, ElectricCurrent, ElectricField, Energy, Force,
    Frequency, HeatCapacity, HeatFlux, Inductance, Jerk, Length, MagneticFlux, MagneticFluxDensity,
    Mass, MassFlowRate, Molality, MolarEnergy, MolarMass, MolarVolume, MomentOfInertia, Momentum,
    Power, Pressure, Resistance, SpecificEnergy, SpecificHeatCapacity, Speed, SurfaceTension,
    Temperature, Time, Torque, Voltage, Volume, VolumetricFlowRate,
};
