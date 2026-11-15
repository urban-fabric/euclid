//! # euclid-quantity — unit and quantity types.
//!
//! The physical-unit primitives: unit conversions and physical units grouped by dimension
//! family (lengths, times, angles, mass, temperature, speed, area, and the wider SI web —
//! force, energy, electromagnetism, chemistry, …), storage/count/timestamp wrappers, and the
//! operations over quantities (linear vector-space ops, affine positions, intervals). Lifted
//! out of `euclid`, which owns the carrier-generic numeric, tensor, and unit-contract
//! foundations. euclid-quantity supplies the concrete `f64` unit/storage types and
//! implements euclid's unit contracts for them.
//!
//! The quantity *contracts* are euclid's, not this crate's. [`Measure`], [`Quantity`],
//! [`ScalarQuantity`], [`LinearQuantity`] and [`DiscreteQuantity`] are re-exported from
//! `euclid-units`, so `use euclid_quantity::Quantity` and `use euclid_units::Quantity` name
//! the same trait. Only [`HostQuantity`] and [`QuantityError`] originate here — see
//! [`quantity`] for why those two cannot live in a carrier-generic crate.

pub mod conversion;
pub mod data;
pub mod ops;
pub mod quantity;
pub mod units;

pub use conversion::UnitOfMeasure;

pub use quantity::{
    Dimension, DiscreteQuantity, HostQuantity, LinearQuantity, Measure, Quantity, QuantityError,
    ScalarQuantity, scalar_mul,
};

pub use data::{
    ByteRange, Bytes, Count, DistanceDecimeters, ThreadCount, TimeCentiseconds, UnixMillis,
    UnixSeconds,
};

pub use ops::{Interval, Point};

pub use units::*;

/// euclid's dimension contracts, re-exported so the cross-dimensional algebra reads ergonomically
/// from euclid-quantity alone: `use euclid_quantity::Length;` then `distance.per_time(elapsed)`.
///
/// The base contracts (`Measure`, `Quantity`, `ScalarQuantity`, `LinearQuantity`,
/// `DiscreteQuantity`) are re-exported alongside them, from [`quantity`]. They are euclid's own
/// traits — this crate no longer defines same-named ones of its own — so
/// `use euclid_quantity::Quantity` and `use euclid_units::Quantity` now name the same trait.
pub use euclid_units::{
    Acceleration, AmountOfSubstance, Angle, AngularAcceleration, AngularMomentum, AngularVelocity,
    Area, Capacitance, CatalyticActivity, Concentration, Conductance, CurrentDensity, Density,
    Dimensionless, DynamicViscosity, ElectricCharge, ElectricCurrent, ElectricField, Energy, Force,
    Frequency, HeatCapacity, HeatFlux, Inductance, Jerk, Length, MagneticFlux, MagneticFluxDensity,
    Mass, MassFlowRate, Molality, MolarEnergy, MolarMass, MolarVolume, MomentOfInertia, Momentum,
    Power, Pressure, Resistance, SpecificEnergy, SpecificHeatCapacity, Speed, SurfaceTension,
    Temperature, Time, Torque, Voltage, Volume, VolumetricFlowRate,
};
