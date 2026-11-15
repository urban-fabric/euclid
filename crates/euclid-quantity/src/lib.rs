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
//!
pub mod conversion;
/// Storage, count and timestamp newtypes: byte sizes, cardinal counts, and epoch instants.
///
/// Dimensionally these are either dimensionless or affine, which is why they sit apart from
/// [`crate::units`] — a count does not scale and an instant does not add.
pub mod data;
/// Operations *over* quantities rather than quantities themselves: uncertainty intervals
/// ([`crate::Interval`]) and affine positions ([`crate::Point`]).
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

// The README's examples, compiled.
//
// `cfg(doctest)` means this item exists only while rustdoc is collecting doctests: it is absent
// from every real build, adds nothing to the public surface, and never appears in the rendered
// docs. What it buys is that `cargo test --doc` compiles the code blocks in README.md itself.
//
// A crate whose deliverable is its trait surface has a README whose example *is* part of that
// surface, and a README example that no longer compiles is worse than none — it documents a
// signature the crate no longer has. Two crates used to check this by hand-copying the example
// into `tests/readme_check.rs`, which compiles the copy and lets the original drift. This checks
// the original.
//
// Illustrative fragments in the README are marked ```rust,ignore, which is the honest marking:
// they reference names that are deliberately undefined.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeExamples;

// The **workspace** README's examples, compiled here because no crate owns that file.
//
// Its remaining live examples need `euclid-numeric` and `euclid-units`, and this is the only
// member that depends on both — which is the whole reason it landed here rather than somewhere
// more obviously appropriate. If that stops being true, move this item; do not delete it, or the
// front page of the repository goes back to being unverified prose.
//
// The two illustrative fragments in it are marked ```rust,ignore for the same reason as above.
#[cfg(doctest)]
#[doc = include_str!("../../../README.md")]
struct WorkspaceReadmeExamples;
