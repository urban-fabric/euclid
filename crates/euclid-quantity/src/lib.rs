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
//! ## The one crate `missing_docs` is not yet enforced on
//!
//! The workspace denies `missing_docs`, because euclid is mostly trait surface and a doc comment
//! is therefore the deliverable. This crate is excepted, and the exception is a **known debt
//! with a number on it rather than a decision**: 270 public items here carry no doc comment.
//!
//! They are not all equal, and whoever picks this up should take them in this order:
//!
//! - **36 structs and 3 modules** — genuine gaps. `pub struct Bytes(u64);` with nothing above it
//!   leaves a reader to infer what a `Bytes` is and what it does not do, and there is no second
//!   place to find out.
//! - **46 associated constants** — worth a line each where the value is not its own explanation
//!   (`TimeSeconds::INFINITY` means something specific about saturation).
//! - **132 methods and 49 associated functions** — the repetitive carrier surface: `as_km`,
//!   `to_mph`, `is_zero`, `new`, repeated across every unit type. These are the ones where a
//!   mechanical `/// The value in kilometres.` would make the docs *longer* without making them
//!   more useful, and they should be written where the conversion has a convention worth stating
//!   (rounding, saturation, an affine offset) and skipped where it does not.
//!
//! Enforcing the lint here before that work is done would mean writing 270 lines of filler to
//! satisfy a tool, which is the opposite of what the lint is for.

#![allow(missing_docs)]

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
