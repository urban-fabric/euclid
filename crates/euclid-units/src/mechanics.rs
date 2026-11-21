//! Mechanical / dynamic dimensions: [`Mass`], [`Force`], [`Pressure`], [`Energy`], [`Power`],
//! [`Momentum`], [`Density`], [`MomentOfInertia`], [`AngularMomentum`], [`Torque`],
//! [`DynamicViscosity`], [`SurfaceTension`], [`MassFlowRate`], [`KinematicViscosity`],
//! [`SpecificEnergy`], [`EnergyDensity`], [`Action`].
//!
//! This is the densest part of the algebra web — Newton's second law (`mass × acceleration =
//! force`), work (`force × length = energy`), power (`energy / time`), impulse (`force × time =
//! momentum`), and the rotational analogues all live here. See the
//! [`geometry`](super::geometry) module docs for the `times_<operand>` / `per_<operand>` naming
//! convention.
//!
//! Note that [`Energy`] and [`Torque`] share a base-dimension signature (`[L² M T⁻²]`) but are
//! distinct *names*: `force × length` is defined to produce [`Energy`] (work), while
//! [`Torque`] is reached through the rotational chain
//! ([`MomentOfInertia::times_angular_acceleration`]). They are
//! [`commensurable`](super::Dimension::commensurable) but not
//! [`same`](super::Dimension::same).

//! # The split
//!
//! This module passed the crate's 400-line budget. The seventeen traits divide into four clusters by
//! the relation that reaches them, and this file is the index — the re-exports below keep
//! `mechanics::Force` and the crate-root paths exactly where they were.

mod energy;
mod flow;
mod kinetics;
mod rotation;

// Re-exported so every public path this module has ever had still resolves: the
// split is an internal rearrangement and no carrier should have to notice it.
pub use energy::{Action, Energy, EnergyDensity, Power, SpecificEnergy};
pub use flow::{DynamicViscosity, KinematicViscosity, MassFlowRate, SurfaceTension};
pub use kinetics::{Density, Force, Mass, Momentum, Pressure};
pub use rotation::{AngularMomentum, MomentOfInertia, Torque};

use super::{
    Acceleration, AmountOfSubstance, AngularAcceleration, AngularVelocity, Area, ElectricCharge,
    ElectricCurrent, ElectricField, Frequency, HeatCapacity, HeatFlux, Length, LinearQuantity,
    MolarEnergy, MolarMass, Speed, Temperature, Time, Voltage, Volume, VolumetricFlowRate,
};
