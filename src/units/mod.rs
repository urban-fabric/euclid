//! Physical-dimension contracts over any [`Real`](crate::Real) carrier.
//!
//! Like the rest of euclid, `units` ships *contracts*, not concrete types. A quantity is any
//! [`Real`](crate::Real)-backed carrier that implements [`Quantity`]; the dimension traits refine
//! it with the operations that dimension supports, including the cross-dimensional algebra
//! (`length × length = area`, `length / time = speed`, `mass × acceleration = force`,
//! `current × resistance = voltage`, …) expressed through associated quantity types. euclid does
//! not ship a `Meters`/`Seconds` struct; a consumer defines its own unit carriers and implements
//! these traits for them.
//!
//! ## Coverage
//!
//! The trait surface spans the seven SI base dimensions and the standard derived set:
//!
//! - **base** — [`Length`], [`Mass`], [`Time`], [`ElectricCurrent`], [`Temperature`],
//!   [`AmountOfSubstance`], [`LuminousIntensity`]
//! - **[`geometry`]** — [`Area`], [`Volume`], [`Angle`], [`SolidAngle`]
//! - **[`kinematics`]** — [`Frequency`], [`Speed`], [`Acceleration`], [`Jerk`],
//!   [`AngularVelocity`], [`AngularAcceleration`], [`VolumetricFlowRate`]
//! - **[`mechanics`]** — [`Force`], [`Pressure`], [`Energy`], [`Power`], [`Momentum`],
//!   [`AngularMomentum`], [`Torque`], [`Density`], [`MomentOfInertia`], [`DynamicViscosity`],
//!   [`SurfaceTension`]
//! - **[`electromagnetism`]** — [`ElectricCharge`], [`Voltage`], [`Resistance`], [`Conductance`],
//!   [`Capacitance`], [`Inductance`], [`MagneticFlux`], [`MagneticFluxDensity`], [`ElectricField`]
//! - **[`thermal`]** — [`HeatCapacity`], [`ThermalConductivity`]
//! - **[`photometry`]** — [`LuminousFlux`], [`Illuminance`]
//! - **[`chemistry`]** — [`CatalyticActivity`], [`Concentration`], [`MolarMass`]
//! - **pure number** — [`Dimensionless`] (factors, gains, percentages)
//!
//! ## Type-level markers
//!
//! Alongside the runtime [`Dimension`] enum and the trait-position dimension contracts, the
//! [`dim`] module provides a zero-sized *type-level* marker per dimension, and [`Dimensioned`]
//! bridges each marker back to its runtime [`Dimension`] (and SI [`BaseExponents`]). A
//! unit-of-measure protocol can carry a `dim::*` marker as an associated type to make
//! "same dimension" a compile-time requirement.
//!
//! ## Trait layering
//!
//! Every dimension trait refines one of three carrier-generic base contracts:
//!
//! - [`Quantity`] — the raw carrier, [`Dimension`] tag, and unit name.
//! - [`ScalarQuantity`] — min/max/clamp/rounding and mask-returning predicates.
//! - [`LinearQuantity`] — displacement algebra (add/sub/neg/scale/lerp). Absolute scalars such
//!   as [`Temperature`] deliberately stop at [`ScalarQuantity`].
//!
//! ## Naming convention
//!
//! Cross-dimensional operations are verbs suffixed by the *operand's* dimension: `times_<operand>`
//! is `self * operand`, `per_<operand>` is `self / operand`. Pure powers get `squared` / `cubed`,
//! and reciprocal pairs (`time ⇄ frequency`, `resistance ⇄ conductance`) are named semantically.
//! Same-unit ratios collapse to the raw carrier through [`LinearQuantity::ratio`], and a
//! [`Dimensionless`] factor scales a dimensioned quantity through [`LinearQuantity::scale`].
//!
//! Quantities are scalar-valued: a length carries one `T`, where `T` can be a host scalar,
//! expression, or graph scalar. Values are `Clone`, comparisons return masks through
//! [`Compare`](crate::Compare), and host literals are lifted through
//! [`Real::scalar`](crate::Real::scalar)/[`scalar_like`](crate::Real::scalar_like).

pub mod chemistry;
pub mod electromagnetism;
pub mod geometry;
pub mod kinematics;
pub mod mechanics;
pub mod photometry;
pub mod thermal;

mod dimension;
mod dimensioned;
mod dimensionless;
mod linear;
mod quantity;
mod scalar;

pub use chemistry::{AmountOfSubstance, CatalyticActivity, Concentration, MolarMass};
pub use dimension::{BaseExponents, Dimension};
pub use dimensioned::{Dimensioned, dim};
pub use dimensionless::Dimensionless;
pub use electromagnetism::{
    Capacitance, Conductance, ElectricCharge, ElectricCurrent, ElectricField, Inductance,
    MagneticFlux, MagneticFluxDensity, Resistance, Voltage,
};
pub use geometry::{Angle, Area, Length, SolidAngle, Volume};
pub use kinematics::{
    Acceleration, AngularAcceleration, AngularVelocity, Frequency, Jerk, Speed, Time,
    VolumetricFlowRate,
};
pub use linear::LinearQuantity;
pub use mechanics::{
    AngularMomentum, Density, DynamicViscosity, Energy, Force, Mass, MomentOfInertia, Momentum,
    Power, Pressure, SurfaceTension, Torque,
};
pub use photometry::{Illuminance, LuminousFlux, LuminousIntensity};
pub use quantity::{Mask, Quantity};
pub use scalar::ScalarQuantity;
pub use thermal::{HeatCapacity, Temperature, ThermalConductivity};

#[cfg(test)]
mod tests;
