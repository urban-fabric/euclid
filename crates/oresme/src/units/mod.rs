//! Physical units, grouped by euclid dimension family.
//!
//! Each submodule owns one family of euclid's dimension web — the concrete `f64` unit newtypes
//! together with their oresme [`Quantity`](crate::Quantity)/[`LinearQuantity`](crate::LinearQuantity)
//! impls, [`UnitOfMeasure`](crate::conversion::UnitOfMeasure) conversions, and the implementations
//! of euclid's base and dimension-specific contracts. Families mirror euclid's own `units` layout
//! (`geometry`, `kinematics`, `mechanics`, `thermal`, `electromagnetism`, `chemistry`, plus the
//! dimensionless ratios), so a unit and the algebra it participates in live in one place.

pub mod chemistry;
pub mod dimensionless;
pub mod electromagnetism;
pub mod geometry;
pub mod kinematics;
pub mod mechanics;
pub mod thermal;

pub use chemistry::*;
pub use dimensionless::*;
pub use electromagnetism::*;
pub use geometry::*;
pub use kinematics::*;
pub use mechanics::*;
pub use thermal::*;

#[cfg(test)]
mod base_tests;
#[cfg(test)]
mod web_tests;
