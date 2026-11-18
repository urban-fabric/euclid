//! Physical units, grouped by euclid dimension family.
//!
//! Each family provides concrete carriers, [`crate::conversion::UnitOfMeasure`] conversions,
//! and implementations of the relevant `euclid-units` contracts. Units for the same quantity
//! share a private module when they have multiple scales. The concrete types are available
//! from their family, this module, and the crate root.

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
mod tests;
