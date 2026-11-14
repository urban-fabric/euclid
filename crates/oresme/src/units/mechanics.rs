//! Mechanical units: mass, force, pressure, energy, power, momentum, torque, and friends.

mod angular_momentum;
mod density;
mod dynamic_viscosity;
mod energy;
mod force;
mod mass_flow_rate;
mod mass_grams;
mod mass_kilograms;
mod mass_milligrams;
mod moment_of_inertia;
mod momentum;
mod power;
mod pressure;
mod specific_energy;
mod surface_tension;
mod torque;

pub use angular_momentum::*;
pub use density::*;
pub use dynamic_viscosity::*;
pub use energy::*;
pub use force::*;
pub use mass_flow_rate::*;
pub use mass_grams::*;
pub use mass_kilograms::*;
pub use mass_milligrams::*;
pub use moment_of_inertia::*;
pub use momentum::*;
pub use power::*;
pub use pressure::*;
pub use specific_energy::*;
pub use surface_tension::*;
pub use torque::*;

pub(crate) const KG_PER_GRAM: f64 = 0.001;
pub(crate) const KG_PER_MILLIGRAM: f64 = 0.000_001;

#[cfg(test)]
mod mass_tests {
    use crate::{MassGrams, MassKilograms, MassMilligrams, Quantity};

    #[test]
    fn mass_round_trips() {
        let kg = MassKilograms::new(2.5);
        assert_eq!(kg.to_grams(), MassGrams::new(2500.0));
        assert_eq!(kg.to_milligrams(), MassMilligrams::new(2_500_000.0));
        assert_eq!(MassGrams::new(2500.0).to_kilograms(), kg);
        assert_eq!(MassMilligrams::new(2_500_000.0).to_kilograms(), kg);
    }

    #[test]
    fn gram_and_milligram_conversions() {
        assert_eq!(
            MassGrams::new(1.25).to_milligrams(),
            MassMilligrams::new(1250.0)
        );
        assert_eq!(MassMilligrams::new(1250.0).to_grams(), MassGrams::new(1.25));
    }

    #[test]
    fn mass_quantity_impls() {
        assert_eq!(MassKilograms::DIMENSION, crate::Dimension::Mass);
        assert_eq!(MassKilograms::UNIT_NAME, "kg");
        assert_eq!(MassGrams::UNIT_NAME, "g");
        assert_eq!(MassMilligrams::UNIT_NAME, "mg");
    }
}
