//! Dimensions laws for the toy unit carriers.

use super::*;

#[test]
fn base_dimension_signatures() {
    // [L, M, T, I, Θ, N, J]
    let len: BaseExponents = Dimension::Length.exponents();
    assert_eq!(len, [1, 0, 0, 0, 0, 0, 0]);
    assert_eq!(Dimension::Force.exponents(), [1, 1, -2, 0, 0, 0, 0]);
    assert_eq!(Dimension::Voltage.exponents(), [2, 1, -3, -1, 0, 0, 0]);

    // energy and torque share a base signature but are distinct names.
    assert!(Dimension::Energy.commensurable(Dimension::Torque));
    assert!(!Dimension::Energy.same(Dimension::Torque));

    // the same pattern recurs among the new derived names.
    assert!(Dimension::EnergyDensity.commensurable(Dimension::Pressure));
    assert!(!Dimension::EnergyDensity.same(Dimension::Pressure));
    assert!(Dimension::Action.commensurable(Dimension::AngularMomentum));
    assert!(!Dimension::Action.same(Dimension::AngularMomentum));
    assert!(Dimension::Luminance.commensurable(Dimension::Illuminance));
    assert!(!Dimension::Luminance.same(Dimension::Illuminance));

    // a couple of the new derived signatures spelled out explicitly.
    assert_eq!(Dimension::Permittivity.exponents(), [-3, -1, 4, 2, 0, 0, 0]);
    assert_eq!(Dimension::MolarEnergy.exponents(), [2, 1, -2, 0, 0, -1, 0]);
    assert_eq!(Dimension::AbsorbedDose.exponents(), [2, 0, -2, 0, 0, 0, 0]);

    // frequency, angular velocity, and radioactivity are all commensurable (`[T⁻¹]`).
    assert!(Dimension::Frequency.commensurable(Dimension::AngularVelocity));
    assert!(Dimension::Frequency.commensurable(Dimension::Radioactivity));

    // angles read as dimensionless in SI even though they are tracked names.
    assert!(Dimension::Angle.is_dimensionless());
    assert!(Dimension::SolidAngle.is_dimensionless());
    assert!(!Dimension::Length.is_dimensionless());
    assert!(!Dimension::Length.commensurable(Dimension::Time));
}

/// Every derived dimension's exponents, asserted as the **definition** that produced them.
///
/// The existing `base_dimension_signatures` spells out five signatures by hand, and mutation
/// testing showed what the other sixty cost: ten sign flips in the exponent table survived the
/// whole workspace suite. `Jerk` could become `[1, 0, +3]`, `Conductance` could stop being the
/// reciprocal of `Resistance`, and `ThermalConductivity` could lose its temperature exponent,
/// with nothing to say so.
///
/// That table is the semantic core of the crate — `commensurable`, `is_dimensionless` and every
/// marker's `EXPONENTS` are derived from it — so a sign error there is a silently wrong unit
/// system rather than a cosmetic slip.
///
/// # Why definitions rather than a second copy of the table
///
/// Writing the sixty vectors out again would be a transcription, and a transcription copies the
/// typo it is meant to catch. Each line below is instead a *physical definition* — pressure is
/// force per area, resistance is voltage per current, concentration is amount per volume — and
/// an exponent vector either satisfies its definition or it does not. The assertions are
/// therefore independent of the table and of each other's mistakes.
#[test]
fn every_derived_dimension_matches_its_physical_definition() {
    use Dimension as D;

    /// The dimension of a product: exponents add.
    fn times(a: BaseExponents, b: BaseExponents) -> BaseExponents {
        let mut out = a;
        for (slot, rhs) in out.iter_mut().zip(b) {
            *slot += rhs;
        }
        out
    }
    /// The dimension of a quotient: exponents subtract.
    fn over(a: BaseExponents, b: BaseExponents) -> BaseExponents {
        let mut out = a;
        for (slot, rhs) in out.iter_mut().zip(b) {
            *slot -= rhs;
        }
        out
    }
    let of = D::exponents;
    let one = of(D::Dimensionless);

    // --- geometry ---
    assert_eq!(of(D::Area), times(of(D::Length), of(D::Length)));
    assert_eq!(of(D::Volume), times(of(D::Area), of(D::Length)));

    // --- kinematics ---
    assert_eq!(of(D::Frequency), over(one, of(D::Time)));
    assert_eq!(of(D::Speed), over(of(D::Length), of(D::Time)));
    assert_eq!(of(D::Acceleration), over(of(D::Speed), of(D::Time)));
    assert_eq!(of(D::Jerk), over(of(D::Acceleration), of(D::Time)));
    assert_eq!(of(D::AngularVelocity), over(of(D::Angle), of(D::Time)));
    assert_eq!(
        of(D::AngularAcceleration),
        over(of(D::AngularVelocity), of(D::Time))
    );
    assert_eq!(of(D::VolumetricFlowRate), over(of(D::Volume), of(D::Time)));

    // --- mechanics ---
    assert_eq!(of(D::Force), times(of(D::Mass), of(D::Acceleration)));
    assert_eq!(of(D::Pressure), over(of(D::Force), of(D::Area)));
    assert_eq!(of(D::Energy), times(of(D::Force), of(D::Length)));
    assert_eq!(of(D::Power), over(of(D::Energy), of(D::Time)));
    assert_eq!(of(D::Momentum), times(of(D::Mass), of(D::Speed)));
    assert_eq!(of(D::MomentOfInertia), times(of(D::Mass), of(D::Area)));
    assert_eq!(
        of(D::AngularMomentum),
        times(of(D::MomentOfInertia), of(D::AngularVelocity))
    );
    assert_eq!(of(D::Torque), times(of(D::Force), of(D::Length)));
    assert_eq!(of(D::Density), over(of(D::Mass), of(D::Volume)));
    assert_eq!(of(D::DynamicViscosity), times(of(D::Pressure), of(D::Time)));
    assert_eq!(of(D::SurfaceTension), over(of(D::Force), of(D::Length)));
    assert_eq!(of(D::MassFlowRate), over(of(D::Mass), of(D::Time)));
    assert_eq!(of(D::KinematicViscosity), over(of(D::Area), of(D::Time)));
    assert_eq!(of(D::SpecificEnergy), over(of(D::Energy), of(D::Mass)));
    assert_eq!(of(D::EnergyDensity), over(of(D::Energy), of(D::Volume)));
    assert_eq!(of(D::Action), times(of(D::Energy), of(D::Time)));

    // --- electromagnetism ---
    assert_eq!(
        of(D::ElectricCharge),
        times(of(D::ElectricCurrent), of(D::Time))
    );
    assert_eq!(of(D::Voltage), over(of(D::Power), of(D::ElectricCurrent)));
    assert_eq!(
        of(D::Resistance),
        over(of(D::Voltage), of(D::ElectricCurrent))
    );
    assert_eq!(of(D::Conductance), over(one, of(D::Resistance)));
    assert_eq!(
        of(D::Capacitance),
        over(of(D::ElectricCharge), of(D::Voltage))
    );
    assert_eq!(of(D::MagneticFlux), times(of(D::Voltage), of(D::Time)));
    assert_eq!(
        of(D::Inductance),
        over(of(D::MagneticFlux), of(D::ElectricCurrent))
    );
    assert_eq!(
        of(D::MagneticFluxDensity),
        over(of(D::MagneticFlux), of(D::Area))
    );
    assert_eq!(of(D::ElectricField), over(of(D::Voltage), of(D::Length)));
    assert_eq!(
        of(D::CurrentDensity),
        over(of(D::ElectricCurrent), of(D::Area))
    );
    assert_eq!(
        of(D::MagneticFieldStrength),
        over(of(D::ElectricCurrent), of(D::Length))
    );
    assert_eq!(of(D::Permittivity), over(of(D::Capacitance), of(D::Length)));
    assert_eq!(of(D::Permeability), over(of(D::Inductance), of(D::Length)));
    assert_eq!(of(D::Resistivity), times(of(D::Resistance), of(D::Length)));
    assert_eq!(of(D::Conductivity), over(one, of(D::Resistivity)));

    // --- thermal ---
    assert_eq!(of(D::HeatCapacity), over(of(D::Energy), of(D::Temperature)));
    assert_eq!(
        of(D::ThermalConductivity),
        over(of(D::Power), times(of(D::Length), of(D::Temperature)))
    );
    assert_eq!(
        of(D::SpecificHeatCapacity),
        over(of(D::HeatCapacity), of(D::Mass))
    );
    assert_eq!(
        of(D::ThermalResistance),
        over(of(D::Temperature), of(D::Power))
    );
    assert_eq!(of(D::HeatFlux), over(of(D::Power), of(D::Area)));

    // --- photometry ---
    assert_eq!(
        of(D::LuminousFlux),
        times(of(D::LuminousIntensity), of(D::SolidAngle))
    );
    assert_eq!(of(D::Illuminance), over(of(D::LuminousFlux), of(D::Area)));
    assert_eq!(
        of(D::Luminance),
        over(of(D::LuminousIntensity), of(D::Area))
    );

    // --- chemistry ---
    assert_eq!(
        of(D::CatalyticActivity),
        over(of(D::AmountOfSubstance), of(D::Time))
    );
    assert_eq!(
        of(D::Concentration),
        over(of(D::AmountOfSubstance), of(D::Volume))
    );
    assert_eq!(
        of(D::MolarMass),
        over(of(D::Mass), of(D::AmountOfSubstance))
    );
    assert_eq!(
        of(D::MolarEnergy),
        over(of(D::Energy), of(D::AmountOfSubstance))
    );
    assert_eq!(
        of(D::MolarVolume),
        over(of(D::Volume), of(D::AmountOfSubstance))
    );
    assert_eq!(of(D::Molality), over(of(D::AmountOfSubstance), of(D::Mass)));

    // --- radiology ---
    assert_eq!(of(D::Radioactivity), over(one, of(D::Time)));
    assert_eq!(of(D::AbsorbedDose), over(of(D::Energy), of(D::Mass)));
    assert_eq!(of(D::DoseEquivalent), over(of(D::Energy), of(D::Mass)));

    // --- the dimensionless ones, which must really be dimensionless ---
    for d in [D::Angle, D::SolidAngle, D::Information, D::Dimensionless] {
        assert_eq!(of(d), [0; 7], "{d:?} must reduce to a pure number");
    }

    // And the base dimensions are the unit vectors, in the documented order.
    assert_eq!(of(D::Length), [1, 0, 0, 0, 0, 0, 0]);
    assert_eq!(of(D::Mass), [0, 1, 0, 0, 0, 0, 0]);
    assert_eq!(of(D::Time), [0, 0, 1, 0, 0, 0, 0]);
    assert_eq!(of(D::ElectricCurrent), [0, 0, 0, 1, 0, 0, 0]);
    assert_eq!(of(D::Temperature), [0, 0, 0, 0, 1, 0, 0]);
    assert_eq!(of(D::AmountOfSubstance), [0, 0, 0, 0, 0, 1, 0]);
    assert_eq!(of(D::LuminousIntensity), [0, 0, 0, 0, 0, 0, 1]);
}
