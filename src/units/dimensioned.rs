//! Type-level dimension markers and the [`Dimensioned`] bridge to the runtime [`Dimension`] enum.
//!
//! The [`Dimension`] enum and the dimension *traits* ([`Length`](super::Length),
//! [`Force`](super::Force), …) name dimensions at runtime and in trait position respectively.
//! The [`dim`] markers add the third form: zero-sized *type-level* tags usable as a generic
//! parameter or associated type, so a unit-of-measure protocol can enforce "same dimension" at
//! compile time (`convert::<To>()` only typechecks when `To::Dim == Self::Dim`).
//!
//! [`Dimensioned`] is the canonical bridge: it maps each marker to its runtime [`Dimension`] (and,
//! for free, its SI [`BaseExponents`]), so a consumer that declares a dimension twice — once as a
//! type-level marker, once as a runtime [`Quantity::DIMENSION`](super::Quantity::DIMENSION) — can
//! assert the two agree instead of letting them drift.

use super::{BaseExponents, Dimension};

/// The bridge from a type-level [`dim`] marker to its runtime [`Dimension`].
pub trait Dimensioned: Copy + 'static {
    /// The runtime dimension this marker stands for.
    const DIMENSION: Dimension;
    /// The SI base-dimension exponents of this marker, derived from [`DIMENSION`](Self::DIMENSION).
    const EXPONENTS: BaseExponents = Self::DIMENSION.exponents();
}

macro_rules! markers {
    ($($name:ident),+ $(,)?) => {
        /// Zero-sized type-level markers, one per [`Dimension`] variant. Two units are
        /// inter-convertible only when they share the same marker, which the compiler enforces.
        pub mod dim {
            $(
                #[doc = concat!("Type-level marker for the `", stringify!($name), "` dimension.")]
                #[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
                pub struct $name;
            )+
        }

        $(
            impl Dimensioned for dim::$name {
                const DIMENSION: Dimension = Dimension::$name;
            }
        )+
    };
}

markers!(
    // base
    Length,
    Mass,
    Time,
    ElectricCurrent,
    Temperature,
    AmountOfSubstance,
    LuminousIntensity,
    // geometric
    Area,
    Volume,
    Angle,
    SolidAngle,
    // kinematic
    Frequency,
    Speed,
    Acceleration,
    Jerk,
    AngularVelocity,
    AngularAcceleration,
    VolumetricFlowRate,
    // mechanical
    Force,
    Pressure,
    Energy,
    Power,
    Momentum,
    AngularMomentum,
    Torque,
    Density,
    MomentOfInertia,
    DynamicViscosity,
    SurfaceTension,
    MassFlowRate,
    KinematicViscosity,
    SpecificEnergy,
    EnergyDensity,
    Action,
    // electromagnetic
    ElectricCharge,
    Voltage,
    Resistance,
    Conductance,
    Capacitance,
    Inductance,
    MagneticFlux,
    MagneticFluxDensity,
    ElectricField,
    CurrentDensity,
    MagneticFieldStrength,
    Permittivity,
    Permeability,
    Resistivity,
    Conductivity,
    // thermal
    HeatCapacity,
    ThermalConductivity,
    SpecificHeatCapacity,
    ThermalResistance,
    HeatFlux,
    // photometric
    LuminousFlux,
    Illuminance,
    Luminance,
    // chemical
    CatalyticActivity,
    Concentration,
    MolarMass,
    MolarEnergy,
    MolarVolume,
    Molality,
    // radiological
    Radioactivity,
    AbsorbedDose,
    DoseEquivalent,
    // pure number
    Dimensionless,
);
