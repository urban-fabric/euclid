/// The physical dimension a quantity measures.
///
/// Every variant is expressed in the SI base dimensions
/// `[length, mass, time, current, temperature, amount, luminous_intensity]`
/// via [`Dimension::exponents`], so two distinct names that share a dimensional
/// signature (energy and torque, frequency and angular velocity) are
/// [`Dimension::commensurable`] even though they are not [`Dimension::same`].
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[non_exhaustive]
pub enum Dimension {
    // --- SI base dimensions ---
    Length,
    Mass,
    Time,
    ElectricCurrent,
    Temperature,
    AmountOfSubstance,
    LuminousIntensity,

    // --- geometric ---
    Area,
    Volume,
    Angle,
    SolidAngle,

    // --- kinematic ---
    Frequency,
    Speed,
    Acceleration,
    Jerk,
    AngularVelocity,
    AngularAcceleration,
    VolumetricFlowRate,

    // --- mechanical / dynamic ---
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

    // --- electromagnetic ---
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

    // --- thermal ---
    HeatCapacity,
    ThermalConductivity,
    SpecificHeatCapacity,
    ThermalResistance,
    HeatFlux,

    // --- photometric ---
    LuminousFlux,
    Illuminance,
    Luminance,

    // --- chemical ---
    CatalyticActivity,
    Concentration,
    MolarMass,
    MolarEnergy,
    MolarVolume,
    Molality,

    // --- radiological ---
    Radioactivity,
    AbsorbedDose,
    DoseEquivalent,

    // --- information ---
    /// Data size (bytes, bits). Dimensionless in SI base dimensions, but tracked as a distinct
    /// name so a byte count is not conflated with a bare number.
    Information,

    // --- pure number ---
    Dimensionless,
}

/// SI base-dimension exponents
/// `[length, mass, time, current, temperature, amount, luminous_intensity]`.
pub type BaseExponents = [i8; 7];

impl Dimension {
    /// Const-evaluable equality on the *named* dimension. The derived `PartialEq` cannot
    /// run in `const` context, where a unit's [`Quantity`](crate::units::Quantity) impl can
    /// assert that its runtime dimension matches its type-level marker.
    ///
    /// Distinct from [`commensurable`](Self::commensurable): `Energy` and `Torque` are not
    /// `same` (they are different physical roles) but they are `commensurable` (identical
    /// base-dimension signature).
    pub const fn same(self, other: Self) -> bool {
        self as u8 == other as u8
    }

    /// The SI base-dimension exponents of this dimension, ordered
    /// `[length, mass, time, current, temperature, amount, luminous_intensity]`.
    ///
    /// Angles and solid angles are dimensionless in SI (`[0; 7]`); they are tracked as
    /// distinct *names* via [`same`](Self::same) but read as dimensionless here.
    pub const fn exponents(self) -> BaseExponents {
        // [L, M, T, I, Θ, N, J]
        match self {
            Self::Length => [1, 0, 0, 0, 0, 0, 0],
            Self::Mass => [0, 1, 0, 0, 0, 0, 0],
            Self::Time => [0, 0, 1, 0, 0, 0, 0],
            Self::ElectricCurrent => [0, 0, 0, 1, 0, 0, 0],
            Self::Temperature => [0, 0, 0, 0, 1, 0, 0],
            Self::AmountOfSubstance => [0, 0, 0, 0, 0, 1, 0],
            Self::LuminousIntensity => [0, 0, 0, 0, 0, 0, 1],

            Self::Area => [2, 0, 0, 0, 0, 0, 0],
            Self::Volume => [3, 0, 0, 0, 0, 0, 0],
            Self::Angle => [0, 0, 0, 0, 0, 0, 0],
            Self::SolidAngle => [0, 0, 0, 0, 0, 0, 0],

            Self::Frequency => [0, 0, -1, 0, 0, 0, 0],
            Self::Speed => [1, 0, -1, 0, 0, 0, 0],
            Self::Acceleration => [1, 0, -2, 0, 0, 0, 0],
            Self::Jerk => [1, 0, -3, 0, 0, 0, 0],
            Self::AngularVelocity => [0, 0, -1, 0, 0, 0, 0],
            Self::AngularAcceleration => [0, 0, -2, 0, 0, 0, 0],
            Self::VolumetricFlowRate => [3, 0, -1, 0, 0, 0, 0],

            Self::Force => [1, 1, -2, 0, 0, 0, 0],
            Self::Pressure => [-1, 1, -2, 0, 0, 0, 0],
            Self::Energy => [2, 1, -2, 0, 0, 0, 0],
            Self::Power => [2, 1, -3, 0, 0, 0, 0],
            Self::Momentum => [1, 1, -1, 0, 0, 0, 0],
            Self::AngularMomentum => [2, 1, -1, 0, 0, 0, 0],
            Self::Torque => [2, 1, -2, 0, 0, 0, 0],
            Self::Density => [-3, 1, 0, 0, 0, 0, 0],
            Self::MomentOfInertia => [2, 1, 0, 0, 0, 0, 0],
            Self::DynamicViscosity => [-1, 1, -1, 0, 0, 0, 0],
            Self::SurfaceTension => [0, 1, -2, 0, 0, 0, 0],
            Self::MassFlowRate => [0, 1, -1, 0, 0, 0, 0],
            Self::KinematicViscosity => [2, 0, -1, 0, 0, 0, 0],
            Self::SpecificEnergy => [2, 0, -2, 0, 0, 0, 0],
            Self::EnergyDensity => [-1, 1, -2, 0, 0, 0, 0],
            Self::Action => [2, 1, -1, 0, 0, 0, 0],

            Self::ElectricCharge => [0, 0, 1, 1, 0, 0, 0],
            Self::Voltage => [2, 1, -3, -1, 0, 0, 0],
            Self::Resistance => [2, 1, -3, -2, 0, 0, 0],
            Self::Conductance => [-2, -1, 3, 2, 0, 0, 0],
            Self::Capacitance => [-2, -1, 4, 2, 0, 0, 0],
            Self::Inductance => [2, 1, -2, -2, 0, 0, 0],
            Self::MagneticFlux => [2, 1, -2, -1, 0, 0, 0],
            Self::MagneticFluxDensity => [0, 1, -2, -1, 0, 0, 0],
            Self::ElectricField => [1, 1, -3, -1, 0, 0, 0],
            Self::CurrentDensity => [-2, 0, 0, 1, 0, 0, 0],
            Self::MagneticFieldStrength => [-1, 0, 0, 1, 0, 0, 0],
            Self::Permittivity => [-3, -1, 4, 2, 0, 0, 0],
            Self::Permeability => [1, 1, -2, -2, 0, 0, 0],
            Self::Resistivity => [3, 1, -3, -2, 0, 0, 0],
            Self::Conductivity => [-3, -1, 3, 2, 0, 0, 0],

            Self::HeatCapacity => [2, 1, -2, 0, -1, 0, 0],
            Self::ThermalConductivity => [1, 1, -3, 0, -1, 0, 0],
            Self::SpecificHeatCapacity => [2, 0, -2, 0, -1, 0, 0],
            Self::ThermalResistance => [-2, -1, 3, 0, 1, 0, 0],
            Self::HeatFlux => [0, 1, -3, 0, 0, 0, 0],

            Self::LuminousFlux => [0, 0, 0, 0, 0, 0, 1],
            Self::Illuminance => [-2, 0, 0, 0, 0, 0, 1],
            Self::Luminance => [-2, 0, 0, 0, 0, 0, 1],

            Self::CatalyticActivity => [0, 0, -1, 0, 0, 1, 0],
            Self::Concentration => [-3, 0, 0, 0, 0, 1, 0],
            Self::MolarMass => [0, 1, 0, 0, 0, -1, 0],
            Self::MolarEnergy => [2, 1, -2, 0, 0, -1, 0],
            Self::MolarVolume => [3, 0, 0, 0, 0, -1, 0],
            Self::Molality => [0, -1, 0, 0, 0, 1, 0],

            Self::Radioactivity => [0, 0, -1, 0, 0, 0, 0],
            Self::AbsorbedDose => [2, 0, -2, 0, 0, 0, 0],
            Self::DoseEquivalent => [2, 0, -2, 0, 0, 0, 0],

            Self::Information => [0, 0, 0, 0, 0, 0, 0],

            Self::Dimensionless => [0, 0, 0, 0, 0, 0, 0],
        }
    }

    /// Whether two dimensions share an SI base-dimension signature, even if their *names*
    /// differ. Energy and torque are `commensurable` (both `[2, 1, -2, …]`) but not
    /// [`same`](Self::same); a length and a time are neither.
    pub const fn commensurable(self, other: Self) -> bool {
        let a = self.exponents();
        let b = other.exponents();
        let mut i = 0;
        while i < 7 {
            if a[i] != b[i] {
                return false;
            }
            i += 1;
        }
        true
    }

    /// Whether this dimension reduces to a pure number in SI base dimensions. True for
    /// [`Dimensionless`](Self::Dimensionless), [`Angle`](Self::Angle), and
    /// [`SolidAngle`](Self::SolidAngle).
    pub const fn is_dimensionless(self) -> bool {
        self.commensurable(Self::Dimensionless)
    }
}
