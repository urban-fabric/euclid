//! Electromagnetic dimensions: [`ElectricCurrent`], [`ElectricCharge`], [`Voltage`],
//! [`Resistance`], [`Conductance`], [`Capacitance`], [`Inductance`], [`MagneticFlux`],
//! [`MagneticFluxDensity`], [`ElectricField`], [`CurrentDensity`], [`MagneticFieldStrength`],
//! [`Permittivity`], [`Permeability`], [`Resistivity`], [`Conductivity`].
//!
//! These carry the circuit and field algebra — Ohm's law (`current × resistance = voltage`),
//! electrical power (`voltage × current = power`), `charge = current × time`, capacitor and
//! inductor relations, and Faraday's `voltage = dΦ/dt`. [`ElectricCurrent`] is the SI base
//! dimension `[I]`. See the [`geometry`](super::geometry) module docs for the naming convention.

use super::{Area, Energy, Force, Length, LinearQuantity, Power, Time};

/// An electric-current dimensioned quantity (`[I]`, the SI base dimension, the ampere).
/// `current × time = charge`, `current × resistance = voltage`, `current × voltage = power`,
/// `current / area = current density`.
pub trait ElectricCurrent: LinearQuantity {
    /// The time that pairs with this current.
    type Time: Time<Repr = Self::Repr>;
    /// The charge produced by `current × time`.
    type ElectricCharge: ElectricCharge<Repr = Self::Repr>;
    /// The resistance that pairs with this current.
    type Resistance: Resistance<Repr = Self::Repr>;
    /// The voltage produced by `current × resistance`.
    type Voltage: Voltage<Repr = Self::Repr>;
    /// The power produced by `current × voltage`.
    type Power: Power<Repr = Self::Repr>;
    /// The inductance that pairs with this current.
    type Inductance: Inductance<Repr = Self::Repr>;
    /// The magnetic flux produced by `current × inductance`.
    type MagneticFlux: MagneticFlux<Repr = Self::Repr>;
    /// The area that pairs with this current.
    type Area: Area<Repr = Self::Repr>;
    /// The current density produced by `current / area`.
    type CurrentDensity: CurrentDensity<Repr = Self::Repr>;

    /// `current × time = charge`.
    fn times_time(self, elapsed: Self::Time) -> Self::ElectricCharge;
    /// `current × resistance = voltage` (Ohm's law).
    fn times_resistance(self, resistance: Self::Resistance) -> Self::Voltage;
    /// `current × voltage = power`.
    fn times_voltage(self, voltage: Self::Voltage) -> Self::Power;
    /// `current × inductance = magnetic flux`.
    fn times_inductance(self, inductance: Self::Inductance) -> Self::MagneticFlux;
    /// `current / area = current density`.
    fn per_area(self, area: Self::Area) -> Self::CurrentDensity;
}

/// An electric-charge dimensioned quantity (`[T I]`, the coulomb). `charge / time = current`,
/// `charge × voltage = energy`, `charge / capacitance = voltage`.
pub trait ElectricCharge: LinearQuantity {
    /// The time that pairs with this charge.
    type Time: Time<Repr = Self::Repr>;
    /// The current produced by `charge / time`.
    type ElectricCurrent: ElectricCurrent<Repr = Self::Repr>;
    /// The voltage that pairs with this charge.
    type Voltage: Voltage<Repr = Self::Repr>;
    /// The energy produced by `charge × voltage`.
    type Energy: Energy<Repr = Self::Repr>;
    /// The capacitance that pairs with this charge.
    type Capacitance: Capacitance<Repr = Self::Repr>;
    /// The electric field that pairs with this charge.
    type ElectricField: ElectricField<Repr = Self::Repr>;
    /// The force produced by `charge × electric field`.
    type Force: Force<Repr = Self::Repr>;

    /// `charge / time = current`.
    fn per_time(self, elapsed: Self::Time) -> Self::ElectricCurrent;
    /// `charge / current = time`.
    fn per_current(self, current: Self::ElectricCurrent) -> Self::Time;
    /// `charge × voltage = energy`.
    fn times_voltage(self, voltage: Self::Voltage) -> Self::Energy;
    /// `charge / capacitance = voltage`.
    fn per_capacitance(self, capacitance: Self::Capacitance) -> Self::Voltage;
    /// `charge / voltage = capacitance`.
    fn per_voltage(self, voltage: Self::Voltage) -> Self::Capacitance;
    /// `charge × electric field = force`.
    fn times_electric_field(self, field: Self::ElectricField) -> Self::Force;
}

/// A voltage (electric potential) dimensioned quantity (`[L² M T⁻³ I⁻¹]`, the volt).
/// `voltage / current = resistance`, `voltage × current = power`, `voltage × charge = energy`,
/// `voltage / length = electric field`, `voltage × time = magnetic flux`.
pub trait Voltage: LinearQuantity {
    /// The current that pairs with this voltage.
    type ElectricCurrent: ElectricCurrent<Repr = Self::Repr>;
    /// The resistance produced by `voltage / current`.
    type Resistance: Resistance<Repr = Self::Repr>;
    /// The power produced by `voltage × current`.
    type Power: Power<Repr = Self::Repr>;
    /// The charge that pairs with this voltage.
    type ElectricCharge: ElectricCharge<Repr = Self::Repr>;
    /// The energy produced by `voltage × charge`.
    type Energy: Energy<Repr = Self::Repr>;
    /// The length that pairs with this voltage.
    type Length: Length<Repr = Self::Repr>;
    /// The electric field produced by `voltage / length`.
    type ElectricField: ElectricField<Repr = Self::Repr>;
    /// The time that pairs with this voltage.
    type Time: Time<Repr = Self::Repr>;
    /// The magnetic flux produced by `voltage × time`.
    type MagneticFlux: MagneticFlux<Repr = Self::Repr>;

    /// `voltage / current = resistance` (Ohm's law, solved for `R`).
    fn per_current(self, current: Self::ElectricCurrent) -> Self::Resistance;
    /// `voltage / resistance = current` (Ohm's law, solved for `I`).
    fn per_resistance(self, resistance: Self::Resistance) -> Self::ElectricCurrent;
    /// `voltage × current = power`.
    fn times_current(self, current: Self::ElectricCurrent) -> Self::Power;
    /// `voltage × charge = energy`.
    fn times_charge(self, charge: Self::ElectricCharge) -> Self::Energy;
    /// `voltage / length = electric field`.
    fn per_length(self, length: Self::Length) -> Self::ElectricField;
    /// `voltage × time = magnetic flux` (Faraday's law, integrated).
    fn times_time(self, elapsed: Self::Time) -> Self::MagneticFlux;
}

/// A resistance dimensioned quantity (`[L² M T⁻³ I⁻²]`, the ohm). `current × resistance =
/// voltage`, `resistance × time = inductance`, and the reciprocal `1 / resistance =
/// conductance`.
pub trait Resistance: LinearQuantity {
    /// The current that pairs with this resistance.
    type ElectricCurrent: ElectricCurrent<Repr = Self::Repr>;
    /// The voltage produced by `resistance × current`.
    type Voltage: Voltage<Repr = Self::Repr>;
    /// The conductance that is this resistance's reciprocal.
    type Conductance: Conductance<Repr = Self::Repr>;
    /// The time that pairs with this resistance.
    type Time: Time<Repr = Self::Repr>;
    /// The inductance produced by `resistance × time`.
    type Inductance: Inductance<Repr = Self::Repr>;

    /// `resistance × current = voltage` (Ohm's law).
    fn times_current(self, current: Self::ElectricCurrent) -> Self::Voltage;
    /// `1 / resistance = conductance`.
    fn conductance(self) -> Self::Conductance;
    /// `resistance × time = inductance` (the `L = Rτ` relation).
    fn times_time(self, elapsed: Self::Time) -> Self::Inductance;
}

/// A conductance dimensioned quantity (`[L⁻² M⁻¹ T³ I²]`, the siemens). The reciprocal of
/// [`Resistance`].
pub trait Conductance: LinearQuantity {
    /// The resistance that is this conductance's reciprocal.
    type Resistance: Resistance<Repr = Self::Repr>;

    /// `1 / conductance = resistance`.
    fn resistance(self) -> Self::Resistance;
}

/// A capacitance dimensioned quantity (`[L⁻² M⁻¹ T⁴ I²]`, the farad). `capacitance × voltage =
/// charge`, `capacitance × resistance = time` (the `RC` time constant).
pub trait Capacitance: LinearQuantity {
    /// The voltage that pairs with this capacitance.
    type Voltage: Voltage<Repr = Self::Repr>;
    /// The charge produced by `capacitance × voltage`.
    type ElectricCharge: ElectricCharge<Repr = Self::Repr>;
    /// The resistance that pairs with this capacitance.
    type Resistance: Resistance<Repr = Self::Repr>;
    /// The time produced by `capacitance × resistance`.
    type Time: Time<Repr = Self::Repr>;

    /// `capacitance × voltage = charge`.
    fn times_voltage(self, voltage: Self::Voltage) -> Self::ElectricCharge;
    /// `capacitance × resistance = time` (the `RC` time constant).
    fn times_resistance(self, resistance: Self::Resistance) -> Self::Time;
}

/// An inductance dimensioned quantity (`[L² M T⁻² I⁻²]`, the henry). `inductance × current =
/// magnetic flux`, `inductance / resistance = time` (the `L/R` time constant).
pub trait Inductance: LinearQuantity {
    /// The current that pairs with this inductance.
    type ElectricCurrent: ElectricCurrent<Repr = Self::Repr>;
    /// The magnetic flux produced by `inductance × current`.
    type MagneticFlux: MagneticFlux<Repr = Self::Repr>;
    /// The resistance that pairs with this inductance.
    type Resistance: Resistance<Repr = Self::Repr>;
    /// The time produced by `inductance / resistance`.
    type Time: Time<Repr = Self::Repr>;

    /// `inductance × current = magnetic flux`.
    fn times_current(self, current: Self::ElectricCurrent) -> Self::MagneticFlux;
    /// `inductance / resistance = time` (the `L/R` time constant).
    fn per_resistance(self, resistance: Self::Resistance) -> Self::Time;
    /// `inductance / time = resistance`.
    fn per_time(self, elapsed: Self::Time) -> Self::Resistance;
}

/// A magnetic-flux dimensioned quantity (`[L² M T⁻² I⁻¹]`, the weber). `flux / current =
/// inductance`, `flux / time = voltage`, `flux / area = magnetic flux density`.
pub trait MagneticFlux: LinearQuantity {
    /// The current that pairs with this flux.
    type ElectricCurrent: ElectricCurrent<Repr = Self::Repr>;
    /// The inductance produced by `flux / current`.
    type Inductance: Inductance<Repr = Self::Repr>;
    /// The time that pairs with this flux.
    type Time: Time<Repr = Self::Repr>;
    /// The voltage produced by `flux / time`.
    type Voltage: Voltage<Repr = Self::Repr>;
    /// The area that pairs with this flux.
    type Area: Area<Repr = Self::Repr>;
    /// The magnetic flux density produced by `flux / area`.
    type MagneticFluxDensity: MagneticFluxDensity<Repr = Self::Repr>;

    /// `flux / current = inductance`.
    fn per_current(self, current: Self::ElectricCurrent) -> Self::Inductance;
    /// `flux / time = voltage` (Faraday's law of induction).
    fn per_time(self, elapsed: Self::Time) -> Self::Voltage;
    /// `flux / area = magnetic flux density`.
    fn per_area(self, area: Self::Area) -> Self::MagneticFluxDensity;
}

/// A magnetic-flux-density dimensioned quantity (`[M T⁻² I⁻¹]`, the tesla). `flux density × area
/// = magnetic flux`.
pub trait MagneticFluxDensity: LinearQuantity {
    /// The area that pairs with this flux density.
    type Area: Area<Repr = Self::Repr>;
    /// The magnetic flux produced by `flux density × area`.
    type MagneticFlux: MagneticFlux<Repr = Self::Repr>;

    /// `flux density × area = magnetic flux`.
    fn times_area(self, area: Self::Area) -> Self::MagneticFlux;
}

/// An electric-field dimensioned quantity (`[L M T⁻³ I⁻¹]`, volts per metre). `field × length =
/// voltage`, `field × charge = force`.
pub trait ElectricField: LinearQuantity {
    /// The length that pairs with this field.
    type Length: Length<Repr = Self::Repr>;
    /// The voltage produced by `field × length`.
    type Voltage: Voltage<Repr = Self::Repr>;
    /// The charge that pairs with this field.
    type ElectricCharge: ElectricCharge<Repr = Self::Repr>;
    /// The force produced by `field × charge`.
    type Force: Force<Repr = Self::Repr>;

    /// `field × length = voltage`.
    fn times_length(self, length: Self::Length) -> Self::Voltage;
    /// `field × charge = force`.
    fn times_charge(self, charge: Self::ElectricCharge) -> Self::Force;
}

/// A current-density dimensioned quantity (`[L⁻² I]`, amperes per square metre). `current density ×
/// area = current`.
pub trait CurrentDensity: LinearQuantity {
    /// The area that pairs with this current density.
    type Area: Area<Repr = Self::Repr>;
    /// The current produced by `current density × area`.
    type ElectricCurrent: ElectricCurrent<Repr = Self::Repr>;

    /// `current density × area = current`.
    fn times_area(self, area: Self::Area) -> Self::ElectricCurrent;
}

/// A magnetic-field-strength (magnetizing field, `H`) dimensioned quantity (`[L⁻¹ I]`, amperes per
/// metre). `field strength × length = current` (Ampère's circuital law),
/// `field strength × permeability = magnetic flux density` (`B = μH`).
pub trait MagneticFieldStrength: LinearQuantity {
    /// The length that pairs with this field strength.
    type Length: Length<Repr = Self::Repr>;
    /// The current produced by `field strength × length`.
    type ElectricCurrent: ElectricCurrent<Repr = Self::Repr>;
    /// The permeability that pairs with this field strength.
    type Permeability: Permeability<Repr = Self::Repr>;
    /// The magnetic flux density produced by `field strength × permeability`.
    type MagneticFluxDensity: MagneticFluxDensity<Repr = Self::Repr>;

    /// `field strength × length = current` (Ampère's circuital law).
    fn times_length(self, length: Self::Length) -> Self::ElectricCurrent;
    /// `field strength × permeability = magnetic flux density` (`B = μH`).
    fn times_permeability(self, permeability: Self::Permeability) -> Self::MagneticFluxDensity;
}

/// A permittivity dimensioned quantity (`[L⁻³ M⁻¹ T⁴ I²]`, farads per metre). The electric
/// constant of a medium. `permittivity × length = capacitance`.
pub trait Permittivity: LinearQuantity {
    /// The length that pairs with this permittivity.
    type Length: Length<Repr = Self::Repr>;
    /// The capacitance produced by `permittivity × length`.
    type Capacitance: Capacitance<Repr = Self::Repr>;

    /// `permittivity × length = capacitance` (the parallel-plate relation, per unit aspect ratio).
    fn times_length(self, length: Self::Length) -> Self::Capacitance;
}

/// A permeability dimensioned quantity (`[L M T⁻² I⁻²]`, henries per metre). The magnetic constant
/// of a medium. `permeability × length = inductance`, `permeability × field strength = magnetic
/// flux density` (`B = μH`).
pub trait Permeability: LinearQuantity {
    /// The length that pairs with this permeability.
    type Length: Length<Repr = Self::Repr>;
    /// The inductance produced by `permeability × length`.
    type Inductance: Inductance<Repr = Self::Repr>;
    /// The magnetic field strength that pairs with this permeability.
    type MagneticFieldStrength: MagneticFieldStrength<Repr = Self::Repr>;
    /// The magnetic flux density produced by `permeability × field strength`.
    type MagneticFluxDensity: MagneticFluxDensity<Repr = Self::Repr>;

    /// `permeability × length = inductance` (per unit aspect ratio).
    fn times_length(self, length: Self::Length) -> Self::Inductance;
    /// `permeability × field strength = magnetic flux density` (`B = μH`).
    fn times_field_strength(
        self,
        field_strength: Self::MagneticFieldStrength,
    ) -> Self::MagneticFluxDensity;
}

/// An (electrical) resistivity dimensioned quantity (`[L³ M T⁻³ I⁻²]`, ohm-metres). The bulk
/// resistance of a material, independent of geometry. `resistivity × current density = electric
/// field` (the field form of Ohm's law, `E = ρJ`), and the reciprocal `1 / resistivity =
/// conductivity`.
pub trait Resistivity: LinearQuantity {
    /// The current density that pairs with this resistivity.
    type CurrentDensity: CurrentDensity<Repr = Self::Repr>;
    /// The electric field produced by `resistivity × current density`.
    type ElectricField: ElectricField<Repr = Self::Repr>;
    /// The conductivity that is this resistivity's reciprocal.
    type Conductivity: Conductivity<Repr = Self::Repr>;

    /// `resistivity × current density = electric field` (`E = ρJ`).
    fn times_current_density(self, current_density: Self::CurrentDensity) -> Self::ElectricField;
    /// `1 / resistivity = conductivity`.
    fn conductivity(self) -> Self::Conductivity;
}

/// An (electrical) conductivity dimensioned quantity (`[L⁻³ M⁻¹ T³ I²]`, siemens per metre). The
/// reciprocal of [`Resistivity`]. `conductivity × electric field = current density` (`J = σE`).
pub trait Conductivity: LinearQuantity {
    /// The electric field that pairs with this conductivity.
    type ElectricField: ElectricField<Repr = Self::Repr>;
    /// The current density produced by `conductivity × electric field`.
    type CurrentDensity: CurrentDensity<Repr = Self::Repr>;
    /// The resistivity that is this conductivity's reciprocal.
    type Resistivity: Resistivity<Repr = Self::Repr>;

    /// `conductivity × electric field = current density` (`J = σE`).
    fn times_electric_field(self, field: Self::ElectricField) -> Self::CurrentDensity;
    /// `1 / conductivity = resistivity`.
    fn resistivity(self) -> Self::Resistivity;
}
