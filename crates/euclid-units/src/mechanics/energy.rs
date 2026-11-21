//! [`Energy`], [`Power`], [`Action`], [`SpecificEnergy`], [`EnergyDensity`] — work and its
//! rates and densities.
//!
//! `force × length = energy` enters from [`Force`], and everything here divides it by something:
//! by time for power, by mass for specific energy, by volume for energy density. [`Action`]
//! multiplies instead, by time.
//!
//! [`Energy`] shares a base-dimension signature with [`Torque`] —
//! both `[L² M T⁻²]` — but they are distinct names reached through different chains, which is
//! the distinction the parent's header explains.

use super::*;

/// An energy / work / heat dimensioned quantity (`[L² M T⁻²]`). `energy / length = force`,
/// `energy / time = power`, `energy / volume = pressure`, `energy / charge = voltage`,
/// `energy / temperature = heat capacity`, `energy / mass = specific energy`,
/// `energy / amount = molar energy`.
pub trait Energy: LinearQuantity {
    /// The force that pairs with this energy.
    type Force: Force<Repr = Self::Repr>;
    /// The length that pairs with this energy.
    type Length: Length<Repr = Self::Repr>;
    /// The time that pairs with this energy.
    type Time: Time<Repr = Self::Repr>;
    /// The power produced by `energy / time`.
    type Power: Power<Repr = Self::Repr>;
    /// The volume that pairs with this energy.
    type Volume: Volume<Repr = Self::Repr>;
    /// The pressure produced by `energy / volume`.
    type Pressure: Pressure<Repr = Self::Repr>;
    /// The electric charge that pairs with this energy.
    type ElectricCharge: ElectricCharge<Repr = Self::Repr>;
    /// The voltage produced by `energy / charge`.
    type Voltage: Voltage<Repr = Self::Repr>;
    /// The temperature that pairs with this energy.
    type Temperature: Temperature<Repr = Self::Repr>;
    /// The heat capacity produced by `energy / temperature`.
    type HeatCapacity: HeatCapacity<Repr = Self::Repr>;
    /// The mass that pairs with this energy.
    type Mass: Mass<Repr = Self::Repr>;
    /// The specific energy produced by `energy / mass`.
    type SpecificEnergy: SpecificEnergy<Repr = Self::Repr>;
    /// The amount of substance that pairs with this energy.
    type AmountOfSubstance: AmountOfSubstance<Repr = Self::Repr>;
    /// The molar energy produced by `energy / amount of substance`.
    type MolarEnergy: MolarEnergy<Repr = Self::Repr>;

    /// `energy / length = force`.
    fn per_length(self, distance: Self::Length) -> Self::Force;
    /// `energy / force = length`.
    fn per_force(self, force: Self::Force) -> Self::Length;
    /// `energy / time = power`.
    fn per_time(self, elapsed: Self::Time) -> Self::Power;
    /// `energy / power = time`.
    fn per_power(self, power: Self::Power) -> Self::Time;
    /// `energy / volume = pressure` (energy density).
    fn per_volume(self, volume: Self::Volume) -> Self::Pressure;
    /// `energy / charge = voltage`.
    fn per_charge(self, charge: Self::ElectricCharge) -> Self::Voltage;
    /// `energy / temperature = heat capacity`. The temperature is read as an interval in the
    /// carrier's units.
    fn per_temperature(self, temperature: Self::Temperature) -> Self::HeatCapacity;
    /// `energy / mass = specific energy`.
    fn per_mass(self, mass: Self::Mass) -> Self::SpecificEnergy;
    /// `energy / amount of substance = molar energy`.
    fn per_amount(self, amount: Self::AmountOfSubstance) -> Self::MolarEnergy;
}

/// A power-dimensioned quantity (`[L² M T⁻³]`). `power × time = energy`, `power / force = speed`,
/// `power / current = voltage`, `power / area = heat flux`.
pub trait Power: LinearQuantity {
    /// The time that pairs with this power.
    type Time: Time<Repr = Self::Repr>;
    /// The energy produced by `power × time`.
    type Energy: Energy<Repr = Self::Repr>;
    /// The force that pairs with this power.
    type Force: Force<Repr = Self::Repr>;
    /// The speed produced by `power / force`.
    type Speed: Speed<Repr = Self::Repr>;
    /// The electric current that pairs with this power.
    type ElectricCurrent: ElectricCurrent<Repr = Self::Repr>;
    /// The voltage produced by `power / current`.
    type Voltage: Voltage<Repr = Self::Repr>;
    /// The area that pairs with this power.
    type Area: Area<Repr = Self::Repr>;
    /// The heat flux produced by `power / area`.
    type HeatFlux: HeatFlux<Repr = Self::Repr>;

    /// `power × time = energy`.
    fn times_time(self, elapsed: Self::Time) -> Self::Energy;
    /// `power / area = heat flux` (heat-flux density).
    fn per_area(self, area: Self::Area) -> Self::HeatFlux;
    /// `power / force = speed`.
    fn per_force(self, force: Self::Force) -> Self::Speed;
    /// `power / speed = force`.
    fn per_speed(self, speed: Self::Speed) -> Self::Force;
    /// `power / current = voltage`.
    fn per_current(self, current: Self::ElectricCurrent) -> Self::Voltage;
    /// `power / voltage = current`.
    fn per_voltage(self, voltage: Self::Voltage) -> Self::ElectricCurrent;
}

/// An action dimensioned quantity (`[L² M T⁻¹]`, joule-seconds). The dimension of Planck's constant
/// and of the action integral. Shares its base signature with
/// [`AngularMomentum`] but is a distinct name. `action / time = energy`,
/// `action × frequency = energy` (the Planck relation `E = hf`).
pub trait Action: LinearQuantity {
    /// The time that pairs with this action.
    type Time: Time<Repr = Self::Repr>;
    /// The energy produced by `action / time`.
    type Energy: Energy<Repr = Self::Repr>;
    /// The frequency that pairs with this action.
    type Frequency: Frequency<Repr = Self::Repr>;

    /// `action / time = energy`.
    fn per_time(self, elapsed: Self::Time) -> Self::Energy;
    /// `action × frequency = energy` (the Planck relation `E = hf`).
    fn times_frequency(self, frequency: Self::Frequency) -> Self::Energy;
}

/// A specific-energy dimensioned quantity (`[L² T⁻²]`, joules per kilogram). Energy per unit mass —
/// the dimension of calorific value, specific enthalpy, and (commensurably) of an
/// [`AbsorbedDose`](crate::AbsorbedDose). Shares its base signature with
/// [`AbsorbedDose`](crate::AbsorbedDose) / [`DoseEquivalent`](crate::DoseEquivalent) but is a
/// distinct name. `specific energy × mass = energy`.
pub trait SpecificEnergy: LinearQuantity {
    /// The mass that pairs with this specific energy.
    type Mass: Mass<Repr = Self::Repr>;
    /// The energy produced by `specific energy × mass`.
    type Energy: Energy<Repr = Self::Repr>;

    /// `specific energy × mass = energy`.
    fn times_mass(self, mass: Self::Mass) -> Self::Energy;
}

/// An energy-density dimensioned quantity (`[L⁻¹ M T⁻²]`, joules per cubic metre). Energy per unit
/// volume — the dimension of an energy field's stored density. Shares its base signature with
/// [`Pressure`] (both `[-1, 1, -2]`) but is a distinct name. `energy density × volume = energy`.
pub trait EnergyDensity: LinearQuantity {
    /// The volume that pairs with this energy density.
    type Volume: Volume<Repr = Self::Repr>;
    /// The energy produced by `energy density × volume`.
    type Energy: Energy<Repr = Self::Repr>;

    /// `energy density × volume = energy`.
    fn times_volume(self, volume: Self::Volume) -> Self::Energy;
}
