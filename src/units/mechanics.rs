//! Mechanical / dynamic dimensions: [`Mass`], [`Force`], [`Pressure`], [`Energy`], [`Power`],
//! [`Momentum`], [`Density`], [`MomentOfInertia`], [`AngularMomentum`], [`Torque`],
//! [`DynamicViscosity`], [`SurfaceTension`].
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

use super::{
    Acceleration, AmountOfSubstance, AngularAcceleration, AngularVelocity, Area, ElectricCharge,
    ElectricCurrent, ElectricField, HeatCapacity, Length, LinearQuantity, MolarMass, Speed,
    Temperature, Time, Voltage, Volume,
};

/// A mass-dimensioned quantity (`[M]`). `mass × acceleration = force`, `mass × speed = momentum`,
/// `mass / volume = density`, `mass × area = moment of inertia`.
pub trait Mass: LinearQuantity {
    /// The force produced by `mass × acceleration`.
    type Force: Force<Repr = Self::Repr>;
    /// The acceleration that pairs with this mass.
    type Acceleration: Acceleration<Repr = Self::Repr>;
    /// The momentum produced by `mass × speed`.
    type Momentum: Momentum<Repr = Self::Repr>;
    /// The speed that pairs with this mass.
    type Speed: Speed<Repr = Self::Repr>;
    /// The density produced by `mass / volume`.
    type Density: Density<Repr = Self::Repr>;
    /// The volume that pairs with this mass.
    type Volume: Volume<Repr = Self::Repr>;
    /// The moment of inertia produced by `mass × area`.
    type MomentOfInertia: MomentOfInertia<Repr = Self::Repr>;
    /// The area (radius-of-gyration squared) that pairs with this mass.
    type Area: Area<Repr = Self::Repr>;
    /// The molar mass produced by `mass / amount of substance`.
    type MolarMass: MolarMass<Repr = Self::Repr>;
    /// The amount of substance that pairs with this mass.
    type AmountOfSubstance: AmountOfSubstance<Repr = Self::Repr>;

    /// `mass × acceleration = force` (Newton's second law).
    fn times_acceleration(self, accel: Self::Acceleration) -> Self::Force;
    /// `mass × speed = momentum`.
    fn times_speed(self, speed: Self::Speed) -> Self::Momentum;
    /// `mass / volume = density`.
    fn per_volume(self, volume: Self::Volume) -> Self::Density;
    /// `mass × area = moment of inertia`.
    fn times_area(self, area: Self::Area) -> Self::MomentOfInertia;
    /// `mass / amount of substance = molar mass`.
    fn per_amount(self, amount: Self::AmountOfSubstance) -> Self::MolarMass;
}

/// A force-dimensioned quantity (`[L M T⁻²]`). The hub of dynamics:
/// `force / mass = acceleration`, `force × length = energy`, `force / area = pressure`,
/// `force × time = momentum`, `force × speed = power`.
pub trait Force: LinearQuantity {
    /// The mass that pairs with this force.
    type Mass: Mass<Repr = Self::Repr>;
    /// The acceleration produced by `force / mass`.
    type Acceleration: Acceleration<Repr = Self::Repr>;
    /// The length that pairs with this force.
    type Length: Length<Repr = Self::Repr>;
    /// The energy (work) produced by `force × length`.
    type Energy: Energy<Repr = Self::Repr>;
    /// The area that pairs with this force.
    type Area: Area<Repr = Self::Repr>;
    /// The pressure produced by `force / area`.
    type Pressure: Pressure<Repr = Self::Repr>;
    /// The time that pairs with this force.
    type Time: Time<Repr = Self::Repr>;
    /// The momentum produced by `force × time` (impulse).
    type Momentum: Momentum<Repr = Self::Repr>;
    /// The speed that pairs with this force.
    type Speed: Speed<Repr = Self::Repr>;
    /// The power produced by `force × speed`.
    type Power: Power<Repr = Self::Repr>;
    /// The surface tension produced by `force / length`.
    type SurfaceTension: SurfaceTension<Repr = Self::Repr>;
    /// The electric charge that pairs with this force.
    type ElectricCharge: ElectricCharge<Repr = Self::Repr>;
    /// The electric field produced by `force / charge`.
    type ElectricField: ElectricField<Repr = Self::Repr>;

    /// `force / mass = acceleration` (Newton's second law, solved for `a`).
    fn per_mass(self, mass: Self::Mass) -> Self::Acceleration;
    /// `force / acceleration = mass`.
    fn per_acceleration(self, accel: Self::Acceleration) -> Self::Mass;
    /// `force × length = energy` (work along a displacement).
    fn times_length(self, distance: Self::Length) -> Self::Energy;
    /// `force / area = pressure`.
    fn per_area(self, area: Self::Area) -> Self::Pressure;
    /// `force × time = momentum` (impulse–momentum theorem).
    fn times_time(self, elapsed: Self::Time) -> Self::Momentum;
    /// `force × speed = power`.
    fn times_speed(self, speed: Self::Speed) -> Self::Power;
    /// `force / length = surface tension`.
    fn per_length(self, length: Self::Length) -> Self::SurfaceTension;
    /// `force / charge = electric field`.
    fn per_charge(self, charge: Self::ElectricCharge) -> Self::ElectricField;
}

/// A pressure (or stress) dimensioned quantity (`[L⁻¹ M T⁻²]`). `pressure × area = force`,
/// `pressure × volume = energy`, `pressure × time = dynamic viscosity`.
pub trait Pressure: LinearQuantity {
    /// The area that pairs with this pressure.
    type Area: Area<Repr = Self::Repr>;
    /// The force produced by `pressure × area`.
    type Force: Force<Repr = Self::Repr>;
    /// The volume that pairs with this pressure.
    type Volume: Volume<Repr = Self::Repr>;
    /// The energy produced by `pressure × volume`.
    type Energy: Energy<Repr = Self::Repr>;
    /// The time that pairs with this pressure.
    type Time: Time<Repr = Self::Repr>;
    /// The dynamic viscosity produced by `pressure × time`.
    type DynamicViscosity: DynamicViscosity<Repr = Self::Repr>;

    /// `pressure × area = force`.
    fn times_area(self, area: Self::Area) -> Self::Force;
    /// `pressure × volume = energy`.
    fn times_volume(self, volume: Self::Volume) -> Self::Energy;
    /// `pressure × time = dynamic viscosity`.
    fn times_time(self, elapsed: Self::Time) -> Self::DynamicViscosity;
}

/// An energy / work / heat dimensioned quantity (`[L² M T⁻²]`). `energy / length = force`,
/// `energy / time = power`, `energy / volume = pressure`, `energy / charge = voltage`,
/// `energy / temperature = heat capacity`.
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
}

/// A power-dimensioned quantity (`[L² M T⁻³]`). `power × time = energy`, `power / force = speed`,
/// `power / current = voltage`.
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

    /// `power × time = energy`.
    fn times_time(self, elapsed: Self::Time) -> Self::Energy;
    /// `power / force = speed`.
    fn per_force(self, force: Self::Force) -> Self::Speed;
    /// `power / speed = force`.
    fn per_speed(self, speed: Self::Speed) -> Self::Force;
    /// `power / current = voltage`.
    fn per_current(self, current: Self::ElectricCurrent) -> Self::Voltage;
    /// `power / voltage = current`.
    fn per_voltage(self, voltage: Self::Voltage) -> Self::ElectricCurrent;
}

/// A (linear) momentum dimensioned quantity (`[L M T⁻¹]`). `momentum / mass = speed`,
/// `momentum / time = force`, `momentum × length = angular momentum`.
pub trait Momentum: LinearQuantity {
    /// The mass that pairs with this momentum.
    type Mass: Mass<Repr = Self::Repr>;
    /// The speed produced by `momentum / mass`.
    type Speed: Speed<Repr = Self::Repr>;
    /// The time that pairs with this momentum.
    type Time: Time<Repr = Self::Repr>;
    /// The force produced by `momentum / time`.
    type Force: Force<Repr = Self::Repr>;
    /// The length (lever arm) that pairs with this momentum.
    type Length: Length<Repr = Self::Repr>;
    /// The angular momentum produced by `momentum × length`.
    type AngularMomentum: AngularMomentum<Repr = Self::Repr>;

    /// `momentum / mass = speed`.
    fn per_mass(self, mass: Self::Mass) -> Self::Speed;
    /// `momentum / speed = mass`.
    fn per_speed(self, speed: Self::Speed) -> Self::Mass;
    /// `momentum / time = force`.
    fn per_time(self, elapsed: Self::Time) -> Self::Force;
    /// `momentum × length = angular momentum` (`r × p`).
    fn times_length(self, lever: Self::Length) -> Self::AngularMomentum;
}

/// A mass-density dimensioned quantity (`[L⁻³ M]`). `density × volume = mass`.
pub trait Density: LinearQuantity {
    /// The mass produced by `density × volume`.
    type Mass: Mass<Repr = Self::Repr>;
    /// The volume that pairs with this density.
    type Volume: Volume<Repr = Self::Repr>;

    /// `density × volume = mass`.
    fn times_volume(self, volume: Self::Volume) -> Self::Mass;
}

/// A moment-of-inertia dimensioned quantity (`[L² M]`). `moment of inertia × angular velocity =
/// angular momentum`, `moment of inertia × angular acceleration = torque`.
pub trait MomentOfInertia: LinearQuantity {
    /// The mass that pairs with this moment of inertia.
    type Mass: Mass<Repr = Self::Repr>;
    /// The area produced by `moment of inertia / mass`.
    type Area: Area<Repr = Self::Repr>;
    /// The angular velocity that pairs with this moment of inertia.
    type AngularVelocity: AngularVelocity<Repr = Self::Repr>;
    /// The angular momentum produced by `moment of inertia × angular velocity`.
    type AngularMomentum: AngularMomentum<Repr = Self::Repr>;
    /// The angular acceleration that pairs with this moment of inertia.
    type AngularAcceleration: AngularAcceleration<Repr = Self::Repr>;
    /// The torque produced by `moment of inertia × angular acceleration`.
    type Torque: Torque<Repr = Self::Repr>;

    /// `moment of inertia / mass = area` (radius of gyration squared).
    fn per_mass(self, mass: Self::Mass) -> Self::Area;
    /// `moment of inertia × angular velocity = angular momentum`.
    fn times_angular_velocity(self, omega: Self::AngularVelocity) -> Self::AngularMomentum;
    /// `moment of inertia × angular acceleration = torque` (rotational Newton's second law).
    fn times_angular_acceleration(self, alpha: Self::AngularAcceleration) -> Self::Torque;
}

/// An angular-momentum dimensioned quantity (`[L² M T⁻¹]`, also the dimension of action).
/// `angular momentum / time = torque`, `angular momentum / moment of inertia = angular velocity`,
/// `angular momentum / length = momentum`.
pub trait AngularMomentum: LinearQuantity {
    /// The moment of inertia that pairs with this angular momentum.
    type MomentOfInertia: MomentOfInertia<Repr = Self::Repr>;
    /// The angular velocity produced by `angular momentum / moment of inertia`.
    type AngularVelocity: AngularVelocity<Repr = Self::Repr>;
    /// The time that pairs with this angular momentum.
    type Time: Time<Repr = Self::Repr>;
    /// The torque produced by `angular momentum / time`.
    type Torque: Torque<Repr = Self::Repr>;
    /// The length (lever arm) that pairs with this angular momentum.
    type Length: Length<Repr = Self::Repr>;
    /// The momentum produced by `angular momentum / length`.
    type Momentum: Momentum<Repr = Self::Repr>;

    /// `angular momentum / moment of inertia = angular velocity`.
    fn per_moment_of_inertia(self, inertia: Self::MomentOfInertia) -> Self::AngularVelocity;
    /// `angular momentum / time = torque`.
    fn per_time(self, elapsed: Self::Time) -> Self::Torque;
    /// `angular momentum / length = momentum`.
    fn per_length(self, lever: Self::Length) -> Self::Momentum;
}

/// A torque (moment of force) dimensioned quantity (`[L² M T⁻²]`). Shares a base signature with
/// [`Energy`] but is a distinct name. `torque / length = force`,
/// `torque × angular velocity = power`, `torque × time = angular momentum`.
pub trait Torque: LinearQuantity {
    /// The length (lever arm) that pairs with this torque.
    type Length: Length<Repr = Self::Repr>;
    /// The force produced by `torque / length`.
    type Force: Force<Repr = Self::Repr>;
    /// The angular velocity that pairs with this torque.
    type AngularVelocity: AngularVelocity<Repr = Self::Repr>;
    /// The power produced by `torque × angular velocity`.
    type Power: Power<Repr = Self::Repr>;
    /// The time that pairs with this torque.
    type Time: Time<Repr = Self::Repr>;
    /// The angular momentum produced by `torque × time`.
    type AngularMomentum: AngularMomentum<Repr = Self::Repr>;
    /// The moment of inertia that pairs with this torque.
    type MomentOfInertia: MomentOfInertia<Repr = Self::Repr>;
    /// The angular acceleration produced by `torque / moment of inertia`.
    type AngularAcceleration: AngularAcceleration<Repr = Self::Repr>;

    /// `torque / length = force`.
    fn per_length(self, lever: Self::Length) -> Self::Force;
    /// `torque × angular velocity = power`.
    fn times_angular_velocity(self, omega: Self::AngularVelocity) -> Self::Power;
    /// `torque × time = angular momentum`.
    fn times_time(self, elapsed: Self::Time) -> Self::AngularMomentum;
    /// `torque / moment of inertia = angular acceleration`.
    fn per_moment_of_inertia(self, inertia: Self::MomentOfInertia) -> Self::AngularAcceleration;
}

/// A dynamic-viscosity dimensioned quantity (`[L⁻¹ M T⁻¹]`, the pascal-second).
/// `dynamic viscosity / time = pressure`.
pub trait DynamicViscosity: LinearQuantity {
    /// The time that pairs with this viscosity.
    type Time: Time<Repr = Self::Repr>;
    /// The pressure (shear stress) produced by `dynamic viscosity / time`.
    type Pressure: Pressure<Repr = Self::Repr>;

    /// `dynamic viscosity / time = pressure`.
    fn per_time(self, elapsed: Self::Time) -> Self::Pressure;
}

/// A surface-tension dimensioned quantity (`[M T⁻²]`, newtons per metre / joules per square
/// metre). `surface tension × length = force`, `surface tension × area = energy`.
pub trait SurfaceTension: LinearQuantity {
    /// The length that pairs with this surface tension.
    type Length: Length<Repr = Self::Repr>;
    /// The force produced by `surface tension × length`.
    type Force: Force<Repr = Self::Repr>;
    /// The area that pairs with this surface tension.
    type Area: Area<Repr = Self::Repr>;
    /// The energy produced by `surface tension × area`.
    type Energy: Energy<Repr = Self::Repr>;

    /// `surface tension × length = force`.
    fn times_length(self, length: Self::Length) -> Self::Force;
    /// `surface tension × area = energy` (surface energy).
    fn times_area(self, area: Self::Area) -> Self::Energy;
}
