//! [`Mass`], [`Force`], [`Pressure`], [`Momentum`], [`Density`] — Newton's second law and
//! what hangs directly off it.
//!
//! `mass × acceleration = force` is the edge the rest of this module is reached through, so
//! these five are the densest cluster in the web: force divided by area is pressure, force times
//! time is momentum, mass over volume is density.

use super::*;

/// A mass-dimensioned quantity (`[M]`). `mass × acceleration = force`, `mass × speed = momentum`,
/// `mass / volume = density`, `mass × area = moment of inertia`, `mass / time = mass flow rate`.
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
    /// The time that pairs with this mass.
    type Time: Time<Repr = Self::Repr>;
    /// The mass flow rate produced by `mass / time`.
    type MassFlowRate: MassFlowRate<Repr = Self::Repr>;

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
    /// `mass / time = mass flow rate`.
    fn per_time(self, elapsed: Self::Time) -> Self::MassFlowRate;
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
