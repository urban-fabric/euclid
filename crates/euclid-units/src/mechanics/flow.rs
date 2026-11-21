//! [`DynamicViscosity`], [`SurfaceTension`], [`MassFlowRate`], [`KinematicViscosity`] — the
//! continuum quantities.
//!
//! The four that describe a material rather than a body. They are the thinnest traits in the
//! module and the ones whose dimensions are hardest to recognise by eye, which is reason enough
//! for them to sit together with their relations stated in one place.

use super::*;

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

/// A mass-flow-rate dimensioned quantity (`[M T⁻¹]`, kilograms per second). `mass flow × time =
/// mass`, `mass flow × speed = force` (thrust), `mass flow / density = volumetric flow rate`.
pub trait MassFlowRate: LinearQuantity {
    /// The time that pairs with this mass flow rate.
    type Time: Time<Repr = Self::Repr>;
    /// The mass produced by `mass flow × time`.
    type Mass: Mass<Repr = Self::Repr>;
    /// The speed that pairs with this mass flow rate.
    type Speed: Speed<Repr = Self::Repr>;
    /// The force (thrust) produced by `mass flow × speed`.
    type Force: Force<Repr = Self::Repr>;
    /// The density that pairs with this mass flow rate.
    type Density: Density<Repr = Self::Repr>;
    /// The volumetric flow rate produced by `mass flow / density`.
    type VolumetricFlowRate: VolumetricFlowRate<Repr = Self::Repr>;

    /// `mass flow × time = mass`.
    fn times_time(self, elapsed: Self::Time) -> Self::Mass;
    /// `mass flow × speed = force` (thrust, `F = ṁv`).
    fn times_speed(self, speed: Self::Speed) -> Self::Force;
    /// `mass flow / density = volumetric flow rate`.
    fn per_density(self, density: Self::Density) -> Self::VolumetricFlowRate;
}

/// A kinematic-viscosity dimensioned quantity (`[L² T⁻¹]`, square metres per second, the stokes).
/// Momentum diffusivity `ν = μ/ρ`. `kinematic viscosity × density = dynamic viscosity`,
/// `kinematic viscosity / length = speed`.
pub trait KinematicViscosity: LinearQuantity {
    /// The density that pairs with this kinematic viscosity.
    type Density: Density<Repr = Self::Repr>;
    /// The dynamic viscosity produced by `kinematic viscosity × density`.
    type DynamicViscosity: DynamicViscosity<Repr = Self::Repr>;
    /// The length that pairs with this kinematic viscosity.
    type Length: Length<Repr = Self::Repr>;
    /// The speed produced by `kinematic viscosity / length`.
    type Speed: Speed<Repr = Self::Repr>;

    /// `kinematic viscosity × density = dynamic viscosity` (`μ = νρ`).
    fn times_density(self, density: Self::Density) -> Self::DynamicViscosity;
    /// `kinematic viscosity / length = speed`.
    fn per_length(self, length: Self::Length) -> Self::Speed;
}
