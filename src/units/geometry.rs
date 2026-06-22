//! Geometric dimensions: [`Length`], [`Area`], [`Volume`], [`Angle`], [`SolidAngle`].
//!
//! These carry the spatial algebra — `length × length = area`, `length × area = volume`,
//! `length / time = speed`, `angle / time = angular velocity` — through associated partner
//! quantities, each bound only on the shared `Repr` carrier (the deliberately thin contract
//! described in [`units`](super)).
//!
//! ## Naming convention
//!
//! Every cross-dimensional operation is a verb suffixed by the *operand's* dimension:
//! `times_<operand>` is `self * operand`, `per_<operand>` is `self / operand`. Pure powers of a
//! single quantity get the ergonomic [`squared`](Length::squared) / [`cubed`](Length::cubed),
//! and reciprocal pairs are named semantically. Same-unit ratios collapse to the raw carrier
//! through [`LinearQuantity::ratio`].

use super::{
    AmountOfSubstance, AngularVelocity, LinearQuantity, LuminousFlux, LuminousIntensity,
    MolarVolume, Speed, Time, VolumetricFlowRate,
};

/// A length-dimensioned quantity (`[L]`). Anchors the spatial algebra:
/// `length × length = area`, `length × area = volume`, `length / time = speed`,
/// `length / speed = time`.
pub trait Length: LinearQuantity {
    /// The area produced by `length × length`.
    type Area: Area<Repr = Self::Repr>;
    /// The volume produced by `length × area`.
    type Volume: Volume<Repr = Self::Repr>;
    /// The time that pairs with this length and speed.
    type Time: Time<Repr = Self::Repr>;
    /// The speed produced by `length / time`.
    type Speed: Speed<Repr = Self::Repr>;

    /// `length × length = area`.
    fn times_length(self, width: Self) -> Self::Area;
    /// `length × area = volume`.
    fn times_area(self, base: Self::Area) -> Self::Volume;
    /// `length / time = speed`.
    fn per_time(self, elapsed: Self::Time) -> Self::Speed;
    /// `length / speed = time`.
    fn per_speed(self, speed: Self::Speed) -> Self::Time;

    /// `length²` — [`times_length`](Self::times_length) with a copy of `self`.
    fn squared(self) -> Self::Area;
    /// `length³`.
    fn cubed(self) -> Self::Volume;
}

/// An area-dimensioned quantity (`[L²]`). `area / length = length`, `area × length = volume`,
/// `area × speed = volumetric flow rate`.
pub trait Area: LinearQuantity {
    /// The length produced by `area / length` (and the edge length of a square of this area).
    type Length: Length<Repr = Self::Repr>;
    /// The volume produced by `area × length`.
    type Volume: Volume<Repr = Self::Repr>;
    /// The flow rate produced by `area × speed`.
    type VolumetricFlowRate: VolumetricFlowRate<Repr = Self::Repr>;
    /// The speed that pairs with this area to make a flow rate.
    type Speed: Speed<Repr = Self::Repr>;

    /// `area / length = length`.
    fn per_length(self, side: Self::Length) -> Self::Length;
    /// `area × length = volume`.
    fn times_length(self, height: Self::Length) -> Self::Volume;
    /// `area × speed = volumetric flow rate`.
    fn times_speed(self, speed: Self::Speed) -> Self::VolumetricFlowRate;
    /// The edge length of a square with this area (`√area`).
    fn side(self) -> Self::Length;
}

/// A volume-dimensioned quantity (`[L³]`). `volume / length = area`, `volume / area = length`,
/// `volume / amount = molar volume`.
pub trait Volume: LinearQuantity {
    /// The length produced by `volume / area` (and the edge of a cube of this volume).
    type Length: Length<Repr = Self::Repr>;
    /// The area produced by `volume / length`.
    type Area: Area<Repr = Self::Repr>;
    /// The amount of substance that pairs with this volume.
    type AmountOfSubstance: AmountOfSubstance<Repr = Self::Repr>;
    /// The molar volume produced by `volume / amount of substance`.
    type MolarVolume: MolarVolume<Repr = Self::Repr>;

    /// `volume / length = area`.
    fn per_length(self, height: Self::Length) -> Self::Area;
    /// `volume / area = length`.
    fn per_area(self, base: Self::Area) -> Self::Length;
    /// The edge length of a cube with this volume (`∛volume`).
    fn edge(self) -> Self::Length;
    /// `volume / amount of substance = molar volume`.
    fn per_amount(self, amount: Self::AmountOfSubstance) -> Self::MolarVolume;
}

/// An angle-dimensioned quantity, conventionally carried in radians. Dimensionless in SI but
/// tracked as a distinct name. `angle / time = angular velocity`.
pub trait Angle: LinearQuantity {
    /// The angular velocity produced by `angle / time`.
    type AngularVelocity: AngularVelocity<Repr = Self::Repr>;
    /// The time that pairs with this angle and angular velocity.
    type Time: Time<Repr = Self::Repr>;

    /// Wrap a carrier value already in radians.
    fn from_radians(radians: Self::Repr) -> Self;
    /// Wrap a carrier value in degrees, converting to radians.
    fn from_degrees(degrees: Self::Repr) -> Self;

    /// `π` radians (a half turn).
    fn half_turn() -> Self;
    /// `2π` radians (a full turn).
    fn full_turn() -> Self;

    /// The angle in radians.
    fn radians(&self) -> Self::Repr;
    /// The angle in degrees.
    fn degrees(&self) -> Self::Repr;

    /// Sine of the angle.
    fn sin(&self) -> Self::Repr;
    /// Cosine of the angle.
    fn cos(&self) -> Self::Repr;
    /// Tangent of the angle.
    fn tan(&self) -> Self::Repr;

    /// Normalize into `[0, 2π)`.
    fn normalized_unsigned(self) -> Self;
    /// Normalize into `(-π, π]`.
    fn normalized_signed(self) -> Self;

    /// `angle / time = angular velocity`.
    fn per_time(self, elapsed: Self::Time) -> Self::AngularVelocity;
}

/// A solid-angle-dimensioned quantity, conventionally carried in steradians. Dimensionless in
/// SI but tracked as a distinct name. `luminous intensity × solid angle = luminous flux`.
pub trait SolidAngle: LinearQuantity {
    /// The luminous flux produced by `luminous intensity × solid angle`.
    type LuminousFlux: LuminousFlux<Repr = Self::Repr>;
    /// The luminous intensity that pairs with this solid angle.
    type LuminousIntensity: LuminousIntensity<Repr = Self::Repr>;

    /// Wrap a carrier value already in steradians.
    fn from_steradians(steradians: Self::Repr) -> Self;
    /// The solid angle in steradians.
    fn steradians(&self) -> Self::Repr;
    /// `4π` steradians (the full sphere).
    fn full_sphere() -> Self;

    /// `luminous intensity × solid angle = luminous flux`.
    fn times_luminous_intensity(self, intensity: Self::LuminousIntensity) -> Self::LuminousFlux;
}
