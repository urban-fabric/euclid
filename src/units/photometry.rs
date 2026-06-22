//! Photometric dimensions: [`LuminousIntensity`], [`LuminousFlux`], [`Illuminance`].
//!
//! [`LuminousIntensity`] is the SI base dimension `[J]` (the candela). The solid-angle factor
//! relating intensity to flux (`flux = intensity × solid angle`) is dimensionless in SI but
//! tracked by name through [`SolidAngle`].

use super::{Area, LinearQuantity, SolidAngle};

/// A luminous-intensity dimensioned quantity (`[J]`, the SI base dimension, the candela).
/// `intensity × solid angle = luminous flux`.
pub trait LuminousIntensity: LinearQuantity {
    /// The solid angle that pairs with this intensity.
    type SolidAngle: SolidAngle<Repr = Self::Repr>;
    /// The luminous flux produced by `intensity × solid angle`.
    type LuminousFlux: LuminousFlux<Repr = Self::Repr>;

    /// `intensity × solid angle = luminous flux`.
    fn times_solid_angle(self, solid_angle: Self::SolidAngle) -> Self::LuminousFlux;
}

/// A luminous-flux dimensioned quantity (the lumen, `cd·sr`). `flux / solid angle = intensity`,
/// `flux / area = illuminance`.
pub trait LuminousFlux: LinearQuantity {
    /// The solid angle that pairs with this flux.
    type SolidAngle: SolidAngle<Repr = Self::Repr>;
    /// The intensity produced by `flux / solid angle`.
    type LuminousIntensity: LuminousIntensity<Repr = Self::Repr>;
    /// The area that pairs with this flux.
    type Area: Area<Repr = Self::Repr>;
    /// The illuminance produced by `flux / area`.
    type Illuminance: Illuminance<Repr = Self::Repr>;

    /// `flux / solid angle = luminous intensity`.
    fn per_solid_angle(self, solid_angle: Self::SolidAngle) -> Self::LuminousIntensity;
    /// `flux / area = illuminance`.
    fn per_area(self, area: Self::Area) -> Self::Illuminance;
}

/// An illuminance dimensioned quantity (`[L⁻² J]`, the lux, `lm/m²`). `illuminance × area =
/// luminous flux`.
pub trait Illuminance: LinearQuantity {
    /// The area that pairs with this illuminance.
    type Area: Area<Repr = Self::Repr>;
    /// The luminous flux produced by `illuminance × area`.
    type LuminousFlux: LuminousFlux<Repr = Self::Repr>;

    /// `illuminance × area = luminous flux`.
    fn times_area(self, area: Self::Area) -> Self::LuminousFlux;
}
