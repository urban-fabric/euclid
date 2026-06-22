//! Photometric dimensions: [`LuminousIntensity`], [`LuminousFlux`], [`Illuminance`],
//! [`Luminance`].
//!
//! [`LuminousIntensity`] is the SI base dimension `[J]` (the candela). The solid-angle factor
//! relating intensity to flux (`flux = intensity × solid angle`) is dimensionless in SI but
//! tracked by name through [`SolidAngle`].

use super::{Area, LinearQuantity, SolidAngle};

/// A luminous-intensity dimensioned quantity (`[J]`, the SI base dimension, the candela).
/// `intensity × solid angle = luminous flux`, `intensity / area = luminance`.
pub trait LuminousIntensity: LinearQuantity {
    /// The solid angle that pairs with this intensity.
    type SolidAngle: SolidAngle<Repr = Self::Repr>;
    /// The luminous flux produced by `intensity × solid angle`.
    type LuminousFlux: LuminousFlux<Repr = Self::Repr>;
    /// The area that pairs with this intensity.
    type Area: Area<Repr = Self::Repr>;
    /// The luminance produced by `intensity / area`.
    type Luminance: Luminance<Repr = Self::Repr>;

    /// `intensity × solid angle = luminous flux`.
    fn times_solid_angle(self, solid_angle: Self::SolidAngle) -> Self::LuminousFlux;
    /// `intensity / area = luminance`.
    fn per_area(self, area: Self::Area) -> Self::Luminance;
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

/// A luminance dimensioned quantity (`[L⁻² J]`, candelas per square metre, the nit). Luminous
/// intensity per unit projected area — perceived surface brightness. Shares its base signature with
/// [`Illuminance`] but is a distinct name. `luminance × area = luminous intensity`.
pub trait Luminance: LinearQuantity {
    /// The area that pairs with this luminance.
    type Area: Area<Repr = Self::Repr>;
    /// The luminous intensity produced by `luminance × area`.
    type LuminousIntensity: LuminousIntensity<Repr = Self::Repr>;

    /// `luminance × area = luminous intensity`.
    fn times_area(self, area: Self::Area) -> Self::LuminousIntensity;
}
