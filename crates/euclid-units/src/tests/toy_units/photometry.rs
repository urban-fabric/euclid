//! Photometric dimension implementations for the toy carriers.
//!
//! Trait impls only; the carriers themselves live in
//! [`carriers`](super::carriers).

use super::carriers::*;
use crate::*;

// --- photometry impls ----------------------------------------------------------------------

impl LuminousIntensity for Lum {
    type SolidAngle = Sld;
    type LuminousFlux = LFl;
    type Area = Ar;
    type Luminance = Lmn;
    fn times_solid_angle(self, solid_angle: Sld) -> LFl {
        LFl(self.0 * solid_angle.0)
    }
    fn per_area(self, area: Ar) -> Lmn {
        Lmn(self.0 / area.0)
    }
}

impl LuminousFlux for LFl {
    type SolidAngle = Sld;
    type LuminousIntensity = Lum;
    type Area = Ar;
    type Illuminance = Ilm;
    fn per_solid_angle(self, solid_angle: Sld) -> Lum {
        Lum(self.0 / solid_angle.0)
    }
    fn per_area(self, area: Ar) -> Ilm {
        Ilm(self.0 / area.0)
    }
}

impl Illuminance for Ilm {
    type Area = Ar;
    type LuminousFlux = LFl;
    fn times_area(self, area: Ar) -> LFl {
        LFl(self.0 * area.0)
    }
}

impl Luminance for Lmn {
    type Area = Ar;
    type LuminousIntensity = Lum;
    fn times_area(self, area: Ar) -> Lum {
        Lum(self.0 * area.0)
    }
}
