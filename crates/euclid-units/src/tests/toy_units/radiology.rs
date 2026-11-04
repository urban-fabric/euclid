//! Radiological dimension implementations for the toy carriers.
//!
//! Trait impls only; the carriers themselves live in
//! [`carriers`](super::carriers).

use super::carriers::*;
use crate::*;

// --- radiology impls -----------------------------------------------------------------------

impl Radioactivity for Rdy {
    type Time = Tim;
    type Count = Rat;
    fn times_time(self, elapsed: Tim) -> Rat {
        Rat(self.0 * elapsed.0)
    }
}

impl AbsorbedDose for AbD {
    type Mass = Mas;
    type Energy = Eng;
    type DoseEquivalent = DsE;
    fn times_mass(self, mass: Mas) -> Eng {
        Eng(self.0 * mass.0)
    }
    fn weighted_by(self, radiation_weight: f64) -> DsE {
        DsE(self.0 * radiation_weight)
    }
}

impl DoseEquivalent for DsE {
    type Mass = Mas;
    type Energy = Eng;
    fn times_mass(self, mass: Mas) -> Eng {
        Eng(self.0 * mass.0)
    }
}

impl Dimensionless for Rat {
    fn unit() -> Self {
        Self(1.0)
    }
    fn times(self, other: Self) -> Self {
        Self(self.0 * other.0)
    }
    fn recip(self) -> Self {
        Self(self.0.recip())
    }
}
