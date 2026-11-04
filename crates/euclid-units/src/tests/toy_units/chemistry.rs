//! Chemical dimension implementations for the toy carriers.
//!
//! Trait impls only; the carriers themselves live in
//! [`carriers`](super::carriers).

use super::carriers::*;
use crate::*;

// --- chemistry impls -----------------------------------------------------------------------

impl AmountOfSubstance for Amt {
    type Time = Tim;
    type CatalyticActivity = Cat;
    type Volume = Vol;
    type Concentration = Cnc;
    type MolarMass = Mlr;
    type Mass = Mas;
    type Molality = Mlt;
    fn per_time(self, elapsed: Tim) -> Cat {
        Cat(self.0 / elapsed.0)
    }
    fn per_volume(self, volume: Vol) -> Cnc {
        Cnc(self.0 / volume.0)
    }
    fn times_molar_mass(self, molar_mass: Mlr) -> Mas {
        Mas(self.0 * molar_mass.0)
    }
    fn per_mass(self, mass: Mas) -> Mlt {
        Mlt(self.0 / mass.0)
    }
}

impl CatalyticActivity for Cat {
    type Time = Tim;
    type AmountOfSubstance = Amt;
    fn times_time(self, elapsed: Tim) -> Amt {
        Amt(self.0 * elapsed.0)
    }
}

impl Concentration for Cnc {
    type Volume = Vol;
    type AmountOfSubstance = Amt;
    fn times_volume(self, volume: Vol) -> Amt {
        Amt(self.0 * volume.0)
    }
}

impl MolarMass for Mlr {
    type AmountOfSubstance = Amt;
    type Mass = Mas;
    fn times_amount(self, amount: Amt) -> Mas {
        Mas(self.0 * amount.0)
    }
}

impl MolarEnergy for MlE {
    type AmountOfSubstance = Amt;
    type Energy = Eng;
    fn times_amount(self, amount: Amt) -> Eng {
        Eng(self.0 * amount.0)
    }
}

impl MolarVolume for MlV {
    type AmountOfSubstance = Amt;
    type Volume = Vol;
    type Concentration = Cnc;
    fn times_amount(self, amount: Amt) -> Vol {
        Vol(self.0 * amount.0)
    }
    fn concentration(self) -> Cnc {
        Cnc(self.0.recip())
    }
}

impl Molality for Mlt {
    type Mass = Mas;
    type AmountOfSubstance = Amt;
    fn times_mass(self, mass: Mas) -> Amt {
        Amt(self.0 * mass.0)
    }
}
