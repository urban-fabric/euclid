//! Electromagnetic dimension implementations for the toy carriers.
//!
//! Trait impls only; the carriers themselves live in
//! [`carriers`](super::carriers).

use super::carriers::*;
use crate::*;

// --- electromagnetism impls ----------------------------------------------------------------

impl ElectricCurrent for Cur {
    type Time = Tim;
    type ElectricCharge = Chg;
    type Resistance = Res;
    type Voltage = Vlt;
    type Power = Pwr;
    type Inductance = Ind;
    type MagneticFlux = Flx;
    type Area = Ar;
    type CurrentDensity = CrD;
    fn times_time(self, elapsed: Tim) -> Chg {
        Chg(self.0 * elapsed.0)
    }
    fn times_resistance(self, resistance: Res) -> Vlt {
        Vlt(self.0 * resistance.0)
    }
    fn times_voltage(self, voltage: Vlt) -> Pwr {
        Pwr(self.0 * voltage.0)
    }
    fn times_inductance(self, inductance: Ind) -> Flx {
        Flx(self.0 * inductance.0)
    }
    fn per_area(self, area: Ar) -> CrD {
        CrD(self.0 / area.0)
    }
}

impl ElectricCharge for Chg {
    type Time = Tim;
    type ElectricCurrent = Cur;
    type Voltage = Vlt;
    type Energy = Eng;
    type Capacitance = Cap;
    type ElectricField = EFd;
    type Force = Frc;
    fn per_time(self, elapsed: Tim) -> Cur {
        Cur(self.0 / elapsed.0)
    }
    fn per_current(self, current: Cur) -> Tim {
        Tim(self.0 / current.0)
    }
    fn times_voltage(self, voltage: Vlt) -> Eng {
        Eng(self.0 * voltage.0)
    }
    fn per_capacitance(self, capacitance: Cap) -> Vlt {
        Vlt(self.0 / capacitance.0)
    }
    fn per_voltage(self, voltage: Vlt) -> Cap {
        Cap(self.0 / voltage.0)
    }
    fn times_electric_field(self, field: EFd) -> Frc {
        Frc(self.0 * field.0)
    }
}

impl Voltage for Vlt {
    type ElectricCurrent = Cur;
    type Resistance = Res;
    type Power = Pwr;
    type ElectricCharge = Chg;
    type Energy = Eng;
    type Length = Len;
    type ElectricField = EFd;
    type Time = Tim;
    type MagneticFlux = Flx;
    fn per_current(self, current: Cur) -> Res {
        Res(self.0 / current.0)
    }
    fn per_resistance(self, resistance: Res) -> Cur {
        Cur(self.0 / resistance.0)
    }
    fn times_current(self, current: Cur) -> Pwr {
        Pwr(self.0 * current.0)
    }
    fn times_charge(self, charge: Chg) -> Eng {
        Eng(self.0 * charge.0)
    }
    fn per_length(self, length: Len) -> EFd {
        EFd(self.0 / length.0)
    }
    fn times_time(self, elapsed: Tim) -> Flx {
        Flx(self.0 * elapsed.0)
    }
}

impl Resistance for Res {
    type ElectricCurrent = Cur;
    type Voltage = Vlt;
    type Conductance = Cnd;
    type Time = Tim;
    type Inductance = Ind;
    fn times_current(self, current: Cur) -> Vlt {
        Vlt(self.0 * current.0)
    }
    fn conductance(self) -> Cnd {
        Cnd(self.0.recip())
    }
    fn times_time(self, elapsed: Tim) -> Ind {
        Ind(self.0 * elapsed.0)
    }
}

impl Conductance for Cnd {
    type Resistance = Res;
    fn resistance(self) -> Res {
        Res(self.0.recip())
    }
}

impl Capacitance for Cap {
    type Voltage = Vlt;
    type ElectricCharge = Chg;
    type Resistance = Res;
    type Time = Tim;
    fn times_voltage(self, voltage: Vlt) -> Chg {
        Chg(self.0 * voltage.0)
    }
    fn times_resistance(self, resistance: Res) -> Tim {
        Tim(self.0 * resistance.0)
    }
}

impl Inductance for Ind {
    type ElectricCurrent = Cur;
    type MagneticFlux = Flx;
    type Resistance = Res;
    type Time = Tim;
    fn times_current(self, current: Cur) -> Flx {
        Flx(self.0 * current.0)
    }
    fn per_resistance(self, resistance: Res) -> Tim {
        Tim(self.0 / resistance.0)
    }
    fn per_time(self, elapsed: Tim) -> Res {
        Res(self.0 / elapsed.0)
    }
}

impl MagneticFlux for Flx {
    type ElectricCurrent = Cur;
    type Inductance = Ind;
    type Time = Tim;
    type Voltage = Vlt;
    type Area = Ar;
    type MagneticFluxDensity = Fld;
    fn per_current(self, current: Cur) -> Ind {
        Ind(self.0 / current.0)
    }
    fn per_time(self, elapsed: Tim) -> Vlt {
        Vlt(self.0 / elapsed.0)
    }
    fn per_area(self, area: Ar) -> Fld {
        Fld(self.0 / area.0)
    }
}

impl MagneticFluxDensity for Fld {
    type Area = Ar;
    type MagneticFlux = Flx;
    fn times_area(self, area: Ar) -> Flx {
        Flx(self.0 * area.0)
    }
}

impl ElectricField for EFd {
    type Length = Len;
    type Voltage = Vlt;
    type ElectricCharge = Chg;
    type Force = Frc;
    fn times_length(self, length: Len) -> Vlt {
        Vlt(self.0 * length.0)
    }
    fn times_charge(self, charge: Chg) -> Frc {
        Frc(self.0 * charge.0)
    }
}

impl CurrentDensity for CrD {
    type Area = Ar;
    type ElectricCurrent = Cur;
    fn times_area(self, area: Ar) -> Cur {
        Cur(self.0 * area.0)
    }
}

impl MagneticFieldStrength for Hfs {
    type Length = Len;
    type ElectricCurrent = Cur;
    type Permeability = Pmb;
    type MagneticFluxDensity = Fld;
    fn times_length(self, length: Len) -> Cur {
        Cur(self.0 * length.0)
    }
    fn times_permeability(self, permeability: Pmb) -> Fld {
        Fld(self.0 * permeability.0)
    }
}

impl Permittivity for Pmt {
    type Length = Len;
    type Capacitance = Cap;
    fn times_length(self, length: Len) -> Cap {
        Cap(self.0 * length.0)
    }
}

impl Permeability for Pmb {
    type Length = Len;
    type Inductance = Ind;
    type MagneticFieldStrength = Hfs;
    type MagneticFluxDensity = Fld;
    fn times_length(self, length: Len) -> Ind {
        Ind(self.0 * length.0)
    }
    fn times_field_strength(self, field_strength: Hfs) -> Fld {
        Fld(self.0 * field_strength.0)
    }
}

impl Resistivity for Rsv {
    type CurrentDensity = CrD;
    type ElectricField = EFd;
    type Conductivity = Cdv;
    fn times_current_density(self, current_density: CrD) -> EFd {
        EFd(self.0 * current_density.0)
    }
    fn conductivity(self) -> Cdv {
        Cdv(self.0.recip())
    }
}

impl Conductivity for Cdv {
    type ElectricField = EFd;
    type CurrentDensity = CrD;
    type Resistivity = Rsv;
    fn times_electric_field(self, field: EFd) -> CrD {
        CrD(self.0 * field.0)
    }
    fn resistivity(self) -> Rsv {
        Rsv(self.0.recip())
    }
}
