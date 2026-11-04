//! Mechanical dimension implementations for the toy carriers.
//!
//! Trait impls only; the carriers themselves live in
//! [`carriers`](super::carriers).

use super::carriers::*;
use crate::*;

// --- mechanics impls -----------------------------------------------------------------------

impl Mass for Mas {
    type Force = Frc;
    type Acceleration = Acc;
    type Momentum = Mom;
    type Speed = Spd;
    type Density = Den;
    type Volume = Vol;
    type MomentOfInertia = Inr;
    type Area = Ar;
    type MolarMass = Mlr;
    type AmountOfSubstance = Amt;
    type Time = Tim;
    type MassFlowRate = Mfr;
    fn times_acceleration(self, accel: Acc) -> Frc {
        Frc(self.0 * accel.0)
    }
    fn times_speed(self, speed: Spd) -> Mom {
        Mom(self.0 * speed.0)
    }
    fn per_volume(self, volume: Vol) -> Den {
        Den(self.0 / volume.0)
    }
    fn times_area(self, area: Ar) -> Inr {
        Inr(self.0 * area.0)
    }
    fn per_amount(self, amount: Amt) -> Mlr {
        Mlr(self.0 / amount.0)
    }
    fn per_time(self, elapsed: Tim) -> Mfr {
        Mfr(self.0 / elapsed.0)
    }
}

impl Force for Frc {
    type Mass = Mas;
    type Acceleration = Acc;
    type Length = Len;
    type Energy = Eng;
    type Area = Ar;
    type Pressure = Prs;
    type Time = Tim;
    type Momentum = Mom;
    type Speed = Spd;
    type Power = Pwr;
    type SurfaceTension = Srf;
    type ElectricCharge = Chg;
    type ElectricField = EFd;
    fn per_mass(self, mass: Mas) -> Acc {
        Acc(self.0 / mass.0)
    }
    fn per_acceleration(self, accel: Acc) -> Mas {
        Mas(self.0 / accel.0)
    }
    fn times_length(self, distance: Len) -> Eng {
        Eng(self.0 * distance.0)
    }
    fn per_area(self, area: Ar) -> Prs {
        Prs(self.0 / area.0)
    }
    fn times_time(self, elapsed: Tim) -> Mom {
        Mom(self.0 * elapsed.0)
    }
    fn times_speed(self, speed: Spd) -> Pwr {
        Pwr(self.0 * speed.0)
    }
    fn per_length(self, length: Len) -> Srf {
        Srf(self.0 / length.0)
    }
    fn per_charge(self, charge: Chg) -> EFd {
        EFd(self.0 / charge.0)
    }
}

impl Pressure for Prs {
    type Area = Ar;
    type Force = Frc;
    type Volume = Vol;
    type Energy = Eng;
    type Time = Tim;
    type DynamicViscosity = Vsc;
    fn times_area(self, area: Ar) -> Frc {
        Frc(self.0 * area.0)
    }
    fn times_volume(self, volume: Vol) -> Eng {
        Eng(self.0 * volume.0)
    }
    fn times_time(self, elapsed: Tim) -> Vsc {
        Vsc(self.0 * elapsed.0)
    }
}

impl Energy for Eng {
    type Force = Frc;
    type Length = Len;
    type Time = Tim;
    type Power = Pwr;
    type Volume = Vol;
    type Pressure = Prs;
    type ElectricCharge = Chg;
    type Voltage = Vlt;
    type Temperature = Tmp;
    type HeatCapacity = Hcp;
    type Mass = Mas;
    type SpecificEnergy = SpE;
    type AmountOfSubstance = Amt;
    type MolarEnergy = MlE;
    fn per_length(self, distance: Len) -> Frc {
        Frc(self.0 / distance.0)
    }
    fn per_force(self, force: Frc) -> Len {
        Len(self.0 / force.0)
    }
    fn per_time(self, elapsed: Tim) -> Pwr {
        Pwr(self.0 / elapsed.0)
    }
    fn per_power(self, power: Pwr) -> Tim {
        Tim(self.0 / power.0)
    }
    fn per_volume(self, volume: Vol) -> Prs {
        Prs(self.0 / volume.0)
    }
    fn per_charge(self, charge: Chg) -> Vlt {
        Vlt(self.0 / charge.0)
    }
    fn per_temperature(self, temperature: Tmp) -> Hcp {
        Hcp(self.0 / temperature.0)
    }
    fn per_mass(self, mass: Mas) -> SpE {
        SpE(self.0 / mass.0)
    }
    fn per_amount(self, amount: Amt) -> MlE {
        MlE(self.0 / amount.0)
    }
}

impl Power for Pwr {
    type Time = Tim;
    type Energy = Eng;
    type Force = Frc;
    type Speed = Spd;
    type ElectricCurrent = Cur;
    type Voltage = Vlt;
    type Area = Ar;
    type HeatFlux = Hfx;
    fn times_time(self, elapsed: Tim) -> Eng {
        Eng(self.0 * elapsed.0)
    }
    fn per_area(self, area: Ar) -> Hfx {
        Hfx(self.0 / area.0)
    }
    fn per_force(self, force: Frc) -> Spd {
        Spd(self.0 / force.0)
    }
    fn per_speed(self, speed: Spd) -> Frc {
        Frc(self.0 / speed.0)
    }
    fn per_current(self, current: Cur) -> Vlt {
        Vlt(self.0 / current.0)
    }
    fn per_voltage(self, voltage: Vlt) -> Cur {
        Cur(self.0 / voltage.0)
    }
}

impl Momentum for Mom {
    type Mass = Mas;
    type Speed = Spd;
    type Time = Tim;
    type Force = Frc;
    type Length = Len;
    type AngularMomentum = AngMom;
    fn per_mass(self, mass: Mas) -> Spd {
        Spd(self.0 / mass.0)
    }
    fn per_speed(self, speed: Spd) -> Mas {
        Mas(self.0 / speed.0)
    }
    fn per_time(self, elapsed: Tim) -> Frc {
        Frc(self.0 / elapsed.0)
    }
    fn times_length(self, lever: Len) -> AngMom {
        AngMom(self.0 * lever.0)
    }
}

impl Density for Den {
    type Mass = Mas;
    type Volume = Vol;
    fn times_volume(self, volume: Vol) -> Mas {
        Mas(self.0 * volume.0)
    }
}

impl MomentOfInertia for Inr {
    type Mass = Mas;
    type Area = Ar;
    type AngularVelocity = AngVel;
    type AngularMomentum = AngMom;
    type AngularAcceleration = AngAcc;
    type Torque = Trq;
    fn per_mass(self, mass: Mas) -> Ar {
        Ar(self.0 / mass.0)
    }
    fn times_angular_velocity(self, omega: AngVel) -> AngMom {
        AngMom(self.0 * omega.0)
    }
    fn times_angular_acceleration(self, alpha: AngAcc) -> Trq {
        Trq(self.0 * alpha.0)
    }
}

impl AngularMomentum for AngMom {
    type MomentOfInertia = Inr;
    type AngularVelocity = AngVel;
    type Time = Tim;
    type Torque = Trq;
    type Length = Len;
    type Momentum = Mom;
    fn per_moment_of_inertia(self, inertia: Inr) -> AngVel {
        AngVel(self.0 / inertia.0)
    }
    fn per_time(self, elapsed: Tim) -> Trq {
        Trq(self.0 / elapsed.0)
    }
    fn per_length(self, lever: Len) -> Mom {
        Mom(self.0 / lever.0)
    }
}

impl Torque for Trq {
    type Length = Len;
    type Force = Frc;
    type AngularVelocity = AngVel;
    type Power = Pwr;
    type Time = Tim;
    type AngularMomentum = AngMom;
    type MomentOfInertia = Inr;
    type AngularAcceleration = AngAcc;
    fn per_length(self, lever: Len) -> Frc {
        Frc(self.0 / lever.0)
    }
    fn times_angular_velocity(self, omega: AngVel) -> Pwr {
        Pwr(self.0 * omega.0)
    }
    fn times_time(self, elapsed: Tim) -> AngMom {
        AngMom(self.0 * elapsed.0)
    }
    fn per_moment_of_inertia(self, inertia: Inr) -> AngAcc {
        AngAcc(self.0 / inertia.0)
    }
}

impl DynamicViscosity for Vsc {
    type Time = Tim;
    type Pressure = Prs;
    fn per_time(self, elapsed: Tim) -> Prs {
        Prs(self.0 / elapsed.0)
    }
}

impl SurfaceTension for Srf {
    type Length = Len;
    type Force = Frc;
    type Area = Ar;
    type Energy = Eng;
    fn times_length(self, length: Len) -> Frc {
        Frc(self.0 * length.0)
    }
    fn times_area(self, area: Ar) -> Eng {
        Eng(self.0 * area.0)
    }
}

impl MassFlowRate for Mfr {
    type Time = Tim;
    type Mass = Mas;
    type Speed = Spd;
    type Force = Frc;
    type Density = Den;
    type VolumetricFlowRate = Flow;
    fn times_time(self, elapsed: Tim) -> Mas {
        Mas(self.0 * elapsed.0)
    }
    fn times_speed(self, speed: Spd) -> Frc {
        Frc(self.0 * speed.0)
    }
    fn per_density(self, density: Den) -> Flow {
        Flow(self.0 / density.0)
    }
}

impl KinematicViscosity for Kvs {
    type Density = Den;
    type DynamicViscosity = Vsc;
    type Length = Len;
    type Speed = Spd;
    fn times_density(self, density: Den) -> Vsc {
        Vsc(self.0 * density.0)
    }
    fn per_length(self, length: Len) -> Spd {
        Spd(self.0 / length.0)
    }
}

impl SpecificEnergy for SpE {
    type Mass = Mas;
    type Energy = Eng;
    fn times_mass(self, mass: Mas) -> Eng {
        Eng(self.0 * mass.0)
    }
}

impl EnergyDensity for EnD {
    type Volume = Vol;
    type Energy = Eng;
    fn times_volume(self, volume: Vol) -> Eng {
        Eng(self.0 * volume.0)
    }
}

impl Action for Act {
    type Time = Tim;
    type Energy = Eng;
    type Frequency = Frq;
    fn per_time(self, elapsed: Tim) -> Eng {
        Eng(self.0 / elapsed.0)
    }
    fn times_frequency(self, frequency: Frq) -> Eng {
        Eng(self.0 * frequency.0)
    }
}
