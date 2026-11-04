//! The toy unit-carrier family: one `f64`-backed carrier per dimension, plus every
//! dimension-trait implementation for them.
//!
//! euclid ships no concrete unit type, so these carriers play the role a consumer crate's
//! units would. Implementing *every* dimension trait for one coherent family is itself the
//! test: the compiler only confirms that a consistent set of implementors can exist once they
//! are actually written, so the impl blocks below are the proof that the cross-dimensional
//! algebra web closes.
//!
//! The tests that exercise these live in [`algebra`](super::algebra).

#![allow(clippy::float_cmp)]

use crate::{
    AbsorbedDose, Acceleration, Action, AmountOfSubstance, Angle, AngularAcceleration,
    AngularMomentum, AngularVelocity, Area, Capacitance, CatalyticActivity, Concentration,
    Conductance, Conductivity, CurrentDensity, Density, Dimension, Dimensionless, DoseEquivalent,
    DynamicViscosity, ElectricCharge, ElectricCurrent, ElectricField, Energy, EnergyDensity, Force,
    Frequency, HeatCapacity, HeatFlux, Illuminance, Inductance, Jerk, KinematicViscosity, Length,
    LinearQuantity, Luminance, LuminousFlux, LuminousIntensity, MagneticFieldStrength,
    MagneticFlux, MagneticFluxDensity, Mass, MassFlowRate, Measure, Molality, MolarEnergy,
    MolarMass, MolarVolume, MomentOfInertia, Momentum, Permeability, Permittivity, Power, Pressure,
    Radioactivity, Resistance, Resistivity, ScalarQuantity, SolidAngle, SpecificEnergy,
    SpecificHeatCapacity, Speed, SurfaceTension, Temperature, ThermalConductivity,
    ThermalResistance, Time, Torque, Voltage, Volume, VolumetricFlowRate,
};

pub(super) const EPS: f64 = 1e-9;

pub(super) fn approx(a: f64, b: f64) -> bool {
    (a - b).abs() <= EPS * (1.0 + a.abs() + b.abs())
}

// --- carrier boilerplate -------------------------------------------------------------------

macro_rules! scalar_carrier {
    ($name:ident, $dim:expr, $sym:literal) => {
        #[derive(Clone, Debug, PartialEq)]
        pub(super) struct $name(pub(super) f64);

        impl Measure for $name {
            type Repr = f64;
            const DIMENSION: Dimension = $dim;
            const UNIT_NAME: &'static str = $sym;
            fn raw(&self) -> f64 {
                self.0
            }
            fn from_raw(v: f64) -> Self {
                Self(v)
            }
        }

        impl ScalarQuantity for $name {
            fn min(self, other: Self) -> Self {
                Self(self.0.min(other.0))
            }
            fn max(self, other: Self) -> Self {
                Self(self.0.max(other.0))
            }
            fn clamp(self, lo: Self, hi: Self) -> Self {
                Self(self.0.clamp(lo.0, hi.0))
            }
            fn floor(self) -> Self {
                Self(self.0.floor())
            }
            fn ceil(self) -> Self {
                Self(self.0.ceil())
            }
            fn round(self) -> Self {
                Self(self.0.round())
            }
            fn trunc(self) -> Self {
                Self(self.0.trunc())
            }
            fn is_finite(&self) -> bool {
                self.0.is_finite()
            }
            fn is_nan(&self) -> bool {
                self.0.is_nan()
            }
            fn is_infinite(&self) -> bool {
                self.0.is_infinite()
            }
            fn approx_eq(&self, other: &Self, tolerance: &Self) -> bool {
                (self.0 - other.0).abs() <= tolerance.0.abs()
            }
        }
    };
}

macro_rules! linear_carrier {
    ($name:ident, $dim:expr, $sym:literal) => {
        scalar_carrier!($name, $dim, $sym);

        impl LinearQuantity for $name {
            fn add(self, other: Self) -> Self {
                Self(self.0 + other.0)
            }
            fn sub(self, other: Self) -> Self {
                Self(self.0 - other.0)
            }
            fn neg(self) -> Self {
                Self(-self.0)
            }
            fn abs(self) -> Self {
                Self(self.0.abs())
            }
            fn signum(self) -> f64 {
                self.0.signum()
            }
            fn ratio(self, other: Self) -> f64 {
                self.0 / other.0
            }
            fn lerp(self, other: Self, t: f64) -> Self {
                Self(self.0 * (1.0 - t) + other.0 * t)
            }
            fn scale(self, factor: f64) -> Self {
                Self(self.0 * factor)
            }
            fn scaled(self, k: f64) -> Self {
                Self(self.0 * k)
            }
        }
    };
}

// --- one carrier per dimension -------------------------------------------------------------

// base
linear_carrier!(Len, Dimension::Length, "m");
linear_carrier!(Mas, Dimension::Mass, "kg");
linear_carrier!(Tim, Dimension::Time, "s");
linear_carrier!(Cur, Dimension::ElectricCurrent, "A");
scalar_carrier!(Tmp, Dimension::Temperature, "K"); // absolute: ScalarQuantity, not LinearQuantity
linear_carrier!(Amt, Dimension::AmountOfSubstance, "mol");
linear_carrier!(Lum, Dimension::LuminousIntensity, "cd");

// geometry
linear_carrier!(Ar, Dimension::Area, "m^2");
linear_carrier!(Vol, Dimension::Volume, "m^3");
linear_carrier!(Ang, Dimension::Angle, "rad");
linear_carrier!(Sld, Dimension::SolidAngle, "sr");

// kinematics
linear_carrier!(Frq, Dimension::Frequency, "Hz");
linear_carrier!(Spd, Dimension::Speed, "m/s");
linear_carrier!(Acc, Dimension::Acceleration, "m/s^2");
linear_carrier!(Jrk, Dimension::Jerk, "m/s^3");
linear_carrier!(AngVel, Dimension::AngularVelocity, "rad/s");
linear_carrier!(AngAcc, Dimension::AngularAcceleration, "rad/s^2");
linear_carrier!(Flow, Dimension::VolumetricFlowRate, "m^3/s");

// mechanics
linear_carrier!(Frc, Dimension::Force, "N");
linear_carrier!(Prs, Dimension::Pressure, "Pa");
linear_carrier!(Eng, Dimension::Energy, "J");
linear_carrier!(Pwr, Dimension::Power, "W");
linear_carrier!(Mom, Dimension::Momentum, "kg.m/s");
linear_carrier!(AngMom, Dimension::AngularMomentum, "kg.m^2/s");
linear_carrier!(Trq, Dimension::Torque, "N.m");
linear_carrier!(Den, Dimension::Density, "kg/m^3");
linear_carrier!(Inr, Dimension::MomentOfInertia, "kg.m^2");
linear_carrier!(Vsc, Dimension::DynamicViscosity, "Pa.s");
linear_carrier!(Srf, Dimension::SurfaceTension, "N/m");
linear_carrier!(Mfr, Dimension::MassFlowRate, "kg/s");
linear_carrier!(Kvs, Dimension::KinematicViscosity, "m^2/s");
linear_carrier!(SpE, Dimension::SpecificEnergy, "J/kg");
linear_carrier!(EnD, Dimension::EnergyDensity, "J/m^3");
linear_carrier!(Act, Dimension::Action, "J.s");

// electromagnetism
linear_carrier!(Chg, Dimension::ElectricCharge, "C");
linear_carrier!(Vlt, Dimension::Voltage, "V");
linear_carrier!(Res, Dimension::Resistance, "ohm");
linear_carrier!(Cnd, Dimension::Conductance, "S");
linear_carrier!(Cap, Dimension::Capacitance, "F");
linear_carrier!(Ind, Dimension::Inductance, "H");
linear_carrier!(Flx, Dimension::MagneticFlux, "Wb");
linear_carrier!(Fld, Dimension::MagneticFluxDensity, "T");
linear_carrier!(EFd, Dimension::ElectricField, "V/m");
linear_carrier!(CrD, Dimension::CurrentDensity, "A/m^2");
linear_carrier!(Hfs, Dimension::MagneticFieldStrength, "A/m");
linear_carrier!(Pmt, Dimension::Permittivity, "F/m");
linear_carrier!(Pmb, Dimension::Permeability, "H/m");
linear_carrier!(Rsv, Dimension::Resistivity, "ohm.m");
linear_carrier!(Cdv, Dimension::Conductivity, "S/m");

// thermal
linear_carrier!(Hcp, Dimension::HeatCapacity, "J/K");
linear_carrier!(Tcd, Dimension::ThermalConductivity, "W/(m.K)");
linear_carrier!(Shc, Dimension::SpecificHeatCapacity, "J/(kg.K)");
linear_carrier!(Thr, Dimension::ThermalResistance, "K/W");
linear_carrier!(Hfx, Dimension::HeatFlux, "W/m^2");

// photometry
linear_carrier!(LFl, Dimension::LuminousFlux, "lm");
linear_carrier!(Ilm, Dimension::Illuminance, "lx");
linear_carrier!(Lmn, Dimension::Luminance, "cd/m^2");

// chemistry
linear_carrier!(Cat, Dimension::CatalyticActivity, "kat");
linear_carrier!(Cnc, Dimension::Concentration, "mol/m^3");
linear_carrier!(Mlr, Dimension::MolarMass, "kg/mol");
linear_carrier!(MlE, Dimension::MolarEnergy, "J/mol");
linear_carrier!(MlV, Dimension::MolarVolume, "m^3/mol");
linear_carrier!(Mlt, Dimension::Molality, "mol/kg");

// radiology
linear_carrier!(Rdy, Dimension::Radioactivity, "Bq");
linear_carrier!(AbD, Dimension::AbsorbedDose, "Gy");
linear_carrier!(DsE, Dimension::DoseEquivalent, "Sv");

// dimensionless
linear_carrier!(Rat, Dimension::Dimensionless, "");

// --- geometry impls ------------------------------------------------------------------------

impl Length for Len {
    type Area = Ar;
    type Volume = Vol;
    type Time = Tim;
    type Speed = Spd;
    fn times_length(self, width: Self) -> Ar {
        Ar(self.0 * width.0)
    }
    fn times_area(self, base: Ar) -> Vol {
        Vol(self.0 * base.0)
    }
    fn per_time(self, elapsed: Tim) -> Spd {
        Spd(self.0 / elapsed.0)
    }
    fn per_speed(self, speed: Spd) -> Tim {
        Tim(self.0 / speed.0)
    }
    fn squared(self) -> Ar {
        Ar(self.0 * self.0)
    }
    fn cubed(self) -> Vol {
        Vol(self.0 * self.0 * self.0)
    }
}

impl Area for Ar {
    type Length = Len;
    type Volume = Vol;
    type VolumetricFlowRate = Flow;
    type Speed = Spd;
    fn per_length(self, side: Len) -> Len {
        Len(self.0 / side.0)
    }
    fn times_length(self, height: Len) -> Vol {
        Vol(self.0 * height.0)
    }
    fn times_speed(self, speed: Spd) -> Flow {
        Flow(self.0 * speed.0)
    }
    fn side(self) -> Len {
        Len(self.0.sqrt())
    }
}

impl Volume for Vol {
    type Length = Len;
    type Area = Ar;
    type AmountOfSubstance = Amt;
    type MolarVolume = MlV;
    fn per_length(self, height: Len) -> Ar {
        Ar(self.0 / height.0)
    }
    fn per_area(self, base: Ar) -> Len {
        Len(self.0 / base.0)
    }
    fn edge(self) -> Len {
        Len(self.0.cbrt())
    }
    fn per_amount(self, amount: Amt) -> MlV {
        MlV(self.0 / amount.0)
    }
}

impl Angle for Ang {
    type AngularVelocity = AngVel;
    type Time = Tim;
    fn from_radians(radians: f64) -> Self {
        Self(radians)
    }
    fn from_degrees(degrees: f64) -> Self {
        Self(degrees.to_radians())
    }
    fn half_turn() -> Self {
        Self(std::f64::consts::PI)
    }
    fn full_turn() -> Self {
        Self(std::f64::consts::TAU)
    }
    fn radians(&self) -> f64 {
        self.0
    }
    fn degrees(&self) -> f64 {
        self.0.to_degrees()
    }
    fn sin(&self) -> f64 {
        self.0.sin()
    }
    fn cos(&self) -> f64 {
        self.0.cos()
    }
    fn tan(&self) -> f64 {
        self.0.tan()
    }
    fn normalized_unsigned(self) -> Self {
        Self(self.0.rem_euclid(std::f64::consts::TAU))
    }
    fn normalized_signed(self) -> Self {
        let r = self.0.rem_euclid(std::f64::consts::TAU);
        if r > std::f64::consts::PI {
            Self(r - std::f64::consts::TAU)
        } else {
            Self(r)
        }
    }
    fn per_time(self, elapsed: Tim) -> AngVel {
        AngVel(self.0 / elapsed.0)
    }
}

impl SolidAngle for Sld {
    type LuminousFlux = LFl;
    type LuminousIntensity = Lum;
    fn from_steradians(steradians: f64) -> Self {
        Self(steradians)
    }
    fn steradians(&self) -> f64 {
        self.0
    }
    fn full_sphere() -> Self {
        Self(4.0 * std::f64::consts::PI)
    }
    fn times_luminous_intensity(self, intensity: Lum) -> LFl {
        LFl(self.0 * intensity.0)
    }
}

// --- kinematics impls ----------------------------------------------------------------------

impl Time for Tim {
    type Length = Len;
    type Speed = Spd;
    type Acceleration = Acc;
    type Frequency = Frq;
    fn times_speed(self, speed: Spd) -> Len {
        Len(self.0 * speed.0)
    }
    fn times_acceleration(self, accel: Acc) -> Spd {
        Spd(self.0 * accel.0)
    }
    fn frequency(self) -> Frq {
        Frq(self.0.recip())
    }
}

impl Frequency for Frq {
    type Time = Tim;
    fn period(self) -> Tim {
        Tim(self.0.recip())
    }
}

impl Speed for Spd {
    type Length = Len;
    type Time = Tim;
    type Acceleration = Acc;
    fn times_time(self, elapsed: Tim) -> Len {
        Len(self.0 * elapsed.0)
    }
    fn per_time(self, elapsed: Tim) -> Acc {
        Acc(self.0 / elapsed.0)
    }
    fn per_acceleration(self, accel: Acc) -> Tim {
        Tim(self.0 / accel.0)
    }
}

impl Acceleration for Acc {
    type Speed = Spd;
    type Time = Tim;
    type Force = Frc;
    type Mass = Mas;
    type Jerk = Jrk;
    fn times_time(self, elapsed: Tim) -> Spd {
        Spd(self.0 * elapsed.0)
    }
    fn times_mass(self, mass: Mas) -> Frc {
        Frc(self.0 * mass.0)
    }
    fn per_time(self, elapsed: Tim) -> Jrk {
        Jrk(self.0 / elapsed.0)
    }
}

impl Jerk for Jrk {
    type Acceleration = Acc;
    type Time = Tim;
    fn times_time(self, elapsed: Tim) -> Acc {
        Acc(self.0 * elapsed.0)
    }
}

impl AngularVelocity for AngVel {
    type Angle = Ang;
    type Time = Tim;
    type AngularAcceleration = AngAcc;
    fn times_time(self, elapsed: Tim) -> Ang {
        Ang(self.0 * elapsed.0)
    }
    fn per_time(self, elapsed: Tim) -> AngAcc {
        AngAcc(self.0 / elapsed.0)
    }
}

impl AngularAcceleration for AngAcc {
    type AngularVelocity = AngVel;
    type Time = Tim;
    fn times_time(self, elapsed: Tim) -> AngVel {
        AngVel(self.0 * elapsed.0)
    }
}

impl VolumetricFlowRate for Flow {
    type Volume = Vol;
    type Time = Tim;
    type Area = Ar;
    type Speed = Spd;
    fn times_time(self, elapsed: Tim) -> Vol {
        Vol(self.0 * elapsed.0)
    }
    fn per_area(self, section: Ar) -> Spd {
        Spd(self.0 / section.0)
    }
    fn per_speed(self, speed: Spd) -> Ar {
        Ar(self.0 / speed.0)
    }
}

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

// --- thermal impls -------------------------------------------------------------------------

impl Temperature for Tmp {
    fn from_kelvin(kelvin: f64) -> Self {
        Self(kelvin)
    }
    fn from_celsius(celsius: f64) -> Self {
        Self(celsius + 273.15)
    }
    fn from_fahrenheit(fahrenheit: f64) -> Self {
        Self((fahrenheit - 32.0) * 5.0 / 9.0 + 273.15)
    }
    fn kelvin(&self) -> f64 {
        self.0
    }
    fn celsius(&self) -> f64 {
        self.0 - 273.15
    }
    fn fahrenheit(&self) -> f64 {
        (self.0 - 273.15) * 9.0 / 5.0 + 32.0
    }
    fn absolute_zero() -> Self {
        Self(0.0)
    }
}

impl HeatCapacity for Hcp {
    type Temperature = Tmp;
    type Energy = Eng;
    type Mass = Mas;
    type SpecificHeatCapacity = Shc;
    fn times_temperature(self, delta: Tmp) -> Eng {
        Eng(self.0 * delta.0)
    }
    fn per_mass(self, mass: Mas) -> Shc {
        Shc(self.0 / mass.0)
    }
}

impl ThermalConductivity for Tcd {}

impl SpecificHeatCapacity for Shc {
    type Mass = Mas;
    type HeatCapacity = Hcp;
    type Temperature = Tmp;
    type SpecificEnergy = SpE;
    fn times_mass(self, mass: Mas) -> Hcp {
        Hcp(self.0 * mass.0)
    }
    fn times_temperature(self, delta: Tmp) -> SpE {
        SpE(self.0 * delta.0)
    }
}

impl ThermalResistance for Thr {
    type Power = Pwr;
    type Temperature = Tmp;
    fn times_power(self, power: Pwr) -> Tmp {
        Tmp(self.0 * power.0)
    }
}

impl HeatFlux for Hfx {
    type Area = Ar;
    type Power = Pwr;
    fn times_area(self, area: Ar) -> Pwr {
        Pwr(self.0 * area.0)
    }
}

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
