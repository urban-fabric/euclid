//! Coverage over the `f64` carrier with toy unit carriers that implement the dimension
//! contracts. `f64: Real`, so this exercises the same generic code paths used by other real
//! carriers, and `<f64 as Compare>::Mask = bool`, so the mask-returning predicates collapse to
//! plain `bool` here. euclid ships no concrete unit type, so the carriers below play the role a
//! consumer crate's units would.
//!
//! Beyond spot-checking arithmetic, implementing *every* dimension trait for this family of
//! carriers is the real proof that the cross-dimensional algebra web is internally consistent:
//! the compiler only verifies that a coherent set of implementors can exist once they are
//! actually written, so the impl blocks below are themselves the test.

#![allow(clippy::float_cmp)]

use super::{
    Acceleration, AmountOfSubstance, Angle, AngularAcceleration, AngularMomentum, AngularVelocity,
    Area, BaseExponents, Capacitance, CatalyticActivity, Concentration, Conductance, Density,
    Dimension, Dimensioned, Dimensionless, DynamicViscosity, ElectricCharge, ElectricCurrent,
    ElectricField, Energy, Force, Frequency, HeatCapacity, Illuminance, Inductance, Jerk, Length,
    LinearQuantity, LuminousFlux, LuminousIntensity, MagneticFlux, MagneticFluxDensity, Mass,
    MolarMass, MomentOfInertia, Momentum, Power, Pressure, Quantity, Resistance, ScalarQuantity,
    SolidAngle, Speed, SurfaceTension, Temperature, ThermalConductivity, Time, Torque, Voltage,
    Volume, VolumetricFlowRate, dim,
};

const EPS: f64 = 1e-9;

fn approx(a: f64, b: f64) -> bool {
    (a - b).abs() <= EPS * (1.0 + a.abs() + b.abs())
}

// --- carrier boilerplate -------------------------------------------------------------------

macro_rules! scalar_carrier {
    ($name:ident, $dim:expr, $sym:literal) => {
        #[derive(Clone, Debug, PartialEq)]
        struct $name(f64);

        impl Quantity for $name {
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

// thermal
linear_carrier!(Hcp, Dimension::HeatCapacity, "J/K");
linear_carrier!(Tcd, Dimension::ThermalConductivity, "W/(m.K)");

// photometry
linear_carrier!(LFl, Dimension::LuminousFlux, "lm");
linear_carrier!(Ilm, Dimension::Illuminance, "lx");

// chemistry
linear_carrier!(Cat, Dimension::CatalyticActivity, "kat");
linear_carrier!(Cnc, Dimension::Concentration, "mol/m^3");
linear_carrier!(Mlr, Dimension::MolarMass, "kg/mol");

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
    fn per_length(self, height: Len) -> Ar {
        Ar(self.0 / height.0)
    }
    fn per_area(self, base: Ar) -> Len {
        Len(self.0 / base.0)
    }
    fn edge(self) -> Len {
        Len(self.0.cbrt())
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
}

impl Power for Pwr {
    type Time = Tim;
    type Energy = Eng;
    type Force = Frc;
    type Speed = Spd;
    type ElectricCurrent = Cur;
    type Voltage = Vlt;
    fn times_time(self, elapsed: Tim) -> Eng {
        Eng(self.0 * elapsed.0)
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

// --- electromagnetism impls ----------------------------------------------------------------

impl ElectricCurrent for Cur {
    type Time = Tim;
    type ElectricCharge = Chg;
    type Resistance = Res;
    type Voltage = Vlt;
    type Power = Pwr;
    type Inductance = Ind;
    type MagneticFlux = Flx;
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
    fn times_temperature(self, delta: Tmp) -> Eng {
        Eng(self.0 * delta.0)
    }
}

impl ThermalConductivity for Tcd {}

// --- photometry impls ----------------------------------------------------------------------

impl LuminousIntensity for Lum {
    type SolidAngle = Sld;
    type LuminousFlux = LFl;
    fn times_solid_angle(self, solid_angle: Sld) -> LFl {
        LFl(self.0 * solid_angle.0)
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

// --- chemistry impls -----------------------------------------------------------------------

impl AmountOfSubstance for Amt {
    type Time = Tim;
    type CatalyticActivity = Cat;
    type Volume = Vol;
    type Concentration = Cnc;
    type MolarMass = Mlr;
    type Mass = Mas;
    fn per_time(self, elapsed: Tim) -> Cat {
        Cat(self.0 / elapsed.0)
    }
    fn per_volume(self, volume: Vol) -> Cnc {
        Cnc(self.0 / volume.0)
    }
    fn times_molar_mass(self, molar_mass: Mlr) -> Mas {
        Mas(self.0 * molar_mass.0)
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

// --- tests ---------------------------------------------------------------------------------

#[test]
fn linear_ops_over_f64() {
    let a = Len(3.0);
    let b = Len(4.0);
    assert!(approx(a.clone().add(b.clone()).raw(), 7.0));
    assert!(approx(b.clone().sub(a.clone()).raw(), 1.0));
    assert!(approx(a.clone().neg().raw(), -3.0));
    assert!(approx(a.clone().scaled(2.5).raw(), 7.5));
    assert!(approx(a.clone().scale(2.0).raw(), 6.0));
    assert!(approx(a.clone().lerp(b.clone(), 0.5).raw(), 3.5));
    assert!(approx(b.ratio(a), 4.0 / 3.0));
}

#[test]
fn selectors_and_masks_collapse_to_bool() {
    let a = Len(2.0);
    let b = Len(5.0);
    assert!(approx(a.clone().min(b.clone()).raw(), 2.0));
    assert!(approx(a.clone().max(b.clone()).raw(), 5.0));
    assert!(approx(Len(9.0).clamp(Len(0.0), Len(5.0)).raw(), 5.0));
    let finite: bool = a.is_finite();
    assert!(finite);
    assert!(!a.is_nan());
    assert!(a.approx_eq(&Len(2.0 + 1e-12), &Len(EPS)));
}

#[test]
fn geometry_algebra() {
    assert!(approx(Len(3.0).times_length(Len(4.0)).raw(), 12.0));
    assert!(approx(Len(3.0).squared().raw(), 9.0));
    assert!(approx(Len(2.0).cubed().raw(), 8.0));
    assert!(approx(Ar(12.0).per_length(Len(4.0)).raw(), 3.0));
    assert!(approx(Ar(16.0).side().raw(), 4.0));
    assert!(approx(Vol(27.0).edge().raw(), 3.0));
    assert!(approx(Len(2.0).times_area(Ar(6.0)).raw(), 12.0));
    assert!(approx(Vol(12.0).per_area(Ar(6.0)).raw(), 2.0));
    assert!(approx(Vol(12.0).per_length(Len(3.0)).raw(), 4.0));
}

#[test]
fn kinematic_ladder() {
    let v: Spd = Len(100.0).per_time(Tim(4.0));
    assert!(approx(v.clone().raw(), 25.0));
    assert!(approx(v.clone().times_time(Tim(4.0)).raw(), 100.0));
    assert!(approx(Len(100.0).per_speed(v.clone()).raw(), 4.0));

    let a: Acc = v.per_time(Tim(5.0));
    assert!(approx(a.clone().raw(), 5.0));
    assert!(approx(a.clone().times_time(Tim(2.0)).raw(), 10.0));
    assert!(approx(a.per_time(Tim(2.0)).raw(), 2.5)); // jerk

    assert!(approx(Tim(2.0).frequency().raw(), 0.5));
    assert!(approx(Frq(0.5).period().raw(), 2.0));
    assert!(approx(Ar(2.0).times_speed(Spd(3.0)).raw(), 6.0)); // flow rate
    assert!(approx(Flow(6.0).per_area(Ar(2.0)).raw(), 3.0));
}

#[test]
fn newtonian_dynamics() {
    let f: Frc = Mas(5.0).times_acceleration(Acc(2.0)); // F = ma
    assert!(approx(f.clone().raw(), 10.0));
    assert!(approx(f.clone().per_mass(Mas(5.0)).raw(), 2.0));
    assert!(approx(f.clone().per_acceleration(Acc(2.0)).raw(), 5.0));

    let work: Eng = f.clone().times_length(Len(3.0)); // W = F·d
    assert!(approx(work.clone().raw(), 30.0));
    assert!(approx(work.clone().per_time(Tim(5.0)).raw(), 6.0)); // P = W/t
    assert!(approx(work.per_force(Frc(10.0)).raw(), 3.0));

    assert!(approx(f.clone().per_area(Ar(2.0)).raw(), 5.0)); // pressure
    assert!(approx(f.clone().times_time(Tim(3.0)).raw(), 30.0)); // impulse → momentum
    assert!(approx(f.times_speed(Spd(4.0)).raw(), 40.0)); // power

    let p: Mom = Mas(5.0).times_speed(Spd(4.0));
    assert!(approx(p.clone().raw(), 20.0));
    assert!(approx(p.clone().per_mass(Mas(5.0)).raw(), 4.0));
    assert!(approx(p.per_time(Tim(4.0)).raw(), 5.0)); // F = dp/dt

    assert!(approx(Mas(12.0).per_volume(Vol(4.0)).raw(), 3.0)); // density
    assert!(approx(Den(3.0).times_volume(Vol(4.0)).raw(), 12.0));
    assert!(approx(Prs(5.0).times_volume(Vol(2.0)).raw(), 10.0)); // P·V = energy
}

#[test]
fn rotational_dynamics() {
    assert!(approx(AngVel(2.0).times_time(Tim(3.0)).raw(), 6.0)); // angle
    assert!(approx(Ang(6.0).per_time(Tim(3.0)).raw(), 2.0));

    let torque: Trq = Inr(4.0).times_angular_acceleration(AngAcc(2.0)); // τ = Iα
    assert!(approx(torque.clone().raw(), 8.0));
    assert!(approx(torque.clone().per_length(Len(2.0)).raw(), 4.0)); // → force
    assert!(approx(torque.times_angular_velocity(AngVel(3.0)).raw(), 24.0)); // → power

    let l: AngMom = Inr(4.0).times_angular_velocity(AngVel(3.0)); // L = Iω
    assert!(approx(l.clone().raw(), 12.0));
    assert!(approx(l.clone().per_moment_of_inertia(Inr(4.0)).raw(), 3.0));
    assert!(approx(l.per_time(Tim(4.0)).raw(), 3.0)); // → torque
    assert!(approx(Mom(20.0).times_length(Len(2.0)).raw(), 40.0)); // r × p
}

#[test]
fn electromagnetic_circuits() {
    let v: Vlt = Cur(2.0).times_resistance(Res(3.0)); // Ohm: V = IR
    assert!(approx(v.clone().raw(), 6.0));
    assert!(approx(v.clone().per_current(Cur(2.0)).raw(), 3.0));
    assert!(approx(v.clone().per_resistance(Res(3.0)).raw(), 2.0));
    assert!(approx(v.clone().times_current(Cur(2.0)).raw(), 12.0)); // P = VI

    let q: Chg = Cur(3.0).times_time(Tim(4.0)); // Q = It
    assert!(approx(q.clone().raw(), 12.0));
    assert!(approx(q.clone().per_time(Tim(4.0)).raw(), 3.0));
    assert!(approx(q.times_voltage(Vlt(2.0)).raw(), 24.0)); // E = QV

    assert!(approx(Cap(5.0).times_voltage(Vlt(2.0)).raw(), 10.0)); // Q = CV
    assert!(approx(Res(4.0).conductance().raw(), 0.25));
    assert!(approx(Cnd(0.25).resistance().raw(), 4.0));
    assert!(approx(Cap(2.0).times_resistance(Res(3.0)).raw(), 6.0)); // RC time constant

    let flux: Flx = Ind(2.0).times_current(Cur(3.0)); // Φ = LI
    assert!(approx(flux.clone().raw(), 6.0));
    assert!(approx(flux.clone().per_time(Tim(2.0)).raw(), 3.0)); // Faraday: V = dΦ/dt
    assert!(approx(flux.per_area(Ar(3.0)).raw(), 2.0)); // B = Φ/A
    assert!(approx(Fld(2.0).times_area(Ar(3.0)).raw(), 6.0));
    assert!(approx(EFd(5.0).times_length(Len(2.0)).raw(), 10.0)); // V = E·d
    assert!(approx(EFd(5.0).times_charge(Chg(2.0)).raw(), 10.0)); // F = qE
}

#[test]
fn thermal_photometric_chemical() {
    // thermal — temperature is absolute, so the energy relation reads ΔT as a carrier interval.
    assert!(approx(Hcp(5.0).times_temperature(Tmp(2.0)).raw(), 10.0));
    assert!(approx(Eng(10.0).per_temperature(Tmp(2.0)).raw(), 5.0));
    assert!(approx(Tmp::from_celsius(0.0).kelvin(), 273.15));
    assert!(approx(Tmp::from_fahrenheit(32.0).celsius(), 0.0));
    assert!(approx(Tmp::absolute_zero().kelvin(), 0.0));

    // photometry
    let flux: LFl = Lum(4.0).times_solid_angle(Sld(2.0));
    assert!(approx(flux.clone().raw(), 8.0));
    assert!(approx(flux.clone().per_solid_angle(Sld(2.0)).raw(), 4.0));
    assert!(approx(flux.per_area(Ar(2.0)).raw(), 4.0)); // illuminance
    assert!(approx(Ilm(4.0).times_area(Ar(2.0)).raw(), 8.0));
    assert!(approx(Sld::full_sphere().steradians(), 4.0 * std::f64::consts::PI));

    // chemistry
    assert!(approx(Amt(6.0).per_time(Tim(2.0)).raw(), 3.0)); // catalytic activity
    assert!(approx(Cat(3.0).times_time(Tim(2.0)).raw(), 6.0));
    assert!(approx(Amt(6.0).per_volume(Vol(2.0)).raw(), 3.0)); // concentration
    assert!(approx(Cnc(3.0).times_volume(Vol(2.0)).raw(), 6.0));
    assert!(approx(Mlr(2.0).times_amount(Amt(3.0)).raw(), 6.0)); // mass
    assert!(approx(Mas(6.0).per_amount(Amt(3.0)).raw(), 2.0)); // molar mass

    // thermal conductivity is a tracked marker with no cross-dimensional product.
    assert_eq!(Tcd(0.5).raw(), 0.5);
    assert_eq!(Tcd::DIMENSION, Dimension::ThermalConductivity);
}

#[test]
fn angle_conversion_and_normalization() {
    let half_turn = Ang::from_degrees(180.0);
    assert!(approx(half_turn.radians(), std::f64::consts::PI));
    assert!(approx(half_turn.degrees(), 180.0));
    assert!(approx(Ang::from_degrees(450.0).normalized_unsigned().degrees(), 90.0));
    assert!(approx(Ang::from_degrees(270.0).normalized_signed().degrees(), -90.0));
}

#[test]
fn unit_metadata() {
    assert_eq!(Len::UNIT_NAME, "m");
    assert_eq!(Frc::UNIT_NAME, "N");
    assert_eq!(Spd::DIMENSION, Dimension::Speed);
    assert_eq!(Cur::DIMENSION, Dimension::ElectricCurrent);
    assert!(Dimension::Length.same(Dimension::Length));
    assert!(!Dimension::Length.same(Dimension::Time));
}

#[test]
fn dimensionless_ratios_compose() {
    let half = Rat(0.5);
    let third = Rat(3.0).recip();
    assert!(approx(third.clone().raw(), 1.0 / 3.0));
    assert!(approx(half.clone().times(Rat(4.0)).raw(), 2.0));
    assert!(approx(Rat(6.0).per(Rat(2.0)).raw(), 3.0));
    assert!(approx(Rat::unit().raw(), 1.0));
    // dimensionless still scales a dimensioned quantity through LinearQuantity::scale.
    assert!(approx(Len(10.0).scale(half.raw()).raw(), 5.0));
}

#[test]
fn type_level_markers_bridge_to_runtime() {
    // each marker maps to its runtime dimension and inherits the matching exponents.
    assert_eq!(<dim::Force as Dimensioned>::DIMENSION, Dimension::Force);
    assert_eq!(
        <dim::Force as Dimensioned>::EXPONENTS,
        Dimension::Force.exponents()
    );
    assert_eq!(<dim::Length as Dimensioned>::DIMENSION, Dimension::Length);
    assert_eq!(
        <dim::Voltage as Dimensioned>::EXPONENTS,
        [2, 1, -3, -1, 0, 0, 0]
    );
    assert_eq!(
        <dim::Dimensionless as Dimensioned>::DIMENSION,
        Dimension::Dimensionless
    );
    // a marker is a zero-sized tag.
    assert_eq!(core::mem::size_of::<dim::Energy>(), 0);
}

#[test]
fn base_dimension_signatures() {
    // [L, M, T, I, Θ, N, J]
    let len: BaseExponents = Dimension::Length.exponents();
    assert_eq!(len, [1, 0, 0, 0, 0, 0, 0]);
    assert_eq!(Dimension::Force.exponents(), [1, 1, -2, 0, 0, 0, 0]);
    assert_eq!(Dimension::Voltage.exponents(), [2, 1, -3, -1, 0, 0, 0]);

    // energy and torque share a base signature but are distinct names.
    assert!(Dimension::Energy.commensurable(Dimension::Torque));
    assert!(!Dimension::Energy.same(Dimension::Torque));

    // frequency and angular velocity are likewise commensurable.
    assert!(Dimension::Frequency.commensurable(Dimension::AngularVelocity));

    // angles read as dimensionless in SI even though they are tracked names.
    assert!(Dimension::Angle.is_dimensionless());
    assert!(Dimension::SolidAngle.is_dimensionless());
    assert!(!Dimension::Length.is_dimensionless());
    assert!(!Dimension::Length.commensurable(Dimension::Time));
}
