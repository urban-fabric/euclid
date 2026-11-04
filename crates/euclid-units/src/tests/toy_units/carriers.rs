//! The toy carriers: the boilerplate macros, and one `f64`-backed carrier per
//! dimension.
//!
//! The macros emit `pub(crate)` items so the per-domain impl modules and the
//! [`algebra`](super::super::algebra) tests can reach them.

use crate::{Dimension, LinearQuantity, Measure, ScalarQuantity};

pub(crate) const EPS: f64 = 1e-9;

pub(crate) fn approx(a: f64, b: f64) -> bool {
    (a - b).abs() <= EPS * (1.0 + a.abs() + b.abs())
}

// --- carrier boilerplate -------------------------------------------------------------------

macro_rules! scalar_carrier {
    ($name:ident, $dim:expr, $sym:literal) => {
        #[derive(Clone, Debug, PartialEq)]
        pub(crate) struct $name(pub(crate) f64);

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
