//! Thermal dimension implementations for the toy carriers.
//!
//! Trait impls only; the carriers themselves live in
//! [`carriers`](super::carriers).

use super::carriers::*;
use crate::*;

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
