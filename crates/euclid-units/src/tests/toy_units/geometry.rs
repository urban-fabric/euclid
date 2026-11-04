//! Geometric dimension implementations for the toy carriers.
//!
//! Trait impls only; the carriers themselves live in
//! [`carriers`](super::carriers).

use super::carriers::*;
use crate::*;

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
