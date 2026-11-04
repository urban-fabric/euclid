//! Kinematic dimension implementations for the toy carriers.
//!
//! Trait impls only; the carriers themselves live in
//! [`carriers`](super::carriers).

use super::carriers::*;
use crate::*;

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
