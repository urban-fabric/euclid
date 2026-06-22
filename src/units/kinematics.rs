//! Kinematic dimensions — rates of change in time: [`Time`], [`Frequency`], [`Speed`],
//! [`Acceleration`], [`Jerk`], [`AngularVelocity`], [`AngularAcceleration`],
//! [`VolumetricFlowRate`].
//!
//! These close the differentiation/integration ladder `length → speed → acceleration → jerk`
//! (each `/ time`) and its angular twin, plus the reciprocal pair `time ⇄ frequency`. See the
//! [`geometry`](super::geometry) module docs for the `times_<operand>` / `per_<operand>` naming
//! convention shared across the units module.

use super::{Angle, Area, Force, LinearQuantity, Length, Mass, Volume};

/// A time-dimensioned quantity (`[T]`). `time × speed = length`, `time × acceleration = speed`,
/// and the reciprocal `1 / time = frequency`.
pub trait Time: LinearQuantity {
    /// The length produced by `time × speed`.
    type Length: Length<Repr = Self::Repr>;
    /// The speed that pairs with this time.
    type Speed: Speed<Repr = Self::Repr>;
    /// The acceleration that pairs with this time.
    type Acceleration: Acceleration<Repr = Self::Repr>;
    /// The frequency that is this time's reciprocal.
    type Frequency: Frequency<Repr = Self::Repr>;

    /// `time × speed = length`.
    fn times_speed(self, speed: Self::Speed) -> Self::Length;
    /// `time × acceleration = speed`.
    fn times_acceleration(self, accel: Self::Acceleration) -> Self::Speed;
    /// `1 / time = frequency`.
    fn frequency(self) -> Self::Frequency;
}

/// A frequency-dimensioned quantity (`[T⁻¹]`). The reciprocal of [`Time`].
pub trait Frequency: LinearQuantity {
    /// The time that is this frequency's reciprocal.
    type Time: Time<Repr = Self::Repr>;

    /// `1 / frequency = time` (the period).
    fn period(self) -> Self::Time;
}

/// A speed-dimensioned quantity (`[L T⁻¹]`). `speed × time = length`,
/// `speed / time = acceleration`.
pub trait Speed: LinearQuantity {
    /// The length produced by `speed × time`.
    type Length: Length<Repr = Self::Repr>;
    /// The time that pairs with this speed.
    type Time: Time<Repr = Self::Repr>;
    /// The acceleration produced by `speed / time`.
    type Acceleration: Acceleration<Repr = Self::Repr>;

    /// `speed × time = length`.
    fn times_time(self, elapsed: Self::Time) -> Self::Length;
    /// `speed / time = acceleration`.
    fn per_time(self, elapsed: Self::Time) -> Self::Acceleration;
    /// `speed / acceleration = time`.
    fn per_acceleration(self, accel: Self::Acceleration) -> Self::Time;
}

/// An acceleration-dimensioned quantity (`[L T⁻²]`). `acceleration × time = speed`,
/// `mass × acceleration = force`, `acceleration / time = jerk`.
pub trait Acceleration: LinearQuantity {
    /// The speed produced by `acceleration × time`.
    type Speed: Speed<Repr = Self::Repr>;
    /// The time that pairs with this acceleration.
    type Time: Time<Repr = Self::Repr>;
    /// The force produced by `acceleration × mass`.
    type Force: Force<Repr = Self::Repr>;
    /// The mass that pairs with this acceleration to make a force.
    type Mass: Mass<Repr = Self::Repr>;
    /// The jerk produced by `acceleration / time`.
    type Jerk: Jerk<Repr = Self::Repr>;

    /// `acceleration × time = speed`.
    fn times_time(self, elapsed: Self::Time) -> Self::Speed;
    /// `acceleration × mass = force` (Newton's second law).
    fn times_mass(self, mass: Self::Mass) -> Self::Force;
    /// `acceleration / time = jerk`.
    fn per_time(self, elapsed: Self::Time) -> Self::Jerk;
}

/// A jerk-dimensioned quantity (`[L T⁻³]`) — the time derivative of acceleration.
/// `jerk × time = acceleration`.
pub trait Jerk: LinearQuantity {
    /// The acceleration produced by `jerk × time`.
    type Acceleration: Acceleration<Repr = Self::Repr>;
    /// The time that pairs with this jerk.
    type Time: Time<Repr = Self::Repr>;

    /// `jerk × time = acceleration`.
    fn times_time(self, elapsed: Self::Time) -> Self::Acceleration;
}

/// An angular-velocity-dimensioned quantity (`[T⁻¹]`). `angular velocity × time = angle`,
/// `angular velocity / time = angular acceleration`.
pub trait AngularVelocity: LinearQuantity {
    /// The angle produced by `angular velocity × time`.
    type Angle: Angle<Repr = Self::Repr>;
    /// The time that pairs with this angular velocity.
    type Time: Time<Repr = Self::Repr>;
    /// The angular acceleration produced by `angular velocity / time`.
    type AngularAcceleration: AngularAcceleration<Repr = Self::Repr>;

    /// `angular velocity × time = angle`.
    fn times_time(self, elapsed: Self::Time) -> Self::Angle;
    /// `angular velocity / time = angular acceleration`.
    fn per_time(self, elapsed: Self::Time) -> Self::AngularAcceleration;
}

/// An angular-acceleration-dimensioned quantity (`[T⁻²]`).
/// `angular acceleration × time = angular velocity`.
pub trait AngularAcceleration: LinearQuantity {
    /// The angular velocity produced by `angular acceleration × time`.
    type AngularVelocity: AngularVelocity<Repr = Self::Repr>;
    /// The time that pairs with this angular acceleration.
    type Time: Time<Repr = Self::Repr>;

    /// `angular acceleration × time = angular velocity`.
    fn times_time(self, elapsed: Self::Time) -> Self::AngularVelocity;
}

/// A volumetric-flow-rate-dimensioned quantity (`[L³ T⁻¹]`). `flow rate × time = volume`,
/// `flow rate / area = speed`.
pub trait VolumetricFlowRate: LinearQuantity {
    /// The volume produced by `flow rate × time`.
    type Volume: Volume<Repr = Self::Repr>;
    /// The time that pairs with this flow rate.
    type Time: Time<Repr = Self::Repr>;
    /// The area that pairs with this flow rate and a speed.
    type Area: Area<Repr = Self::Repr>;
    /// The speed produced by `flow rate / area`.
    type Speed: Speed<Repr = Self::Repr>;

    /// `flow rate × time = volume`.
    fn times_time(self, elapsed: Self::Time) -> Self::Volume;
    /// `flow rate / area = speed`.
    fn per_area(self, section: Self::Area) -> Self::Speed;
    /// `flow rate / speed = area`.
    fn per_speed(self, speed: Self::Speed) -> Self::Area;
}
