//! [`MomentOfInertia`], [`AngularMomentum`], [`Torque`] — the rotational analogues.
//!
//! The same three relations as [`Mass`] and [`Force`], with the linear
//! quantities replaced by their angular counterparts: inertia times angular acceleration is
//! torque, inertia times angular velocity is angular momentum. Grouped because that parallel is
//! the thing worth seeing, and because [`Torque`]'s separation from `Energy` only makes sense
//! beside the chain that produces it.

use super::*;

/// A moment-of-inertia dimensioned quantity (`[L² M]`). `moment of inertia × angular velocity =
/// angular momentum`, `moment of inertia × angular acceleration = torque`.
pub trait MomentOfInertia: LinearQuantity {
    /// The mass that pairs with this moment of inertia.
    type Mass: Mass<Repr = Self::Repr>;
    /// The area produced by `moment of inertia / mass`.
    type Area: Area<Repr = Self::Repr>;
    /// The angular velocity that pairs with this moment of inertia.
    type AngularVelocity: AngularVelocity<Repr = Self::Repr>;
    /// The angular momentum produced by `moment of inertia × angular velocity`.
    type AngularMomentum: AngularMomentum<Repr = Self::Repr>;
    /// The angular acceleration that pairs with this moment of inertia.
    type AngularAcceleration: AngularAcceleration<Repr = Self::Repr>;
    /// The torque produced by `moment of inertia × angular acceleration`.
    type Torque: Torque<Repr = Self::Repr>;

    /// `moment of inertia / mass = area` (radius of gyration squared).
    fn per_mass(self, mass: Self::Mass) -> Self::Area;
    /// `moment of inertia × angular velocity = angular momentum`.
    fn times_angular_velocity(self, omega: Self::AngularVelocity) -> Self::AngularMomentum;
    /// `moment of inertia × angular acceleration = torque` (rotational Newton's second law).
    fn times_angular_acceleration(self, alpha: Self::AngularAcceleration) -> Self::Torque;
}

/// An angular-momentum dimensioned quantity (`[L² M T⁻¹]`, also the dimension of action).
/// `angular momentum / time = torque`, `angular momentum / moment of inertia = angular velocity`,
/// `angular momentum / length = momentum`.
pub trait AngularMomentum: LinearQuantity {
    /// The moment of inertia that pairs with this angular momentum.
    type MomentOfInertia: MomentOfInertia<Repr = Self::Repr>;
    /// The angular velocity produced by `angular momentum / moment of inertia`.
    type AngularVelocity: AngularVelocity<Repr = Self::Repr>;
    /// The time that pairs with this angular momentum.
    type Time: Time<Repr = Self::Repr>;
    /// The torque produced by `angular momentum / time`.
    type Torque: Torque<Repr = Self::Repr>;
    /// The length (lever arm) that pairs with this angular momentum.
    type Length: Length<Repr = Self::Repr>;
    /// The momentum produced by `angular momentum / length`.
    type Momentum: Momentum<Repr = Self::Repr>;

    /// `angular momentum / moment of inertia = angular velocity`.
    fn per_moment_of_inertia(self, inertia: Self::MomentOfInertia) -> Self::AngularVelocity;
    /// `angular momentum / time = torque`.
    fn per_time(self, elapsed: Self::Time) -> Self::Torque;
    /// `angular momentum / length = momentum`.
    fn per_length(self, lever: Self::Length) -> Self::Momentum;
}

/// A torque (moment of force) dimensioned quantity (`[L² M T⁻²]`). Shares a base signature with
/// [`Energy`] but is a distinct name. `torque / length = force`,
/// `torque × angular velocity = power`, `torque × time = angular momentum`.
pub trait Torque: LinearQuantity {
    /// The length (lever arm) that pairs with this torque.
    type Length: Length<Repr = Self::Repr>;
    /// The force produced by `torque / length`.
    type Force: Force<Repr = Self::Repr>;
    /// The angular velocity that pairs with this torque.
    type AngularVelocity: AngularVelocity<Repr = Self::Repr>;
    /// The power produced by `torque × angular velocity`.
    type Power: Power<Repr = Self::Repr>;
    /// The time that pairs with this torque.
    type Time: Time<Repr = Self::Repr>;
    /// The angular momentum produced by `torque × time`.
    type AngularMomentum: AngularMomentum<Repr = Self::Repr>;
    /// The moment of inertia that pairs with this torque.
    type MomentOfInertia: MomentOfInertia<Repr = Self::Repr>;
    /// The angular acceleration produced by `torque / moment of inertia`.
    type AngularAcceleration: AngularAcceleration<Repr = Self::Repr>;

    /// `torque / length = force`.
    fn per_length(self, lever: Self::Length) -> Self::Force;
    /// `torque × angular velocity = power`.
    fn times_angular_velocity(self, omega: Self::AngularVelocity) -> Self::Power;
    /// `torque × time = angular momentum`.
    fn times_time(self, elapsed: Self::Time) -> Self::AngularMomentum;
    /// `torque / moment of inertia = angular acceleration`.
    fn per_moment_of_inertia(self, inertia: Self::MomentOfInertia) -> Self::AngularAcceleration;
}
