//! Radiological dimensions: [`Radioactivity`], [`AbsorbedDose`], [`DoseEquivalent`].
//!
//! These are the three SI named derived units the rest of `units` did not yet cover — the
//! becquerel, gray, and sievert — completing the standard set of 22. [`AbsorbedDose`] and
//! [`DoseEquivalent`] share a base-dimension signature (`[L² T⁻²]`, energy per mass) with each
//! other and with [`SpecificEnergy`](super::SpecificEnergy): all three are
//! [`commensurable`](super::Dimension::commensurable) but distinct *names*, exactly as
//! [`Energy`] and [`Torque`](super::Torque) are. The biological weighting that
//! turns an absorbed dose (gray) into a dose equivalent (sievert) is a dimensionless radiation
//! factor, carried here as a scalar on [`AbsorbedDose::weighted_by`].
//!
//! See the [`geometry`](super::geometry) module docs for the `times_<operand>` / `per_<operand>`
//! naming convention.

use super::{Dimensionless, Energy, LinearQuantity, Mass, Time};

/// A radioactivity (activity) dimensioned quantity (`[T⁻¹]`, the becquerel, one decay per second).
/// Commensurable with [`Frequency`](super::Frequency) and [`AngularVelocity`](super::AngularVelocity)
/// but tracked as a distinct name. `activity × time = expected number of decays`.
pub trait Radioactivity: LinearQuantity {
    /// The time that pairs with this activity.
    type Time: Time<Repr = Self::Repr>;
    /// The dimensionless decay count produced by `activity × time`.
    type Count: Dimensionless<Repr = Self::Repr>;

    /// `activity × time = expected number of decays` (a pure count).
    fn times_time(self, elapsed: Self::Time) -> Self::Count;
}

/// An absorbed-dose dimensioned quantity (`[L² T⁻²]`, the gray, joules per kilogram). The energy
/// imparted by ionizing radiation per unit mass. `absorbed dose × mass = energy`, and weighting by
/// a dimensionless radiation factor yields a [`DoseEquivalent`] (sievert).
pub trait AbsorbedDose: LinearQuantity {
    /// The mass that pairs with this dose.
    type Mass: Mass<Repr = Self::Repr>;
    /// The energy produced by `absorbed dose × mass`.
    type Energy: Energy<Repr = Self::Repr>;
    /// The dose equivalent produced by weighting this absorbed dose.
    type DoseEquivalent: DoseEquivalent<Repr = Self::Repr>;

    /// `absorbed dose × mass = energy` (the energy deposited in that mass).
    fn times_mass(self, mass: Self::Mass) -> Self::Energy;
    /// `absorbed dose × wᵣ = dose equivalent`, scaling the gray value by a dimensionless
    /// radiation-weighting factor to obtain sieverts.
    fn weighted_by(self, radiation_weight: Self::Repr) -> Self::DoseEquivalent;
}

/// A dose-equivalent dimensioned quantity (`[L² T⁻²]`, the sievert). Shares a base signature with
/// [`AbsorbedDose`] (the gray) but is a distinct name: it carries the biological weighting that
/// makes doses of different radiation types comparable. `dose equivalent × mass = energy`.
pub trait DoseEquivalent: LinearQuantity {
    /// The mass that pairs with this dose equivalent.
    type Mass: Mass<Repr = Self::Repr>;
    /// The energy produced by `dose equivalent × mass`.
    type Energy: Energy<Repr = Self::Repr>;

    /// `dose equivalent × mass = energy` (the weighted energy over that mass).
    fn times_mass(self, mass: Self::Mass) -> Self::Energy;
}
