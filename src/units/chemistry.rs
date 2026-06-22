//! Chemical dimensions: [`AmountOfSubstance`], [`CatalyticActivity`], [`Concentration`],
//! [`MolarMass`].
//!
//! [`AmountOfSubstance`] is the SI base dimension `[N]` (the mole). The derived quantities relate
//! it to time (reaction rate), volume (concentration), and mass (molar mass). See the
//! [`geometry`](super::geometry) module docs for the naming convention.

use super::{LinearQuantity, Mass, Time, Volume};

/// An amount-of-substance dimensioned quantity (`[N]`, the SI base dimension, the mole).
/// `amount / time = catalytic activity`, `amount / volume = concentration`,
/// `amount × molar mass = mass`.
pub trait AmountOfSubstance: LinearQuantity {
    /// The time that pairs with this amount.
    type Time: Time<Repr = Self::Repr>;
    /// The catalytic activity produced by `amount / time`.
    type CatalyticActivity: CatalyticActivity<Repr = Self::Repr>;
    /// The volume that pairs with this amount.
    type Volume: Volume<Repr = Self::Repr>;
    /// The concentration produced by `amount / volume`.
    type Concentration: Concentration<Repr = Self::Repr>;
    /// The molar mass that pairs with this amount.
    type MolarMass: MolarMass<Repr = Self::Repr>;
    /// The mass produced by `amount × molar mass`.
    type Mass: Mass<Repr = Self::Repr>;

    /// `amount / time = catalytic activity` (reaction rate).
    fn per_time(self, elapsed: Self::Time) -> Self::CatalyticActivity;
    /// `amount / volume = concentration`.
    fn per_volume(self, volume: Self::Volume) -> Self::Concentration;
    /// `amount × molar mass = mass`.
    fn times_molar_mass(self, molar_mass: Self::MolarMass) -> Self::Mass;
}

/// A catalytic-activity dimensioned quantity (`[N T⁻¹]`, the katal, `mol/s`). `catalytic activity
/// × time = amount of substance`.
pub trait CatalyticActivity: LinearQuantity {
    /// The time that pairs with this activity.
    type Time: Time<Repr = Self::Repr>;
    /// The amount of substance produced by `catalytic activity × time`.
    type AmountOfSubstance: AmountOfSubstance<Repr = Self::Repr>;

    /// `catalytic activity × time = amount of substance`.
    fn times_time(self, elapsed: Self::Time) -> Self::AmountOfSubstance;
}

/// A (molar) concentration dimensioned quantity (`[L⁻³ N]`, `mol/m³`). `concentration × volume =
/// amount of substance`.
pub trait Concentration: LinearQuantity {
    /// The volume that pairs with this concentration.
    type Volume: Volume<Repr = Self::Repr>;
    /// The amount of substance produced by `concentration × volume`.
    type AmountOfSubstance: AmountOfSubstance<Repr = Self::Repr>;

    /// `concentration × volume = amount of substance`.
    fn times_volume(self, volume: Self::Volume) -> Self::AmountOfSubstance;
}

/// A molar-mass dimensioned quantity (`[M N⁻¹]`, `kg/mol`). `molar mass × amount of substance =
/// mass`.
pub trait MolarMass: LinearQuantity {
    /// The amount of substance that pairs with this molar mass.
    type AmountOfSubstance: AmountOfSubstance<Repr = Self::Repr>;
    /// The mass produced by `molar mass × amount of substance`.
    type Mass: Mass<Repr = Self::Repr>;

    /// `molar mass × amount of substance = mass`.
    fn times_amount(self, amount: Self::AmountOfSubstance) -> Self::Mass;
}
