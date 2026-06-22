//! Chemical dimensions: [`AmountOfSubstance`], [`CatalyticActivity`], [`Concentration`],
//! [`MolarMass`], [`MolarEnergy`], [`MolarVolume`], [`Molality`].
//!
//! [`AmountOfSubstance`] is the SI base dimension `[N]` (the mole). The derived quantities relate
//! it to time (reaction rate), volume (concentration and molar volume), mass (molar mass and
//! molality), and energy (molar energy). See the [`geometry`](super::geometry) module docs for the
//! naming convention.

use super::{Energy, LinearQuantity, Mass, Time, Volume};

/// An amount-of-substance dimensioned quantity (`[N]`, the SI base dimension, the mole).
/// `amount / time = catalytic activity`, `amount / volume = concentration`,
/// `amount × molar mass = mass`, `amount / mass = molality`.
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
    /// The molality produced by `amount / mass` (of solvent).
    type Molality: Molality<Repr = Self::Repr>;

    /// `amount / time = catalytic activity` (reaction rate).
    fn per_time(self, elapsed: Self::Time) -> Self::CatalyticActivity;
    /// `amount / volume = concentration`.
    fn per_volume(self, volume: Self::Volume) -> Self::Concentration;
    /// `amount × molar mass = mass`.
    fn times_molar_mass(self, molar_mass: Self::MolarMass) -> Self::Mass;
    /// `amount / mass = molality` (moles of solute per mass of solvent).
    fn per_mass(self, mass: Self::Mass) -> Self::Molality;
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

/// A molar-energy dimensioned quantity (`[L² M T⁻² N⁻¹]`, joules per mole). Energy per amount of
/// substance — the dimension of molar enthalpy, molar Gibbs energy, and bond energies.
/// `molar energy × amount of substance = energy`.
pub trait MolarEnergy: LinearQuantity {
    /// The amount of substance that pairs with this molar energy.
    type AmountOfSubstance: AmountOfSubstance<Repr = Self::Repr>;
    /// The energy produced by `molar energy × amount of substance`.
    type Energy: Energy<Repr = Self::Repr>;

    /// `molar energy × amount of substance = energy`.
    fn times_amount(self, amount: Self::AmountOfSubstance) -> Self::Energy;
}

/// A molar-volume dimensioned quantity (`[L³ N⁻¹]`, cubic metres per mole). Volume per amount of
/// substance — the reciprocal of (molar) [`Concentration`]. `molar volume × amount of substance =
/// volume`, and `1 / molar volume = concentration`.
pub trait MolarVolume: LinearQuantity {
    /// The amount of substance that pairs with this molar volume.
    type AmountOfSubstance: AmountOfSubstance<Repr = Self::Repr>;
    /// The volume produced by `molar volume × amount of substance`.
    type Volume: Volume<Repr = Self::Repr>;
    /// The concentration that is this molar volume's reciprocal.
    type Concentration: Concentration<Repr = Self::Repr>;

    /// `molar volume × amount of substance = volume`.
    fn times_amount(self, amount: Self::AmountOfSubstance) -> Self::Volume;
    /// `1 / molar volume = concentration`.
    fn concentration(self) -> Self::Concentration;
}

/// A molality dimensioned quantity (`[M⁻¹ N]`, moles per kilogram). Amount of solute per mass of
/// solvent — the temperature-independent companion to (molar) [`Concentration`]. `molality × mass =
/// amount of substance`.
pub trait Molality: LinearQuantity {
    /// The mass (of solvent) that pairs with this molality.
    type Mass: Mass<Repr = Self::Repr>;
    /// The amount of substance produced by `molality × mass`.
    type AmountOfSubstance: AmountOfSubstance<Repr = Self::Repr>;

    /// `molality × mass = amount of substance`.
    fn times_mass(self, mass: Self::Mass) -> Self::AmountOfSubstance;
}
