use bytemuck::{Pod, Zeroable};

use super::{Dimension, LinearQuantity, Quantity};

pub(crate) const KG_PER_GRAM: f64 = 0.001;
pub(crate) const KG_PER_MILLIGRAM: f64 = 0.000_001;

#[repr(transparent)]
#[derive(
    Copy,
    Clone,
    Debug,
    Default,
    PartialEq,
    PartialOrd,
    Pod,
    Zeroable,
    serde::Serialize,
    serde::Deserialize,
)]
#[serde(transparent)]
pub struct MassKilograms(f64);

#[repr(transparent)]
#[derive(
    Copy,
    Clone,
    Debug,
    Default,
    PartialEq,
    PartialOrd,
    Pod,
    Zeroable,
    serde::Serialize,
    serde::Deserialize,
)]
#[serde(transparent)]
pub struct MassGrams(f64);

#[repr(transparent)]
#[derive(
    Copy,
    Clone,
    Debug,
    Default,
    PartialEq,
    PartialOrd,
    Pod,
    Zeroable,
    serde::Serialize,
    serde::Deserialize,
)]
#[serde(transparent)]
pub struct MassMilligrams(f64);

impl MassKilograms {
    pub const ZERO: Self = Self(0.0);

    pub const fn new(v: f64) -> Self {
        debug_assert!(!v.is_nan());
        Self(v)
    }

    pub const fn as_kilograms(self) -> f64 {
        self.0
    }

    pub fn to_grams(self) -> MassGrams {
        MassGrams(self.0 / KG_PER_GRAM)
    }

    pub fn to_milligrams(self) -> MassMilligrams {
        MassMilligrams(self.0 / KG_PER_MILLIGRAM)
    }
}

impl MassGrams {
    pub const ZERO: Self = Self(0.0);

    pub const fn new(v: f64) -> Self {
        debug_assert!(!v.is_nan());
        Self(v)
    }

    pub const fn as_grams(self) -> f64 {
        self.0
    }

    pub fn to_kilograms(self) -> MassKilograms {
        MassKilograms(self.0 * KG_PER_GRAM)
    }

    pub fn to_milligrams(self) -> MassMilligrams {
        MassMilligrams(self.0 * 1_000.0)
    }
}

impl MassMilligrams {
    pub const ZERO: Self = Self(0.0);

    pub const fn new(v: f64) -> Self {
        debug_assert!(!v.is_nan());
        Self(v)
    }

    pub const fn as_milligrams(self) -> f64 {
        self.0
    }

    pub fn to_kilograms(self) -> MassKilograms {
        MassKilograms(self.0 * KG_PER_MILLIGRAM)
    }

    pub fn to_grams(self) -> MassGrams {
        MassGrams(self.0 / 1_000.0)
    }
}

impl Quantity for MassKilograms {
    type Repr = f64;
    const DIMENSION: Dimension = Dimension::Mass;
    const UNIT_NAME: &'static str = "kg";
    fn raw(self) -> f64 {
        self.0
    }
    fn from_raw(v: f64) -> Self {
        Self(v)
    }
}

impl LinearQuantity for MassKilograms {}

impl Quantity for MassGrams {
    type Repr = f64;
    const DIMENSION: Dimension = Dimension::Mass;
    const UNIT_NAME: &'static str = "g";
    fn raw(self) -> f64 {
        self.0
    }
    fn from_raw(v: f64) -> Self {
        Self(v)
    }
}

impl LinearQuantity for MassGrams {}

impl Quantity for MassMilligrams {
    type Repr = f64;
    const DIMENSION: Dimension = Dimension::Mass;
    const UNIT_NAME: &'static str = "mg";
    fn raw(self) -> f64 {
        self.0
    }
    fn from_raw(v: f64) -> Self {
        Self(v)
    }
}

impl LinearQuantity for MassMilligrams {}

#[cfg(test)]
mod tests;
