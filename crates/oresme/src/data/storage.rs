//! Raw persisted unit wrappers.
//!
//! These types name compact storage fields such as centiseconds and decimeters.
//! They do not carry route-cost sentinel semantics themselves; the graph crate's
//! route-weight types convert from these when a value should treat `u32::MAX` as
//! infinity.

use std::fmt;

use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};

/// Raw centisecond storage value.
#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Pod,
    Zeroable,
    Serialize,
    Deserialize,
)]
#[serde(transparent)]
#[repr(transparent)]
pub struct TimeCentiseconds(u32);

impl TimeCentiseconds {
    pub const ZERO: Self = Self(0);

    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    pub const fn as_u32(self) -> u32 {
        self.0
    }

    pub const fn as_centiseconds(self) -> u32 {
        self.0
    }
}

impl fmt::Display for TimeCentiseconds {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} cs", self.0)
    }
}

impl From<u32> for TimeCentiseconds {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

impl From<TimeCentiseconds> for u32 {
    fn from(value: TimeCentiseconds) -> Self {
        value.0
    }
}

/// Raw decimeter storage value.
#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Pod,
    Zeroable,
    Serialize,
    Deserialize,
)]
#[serde(transparent)]
#[repr(transparent)]
pub struct DistanceDecimeters(u32);

impl DistanceDecimeters {
    pub const ZERO: Self = Self(0);

    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    pub const fn as_u32(self) -> u32 {
        self.0
    }

    pub const fn as_decimeters(self) -> u32 {
        self.0
    }
}

impl fmt::Display for DistanceDecimeters {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} dm", self.0)
    }
}

impl From<u32> for DistanceDecimeters {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

impl From<DistanceDecimeters> for u32 {
    fn from(value: DistanceDecimeters) -> Self {
        value.0
    }
}

impl euclid_units::Measure for TimeCentiseconds {
    type Repr = u32;
    const DIMENSION: euclid_units::Dimension = euclid_units::Dimension::Time;
    const UNIT_NAME: &'static str = "cs";
    fn raw(&self) -> u32 {
        self.0
    }
    fn from_raw(v: u32) -> Self {
        Self(v)
    }
}

impl euclid_units::Measure for DistanceDecimeters {
    type Repr = u32;
    const DIMENSION: euclid_units::Dimension = euclid_units::Dimension::Length;
    const UNIT_NAME: &'static str = "dm";
    fn raw(&self) -> u32 {
        self.0
    }
    fn from_raw(v: u32) -> Self {
        Self(v)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
    use super::*;

    #[test]
    fn storage_wrappers_are_dimensioned_measures() {
        use euclid_units::Measure;
        assert_eq!(
            <TimeCentiseconds as Measure>::DIMENSION,
            euclid_units::Dimension::Time
        );
        assert_eq!(
            <DistanceDecimeters as Measure>::DIMENSION,
            euclid_units::Dimension::Length
        );
        assert_eq!(Measure::raw(&DistanceDecimeters::new(7)), 7);
    }

    #[test]
    fn time_cs_display_carries_unit() {
        assert_eq!(TimeCentiseconds::new(150).to_string(), "150 cs");
    }

    #[test]
    fn time_cs_serde_is_bare_u32() {
        let t = TimeCentiseconds::new(150);
        let json = serde_json::to_string(&t).unwrap();
        assert_eq!(json, "150");
        let back: TimeCentiseconds = serde_json::from_str(&json).unwrap();
        assert_eq!(back, t);
    }

    #[test]
    fn distance_dm_display_carries_unit() {
        assert_eq!(DistanceDecimeters::new(420).to_string(), "420 dm");
    }

    #[test]
    fn distance_dm_serde_is_bare_u32() {
        let d = DistanceDecimeters::new(420);
        let json = serde_json::to_string(&d).unwrap();
        assert_eq!(json, "420");
        let back: DistanceDecimeters = serde_json::from_str(&json).unwrap();
        assert_eq!(back, d);
    }
}
