//! Raw persisted unit wrappers.
//!
//! These types name compact storage fields such as centiseconds and decimeters.
//! They do not carry route-cost sentinel semantics themselves. Convert to
//! `crate::weights` when a value should participate in route-cost ordering or
//! treat `u32::MAX` as infinity.

use std::fmt;

use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};

use crate::{DistanceWeight, TimeWeight};

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

    pub const fn to_weight(self) -> TimeWeight {
        TimeWeight::from_centiseconds(self.0)
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

impl From<TimeCentiseconds> for TimeWeight {
    fn from(value: TimeCentiseconds) -> Self {
        TimeWeight::from_centiseconds(value.0)
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

    pub const fn to_weight(self) -> DistanceWeight {
        DistanceWeight::from_decimeters(self.0)
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

impl From<DistanceDecimeters> for DistanceWeight {
    fn from(value: DistanceDecimeters) -> Self {
        DistanceWeight::from_decimeters(value.0)
    }
}
#[cfg(test)]
mod tests;
