use std::fmt;

use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};

use crate::{DistanceWeight, TimeWeight};

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
pub struct TimeCentiseconds(pub u32);

impl TimeCentiseconds {
    pub const ZERO: Self = Self(0);

    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    pub const fn as_u32(self) -> u32 {
        self.0
    }

    pub const fn to_weight(self) -> TimeWeight {
        TimeWeight(self.0)
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
        TimeWeight(value.0)
    }
}

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
pub struct DistanceDecimeters(pub u32);

impl DistanceDecimeters {
    pub const ZERO: Self = Self(0);

    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    pub const fn as_u32(self) -> u32 {
        self.0
    }

    pub const fn to_weight(self) -> DistanceWeight {
        DistanceWeight(self.0)
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
        DistanceWeight(value.0)
    }
}
#[cfg(test)]
mod tests;
