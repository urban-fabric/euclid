//! Route-cost weight types.
//!
//! These wrappers store compact integer costs for routing-style algorithms.
//! They reserve `u32::MAX` as an infinity sentinel, so they are semantically
//! different from the raw persisted unit wrappers in `crate::storage`.

use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};

use crate::{Dimension, Quantity};

/// Route time cost stored in centiseconds with `u32::MAX` reserved as infinity.
#[derive(
    Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Pod, Zeroable, Serialize, Deserialize,
)]
#[serde(transparent)]
#[repr(transparent)]
pub struct TimeWeight(u32);

impl TimeWeight {
    pub const INF: Self = Self(u32::MAX);

    pub const ZERO: Self = Self(0);

    pub fn from_seconds(s: f64) -> Self {
        if !s.is_finite() {
            return Self::INF;
        }
        debug_assert!(s >= 0.0, "negative time {s}");
        let cs = (s * 100.0).round();
        if cs >= u32::MAX as f64 {
            Self::INF
        } else {
            Self(cs as u32)
        }
    }

    pub fn as_seconds(self) -> f64 {
        f64::from(self.0) / 100.0
    }

    pub const fn from_centiseconds(cs: u32) -> Self {
        Self(cs)
    }

    pub const fn as_centiseconds(self) -> u32 {
        self.0
    }

    #[inline]
    pub const fn as_centiseconds_usize(self) -> usize {
        self.0 as usize
    }

    pub fn saturating_add(self, rhs: Self) -> Self {
        Self(self.0.saturating_add(rhs.0))
    }

    pub fn is_inf(self) -> bool {
        self.0 == u32::MAX
    }

    pub const fn is_zero(self) -> bool {
        self.0 == 0
    }

    pub const fn is_positive(self) -> bool {
        self.0 > 0 && self.0 < u32::MAX
    }
}

impl Quantity for TimeWeight {
    type Repr = u32;

    const DIMENSION: Dimension = Dimension::Time;
    const UNIT_NAME: &'static str = "cs";

    fn raw(self) -> u32 {
        self.0
    }

    fn from_raw(v: u32) -> Self {
        Self(v)
    }
}

/// Route distance cost stored in decimeters with `u32::MAX` reserved as infinity.
#[derive(
    Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Pod, Zeroable, Serialize, Deserialize,
)]
#[serde(transparent)]
#[repr(transparent)]
pub struct DistanceWeight(u32);

impl DistanceWeight {
    pub const INF: Self = Self(u32::MAX);

    pub const ZERO: Self = Self(0);

    pub fn from_meters(m: f64) -> Self {
        if !m.is_finite() {
            return Self::INF;
        }
        debug_assert!(m >= 0.0, "negative distance {m}");
        let dm = (m * 10.0).round();
        if dm >= u32::MAX as f64 {
            Self::INF
        } else {
            Self(dm as u32)
        }
    }

    pub fn as_meters(self) -> f64 {
        f64::from(self.0) / 10.0
    }

    pub const fn from_decimeters(dm: u32) -> Self {
        Self(dm)
    }

    pub const fn as_decimeters(self) -> u32 {
        self.0
    }

    pub fn saturating_add(self, rhs: Self) -> Self {
        Self(self.0.saturating_add(rhs.0))
    }

    pub fn is_inf(self) -> bool {
        self.0 == u32::MAX
    }

    pub const fn is_zero(self) -> bool {
        self.0 == 0
    }

    pub const fn is_positive(self) -> bool {
        self.0 > 0 && self.0 < u32::MAX
    }
}

impl Quantity for DistanceWeight {
    type Repr = u32;

    const DIMENSION: Dimension = Dimension::Length;
    const UNIT_NAME: &'static str = "dm";

    fn raw(self) -> u32 {
        self.0
    }

    fn from_raw(v: u32) -> Self {
        Self(v)
    }
}

#[cfg(test)]
mod tests;
