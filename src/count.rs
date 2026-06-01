//! Dimensionless count newtypes (`Count`, `ThreadCount`). Relocated from
//! kolmogorov-primitives (`ResettableCounter` stays behind in kolmogorov — it's
//! a control-flow helper, not a metric).

use std::fmt;
use std::ops::{Add, AddAssign, Sub, SubAssign};

use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};

#[repr(transparent)]
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
pub struct Count(pub u64);

impl Count {
    pub const ZERO: Self = Self(0);

    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    pub const fn as_u64(self) -> u64 {
        self.0
    }

    pub fn as_f32(self) -> f32 {
        self.0 as f32
    }

    pub fn as_f64(self) -> f64 {
        self.0 as f64
    }

    pub fn saturating_sub(self, rhs: Self) -> Self {
        Self(self.0.saturating_sub(rhs.0))
    }
}

impl fmt::Display for Count {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Add for Count {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl AddAssign for Count {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl Sub for Count {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}

impl SubAssign for Count {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}

impl AddAssign<u64> for Count {
    fn add_assign(&mut self, rhs: u64) {
        self.0 += rhs;
    }
}

impl From<u64> for Count {
    fn from(value: u64) -> Self {
        Self(value)
    }
}

impl std::str::FromStr for Count {
    type Err = std::num::ParseIntError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        s.parse::<u64>().map(Self)
    }
}

impl From<Count> for u64 {
    fn from(value: Count) -> Self {
        value.0
    }
}

impl PartialEq<u64> for Count {
    fn eq(&self, other: &u64) -> bool {
        self.0 == *other
    }
}

impl PartialEq<Count> for u64 {
    fn eq(&self, other: &Count) -> bool {
        *self == other.0
    }
}

impl PartialOrd<u64> for Count {
    fn partial_cmp(&self, other: &u64) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(other)
    }
}

#[repr(transparent)]
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
pub struct ThreadCount(pub u32);

impl ThreadCount {
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    pub const fn as_u32(self) -> u32 {
        self.0
    }

    pub fn as_usize(self) -> usize {
        self.0 as usize
    }
}

impl fmt::Display for ThreadCount {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0 == 1 {
            write!(f, "1 thread")
        } else {
            write!(f, "{} threads", self.0)
        }
    }
}

impl From<u32> for ThreadCount {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

impl From<ThreadCount> for u32 {
    fn from(value: ThreadCount) -> Self {
        value.0
    }
}
