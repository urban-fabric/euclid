//! Unix epoch timestamp wrappers.

use bytemuck::{Pod, Zeroable};

use super::{Dimension, Quantity};

#[repr(transparent)]
#[derive(
    Copy,
    Clone,
    Debug,
    Default,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Pod,
    Zeroable,
    serde::Serialize,
    serde::Deserialize,
)]
#[serde(transparent)]
pub struct UnixSeconds(i64);

#[repr(transparent)]
#[derive(
    Copy,
    Clone,
    Debug,
    Default,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Pod,
    Zeroable,
    serde::Serialize,
    serde::Deserialize,
)]
#[serde(transparent)]
pub struct UnixMillis(i64);

impl UnixSeconds {
    pub const EPOCH: Self = Self(0);

    pub const fn new(secs: i64) -> Self {
        Self(secs)
    }

    pub const fn as_secs(self) -> i64 {
        self.0
    }

    pub fn now() -> Self {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| {
                #[allow(clippy::cast_possible_wrap)]
                Self(d.as_secs() as i64)
            })
            .unwrap_or(Self::EPOCH)
    }
}

impl UnixMillis {
    pub const EPOCH: Self = Self(0);

    pub const MAX: Self = Self(i64::MAX);

    pub const MIN: Self = Self(i64::MIN);

    pub const fn new(ms: i64) -> Self {
        Self(ms)
    }

    pub const fn as_millis(self) -> i64 {
        self.0
    }
}

impl Quantity for UnixSeconds {
    type Repr = i64;

    const DIMENSION: Dimension = Dimension::Time;
    const UNIT_NAME: &'static str = "s (since epoch)";

    fn raw(self) -> i64 {
        self.0
    }

    fn from_raw(v: i64) -> Self {
        Self(v)
    }
}

impl Quantity for UnixMillis {
    type Repr = i64;

    const DIMENSION: Dimension = Dimension::Time;
    const UNIT_NAME: &'static str = "ms (since epoch)";

    fn raw(self) -> i64 {
        self.0
    }

    fn from_raw(v: i64) -> Self {
        Self(v)
    }
}

#[cfg(test)]
mod tests;
