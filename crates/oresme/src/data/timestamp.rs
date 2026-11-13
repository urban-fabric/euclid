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

impl euclid_units::Measure for UnixSeconds {
    type Repr = i64;
    const DIMENSION: euclid_units::Dimension = euclid_units::Dimension::Time;
    const UNIT_NAME: &'static str = "s (since epoch)";
    fn raw(&self) -> i64 {
        self.0
    }
    fn from_raw(v: i64) -> Self {
        Self(v)
    }
}

impl euclid_units::Instant for UnixSeconds {}

impl euclid_units::Measure for UnixMillis {
    type Repr = i64;
    const DIMENSION: euclid_units::Dimension = euclid_units::Dimension::Time;
    const UNIT_NAME: &'static str = "ms (since epoch)";
    fn raw(&self) -> i64 {
        self.0
    }
    fn from_raw(v: i64) -> Self {
        Self(v)
    }
}

impl euclid_units::Instant for UnixMillis {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Dimension, Quantity};

    #[test]
    fn unix_seconds_is_an_affine_instant() {
        use euclid_units::Instant;
        let t0 = UnixSeconds::new(1_000);
        let t1 = UnixSeconds::new(1_060);
        assert_eq!(t1.since(t0), 60);
        assert_eq!(t0.advance(60), t1);
    }

    #[test]
    fn unix_seconds_is_epoch_time_quantity() {
        assert_eq!(UnixSeconds::EPOCH.as_secs(), 0);
        assert_eq!(UnixSeconds::DIMENSION, Dimension::Time);
        assert_eq!(UnixSeconds::UNIT_NAME, "s (since epoch)");
        assert_eq!(UnixSeconds::from_raw(42), UnixSeconds::new(42));
        assert_eq!(UnixSeconds::new(42).raw(), 42);
    }

    #[test]
    fn unix_millis_is_epoch_time_quantity() {
        assert_eq!(UnixMillis::EPOCH.as_millis(), 0);
        assert_eq!(UnixMillis::MIN.as_millis(), i64::MIN);
        assert_eq!(UnixMillis::MAX.as_millis(), i64::MAX);
        assert_eq!(UnixMillis::DIMENSION, Dimension::Time);
        assert_eq!(UnixMillis::UNIT_NAME, "ms (since epoch)");
        assert_eq!(UnixMillis::from_raw(42), UnixMillis::new(42));
        assert_eq!(UnixMillis::new(42).raw(), 42);
    }
}
