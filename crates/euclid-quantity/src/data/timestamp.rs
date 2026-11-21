//! Unix epoch timestamp wrappers.

use bytemuck::{Pod, Zeroable};

#[repr(transparent)]
/// An instant, as **whole seconds** since the Unix epoch, backed by `i64`.
///
/// A point on the timeline rather than a duration, so it is **affine**: two instants
/// subtract to a duration, but adding two instants is meaningless. Signed, so instants
/// before 1970 are representable.
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
/// An instant, as **whole milliseconds** since the Unix epoch, backed by `i64`. The
/// affine counterpart of [`crate::TimeMillis`], as
/// [`UnixSeconds`] is of [`crate::TimeSeconds`].
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
    /// The Unix epoch, 1970-01-01T00:00:00Z — this type's origin rather than a zero, since an
    /// instant is affine.
    pub const EPOCH: Self = Self(0);

    /// Wrap a raw second count since the epoch. Asserted rather than checked.
    pub const fn new(secs: i64) -> Self {
        Self(secs)
    }

    /// Whole seconds since the Unix epoch; negative before 1970.
    pub const fn as_secs(self) -> i64 {
        self.0
    }

    /// The current instant, from the system clock.
    ///
    /// The system clock is not monotonic: it can jump backwards across an NTP correction, so two
    /// successive calls are not ordered. Use a monotonic source to measure an elapsed interval.
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
    /// The Unix epoch, 1970-01-01T00:00:00Z — this type's origin rather than a zero, since an
    /// instant is affine.
    pub const EPOCH: Self = Self(0);

    /// The latest representable instant.
    pub const MAX: Self = Self(i64::MAX);

    /// The earliest representable instant, well before the epoch.
    pub const MIN: Self = Self(i64::MIN);

    /// Wrap a raw millisecond count since the epoch. Asserted rather than checked.
    pub const fn new(ms: i64) -> Self {
        Self(ms)
    }

    /// Whole milliseconds since the Unix epoch; negative before 1970.
    pub const fn as_millis(self) -> i64 {
        self.0
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
    use euclid_units::{Dimension, Measure};

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

    /// The epoch accessor and the wall clock.
    ///
    /// `as_secs` is a one-line unwrap and could return `0` — making every timestamp the epoch.
    /// `now()` cannot be pinned to a value, but it can be pinned to a *range*: a clock reading
    /// that falls outside a plausible window is either a broken conversion or a fallback to
    /// `EPOCH`, which is exactly what the mutation does.
    #[test]
    fn as_secs_is_the_raw_second_count() {
        for raw in [0_i64, 1, -1, 1_700_000_000, i64::MIN, i64::MAX] {
            assert_eq!(UnixSeconds::new(raw).as_secs(), raw);
        }
        assert_eq!(UnixSeconds::EPOCH.as_secs(), 0, "the epoch is zero");
    }

    #[test]
    fn now_reads_a_plausible_wall_clock() {
        let now = UnixSeconds::now().as_secs();

        // 2024-01-01 and 2100-01-01. Wide enough never to be a maintenance burden, narrow
        // enough that the epoch fallback and a units mix-up both fall outside it.
        const Y2024: i64 = 1_704_067_200;
        const Y2100: i64 = 4_102_444_800;
        assert!(
            (Y2024..Y2100).contains(&now),
            "now() returned {now}, outside a plausible window — a broken conversion or a \
             fallback to EPOCH"
        );

        // And it is not the epoch, which is the value the fallback path returns.
        assert_ne!(
            UnixSeconds::now(),
            UnixSeconds::EPOCH,
            "now() must read the clock rather than fall back"
        );
    }
}
