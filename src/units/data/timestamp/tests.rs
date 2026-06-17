use super::*;
use crate::{Dimension, Quantity};

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
