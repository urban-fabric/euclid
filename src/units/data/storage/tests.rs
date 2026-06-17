#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
use super::*;

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
