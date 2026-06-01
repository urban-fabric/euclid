#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
use super::*;

#[test]
fn time_cs_display_carries_unit() {
    assert_eq!(TimeCentiseconds(150).to_string(), "150 cs");
}

#[test]
fn time_cs_serde_is_bare_u32() {
    let t = TimeCentiseconds(150);
    let json = serde_json::to_string(&t).unwrap();
    assert_eq!(json, "150");
    let back: TimeCentiseconds = serde_json::from_str(&json).unwrap();
    assert_eq!(back, t);
}

#[test]
fn time_cs_to_weight_preserves_repr() {
    let t = TimeCentiseconds(150);
    let w: TimeWeight = t.into();
    assert_eq!(w.0, 150);
}

#[test]
fn distance_dm_display_carries_unit() {
    assert_eq!(DistanceDecimeters(420).to_string(), "420 dm");
}

#[test]
fn distance_dm_serde_is_bare_u32() {
    let d = DistanceDecimeters(420);
    let json = serde_json::to_string(&d).unwrap();
    assert_eq!(json, "420");
    let back: DistanceDecimeters = serde_json::from_str(&json).unwrap();
    assert_eq!(back, d);
}

#[test]
fn distance_dm_to_weight_preserves_repr() {
    let d = DistanceDecimeters(420);
    let w: DistanceWeight = d.into();
    assert_eq!(w.0, 420);
}
