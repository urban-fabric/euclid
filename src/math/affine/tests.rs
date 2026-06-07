#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use crate::{LengthMeters, Point, TimeSeconds};

type Instant = Point<TimeSeconds>;

#[test]
fn position_minus_position_is_a_displacement() {
    let start = Instant::new(TimeSeconds::new(10.0));
    let end = Instant::new(TimeSeconds::new(25.0));
    let elapsed: TimeSeconds = end - start;
    assert_eq!(elapsed, TimeSeconds::new(15.0));
}

#[test]
fn translating_a_position_yields_a_position() {
    let start = Instant::new(TimeSeconds::new(10.0));
    let later = start + TimeSeconds::new(5.0);
    assert_eq!(later, Instant::new(TimeSeconds::new(15.0)));
    assert_eq!(later - TimeSeconds::new(5.0), start);
}

#[test]
fn assignment_translation() {
    let mut p = Point::new(LengthMeters::new(100.0));
    p += LengthMeters::new(25.0);
    p -= LengthMeters::new(5.0);
    assert_eq!(p.value(), LengthMeters::new(120.0));
}

#[test]
fn midpoint_is_an_affine_combination() {
    let a = Point::new(LengthMeters::new(0.0));
    let b = Point::new(LengthMeters::new(10.0));
    assert_eq!(a.midpoint(b), Point::new(LengthMeters::new(5.0)));
    assert_eq!(a.lerp(b, 0.25), Point::new(LengthMeters::new(2.5)));
}

#[test]
fn ordering_follows_the_coordinate() {
    let a = Instant::new(TimeSeconds::new(1.0));
    let b = Instant::new(TimeSeconds::new(2.0));
    assert!(a < b);
}

#[test]
fn serde_round_trips_transparently() {
    let p = Point::new(LengthMeters::new(42.5));
    let json = serde_json::to_string(&p).unwrap();
    assert_eq!(json, "42.5");
    let back: Point<LengthMeters> = serde_json::from_str(&json).unwrap();
    assert_eq!(back, p);
}
