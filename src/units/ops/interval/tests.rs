use crate::{Interval, LengthMeters, SpeedMs, TimeSeconds};

fn meters(lo: f64, hi: f64) -> Interval<LengthMeters> {
    Interval::new(LengthMeters::new(lo), LengthMeters::new(hi))
}

#[test]
fn new_orders_its_bounds() {
    let i = meters(5.0, 1.0);
    assert_eq!(i.lo(), LengthMeters::new(1.0));
    assert_eq!(i.hi(), LengthMeters::new(5.0));
}

#[test]
fn midpoint_radius_round_trip() {
    let i = Interval::from_midpoint_radius(LengthMeters::new(10.0), LengthMeters::new(2.0));
    assert_eq!(i.lo(), LengthMeters::new(8.0));
    assert_eq!(i.hi(), LengthMeters::new(12.0));
    assert_eq!(i.midpoint(), LengthMeters::new(10.0));
    assert_eq!(i.radius(), LengthMeters::new(2.0));
    assert_eq!(i.width(), LengthMeters::new(4.0));
}

#[test]
fn contains_is_closed() {
    let i = meters(1.0, 5.0);
    assert!(i.contains(LengthMeters::new(1.0)));
    assert!(i.contains(LengthMeters::new(5.0)));
    assert!(i.contains(LengthMeters::new(3.0)));
    assert!(!i.contains(LengthMeters::new(5.5)));
}

#[test]
fn addition_adds_endpoints() {
    let sum = meters(1.0, 2.0) + meters(10.0, 20.0);
    assert_eq!(sum, meters(11.0, 22.0));
}

#[test]
fn subtraction_crosses_endpoints() {
    let diff = meters(1.0, 2.0) - meters(10.0, 20.0);
    assert_eq!(diff, meters(-19.0, -8.0));
}

#[test]
fn point_minus_itself_is_not_necessarily_zero_width() {
    let x = meters(1.0, 2.0);
    assert_eq!(x - x, meters(-1.0, 1.0));
}

#[test]
fn scalar_multiplication_handles_sign() {
    assert_eq!(meters(1.0, 2.0) * 3.0, meters(3.0, 6.0));
    assert_eq!(meters(1.0, 2.0) * -3.0, meters(-6.0, -3.0));
    assert_eq!(2.0 * meters(1.0, 2.0), meters(2.0, 4.0));
}

#[test]
fn speed_band_from_distance_and_time_bands() {
    let distance = meters(100.0, 110.0);
    let time = Interval::new(TimeSeconds::new(9.0), TimeSeconds::new(10.0));
    let speed = distance / time;
    assert_eq!(speed.lo(), SpeedMs::new(10.0));
    assert!((speed.hi().as_ms() - (110.0 / 9.0)).abs() < 1e-9);
}

#[test]
fn distance_band_from_speed_and_time_bands() {
    let speed = Interval::new(SpeedMs::new(10.0), SpeedMs::new(12.0));
    let time = Interval::new(TimeSeconds::new(2.0), TimeSeconds::new(3.0));
    assert_eq!(speed * time, meters(20.0, 36.0));
    assert_eq!(time * speed, meters(20.0, 36.0));
}

#[test]
fn time_band_from_distance_over_speed() {
    let distance = meters(100.0, 120.0);
    let speed = Interval::new(SpeedMs::new(10.0), SpeedMs::new(12.0));
    let time = distance / speed;
    assert_eq!(time.hi(), TimeSeconds::new(12.0));
    assert!((time.lo().as_seconds() - (100.0 / 12.0)).abs() < 1e-9);
}

#[test]
fn clamp_pins_values_to_the_interval() {
    let i = meters(1.0, 5.0);
    assert_eq!(i.clamp(LengthMeters::new(-3.0)), LengthMeters::new(1.0));
    assert_eq!(i.clamp(LengthMeters::new(9.0)), LengthMeters::new(5.0));
    assert_eq!(i.clamp(LengthMeters::new(3.0)), LengthMeters::new(3.0));
}

#[test]
fn set_relations_between_intervals() {
    let outer = meters(0.0, 10.0);
    let inner = meters(2.0, 4.0);
    let overlapping = meters(8.0, 12.0);
    let disjoint = meters(20.0, 30.0);

    assert!(outer.contains_interval(inner));
    assert!(!inner.contains_interval(outer));

    assert!(outer.intersects(overlapping));
    assert!(!outer.intersects(disjoint));

    assert_eq!(outer.intersection(overlapping), Some(meters(8.0, 10.0)));
    assert_eq!(outer.intersection(disjoint), None);

    assert_eq!(inner.hull(overlapping), meters(2.0, 12.0));
}

#[test]
fn divisor_straddling_zero_is_fully_unbounded() {
    let distance = meters(1.0, 2.0);
    let time = Interval::new(TimeSeconds::new(-1.0), TimeSeconds::new(1.0));
    let speed = distance / time;
    assert_eq!(speed.lo(), SpeedMs::new(f64::NEG_INFINITY));
    assert_eq!(speed.hi(), SpeedMs::new(f64::INFINITY));
}
