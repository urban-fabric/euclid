#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
use crate::{Dimension, Quantity};

use super::*;

#[test]
fn time_seconds_constants() {
    assert_eq!(TimeSeconds::ZERO.as_seconds(), 0.0);
    assert_eq!(TimeSeconds::DIMENSION, Dimension::Time);
    assert_eq!(TimeSeconds::UNIT_NAME, "s");
}

#[test]
fn time_seconds_arithmetic() {
    let a = TimeSeconds::new(60.0);
    let b = TimeSeconds::new(15.0);
    assert_eq!((a + b).as_seconds(), 75.0);
    assert_eq!((a - b).as_seconds(), 45.0);
    assert_eq!((a * 2.0).as_seconds(), 120.0);
    let sum: TimeSeconds = [a, b].into_iter().sum();
    assert_eq!(sum.as_seconds(), 75.0);
}

#[test]
fn time_round_trips() {
    let s = TimeSeconds::new(3_600.0);
    assert!((s.to_minutes().as_minutes() - 60.0).abs() < 1e-12);
    assert!((s.to_hours().as_hours() - 1.0).abs() < 1e-12);
    assert!((s.to_hours().to_seconds().as_seconds() - 3_600.0).abs() < 1e-9);
}

#[test]
fn time_nanos_constants() {
    assert_eq!(TimeNanos::ZERO.as_nanos(), 0);
    assert_eq!(TimeNanos::DIMENSION, Dimension::Time);
    assert_eq!(TimeNanos::UNIT_NAME, "ns");
    assert_eq!(TimeNanosDelta::UNIT_NAME, "ns (signed)");
}

#[test]
fn sub_second_units_display_with_unit_suffix() {
    assert_eq!(format!("{}", TimeNanos::new(100)), "100 ns");
    assert_eq!(format!("{}", TimeNanosDelta::new(-5)), "-5 ns (signed)");
    assert_eq!(format!("{}", TimePicos::new(7)), "7 ps");
    assert_eq!(format!("{}", TimePicosDelta::new(-7)), "-7 ps (signed)");
    assert_eq!(format!("{}", TimeMicros::new(3)), "3 µs");
    assert_eq!(format!("{}", TimeMicrosDelta::new(-3)), "-3 µs (signed)");
}

#[test]
fn time_nanos_arithmetic() {
    let a = TimeNanos::new(100);
    let b = TimeNanos::new(250);
    assert_eq!((a + b).as_nanos(), 350);
    assert_eq!(a.saturating_add(b).as_nanos(), 350);
    assert_eq!(a.saturating_sub(b).as_nanos(), 0);
    assert_eq!(b.saturating_sub(a).as_nanos(), 150);

    let mut acc = TimeNanos::ZERO;
    acc += a;
    assert_eq!(acc, a);

    let sum: TimeNanos = [a, b].into_iter().sum();
    assert_eq!(sum.as_nanos(), 350);

    assert_eq!(
        TimeNanos::new(u64::MAX).saturating_add(a).as_nanos(),
        u64::MAX
    );
}

#[test]
fn time_nanos_conversions() {
    assert_eq!(TimeNanos::new(2_500).to_micros().as_micros(), 2); // truncating
    assert_eq!(
        TimeNanos::from_duration(std::time::Duration::from_micros(5)).as_nanos(),
        5_000
    );
    assert!((TimeNanos::new(1_000_000_000).to_seconds().as_seconds() - 1.0).abs() < 1e-12);
    assert!((TimeNanos::new(1_000_000).to_millis().as_millis() - 1.0).abs() < 1e-9);
}

#[test]
fn time_nanos_delta_between() {
    let before = TimeNanos::new(100);
    let after = TimeNanos::new(450);
    assert_eq!(TimeNanosDelta::between(before, after).as_nanos(), 350);
    assert_eq!(TimeNanosDelta::between(after, before).as_nanos(), -350);
}

#[test]
fn time_picos_constants() {
    assert_eq!(TimePicos::ZERO.as_picos(), 0);
    assert_eq!(TimePicos::DIMENSION, Dimension::Time);
    assert_eq!(TimePicos::UNIT_NAME, "ps");
    assert_eq!(TimePicosDelta::UNIT_NAME, "ps (signed)");
}

#[test]
fn time_picos_arithmetic() {
    let a = TimePicos::new(1_500);
    let b = TimePicos::new(500);
    assert_eq!((a + b).as_picos(), 2_000);
    assert_eq!(a.saturating_sub(b).as_picos(), 1_000);
    assert_eq!(b.saturating_sub(a).as_picos(), 0);
    let sum: TimePicos = [a, b].into_iter().sum();
    assert_eq!(sum.as_picos(), 2_000);
}

#[test]
fn time_picos_conversions() {
    // 1_500 ps -> 1 ns (truncating), the picosecond->nanosecond step huygens needs.
    assert_eq!(TimePicos::new(1_500).to_nanos().as_nanos(), 1);
    assert_eq!(TimePicos::new(2_500_000).to_micros().as_micros(), 2);
    assert!((TimePicos::new(1_000_000_000_000).to_seconds().as_seconds() - 1.0).abs() < 1e-12);
    assert_eq!(
        TimePicos::from_duration(std::time::Duration::from_nanos(7)).as_picos(),
        7_000
    );
}

#[test]
fn time_picos_delta_between() {
    let before = TimePicos::new(1_000);
    let after = TimePicos::new(1_350);
    assert_eq!(TimePicosDelta::between(before, after).as_picos(), 350);
    assert_eq!(TimePicosDelta::between(after, before).as_picos(), -350);
}
