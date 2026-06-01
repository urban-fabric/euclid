#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
use super::*;

#[test]
fn speed_constants_and_dims() {
    assert_eq!(SpeedMs::DIMENSION, Dimension::Speed);
    assert_eq!(SpeedKph::DIMENSION, Dimension::Speed);
    assert_eq!(SpeedMph::DIMENSION, Dimension::Speed);
    assert_eq!(SpeedMs::UNIT_NAME, "m/s");
    assert_eq!(SpeedKph::UNIT_NAME, "km/h");
    assert_eq!(SpeedMph::UNIT_NAME, "mph");
}

#[test]
fn speed_pair_round_trips() {
    let kph = SpeedKph::new(50.0);
    let ms = kph.to_ms();
    assert!((ms.as_ms() - 13.888_888_888_888_89).abs() < 1e-12);
    assert!((ms.to_kph().as_kph() - 50.0).abs() < 1e-9);

    let mph = SpeedMph::new(30.0);
    let ms2 = mph.to_ms();
    assert!((ms2.as_ms() - 13.4112).abs() < 1e-9);
    assert!((ms2.to_mph().as_mph() - 30.0).abs() < 1e-9);

    assert!((SpeedKph::new(100.0).to_mph().as_mph() - 62.137_119_223_733_4).abs() < 1e-9);
    assert!((SpeedMph::new(60.0).to_kph().as_kph() - 96.560_64).abs() < 1e-9);
}

#[test]
fn speed_in_dimension_arithmetic() {
    let a = SpeedKph::new(50.0);
    let b = SpeedKph::new(10.0);
    assert_eq!((a + b).as_kph(), 60.0);
    assert_eq!((a - b).as_kph(), 40.0);
    assert_eq!((a * 1.5).as_kph(), 75.0);
}
