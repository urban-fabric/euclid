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
    let kph = SpeedKph(50.0);
    let ms = kph.to_ms();
    assert!((ms.0 - 13.888_888_888_888_89).abs() < 1e-12);
    assert!((ms.to_kph().0 - 50.0).abs() < 1e-9);

    let mph = SpeedMph(30.0);
    let ms2 = mph.to_ms();
    assert!((ms2.0 - 13.4112).abs() < 1e-9);
    assert!((ms2.to_mph().0 - 30.0).abs() < 1e-9);

    assert!((SpeedKph(100.0).to_mph().0 - 62.137_119_223_733_4).abs() < 1e-9);
    assert!((SpeedMph(60.0).to_kph().0 - 96.560_64).abs() < 1e-9);
}

#[test]
fn speed_in_dimension_arithmetic() {
    let a = SpeedKph(50.0);
    let b = SpeedKph(10.0);
    assert_eq!((a + b).0, 60.0);
    assert_eq!((a - b).0, 40.0);
    assert_eq!((a * 1.5).0, 75.0);
}
