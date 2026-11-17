//! Vocabulary laws for the toy unit carriers.

use super::*;

#[test]
fn dimensionless_ratios_compose() {
    let half = Rat(0.5);
    let third = Rat(3.0).recip();
    assert!(approx(third.clone().raw(), 1.0 / 3.0));
    assert!(approx(half.clone().times(Rat(4.0)).raw(), 2.0));
    assert!(approx(Rat(6.0).per(Rat(2.0)).raw(), 3.0));
    assert!(approx(Rat::unit().raw(), 1.0));
    // dimensionless still scales a dimensioned quantity through LinearQuantity::scale.
    assert!(approx(Len(10.0).scale(half.raw()).raw(), 5.0));
}

#[test]
fn type_level_markers_bridge_to_runtime() {
    // each marker maps to its runtime dimension and inherits the matching exponents.
    assert_eq!(<dim::Force as Dimensioned>::DIMENSION, Dimension::Force);
    assert_eq!(
        <dim::Force as Dimensioned>::EXPONENTS,
        Dimension::Force.exponents()
    );
    assert_eq!(<dim::Length as Dimensioned>::DIMENSION, Dimension::Length);
    assert_eq!(
        <dim::Voltage as Dimensioned>::EXPONENTS,
        [2, 1, -3, -1, 0, 0, 0]
    );
    assert_eq!(
        <dim::Dimensionless as Dimensioned>::DIMENSION,
        Dimension::Dimensionless
    );
    // a marker is a zero-sized tag.
    assert_eq!(core::mem::size_of::<dim::Energy>(), 0);
}

#[test]
fn angle_conversion_and_normalization() {
    let half_turn = Ang::from_degrees(180.0);
    assert!(approx(half_turn.radians(), std::f64::consts::PI));
    assert!(approx(half_turn.degrees(), 180.0));
    assert!(approx(
        Ang::from_degrees(450.0).normalized_unsigned().degrees(),
        90.0
    ));
    assert!(approx(
        Ang::from_degrees(270.0).normalized_signed().degrees(),
        -90.0
    ));
}

#[test]
fn unit_metadata() {
    assert_eq!(Len::UNIT_NAME, "m");
    assert_eq!(Frc::UNIT_NAME, "N");
    assert_eq!(Spd::DIMENSION, Dimension::Speed);
    assert_eq!(Cur::DIMENSION, Dimension::ElectricCurrent);
    assert!(Dimension::Length.same(Dimension::Length));
    assert!(!Dimension::Length.same(Dimension::Time));
}
