use crate::{
    AccelerationCentimetersPerSecondSquared, AccelerationMetersPerSecondSquared, Quantity,
};

#[test]
fn acceleration_round_trips() {
    let meters = AccelerationMetersPerSecondSquared::new(9.81);
    let centimeters = meters.to_centimeters_per_second_squared();
    assert_eq!(
        centimeters,
        AccelerationCentimetersPerSecondSquared::new(981.0)
    );
    assert_eq!(centimeters.to_meters_per_second_squared(), meters);
}

#[test]
fn acceleration_quantity_impls() {
    assert_eq!(
        AccelerationMetersPerSecondSquared::DIMENSION,
        crate::Dimension::Acceleration
    );
    assert_eq!(AccelerationMetersPerSecondSquared::UNIT_NAME, "m/s^2");
    assert_eq!(AccelerationCentimetersPerSecondSquared::UNIT_NAME, "cm/s^2");
}
