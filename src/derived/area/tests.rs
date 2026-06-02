use crate::{AreaSquareKilometers, AreaSquareMeters, Quantity};

#[test]
fn area_round_trips() {
    let sq_km = AreaSquareKilometers::new(1.5);
    let sq_m = sq_km.to_square_meters();
    assert_eq!(sq_m, AreaSquareMeters::new(1_500_000.0));
    assert_eq!(sq_m.to_square_kilometers(), sq_km);
}

#[test]
fn area_quantity_impls() {
    assert_eq!(AreaSquareMeters::DIMENSION, crate::Dimension::Area);
    assert_eq!(AreaSquareMeters::UNIT_NAME, "m^2");
    assert_eq!(AreaSquareKilometers::UNIT_NAME, "km^2");
}
