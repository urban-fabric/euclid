#[test]
fn angle_degrees() {
    accessor_agrees!(crate::AngleDegrees, as_degrees);
    display_distinguishes!(crate::AngleDegrees);
}

#[test]
fn angle_radians() {
    accessor_agrees!(crate::AngleRadians, as_radians);
    abs_agrees!(crate::AngleRadians, as_radians);
    display_distinguishes!(crate::AngleRadians);
}

#[test]
fn area_square_kilometers() {
    accessor_agrees!(crate::AreaSquareKilometers, as_square_kilometers);
    display_distinguishes!(crate::AreaSquareKilometers);
}

#[test]
fn area_square_meters() {
    accessor_agrees!(crate::AreaSquareMeters, as_square_meters);
    display_distinguishes!(crate::AreaSquareMeters);
}

#[test]
fn length_feet() {
    accessor_agrees!(crate::LengthFeet, as_feet);
    display_distinguishes!(crate::LengthFeet);
}

#[test]
fn length_km() {
    accessor_agrees!(crate::LengthKm, as_km);
    display_distinguishes!(crate::LengthKm);
}

full_surface!(length_meters, crate::LengthMeters, as_meters);
full_surface!(
    volume_cubic_meters,
    crate::VolumeCubicMeters,
    as_cubic_meters
);
