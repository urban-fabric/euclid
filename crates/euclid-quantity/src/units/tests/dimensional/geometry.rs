#[test]
fn angle_radians_dimensional_algebra() {
    use euclid_units::Angle;
    per_method!(crate::AngleRadians, Angle, per_time, Time, AngularVelocity);
}

#[test]
fn area_square_meters_dimensional_algebra() {
    use euclid_units::Area;
    per_method!(crate::AreaSquareMeters, Area, per_length, Length, Length);
    times_method!(crate::AreaSquareMeters, Area, times_length, Length, Volume);
    times_method!(
        crate::AreaSquareMeters,
        Area,
        times_speed,
        Speed,
        VolumetricFlowRate
    );
}

#[test]
fn length_meters_dimensional_algebra() {
    use euclid_units::Length;
    times_method!(crate::LengthMeters, Length, times_area, Area, Volume);
    per_method!(crate::LengthMeters, Length, per_time, Time, Speed);
    per_method!(crate::LengthMeters, Length, per_speed, Speed, Time);
    power_method!(crate::LengthMeters, Length, squared, Area, 9.0);
    power_method!(crate::LengthMeters, Length, cubed, Volume, 27.0);
}

#[test]
fn volume_cubic_meters_dimensional_algebra() {
    use euclid_units::Volume;
    per_method!(crate::VolumeCubicMeters, Volume, per_length, Length, Area);
    per_method!(crate::VolumeCubicMeters, Volume, per_area, Area, Length);
    per_method!(
        crate::VolumeCubicMeters,
        Volume,
        per_amount,
        AmountOfSubstance,
        MolarVolume
    );
}
