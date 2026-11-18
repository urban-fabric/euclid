#[test]
fn acceleration_centimeters_per_second_squared() {
    accessor_agrees!(
        crate::AccelerationCentimetersPerSecondSquared,
        as_centimeters_per_second_squared
    );
    display_distinguishes!(crate::AccelerationCentimetersPerSecondSquared);
}

#[test]
fn acceleration_meters_per_second_squared() {
    accessor_agrees!(
        crate::AccelerationMetersPerSecondSquared,
        as_meters_per_second_squared
    );
    display_distinguishes!(crate::AccelerationMetersPerSecondSquared);
}

full_surface!(
    angular_acceleration_radians_per_second_squared,
    crate::AngularAccelerationRadiansPerSecondSquared,
    as_radians_per_second_squared
);
full_surface!(
    angular_velocity_radians_per_second,
    crate::AngularVelocityRadiansPerSecond,
    as_radians_per_second
);
full_surface!(frequency_hertz, crate::FrequencyHertz, as_hertz);
full_surface!(
    jerk_meters_per_second_cubed,
    crate::JerkMetersPerSecondCubed,
    as_meters_per_second_cubed
);
full_surface!(speed_kph, crate::SpeedKph, as_kph);
#[test]
fn speed_mph() {
    accessor_agrees!(crate::SpeedMph, as_mph);
    display_distinguishes!(crate::SpeedMph);
}

#[test]
fn speed_ms() {
    accessor_agrees!(crate::SpeedMs, as_ms);
    display_distinguishes!(crate::SpeedMs);
}

#[test]
fn time_hours() {
    accessor_agrees!(crate::TimeHours, as_hours);
    display_distinguishes!(crate::TimeHours);
}

#[test]
fn time_millis() {
    accessor_agrees!(crate::TimeMillis, as_millis);
    display_distinguishes!(crate::TimeMillis);
}

#[test]
fn time_minutes() {
    accessor_agrees!(crate::TimeMinutes, as_minutes);
    display_distinguishes!(crate::TimeMinutes);
}

#[test]
fn time_seconds() {
    accessor_agrees!(crate::TimeSeconds, as_seconds);
    sign_predicates_agree!(crate::TimeSeconds, as_seconds);
    abs_agrees!(crate::TimeSeconds, as_seconds);
    display_distinguishes!(crate::TimeSeconds);
    epsilon_predicate_agrees!(crate::TimeSeconds);
}

full_surface!(
    volumetric_flow_rate_cubic_meters_per_second,
    crate::VolumetricFlowRateCubicMetersPerSecond,
    as_cubic_meters_per_second
);
