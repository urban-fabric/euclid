#[test]
fn acceleration_meters_per_second_squared_dimensional_algebra() {
    use euclid_units::Acceleration;
    times_method!(
        crate::AccelerationMetersPerSecondSquared,
        Acceleration,
        times_time,
        Time,
        Speed
    );
    times_method!(
        crate::AccelerationMetersPerSecondSquared,
        Acceleration,
        times_mass,
        Mass,
        Force
    );
    per_method!(
        crate::AccelerationMetersPerSecondSquared,
        Acceleration,
        per_time,
        Time,
        Jerk
    );
}

#[test]
fn angular_acceleration_radians_per_second_squared_dimensional_algebra() {
    use euclid_units::AngularAcceleration;
    times_method!(
        crate::AngularAccelerationRadiansPerSecondSquared,
        AngularAcceleration,
        times_time,
        Time,
        AngularVelocity
    );
}

#[test]
fn angular_velocity_radians_per_second_dimensional_algebra() {
    use euclid_units::AngularVelocity;
    times_method!(
        crate::AngularVelocityRadiansPerSecond,
        AngularVelocity,
        times_time,
        Time,
        Angle
    );
    per_method!(
        crate::AngularVelocityRadiansPerSecond,
        AngularVelocity,
        per_time,
        Time,
        AngularAcceleration
    );
}

#[test]
fn jerk_meters_per_second_cubed_dimensional_algebra() {
    use euclid_units::Jerk;
    times_method!(
        crate::JerkMetersPerSecondCubed,
        Jerk,
        times_time,
        Time,
        Acceleration
    );
}

#[test]
fn speed_ms_dimensional_algebra() {
    use euclid_units::Speed;
    times_method!(crate::SpeedMs, Speed, times_time, Time, Length);
    per_method!(crate::SpeedMs, Speed, per_time, Time, Acceleration);
    per_method!(crate::SpeedMs, Speed, per_acceleration, Acceleration, Time);
}

#[test]
fn time_seconds_dimensional_algebra() {
    use euclid_units::Time;
    times_method!(crate::TimeSeconds, Time, times_speed, Speed, Length);
    times_method!(
        crate::TimeSeconds,
        Time,
        times_acceleration,
        Acceleration,
        Speed
    );
}

#[test]
fn volumetric_flow_rate_cubic_meters_per_second_dimensional_algebra() {
    use euclid_units::VolumetricFlowRate;
    times_method!(
        crate::VolumetricFlowRateCubicMetersPerSecond,
        VolumetricFlowRate,
        times_time,
        Time,
        Volume
    );
    per_method!(
        crate::VolumetricFlowRateCubicMetersPerSecond,
        VolumetricFlowRate,
        per_area,
        Area,
        Speed
    );
    per_method!(
        crate::VolumetricFlowRateCubicMetersPerSecond,
        VolumetricFlowRate,
        per_speed,
        Speed,
        Area
    );
}
