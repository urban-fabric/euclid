use crate::{
    AccelerationCentimetersPerSecondSquared, AccelerationMetersPerSecondSquared, AngleDegrees,
    AngleRadians, AreaSquareKilometers, AreaSquareMeters, Factor, LengthFeet, LengthKm,
    LengthMeters, MassGrams, MassKilograms, MassMilligrams, Percentage, ScalarQuantity, SpeedKph,
    SpeedMph, SpeedMs, TemperatureCelsius, TemperatureFahrenheit, TemperatureKelvin, TimeHours,
    TimeMillis, TimeMinutes, TimeSeconds, UnitOfMeasure,
};

#[test]
fn convert_within_a_dimension() {
    assert_eq!(
        LengthKm::new(2.5).convert::<LengthMeters>(),
        LengthMeters::new(2500.0)
    );
    assert_eq!(
        LengthMeters::new(1000.0).convert::<LengthKm>(),
        LengthKm::new(1.0)
    );
    // One km is ~3280.84 feet (routed through the primary unit, meters).
    let feet = LengthKm::new(1.0).convert::<LengthFeet>();
    assert!(feet.approx_eq(LengthFeet::new(3280.839895), LengthFeet::new(1e-3)));

    assert_eq!(
        MassKilograms::new(2.5).convert::<MassGrams>(),
        MassGrams::new(2500.0)
    );
    assert_eq!(
        AreaSquareKilometers::new(1.5).convert::<AreaSquareMeters>(),
        AreaSquareMeters::new(1_500_000.0)
    );
    assert_eq!(
        AccelerationMetersPerSecondSquared::new(9.81)
            .convert::<AccelerationCentimetersPerSecondSquared>(),
        AccelerationCentimetersPerSecondSquared::new(981.0)
    );
}

#[test]
fn convert_to_same_unit_is_identity() {
    assert_eq!(
        LengthMeters::new(5.0).convert::<LengthMeters>(),
        LengthMeters::new(5.0)
    );
}

#[test]
fn value_in_returns_the_bare_number() {
    assert_eq!(LengthKm::new(1.0).value_in::<LengthMeters>(), 1000.0);
    assert_eq!(TimeHours::new(2.0).value_in::<TimeMinutes>(), 120.0);
    assert_eq!(MassMilligrams::new(2500.0).value_in::<MassGrams>(), 2.5);
    assert!(
        (TemperatureCelsius::new(100.0).value_in::<TemperatureFahrenheit>() - 212.0).abs() < 1e-12
    );
}

#[test]
fn in_primary_uses_the_dimension_primary_unit() {
    assert_eq!(
        LengthFeet::new(10.0).in_primary(),
        LengthFeet::new(10.0).to_meters().as_meters()
    );
    assert_eq!(SpeedKph::new(36.0).in_primary(), 10.0);
    assert_eq!(TemperatureCelsius::new(0.0).in_primary(), 273.15);
}

#[test]
fn convert_agrees_with_the_bespoke_conversions() {
    // The protocol must never drift from the hand-written `to_*` methods: both are
    // anchored to the same crate-internal constants.
    let km = LengthKm::new(2.5);
    assert!(
        km.convert::<LengthMeters>()
            .approx_eq(km.to_meters(), LengthMeters::new(1e-9))
    );

    let ft = LengthFeet::new(10.0);
    assert!(
        ft.convert::<LengthMeters>()
            .approx_eq(ft.to_meters(), LengthMeters::new(1e-9))
    );

    let mins = TimeMinutes::new(3.0);
    assert!(
        mins.convert::<TimeSeconds>()
            .approx_eq(mins.to_seconds(), TimeSeconds::new(1e-9))
    );

    let millis = TimeMillis::new(2500.0);
    assert!(
        millis
            .convert::<TimeSeconds>()
            .approx_eq(millis.to_seconds(), TimeSeconds::new(1e-9))
    );

    let ms = SpeedMs::new(10.0);
    assert!(
        ms.convert::<SpeedKph>()
            .approx_eq(ms.to_kph(), SpeedKph::new(1e-9))
    );
    assert!(
        ms.convert::<SpeedMph>()
            .approx_eq(ms.to_mph(), SpeedMph::new(1e-9))
    );

    let mph = SpeedMph::new(60.0);
    assert!(
        mph.convert::<SpeedMs>()
            .approx_eq(mph.to_ms(), SpeedMs::new(1e-9))
    );
    assert!(
        mph.convert::<SpeedKph>()
            .approx_eq(mph.to_kph(), SpeedKph::new(1e-9))
    );

    assert!(
        AngleDegrees::new(180.0)
            .convert::<AngleRadians>()
            .approx_eq(AngleRadians::PI, AngleRadians::new(1e-12))
    );

    let grams = MassGrams::new(2500.0);
    assert!(
        grams
            .convert::<MassKilograms>()
            .approx_eq(grams.to_kilograms(), MassKilograms::new(1e-12))
    );

    let sq_km = AreaSquareKilometers::new(1.25);
    assert!(
        sq_km
            .convert::<AreaSquareMeters>()
            .approx_eq(sq_km.to_square_meters(), AreaSquareMeters::new(1e-9))
    );

    let acceleration = AccelerationCentimetersPerSecondSquared::new(981.0);
    assert!(
        acceleration
            .convert::<AccelerationMetersPerSecondSquared>()
            .approx_eq(
                acceleration.to_meters_per_second_squared(),
                AccelerationMetersPerSecondSquared::new(1e-12),
            )
    );

    assert!(
        TemperatureFahrenheit::new(32.0)
            .convert::<TemperatureCelsius>()
            .approx_eq(TemperatureCelsius::new(0.0), TemperatureCelsius::new(1e-12))
    );
}

#[test]
fn dimensionless_factor_and_percentage_interconvert() {
    assert!(
        Percentage::new(50.0)
            .convert::<Factor>()
            .approx_eq(Factor::new(0.5), Factor::new(1e-12))
    );
    assert!(
        Factor::new(0.25)
            .convert::<Percentage>()
            .approx_eq(Percentage::new(25.0), Percentage::new(1e-9))
    );
}

#[test]
fn round_trip_preserves_value() {
    let original = LengthMeters::new(123.456);
    let round_tripped = original.convert::<LengthFeet>().convert::<LengthMeters>();
    assert!(round_tripped.approx_eq(original, LengthMeters::new(1e-9)));
}

#[test]
fn display_prints_value_with_unit_symbol() {
    assert_eq!(LengthMeters::new(5.0).to_string(), "5 m");
    assert_eq!(LengthKm::new(1.0).to_string(), "1 km");
    assert_eq!(LengthFeet::new(3.0).to_string(), "3 ft");
    assert_eq!(TimeSeconds::new(2.5).to_string(), "2.5 s");
    assert_eq!(SpeedKph::new(90.0).to_string(), "90 km/h");
    assert_eq!(SpeedMs::new(10.0).to_string(), "10 m/s");
    assert_eq!(MassGrams::new(5.0).to_string(), "5 g");
    assert_eq!(AreaSquareMeters::new(12.0).to_string(), "12 m^2");
    assert_eq!(
        AccelerationMetersPerSecondSquared::new(9.81).to_string(),
        "9.81 m/s^2"
    );
    assert_eq!(TemperatureCelsius::new(20.0).to_string(), "20 °C");
    assert_eq!(TemperatureKelvin::new(273.15).to_string(), "273.15 K");
    assert_eq!(Percentage::new(50.0).to_string(), "50 %");
    assert_eq!(Factor::new(1.5).to_string(), "1.5");
}
