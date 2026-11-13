//! Conformance of oresme's `f64` units to euclid's base contracts (`Quantity`,
//! `ScalarQuantity`, `LinearQuantity`, `Dimensionless`, `Temperature`).

use crate::{
    Factor, LengthMeters, Percentage, SpeedMs, TemperatureCelsius, TemperatureFahrenheit,
    TemperatureKelvin, TimeSeconds,
};

fn euclid_linear_midpoint<Q>(a: Q, b: Q) -> Q
where
    Q: euclid_units::LinearQuantity<Repr = f64>,
{
    euclid_units::LinearQuantity::lerp(a, b, 0.5)
}

#[test]
fn f64_units_satisfy_euclid_base_quantity_contracts() {
    fn assert_quantity<Q>(raw: f64, dimension: euclid_units::Dimension, unit_name: &'static str)
    where
        Q: euclid_units::Measure<Repr = f64>,
    {
        let q = Q::from_raw(raw);
        assert_eq!(euclid_units::Measure::raw(&q), raw);
        assert_eq!(Q::DIMENSION, dimension);
        assert_eq!(Q::UNIT_NAME, unit_name);
    }

    assert_quantity::<LengthMeters>(3.0, euclid_units::Dimension::Length, "m");
    assert_quantity::<SpeedMs>(4.0, euclid_units::Dimension::Speed, "m/s");
    assert_quantity::<TemperatureKelvin>(293.15, euclid_units::Dimension::Temperature, "K");
}

#[test]
fn f64_linear_units_satisfy_euclid_linear_helpers() {
    let midpoint = euclid_linear_midpoint(LengthMeters::new(2.0), LengthMeters::new(8.0));
    assert_eq!(midpoint, LengthMeters::new(5.0));

    let speed = euclid_units::LinearQuantity::scale(SpeedMs::new(3.0), 2.0);
    assert_eq!(speed, SpeedMs::new(6.0));

    let ratio = euclid_units::LinearQuantity::ratio(TimeSeconds::new(12.0), TimeSeconds::new(3.0));
    assert_eq!(ratio, 4.0);
}

#[test]
fn dimensionless_impls_use_ratio_semantics() {
    let factor = euclid_units::Dimensionless::times(Factor::new(0.5), Factor::new(0.5));
    assert_eq!(factor, Factor::new(0.25));

    let percent = euclid_units::Dimensionless::times(Percentage::new(50.0), Percentage::new(50.0));
    assert_eq!(percent, Percentage::new(25.0));

    assert_eq!(
        <Percentage as euclid_units::Dimensionless>::unit(),
        Percentage::FULL
    );
}

#[test]
fn temperature_impls_preserve_absolute_scale_conversions() {
    let celsius = <TemperatureCelsius as euclid_units::Temperature>::from_kelvin(273.15);
    assert_eq!(celsius, TemperatureCelsius::ZERO);
    assert_eq!(euclid_units::Temperature::kelvin(&celsius), 273.15);

    let fahrenheit = <TemperatureFahrenheit as euclid_units::Temperature>::from_celsius(100.0);
    assert!((euclid_units::Temperature::fahrenheit(&fahrenheit) - 212.0).abs() < 1e-12);

    assert_eq!(
        <TemperatureKelvin as euclid_units::Temperature>::absolute_zero(),
        TemperatureKelvin::ZERO
    );
}
