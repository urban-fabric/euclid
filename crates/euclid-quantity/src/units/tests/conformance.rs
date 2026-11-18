//! Exercises the public euclid-units laws from a downstream crate.

mod angle;
mod dimensionless;
mod linear;

use euclid_units::conformance::{self as laws, ConformanceSample};

use crate::{
    Factor, LengthFeet, LengthMeters, Measure, SpeedMs, TemperatureCelsius, TemperatureFahrenheit,
    TemperatureKelvin,
};

/// A tolerance small enough that `approx_eq` reflexivity is not trivially satisfied by a
/// permissive bound.
const TOL: f64 = 1e-12;

impl ConformanceSample for TemperatureCelsius {
    fn sample() -> Vec<Self> {
        vec![
            Self::new(0.0),
            Self::new(20.0),
            Self::new(-40.0),
            Self::new(100.0),
        ]
    }
    fn tolerance() -> Self {
        Self::new(TOL)
    }
}

impl ConformanceSample for TemperatureKelvin {
    fn sample() -> Vec<Self> {
        vec![Self::new(0.0), Self::new(273.15), Self::new(1.0)]
    }
    fn tolerance() -> Self {
        Self::new(TOL)
    }
}

#[test]
fn absolute_scalars_get_the_measure_and_scalar_laws_only() {
    // *Every* temperature withholds `LinearQuantity`, Kelvin included — adding two absolute
    // temperatures is meaningless whatever the scale's zero happens to be. So they take the two
    // suites that apply and not the third, and `quantity_laws` would not compile for them.
    // That asymmetry is the contract working as intended.
    let celsius = [
        TemperatureCelsius::new(0.0),
        TemperatureCelsius::new(20.0),
        TemperatureCelsius::new(-40.0),
        TemperatureCelsius::new(100.0),
    ];
    laws::measure_laws(&celsius, "TemperatureCelsius");
    laws::scalar_laws(
        &celsius,
        &TemperatureCelsius::new(TOL),
        "TemperatureCelsius",
    );

    let kelvin = [
        TemperatureKelvin::new(0.0),
        TemperatureKelvin::new(273.15),
        TemperatureKelvin::new(1.0),
    ];
    laws::measure_laws(&kelvin, "TemperatureKelvin");
    laws::scalar_laws(&kelvin, &TemperatureKelvin::new(TOL), "TemperatureKelvin");
}

#[test]
fn integer_backed_storage_satisfies_the_discrete_laws() {
    use crate::{Bytes, Count, ThreadCount};

    laws::discrete_laws(&[Bytes::new(0), Bytes::new(1), Bytes::new(4096)], "Bytes");
    laws::discrete_laws(&[Count::new(0), Count::new(1), Count::new(999)], "Count");
    laws::discrete_laws(
        &[
            ThreadCount::new(0),
            ThreadCount::new(1),
            ThreadCount::new(8),
        ],
        "ThreadCount",
    );
}

#[test]
fn raw_round_trips_across_every_structural_shape() {
    // `measure_laws` on its own, over the widest spread — the one law every unit must satisfy
    // regardless of which tiers it reaches.
    laws::measure_laws(&[LengthMeters::new(1.5)], "LengthMeters");
    laws::measure_laws(&[LengthFeet::new(1.5)], "LengthFeet");
    laws::measure_laws(&[SpeedMs::new(1.5)], "SpeedMs");
    laws::measure_laws(&[Factor::new(1.5)], "Factor");
    laws::measure_laws(&[TemperatureCelsius::new(1.5)], "TemperatureCelsius");
    assert_eq!(LengthMeters::from_raw(2.0).raw(), 2.0);
}

/// The three temperature scales, against physical fixed points.
///
/// The absolute scalars take `measure_laws` and `scalar_laws` above, and neither says anything
/// about the *conversions* between scales — which are affine, and so the one place in the crate
/// where an offset can be dropped without a round trip noticing. Mutation testing found
/// twenty-one survivors here: every `from_*` constructor and every accessor across the three
/// scales.
///
/// See `euclid_units::conformance::temperature_laws` for why the anchors are water's phase
/// changes and absolute zero rather than round trips, and why `-40` is in the list.
#[test]
fn the_temperature_scales_agree_on_the_fixed_points() {
    laws::temperature_laws::<TemperatureCelsius>("TemperatureCelsius");
    laws::temperature_laws::<TemperatureFahrenheit>("TemperatureFahrenheit");
    laws::temperature_laws::<TemperatureKelvin>("TemperatureKelvin");
}
