//! The real `f64` units, checked against euclid's own contracts.
//!
//! Every suite here is [`euclid_units::conformance`] — public API that a carrier author in any
//! repository calls. This crate is the first such author, so these tests are also the proof
//! that the suites are usable from outside the crate that defines them.
//!
//! The spread is deliberate: one unit per structural shape rather than all 72. A conversion
//! factor cannot break a law that holds for `LengthMeters`, so the coverage that matters is
//! across *shapes* — a primary unit, a non-primary one, a dimensionless factor, a derived unit
//! with a compound symbol, and an absolute scalar that withholds `LinearQuantity`.

use euclid_units::conformance::{self as laws, ConformanceSample};

use crate::{
    AreaSquareMeters, Factor, LengthFeet, LengthMeters, Measure, Percentage, SpeedMs,
    TemperatureCelsius, TemperatureKelvin, TimeSeconds,
};

/// A tolerance small enough that `approx_eq` reflexivity is not trivially satisfied by a
/// permissive bound.
const TOL: f64 = 1e-12;

/// Declare a unit's sample and tolerance once, then run every applicable suite from it.
///
/// This is the external-consumer side of `ConformanceSample`: the trait is euclid-units', the
/// impls are ours, and the suites are called with no sample argument at all. Adding a suite to
/// euclid-units now covers every unit declared here without touching this file.
macro_rules! linear_unit_laws {
    ($name:ident, $ty:ty, [$($v:expr),+ $(,)?]) => {
        impl ConformanceSample for $ty {
            fn sample() -> Vec<Self> {
                vec![$(<$ty>::new($v)),+]
            }
            fn tolerance() -> Self {
                <$ty>::new(TOL)
            }
        }

        #[test]
        fn $name() {
            laws::quantity_laws_for::<$ty>(stringify!($ty));
        }
    };
}

// A primary unit of a base dimension.
linear_unit_laws!(length_meters, LengthMeters, [0.0, 1.0, -2.5, 4.0, 0.5]);
// A non-primary unit of the same dimension: the conversion factor must not disturb the laws.
linear_unit_laws!(length_feet, LengthFeet, [0.0, 1.0, -3.0, 10.0, 0.25]);
// A primary unit of a derived dimension, with a compound symbol.
linear_unit_laws!(speed_ms, SpeedMs, [0.0, 1.0, -9.8, 100.0, 0.125]);
// A squared dimension, where a sign error in `abs` or `neg` would be easy to miss.
linear_unit_laws!(area_square_meters, AreaSquareMeters, [0.0, 1.0, 2.5, 16.0]);
// Time, the dimension most used in cross-dimensional algebra.
linear_unit_laws!(time_seconds, TimeSeconds, [0.0, 1.0, -0.5, 60.0, 0.25]);
// Dimensionless, where `UNIT_NAME` is empty and the ratio laws are most exercised.
linear_unit_laws!(factor, Factor, [0.0, 1.0, -1.0, 2.5, 0.5]);
// A dimensionless unit carrying a scale, so `in_primary` is not the identity.
linear_unit_laws!(percentage, Percentage, [0.0, 50.0, -25.0, 100.0]);
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
