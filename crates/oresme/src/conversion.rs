//! A uniform unit-of-measure protocol: dimension-checked conversions and symbols.
//!
//! oresme models each unit as its own newtype (`LengthMeters`, `LengthKm`,
//! `LengthFeet`, ...). That keeps the unit of a value visible in its type, but
//! without a shared protocol every conversion is a bespoke method and nothing
//! stops a `LengthMeters` from being handed where a `TimeSeconds` was meant.
//!
//! [`UnitOfMeasure`] gives the unit newtypes a common backbone, modelled on
//! `squants`: each scalar unit declares the [dimension](dim) it measures and how
//! to express it in that dimension's *primary* unit (meters, seconds, m/s, kelvin,
//! or a bare factor). From that, one generic [`convert`](UnitOfMeasure::convert)
//! works across every unit, and the compiler rejects cross-dimension conversions
//! because the `Dim` associated types would not match.
//!
//! Each scalar unit also gains a [`Display`](std::fmt::Display) that prints the
//! value with its unit symbol (`5 m`, `90 km/h`, `50 %`), so a logged or printed
//! quantity always states what unit it is in.
//!
//! ```
//! use oresme::{LengthKm, LengthFeet, UnitOfMeasure};
//!
//! let one_km = LengthKm::new(1.0);
//! let feet: LengthFeet = one_km.convert();
//! assert!((feet.as_feet() - 3280.84).abs() < 0.01);
//! assert_eq!(one_km.to_string(), "1 km");
//! ```
//!
//! Cross-dimension conversions do not compile:
//!
//! ```compile_fail
//! use oresme::{LengthMeters, TimeSeconds, UnitOfMeasure};
//! // error[E0271]: type mismatch — `Length` is not `Time`
//! let _: TimeSeconds = LengthMeters::new(1.0).convert();
//! ```

use core::fmt;

use crate::{Quantity, ScalarQuantity};

/// Type-level markers identifying a physical dimension, re-exported from
/// [`euclid_units::dim`].
///
/// Two units convert into one another only when they share the same marker, which the compiler
/// enforces. These are the type-level counterparts of the runtime [`crate::Dimension`] enum;
/// [`euclid_units::Dimensioned`] is the canonical bridge from a marker to its runtime
/// [`Dimension`](crate::Dimension). euclid owns both, so the concrete units here and euclid's
/// carrier-generic contracts share exactly one marker set instead of maintaining parallel copies.
pub use euclid_units::dim;

/// A scalar unit of measure that belongs to a dimension and has a known
/// scale and offset to that dimension's primary unit.
///
/// Implemented for every `f64`-backed unit newtype. The provided methods give a
/// single, dimension-checked conversion path that replaces ad-hoc `to_*` methods
/// with one uniform protocol.
pub trait UnitOfMeasure: ScalarQuantity {
    /// The dimension this unit measures, as a [`euclid_units::dim`] marker. Conversions are
    /// only possible between units that share the same `Dim`.
    type Dim: euclid_units::Dimensioned;

    /// Multiply a value in this unit by `TO_PRIMARY` and add `PRIMARY_OFFSET` to
    /// express it in the dimension's primary unit. The primary unit therefore has
    /// `TO_PRIMARY == 1.0` and `PRIMARY_OFFSET == 0.0`.
    const TO_PRIMARY: f64;

    /// Offset added after scaling into the primary unit.
    const PRIMARY_OFFSET: f64 = 0.0;

    /// This quantity's value expressed in the dimension's primary unit.
    fn in_primary(self) -> f64 {
        self.raw() * Self::TO_PRIMARY + Self::PRIMARY_OFFSET
    }

    /// Convert a primary-unit value into this unit's raw numeric value.
    fn value_from_primary(primary: f64) -> f64 {
        (primary - Self::PRIMARY_OFFSET) / Self::TO_PRIMARY
    }

    /// This quantity's numeric value expressed in the unit `To`.
    fn value_in<To: UnitOfMeasure<Dim = Self::Dim>>(self) -> f64 {
        To::value_from_primary(self.in_primary())
    }

    /// Convert to another unit of the same dimension.
    fn convert<To: UnitOfMeasure<Dim = Self::Dim>>(self) -> To {
        To::from_raw(self.value_in::<To>())
    }
}

/// Format a quantity as its numeric value followed by its unit symbol (`5 m`, `90 km/h`,
/// `50 %`); a dimensionless unit with an empty symbol prints just the number. Used by the
/// `Display` impl each unit delegates to.
pub(crate) fn fmt_quantity<Q: Quantity<Repr = f64>>(
    value: Q,
    f: &mut fmt::Formatter<'_>,
) -> fmt::Result {
    if Q::UNIT_NAME.is_empty() {
        write!(f, "{}", value.raw())
    } else {
        write!(f, "{} {}", value.raw(), Q::UNIT_NAME)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        AccelerationCentimetersPerSecondSquared, AccelerationMetersPerSecondSquared, AngleDegrees,
        AngleRadians, AreaSquareKilometers, AreaSquareMeters, Factor, LengthFeet, LengthKm,
        LengthMeters, MassGrams, MassKilograms, MassMilligrams, Percentage, ScalarQuantity,
        SpeedKph, SpeedMph, SpeedMs, TemperatureCelsius, TemperatureFahrenheit, TemperatureKelvin,
        TimeHours, TimeMillis, TimeMinutes, TimeSeconds, UnitOfMeasure,
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
            (TemperatureCelsius::new(100.0).value_in::<TemperatureFahrenheit>() - 212.0).abs()
                < 1e-12
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
}
