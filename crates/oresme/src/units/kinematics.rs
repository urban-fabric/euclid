//! Kinematic units: time, frequency, speed, acceleration, jerk, angular
//! velocity/acceleration, and volumetric flow rate.

mod acceleration_centimeters_per_second_squared;
mod acceleration_meters_per_second_squared;
mod angular_acceleration;
mod angular_velocity;
mod frequency;
mod jerk;
mod speed_kilometers_per_hour;
mod speed_meters_per_second;
mod speed_miles_per_hour;
mod time_hours;
mod time_micros;
mod time_millis;
mod time_minutes;
mod time_nanos;
mod time_picos;
mod time_seconds;
mod volumetric_flow_rate;

pub use acceleration_centimeters_per_second_squared::*;
pub use acceleration_meters_per_second_squared::*;
pub use angular_acceleration::*;
pub use angular_velocity::*;
pub use frequency::*;
pub use jerk::*;
pub use speed_kilometers_per_hour::*;
pub use speed_meters_per_second::*;
pub use speed_miles_per_hour::*;
pub use time_hours::*;
pub use time_micros::*;
pub use time_millis::*;
pub use time_minutes::*;
pub use time_nanos::*;
pub use time_picos::*;
pub use time_seconds::*;
pub use volumetric_flow_rate::*;

pub(crate) const S_PER_MIN: f64 = 60.0;
pub(crate) const S_PER_HOUR: f64 = 3_600.0;
pub(crate) const MS_PER_S: f64 = 1_000.0;
pub(crate) const NS_PER_US: u64 = 1_000;
pub(crate) const NS_PER_S: f64 = 1_000_000_000.0;
pub(crate) const PS_PER_NS: u64 = 1_000;
pub(crate) const PS_PER_S: f64 = 1_000_000_000_000.0;
pub(crate) const MS_PER_KPH: f64 = 1_000.0 / 3_600.0;
pub(crate) const MS_PER_MPH: f64 = 0.447_04;
pub(crate) const KPH_PER_MPH: f64 = 1.609_344;
pub(crate) const MPH_PER_KPH: f64 = 1.0 / KPH_PER_MPH;
pub(crate) const MPS2_PER_CMPS2: f64 = 0.01;

#[cfg(test)]
mod time_tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
    use crate::{Dimension, Quantity};

    use crate::*;

    #[test]
    fn time_seconds_constants() {
        assert_eq!(TimeSeconds::ZERO.as_seconds(), 0.0);
        assert_eq!(TimeSeconds::DIMENSION, Dimension::Time);
        assert_eq!(TimeSeconds::UNIT_NAME, "s");
    }

    #[test]
    fn time_seconds_arithmetic() {
        let a = TimeSeconds::new(60.0);
        let b = TimeSeconds::new(15.0);
        assert_eq!((a + b).as_seconds(), 75.0);
        assert_eq!((a - b).as_seconds(), 45.0);
        assert_eq!((a * 2.0).as_seconds(), 120.0);
        let sum: TimeSeconds = [a, b].into_iter().sum();
        assert_eq!(sum.as_seconds(), 75.0);
    }

    #[test]
    fn time_round_trips() {
        let s = TimeSeconds::new(3_600.0);
        assert!((s.to_minutes().as_minutes() - 60.0).abs() < 1e-12);
        assert!((s.to_hours().as_hours() - 1.0).abs() < 1e-12);
        assert!((s.to_hours().to_seconds().as_seconds() - 3_600.0).abs() < 1e-9);
    }

    #[test]
    fn time_nanos_constants() {
        assert_eq!(TimeNanos::ZERO.as_nanos(), 0);
        assert_eq!(TimeNanos::DIMENSION, Dimension::Time);
        assert_eq!(TimeNanos::UNIT_NAME, "ns");
        assert_eq!(TimeNanosDelta::UNIT_NAME, "ns (signed)");
    }

    #[test]
    fn sub_second_units_display_with_unit_suffix() {
        assert_eq!(format!("{}", TimeNanos::new(100)), "100 ns");
        assert_eq!(format!("{}", TimeNanosDelta::new(-5)), "-5 ns (signed)");
        assert_eq!(format!("{}", TimePicos::new(7)), "7 ps");
        assert_eq!(format!("{}", TimePicosDelta::new(-7)), "-7 ps (signed)");
        assert_eq!(format!("{}", TimeMicros::new(3)), "3 µs");
        assert_eq!(format!("{}", TimeMicrosDelta::new(-3)), "-3 µs (signed)");
    }

    #[test]
    fn time_nanos_arithmetic() {
        let a = TimeNanos::new(100);
        let b = TimeNanos::new(250);
        assert_eq!((a + b).as_nanos(), 350);
        assert_eq!(a.saturating_add(b).as_nanos(), 350);
        assert_eq!(a.saturating_sub(b).as_nanos(), 0);
        assert_eq!(b.saturating_sub(a).as_nanos(), 150);

        let mut acc = TimeNanos::ZERO;
        acc += a;
        assert_eq!(acc, a);

        let sum: TimeNanos = [a, b].into_iter().sum();
        assert_eq!(sum.as_nanos(), 350);

        assert_eq!(
            TimeNanos::new(u64::MAX).saturating_add(a).as_nanos(),
            u64::MAX
        );
    }

    #[test]
    fn time_nanos_conversions() {
        assert_eq!(TimeNanos::new(2_500).to_micros().as_micros(), 2);
        assert_eq!(
            TimeNanos::from_duration(std::time::Duration::from_micros(5)).as_nanos(),
            5_000
        );
        assert!((TimeNanos::new(1_000_000_000).to_seconds().as_seconds() - 1.0).abs() < 1e-12);
        assert!((TimeNanos::new(1_000_000).to_millis().as_millis() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn time_nanos_delta_between() {
        let before = TimeNanos::new(100);
        let after = TimeNanos::new(450);
        assert_eq!(TimeNanosDelta::between(before, after).as_nanos(), 350);
        assert_eq!(TimeNanosDelta::between(after, before).as_nanos(), -350);
    }

    #[test]
    fn time_picos_constants() {
        assert_eq!(TimePicos::ZERO.as_picos(), 0);
        assert_eq!(TimePicos::DIMENSION, Dimension::Time);
        assert_eq!(TimePicos::UNIT_NAME, "ps");
        assert_eq!(TimePicosDelta::UNIT_NAME, "ps (signed)");
    }

    #[test]
    fn time_picos_arithmetic() {
        let a = TimePicos::new(1_500);
        let b = TimePicos::new(500);
        assert_eq!((a + b).as_picos(), 2_000);
        assert_eq!(a.saturating_sub(b).as_picos(), 1_000);
        assert_eq!(b.saturating_sub(a).as_picos(), 0);
        let sum: TimePicos = [a, b].into_iter().sum();
        assert_eq!(sum.as_picos(), 2_000);
    }

    #[test]
    fn time_picos_conversions() {
        assert_eq!(TimePicos::new(1_500).to_nanos().as_nanos(), 1);
        assert_eq!(TimePicos::new(2_500_000).to_micros().as_micros(), 2);
        assert!((TimePicos::new(1_000_000_000_000).to_seconds().as_seconds() - 1.0).abs() < 1e-12);
        assert_eq!(
            TimePicos::from_duration(std::time::Duration::from_nanos(7)).as_picos(),
            7_000
        );
    }

    #[test]
    fn time_picos_delta_between() {
        let before = TimePicos::new(1_000);
        let after = TimePicos::new(1_350);
        assert_eq!(TimePicosDelta::between(before, after).as_picos(), 350);
        assert_eq!(TimePicosDelta::between(after, before).as_picos(), -350);
    }
}
#[cfg(test)]
mod speed_tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
    use crate::*;

    #[test]
    fn speed_constants_and_dims() {
        assert_eq!(SpeedMs::DIMENSION, Dimension::Speed);
        assert_eq!(SpeedKph::DIMENSION, Dimension::Speed);
        assert_eq!(SpeedMph::DIMENSION, Dimension::Speed);
        assert_eq!(SpeedMs::UNIT_NAME, "m/s");
        assert_eq!(SpeedKph::UNIT_NAME, "km/h");
        assert_eq!(SpeedMph::UNIT_NAME, "mph");
    }

    #[test]
    fn speed_pair_round_trips() {
        let kph = SpeedKph::new(50.0);
        let ms = kph.to_ms();
        assert!((ms.as_ms() - 13.888_888_888_888_89).abs() < 1e-12);
        assert!((ms.to_kph().as_kph() - 50.0).abs() < 1e-9);

        let mph = SpeedMph::new(30.0);
        let ms2 = mph.to_ms();
        assert!((ms2.as_ms() - 13.4112).abs() < 1e-9);
        assert!((ms2.to_mph().as_mph() - 30.0).abs() < 1e-9);

        assert!((SpeedKph::new(100.0).to_mph().as_mph() - 62.137_119_223_733_4).abs() < 1e-9);
        assert!((SpeedMph::new(60.0).to_kph().as_kph() - 96.560_64).abs() < 1e-9);
    }

    #[test]
    fn speed_in_dimension_arithmetic() {
        let a = SpeedKph::new(50.0);
        let b = SpeedKph::new(10.0);
        assert_eq!((a + b).as_kph(), 60.0);
        assert_eq!((a - b).as_kph(), 40.0);
        assert_eq!((a * 1.5).as_kph(), 75.0);
    }
}
#[cfg(test)]
mod acceleration_tests {
    use crate::{
        AccelerationCentimetersPerSecondSquared, AccelerationMetersPerSecondSquared, Quantity,
    };

    #[test]
    fn acceleration_round_trips() {
        let meters = AccelerationMetersPerSecondSquared::new(9.81);
        let centimeters = meters.to_centimeters_per_second_squared();
        assert_eq!(
            centimeters,
            AccelerationCentimetersPerSecondSquared::new(981.0)
        );
        assert_eq!(centimeters.to_meters_per_second_squared(), meters);
    }

    #[test]
    fn acceleration_quantity_impls() {
        assert_eq!(
            AccelerationMetersPerSecondSquared::DIMENSION,
            crate::Dimension::Acceleration
        );
        assert_eq!(AccelerationMetersPerSecondSquared::UNIT_NAME, "m/s^2");
        assert_eq!(AccelerationCentimetersPerSecondSquared::UNIT_NAME, "cm/s^2");
    }
}
