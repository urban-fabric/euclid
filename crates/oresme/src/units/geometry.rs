//! Geometric units: length, area, volume, and angle.

mod angle_degrees;
mod angle_radians;
mod area_square_kilometers;
mod area_square_meters;
mod length_feet;
mod length_kilometers;
mod length_meters;
mod volume;

pub use angle_degrees::*;
pub use angle_radians::*;
pub use area_square_kilometers::*;
pub use area_square_meters::*;
pub use length_feet::*;
pub use length_kilometers::*;
pub use length_meters::*;
pub use volume::*;

pub(crate) const M_PER_KM: f64 = 1_000.0;
pub(crate) const M_PER_FOOT: f64 = 0.304_8;
pub(crate) const SQUARE_METERS_PER_SQUARE_KILOMETER: f64 = 1_000_000.0;

#[cfg(test)]
mod length_tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
    use crate::*;

    #[test]
    fn length_meters_constants_and_quantity_impl() {
        assert_eq!(LengthMeters::ZERO.as_meters(), 0.0);
        assert_eq!(LengthMeters::INFINITY.as_meters(), f64::INFINITY);
        assert_eq!(LengthMeters::DIMENSION, Dimension::Length);
        assert_eq!(LengthMeters::UNIT_NAME, "m");
        assert_eq!(LengthMeters::new(42.5).raw(), 42.5);
        assert_eq!(LengthMeters::from_raw(7.0), LengthMeters::new(7.0));
    }

    #[test]
    fn length_meters_arithmetic() {
        let a = LengthMeters::new(100.0);
        let b = LengthMeters::new(25.0);
        assert_eq!((a + b).as_meters(), 125.0);
        assert_eq!((a - b).as_meters(), 75.0);
        assert_eq!((-a).as_meters(), -100.0);
        let mut c = a;
        c += b;
        assert_eq!(c.as_meters(), 125.0);
        c -= b;
        assert_eq!(c.as_meters(), 100.0);
        assert_eq!((a * 2.0).as_meters(), 200.0);
        assert_eq!((2.0 * a).as_meters(), 200.0);
        assert_eq!((a / 4.0).as_meters(), 25.0);
        let sum: LengthMeters = [a, b, b].into_iter().sum();
        assert_eq!(sum.as_meters(), 150.0);
    }

    #[test]
    fn length_km_round_trip() {
        let m = LengthMeters::new(1500.0);
        let km = m.to_km();
        assert!((km.as_km() - 1.5).abs() < 1e-12);
        assert!((km.to_meters().as_meters() - 1500.0).abs() < 1e-9);
    }

    #[test]
    fn length_feet_round_trip() {
        let m = LengthMeters::new(1.0);
        let ft = m.to_feet();
        assert!((ft.as_feet() - 3.280_839_895_013_123).abs() < 1e-12);
        assert!((ft.to_meters().as_meters() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn length_km_arithmetic_is_in_dimension() {
        let a = LengthKm::new(1.0);
        let b = LengthKm::new(0.5);
        assert_eq!((a + b).as_km(), 1.5);
        assert_eq!((a * 2.0).as_km(), 2.0);
    }
}
#[cfg(test)]
mod area_tests {
    use crate::{AreaSquareKilometers, AreaSquareMeters, Quantity};

    #[test]
    fn area_round_trips() {
        let sq_km = AreaSquareKilometers::new(1.5);
        let sq_m = sq_km.to_square_meters();
        assert_eq!(sq_m, AreaSquareMeters::new(1_500_000.0));
        assert_eq!(sq_m.to_square_kilometers(), sq_km);
    }

    #[test]
    fn area_quantity_impls() {
        assert_eq!(AreaSquareMeters::DIMENSION, crate::Dimension::Area);
        assert_eq!(AreaSquareMeters::UNIT_NAME, "m^2");
        assert_eq!(AreaSquareKilometers::UNIT_NAME, "km^2");
    }
}
#[cfg(test)]
mod angle_tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
    use crate::*;

    #[test]
    fn angle_round_trip() {
        let r = AngleRadians::new(std::f64::consts::PI);
        let d = r.to_degrees();
        assert!((d.as_degrees() - 180.0).abs() < 1e-12);
        assert!((d.to_radians().as_radians() - std::f64::consts::PI).abs() < 1e-12);
    }

    #[test]
    fn angle_quantity_impl() {
        assert_eq!(AngleRadians::DIMENSION, Dimension::Angle);
        assert_eq!(AngleRadians::UNIT_NAME, "rad");
        assert_eq!(AngleDegrees::UNIT_NAME, "deg");
    }
}
