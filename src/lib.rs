#![doc = include_str!("../README.md")]

pub mod algebra;
pub mod angle;
pub mod bytes;
pub mod count;
pub mod factor;
pub mod length;
pub mod percentage;
pub mod speed;
pub mod storage;
pub mod time;
pub mod timestamp;
pub mod weights;

mod conversions;
mod ops;

pub use angle::{AngleDegrees, AngleRadians};
pub use bytes::{ByteRange, Bytes};
pub use count::{Count, ThreadCount};
pub use factor::Factor;
pub use length::{LengthFeet, LengthKm, LengthMeters};
pub use percentage::Percentage;
pub use speed::{SpeedKph, SpeedMph, SpeedMs};
pub use storage::{DistanceDecimeters, TimeCentiseconds};
pub use time::{TimeHours, TimeMicros, TimeMicrosDelta, TimeMillis, TimeMinutes, TimeSeconds};
pub use timestamp::{UnixMillis, UnixSeconds};
pub use weights::{DistanceWeight, TimeWeight};

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum Dimension {
    Length,
    Time,
    Speed,
    Angle,
    Dimensionless,
}

pub trait Quantity: Copy + PartialOrd {
    type Repr: Copy;
    const DIMENSION: Dimension;
    const UNIT_NAME: &'static str;
    fn raw(self) -> Self::Repr;
    fn from_raw(v: Self::Repr) -> Self;
}

pub trait ScalarQuantity: Quantity<Repr = f64> {}

impl<T> ScalarQuantity for T where T: Quantity<Repr = f64> {}

pub fn scalar_add<T: ScalarQuantity>(lhs: T, rhs: T) -> T {
    T::from_raw(lhs.raw() + rhs.raw())
}

pub fn scalar_sub<T: ScalarQuantity>(lhs: T, rhs: T) -> T {
    T::from_raw(lhs.raw() - rhs.raw())
}

pub fn scalar_neg<T: ScalarQuantity>(value: T) -> T {
    T::from_raw(-value.raw())
}

pub fn scalar_mul<T: ScalarQuantity>(value: T, factor: f64) -> T {
    T::from_raw(value.raw() * factor)
}

pub fn scalar_div<T: ScalarQuantity>(value: T, divisor: f64) -> T {
    T::from_raw(value.raw() / divisor)
}

pub fn scalar_sum<T, I>(iter: I) -> T
where
    T: ScalarQuantity,
    I: Iterator<Item = T>,
{
    T::from_raw(iter.map(Quantity::raw).sum())
}
