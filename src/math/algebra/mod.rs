//! Dimensional cross-products between quantities: a length over a time is a
//! speed, a speed over a time is an acceleration, two lengths make an area, and a
//! [`Factor`](crate::Factor) scales any of them.
//!
//! These operators are defined only between each dimension's **primary** unit
//! ([`LengthMeters`](crate::LengthMeters), [`TimeSeconds`](crate::TimeSeconds),
//! [`SpeedMs`](crate::SpeedMs), [`AreaSquareMeters`](crate::AreaSquareMeters),
//! [`AccelerationMetersPerSecondSquared`](crate::AccelerationMetersPerSecondSquared)).
//! That keeps the impls linear in the number of dimensions instead of
//! combinatorial in the number of units, and keeps every result in a canonical
//! unit. To combine non-primary units, convert to the primary unit first with
//! [`UnitOfMeasure::convert`](crate::UnitOfMeasure::convert):
//!
//! ```
//! use euclid::{LengthKm, LengthMeters, SpeedMs, TimeHours, TimeSeconds, UnitOfMeasure};
//!
//! let distance: LengthMeters = LengthKm::new(5.0).convert();
//! let elapsed: TimeSeconds = TimeHours::new(0.5).convert();
//! let speed: SpeedMs = distance / elapsed; // 5 km over half an hour
//! assert!((speed.as_ms() - 5000.0 / 1800.0).abs() < 1e-9);
//! ```

use std::ops::{Div, Mul};

use super::{
    AccelerationMetersPerSecondSquared, AreaSquareMeters, Factor, LengthMeters, SpeedKph, SpeedMph,
    SpeedMs, TimeSeconds,
};

impl Div<TimeSeconds> for LengthMeters {
    type Output = SpeedMs;
    fn div(self, t: TimeSeconds) -> SpeedMs {
        SpeedMs::new(self.as_meters() / t.as_seconds())
    }
}

impl Div<SpeedMs> for LengthMeters {
    type Output = TimeSeconds;
    fn div(self, v: SpeedMs) -> TimeSeconds {
        TimeSeconds::new(self.as_meters() / v.as_ms())
    }
}

impl Mul<LengthMeters> for LengthMeters {
    type Output = AreaSquareMeters;
    fn mul(self, rhs: LengthMeters) -> AreaSquareMeters {
        AreaSquareMeters::new(self.as_meters() * rhs.as_meters())
    }
}

impl Div<LengthMeters> for AreaSquareMeters {
    type Output = LengthMeters;
    fn div(self, rhs: LengthMeters) -> LengthMeters {
        LengthMeters::new(self.as_square_meters() / rhs.as_meters())
    }
}

impl Div<SpeedKph> for SpeedKph {
    type Output = f64;
    fn div(self, rhs: SpeedKph) -> f64 {
        self.as_kph() / rhs.as_kph()
    }
}

impl Div<SpeedMs> for SpeedMs {
    type Output = f64;
    fn div(self, rhs: SpeedMs) -> f64 {
        self.as_ms() / rhs.as_ms()
    }
}

impl Mul<TimeSeconds> for SpeedMs {
    type Output = LengthMeters;
    fn mul(self, t: TimeSeconds) -> LengthMeters {
        LengthMeters::new(self.as_ms() * t.as_seconds())
    }
}
impl Mul<SpeedMs> for TimeSeconds {
    type Output = LengthMeters;
    fn mul(self, v: SpeedMs) -> LengthMeters {
        LengthMeters::new(self.as_seconds() * v.as_ms())
    }
}

impl Div<TimeSeconds> for SpeedMs {
    type Output = AccelerationMetersPerSecondSquared;
    fn div(self, rhs: TimeSeconds) -> AccelerationMetersPerSecondSquared {
        AccelerationMetersPerSecondSquared::new(self.as_ms() / rhs.as_seconds())
    }
}

impl Div<AccelerationMetersPerSecondSquared> for SpeedMs {
    type Output = TimeSeconds;
    fn div(self, rhs: AccelerationMetersPerSecondSquared) -> TimeSeconds {
        TimeSeconds::new(self.as_ms() / rhs.as_meters_per_second_squared())
    }
}

impl Mul<TimeSeconds> for AccelerationMetersPerSecondSquared {
    type Output = SpeedMs;
    fn mul(self, rhs: TimeSeconds) -> SpeedMs {
        SpeedMs::new(self.as_meters_per_second_squared() * rhs.as_seconds())
    }
}

impl Mul<AccelerationMetersPerSecondSquared> for TimeSeconds {
    type Output = SpeedMs;
    fn mul(self, rhs: AccelerationMetersPerSecondSquared) -> SpeedMs {
        SpeedMs::new(self.as_seconds() * rhs.as_meters_per_second_squared())
    }
}

impl Mul<LengthMeters> for Factor {
    type Output = LengthMeters;
    fn mul(self, x: LengthMeters) -> LengthMeters {
        LengthMeters::new(self.as_f64() * x.as_meters())
    }
}
impl Mul<Factor> for LengthMeters {
    type Output = LengthMeters;
    fn mul(self, f: Factor) -> LengthMeters {
        LengthMeters::new(self.as_meters() * f.as_f64())
    }
}

impl Mul<TimeSeconds> for Factor {
    type Output = TimeSeconds;
    fn mul(self, x: TimeSeconds) -> TimeSeconds {
        TimeSeconds::new(self.as_f64() * x.as_seconds())
    }
}
impl Mul<Factor> for TimeSeconds {
    type Output = TimeSeconds;
    fn mul(self, f: Factor) -> TimeSeconds {
        TimeSeconds::new(self.as_seconds() * f.as_f64())
    }
}

impl Mul<SpeedMs> for Factor {
    type Output = SpeedMs;
    fn mul(self, x: SpeedMs) -> SpeedMs {
        SpeedMs::new(self.as_f64() * x.as_ms())
    }
}
impl Mul<Factor> for SpeedMs {
    type Output = SpeedMs;
    fn mul(self, f: Factor) -> SpeedMs {
        SpeedMs::new(self.as_ms() * f.as_f64())
    }
}

impl Mul<SpeedKph> for Factor {
    type Output = SpeedKph;
    fn mul(self, x: SpeedKph) -> SpeedKph {
        SpeedKph::new(self.as_f64() * x.as_kph())
    }
}
impl Mul<Factor> for SpeedKph {
    type Output = SpeedKph;
    fn mul(self, f: Factor) -> SpeedKph {
        SpeedKph::new(self.as_kph() * f.as_f64())
    }
}

impl Mul<SpeedMph> for Factor {
    type Output = SpeedMph;
    fn mul(self, x: SpeedMph) -> SpeedMph {
        SpeedMph::new(self.as_f64() * x.as_mph())
    }
}
impl Mul<Factor> for SpeedMph {
    type Output = SpeedMph;
    fn mul(self, f: Factor) -> SpeedMph {
        SpeedMph::new(self.as_mph() * f.as_f64())
    }
}
#[cfg(test)]
mod tests;
