//! Cross-quantity bridges between distinct euclid quantity types (e.g. length to
//! travel time at a speed). Separate from the same-dimension
//! [`UnitOfMeasure::convert`](crate::UnitOfMeasure::convert) protocol.
//!
//! Route-weight bridges (length/time → `TimeWeight`/`DistanceWeight`) live with
//! the weight types in the graph crate, not here — euclid owns only fundamental
//! units.

use crate::{LengthMeters, SpeedKph, TimeSeconds};

impl LengthMeters {
    pub fn travel_time_at(self, speed: SpeedKph) -> TimeSeconds {
        if !speed.is_finite() || !speed.is_positive() {
            return TimeSeconds::INFINITY;
        }
        TimeSeconds::new(self.as_meters() * 3.6 / speed.as_kph())
    }
}

impl TimeSeconds {
    pub fn distance_at(self, speed: SpeedKph) -> LengthMeters {
        LengthMeters::new(speed.as_kph() * self.as_seconds() / 3.6)
    }
}

#[cfg(test)]
mod tests;
