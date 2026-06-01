//! Cross-unit conversions that are implemented as inherent methods.

use crate::{DistanceWeight, LengthMeters, SpeedKph, TimeSeconds, TimeWeight};

impl LengthMeters {
    pub fn to_distance_weight(self) -> DistanceWeight {
        DistanceWeight::from_meters(self.as_meters())
    }

    pub fn travel_time_at(self, speed: SpeedKph) -> TimeSeconds {
        if !speed.is_finite() || !speed.is_positive() {
            return TimeSeconds::INFINITY;
        }
        TimeSeconds::new(self.as_meters() * 3.6 / speed.as_kph())
    }
}

impl TimeSeconds {
    pub fn to_time_weight(self) -> TimeWeight {
        TimeWeight::from_seconds(self.as_seconds())
    }

    pub fn distance_at(self, speed: SpeedKph) -> LengthMeters {
        LengthMeters::new(speed.as_kph() * self.as_seconds() / 3.6)
    }
}

impl From<DistanceWeight> for LengthMeters {
    fn from(d: DistanceWeight) -> Self {
        Self::new(d.as_meters())
    }
}

impl From<TimeWeight> for TimeSeconds {
    fn from(t: TimeWeight) -> Self {
        Self::new(t.as_seconds())
    }
}

#[cfg(test)]
mod tests;
