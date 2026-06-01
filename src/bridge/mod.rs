use super::{Dimension, LengthMeters, Quantity, SpeedKph, TimeSeconds};
use crate::{DistanceWeight, TimeWeight};

impl Quantity for DistanceWeight {
    type Repr = u32;
    const DIMENSION: Dimension = Dimension::Length;
    const UNIT_NAME: &'static str = "dm";
    fn raw(self) -> u32 {
        self.0
    }
    fn from_raw(v: u32) -> Self {
        Self(v)
    }
}

impl Quantity for TimeWeight {
    type Repr = u32;
    const DIMENSION: Dimension = Dimension::Time;
    const UNIT_NAME: &'static str = "cs";
    fn raw(self) -> u32 {
        self.0
    }
    fn from_raw(v: u32) -> Self {
        Self(v)
    }
}

impl LengthMeters {
    pub fn to_distance_weight(self) -> DistanceWeight {
        DistanceWeight::from_meters(self.0)
    }

    pub fn travel_time_at(self, speed: SpeedKph) -> TimeSeconds {
        if !speed.0.is_finite() || speed.0 <= 0.0 {
            return TimeSeconds::INFINITY;
        }
        TimeSeconds(self.0 * 3.6 / speed.0)
    }
}

impl TimeSeconds {
    pub fn to_time_weight(self) -> TimeWeight {
        TimeWeight::from_seconds(self.0)
    }

    pub fn distance_at(self, speed: SpeedKph) -> LengthMeters {
        LengthMeters(speed.0 * self.0 / 3.6)
    }
}

impl From<DistanceWeight> for LengthMeters {
    fn from(d: DistanceWeight) -> Self {
        Self(d.as_meters())
    }
}

impl From<TimeWeight> for TimeSeconds {
    fn from(t: TimeWeight) -> Self {
        Self(t.as_seconds())
    }
}
#[cfg(test)]
mod tests;
