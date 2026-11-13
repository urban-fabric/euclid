//! Cross-quantity bridges between distinct oresme quantity types (e.g. length to
//! travel time at a speed). Separate from the same-dimension
//! [`UnitOfMeasure::convert`](crate::UnitOfMeasure::convert) protocol.
//!
//! Route-weight bridges (length/time → `TimeWeight`/`DistanceWeight`) live with
//! the weight types in the graph crate, not here — oresme owns only fundamental
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
mod tests {
    use crate::{LengthMeters, Quantity, SpeedKph, TimeSeconds};

    #[test]
    fn travel_time_at_helper_uses_canonical_operand_order() {
        let l = LengthMeters::new(1500.0);
        let speed = SpeedKph::new(50.0);
        let helper = l.travel_time_at(speed);
        let expected = TimeSeconds::new(1500.0 * 3.6 / 50.0);
        assert_eq!(helper.as_seconds(), expected.as_seconds());
    }

    #[test]
    fn distance_at_helper_uses_canonical_operand_order() {
        let t = TimeSeconds::new(120.0);
        let speed = SpeedKph::new(60.0);
        let helper = t.distance_at(speed);
        let expected = LengthMeters::new(60.0 * 120.0 / 3.6);
        assert_eq!(helper.as_meters(), expected.as_meters());
    }

    #[test]
    fn travel_time_at_handles_zero_and_negative_speed() {
        let l = LengthMeters::new(100.0);
        assert_eq!(l.travel_time_at(SpeedKph::new(0.0)), TimeSeconds::INFINITY);
        assert_eq!(l.travel_time_at(SpeedKph::new(-5.0)), TimeSeconds::INFINITY);
        assert_eq!(
            l.travel_time_at(SpeedKph::from_raw(f64::NAN)),
            TimeSeconds::INFINITY,
        );
    }
}
