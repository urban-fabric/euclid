use std::ops::{Div, Mul};

use super::{Factor, LengthMeters, SpeedKph, SpeedMph, SpeedMs, TimeSeconds};

impl Div<TimeSeconds> for LengthMeters {
    type Output = SpeedMs;
    fn div(self, t: TimeSeconds) -> SpeedMs {
        SpeedMs(self.0 / t.0)
    }
}

impl Div<SpeedMs> for LengthMeters {
    type Output = TimeSeconds;
    fn div(self, v: SpeedMs) -> TimeSeconds {
        TimeSeconds(self.0 / v.0)
    }
}

impl Div<SpeedKph> for SpeedKph {
    type Output = f64;
    fn div(self, rhs: SpeedKph) -> f64 {
        self.0 / rhs.0
    }
}

impl Div<SpeedMs> for SpeedMs {
    type Output = f64;
    fn div(self, rhs: SpeedMs) -> f64 {
        self.0 / rhs.0
    }
}

impl Mul<TimeSeconds> for SpeedMs {
    type Output = LengthMeters;
    fn mul(self, t: TimeSeconds) -> LengthMeters {
        LengthMeters(self.0 * t.0)
    }
}
impl Mul<SpeedMs> for TimeSeconds {
    type Output = LengthMeters;
    fn mul(self, v: SpeedMs) -> LengthMeters {
        LengthMeters(self.0 * v.0)
    }
}

impl Mul<LengthMeters> for Factor {
    type Output = LengthMeters;
    fn mul(self, x: LengthMeters) -> LengthMeters {
        LengthMeters(self.0 * x.0)
    }
}
impl Mul<Factor> for LengthMeters {
    type Output = LengthMeters;
    fn mul(self, f: Factor) -> LengthMeters {
        LengthMeters(self.0 * f.0)
    }
}

impl Mul<TimeSeconds> for Factor {
    type Output = TimeSeconds;
    fn mul(self, x: TimeSeconds) -> TimeSeconds {
        TimeSeconds(self.0 * x.0)
    }
}
impl Mul<Factor> for TimeSeconds {
    type Output = TimeSeconds;
    fn mul(self, f: Factor) -> TimeSeconds {
        TimeSeconds(self.0 * f.0)
    }
}

impl Mul<SpeedMs> for Factor {
    type Output = SpeedMs;
    fn mul(self, x: SpeedMs) -> SpeedMs {
        SpeedMs(self.0 * x.0)
    }
}
impl Mul<Factor> for SpeedMs {
    type Output = SpeedMs;
    fn mul(self, f: Factor) -> SpeedMs {
        SpeedMs(self.0 * f.0)
    }
}

impl Mul<SpeedKph> for Factor {
    type Output = SpeedKph;
    fn mul(self, x: SpeedKph) -> SpeedKph {
        SpeedKph(self.0 * x.0)
    }
}
impl Mul<Factor> for SpeedKph {
    type Output = SpeedKph;
    fn mul(self, f: Factor) -> SpeedKph {
        SpeedKph(self.0 * f.0)
    }
}

impl Mul<SpeedMph> for Factor {
    type Output = SpeedMph;
    fn mul(self, x: SpeedMph) -> SpeedMph {
        SpeedMph(self.0 * x.0)
    }
}
impl Mul<Factor> for SpeedMph {
    type Output = SpeedMph;
    fn mul(self, f: Factor) -> SpeedMph {
        SpeedMph(self.0 * f.0)
    }
}
#[cfg(test)]
mod tests;
