//! Dimensional cross-products over unit wrappers. The algebra is generic over the carrier
//! `T`, so it holds for `f64`, expression carriers, and tensor scalar carriers.

use core::ops::{Div, Mul};

use crate::Real;

use super::measures::{Meters, Seconds, SpeedMs, SquareMeters};

/// meters ÷ seconds = m/s
impl<T: Real> Div<Seconds<T>> for Meters<T> {
    type Output = SpeedMs<T>;
    fn div(self, t: Seconds<T>) -> SpeedMs<T> {
        SpeedMs(self.0 / t.0)
    }
}

/// meters ÷ (m/s) = seconds
impl<T: Real> Div<SpeedMs<T>> for Meters<T> {
    type Output = Seconds<T>;
    fn div(self, v: SpeedMs<T>) -> Seconds<T> {
        Seconds(self.0 / v.0)
    }
}

/// meters × meters = m²
impl<T: Real> Mul<Meters<T>> for Meters<T> {
    type Output = SquareMeters<T>;
    fn mul(self, rhs: Meters<T>) -> SquareMeters<T> {
        SquareMeters(self.0 * rhs.0)
    }
}

/// m² ÷ meters = meters
impl<T: Real> Div<Meters<T>> for SquareMeters<T> {
    type Output = Meters<T>;
    fn div(self, rhs: Meters<T>) -> Meters<T> {
        Meters(self.0 / rhs.0)
    }
}

/// (m/s) × seconds = meters
impl<T: Real> Mul<Seconds<T>> for SpeedMs<T> {
    type Output = Meters<T>;
    fn mul(self, t: Seconds<T>) -> Meters<T> {
        Meters(self.0 * t.0)
    }
}

/// seconds × (m/s) = meters
impl<T: Real> Mul<SpeedMs<T>> for Seconds<T> {
    type Output = Meters<T>;
    fn mul(self, v: SpeedMs<T>) -> Meters<T> {
        Meters(self.0 * v.0)
    }
}
