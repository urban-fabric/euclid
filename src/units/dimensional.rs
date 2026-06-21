//! Dimensional cross-products over the lazy units — canonical-unit discipline (length
//! over time is a speed, two lengths an area), generic over the carrier `T`, so the
//! algebra holds whether `T` is `f64`, a polars `Expr`, or a backend `Tensor` scalar.

use core::ops::{Div, Mul};

use crate::LazyReal;

use super::measures::{LazyMeters, LazySeconds, LazySpeedMs, LazySquareMeters};

/// meters ÷ seconds = m/s
impl<T: LazyReal> Div<LazySeconds<T>> for LazyMeters<T> {
    type Output = LazySpeedMs<T>;
    fn div(self, t: LazySeconds<T>) -> LazySpeedMs<T> {
        LazySpeedMs(self.0 / t.0)
    }
}

/// meters ÷ (m/s) = seconds
impl<T: LazyReal> Div<LazySpeedMs<T>> for LazyMeters<T> {
    type Output = LazySeconds<T>;
    fn div(self, v: LazySpeedMs<T>) -> LazySeconds<T> {
        LazySeconds(self.0 / v.0)
    }
}

/// meters × meters = m²
impl<T: LazyReal> Mul<LazyMeters<T>> for LazyMeters<T> {
    type Output = LazySquareMeters<T>;
    fn mul(self, rhs: LazyMeters<T>) -> LazySquareMeters<T> {
        LazySquareMeters(self.0 * rhs.0)
    }
}

/// m² ÷ meters = meters
impl<T: LazyReal> Div<LazyMeters<T>> for LazySquareMeters<T> {
    type Output = LazyMeters<T>;
    fn div(self, rhs: LazyMeters<T>) -> LazyMeters<T> {
        LazyMeters(self.0 / rhs.0)
    }
}

/// (m/s) × seconds = meters
impl<T: LazyReal> Mul<LazySeconds<T>> for LazySpeedMs<T> {
    type Output = LazyMeters<T>;
    fn mul(self, t: LazySeconds<T>) -> LazyMeters<T> {
        LazyMeters(self.0 * t.0)
    }
}

/// seconds × (m/s) = meters
impl<T: LazyReal> Mul<LazySpeedMs<T>> for LazySeconds<T> {
    type Output = LazyMeters<T>;
    fn mul(self, v: LazySpeedMs<T>) -> LazyMeters<T> {
        LazyMeters(self.0 * v.0)
    }
}
