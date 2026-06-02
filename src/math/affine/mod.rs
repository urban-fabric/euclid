//! Affine positions versus vector displacements.
//!
//! A quantity newtype like [`TimeSeconds`](crate::TimeSeconds) is a *displacement*:
//! a magnitude you can add, negate, and scale. An *absolute position* is different.
//! Two timestamps can be subtracted to get a duration, and a duration can be added
//! to a timestamp to get another timestamp, but adding two timestamps is
//! meaningless. That distinction is the difference between a point in an affine
//! space and a vector in the space of displacements.
//!
//! [`Point<Q>`] makes that distinction first-class: it is an absolute position
//! whose displacement type is the scalar quantity `Q`. The arithmetic is the
//! affine algebra and nothing more:
//!
//! - `Point<Q, Space> - Point<Q, Space> = Q` (the displacement between two positions)
//! - `Point<Q, Space> + Q = Point<Q, Space>` (translate a position by a displacement)
//! - `Point<Q, Space> - Q = Point<Q, Space>`
//! - `Point<Q, Space> + Point<Q, Space>` is intentionally not implemented
//!
//! `Point<Q>` is the reusable floating-point form of this affine pattern for any
//! [`LinearQuantity`]. It is *not* the same thing as the integer time newtypes
//! [`TimeMicros`](crate::TimeMicros) / [`TimeMicrosDelta`](crate::TimeMicrosDelta):
//! `TimeMicros` is a summable magnitude (it implements `Add` and `Sum`), and
//! `TimeMicrosDelta` is its signed difference, produced with
//! [`TimeMicrosDelta::between`](crate::TimeMicrosDelta::between) — not an affine
//! position-versus-displacement split. Reach for `Point<Q>` specifically when you
//! need positions that must not be added together.
//!
//! ```compile_fail
//! use euclid::{LengthMeters, Point};
//!
//! struct RoadSpace;
//! struct ScreenSpace;
//!
//! let road = Point::<LengthMeters, RoadSpace>::new_in_space(LengthMeters::new(10.0));
//! let screen = Point::<LengthMeters, ScreenSpace>::new_in_space(LengthMeters::new(10.0));
//! let _ = road - screen;
//! ```

use core::marker::PhantomData;
use core::ops::{Add, AddAssign, Sub, SubAssign};

use crate::LinearQuantity;

/// An absolute position in the affine space whose displacements are `Q`.
///
/// The wrapped `Q` is the position measured from an implicit origin. Construct
/// with [`Point::new`] and recover the displacement-from-origin with
/// [`Point::value`]. The optional `Space` marker distinguishes unrelated affine
/// spaces that happen to share the same displacement unit.
#[repr(transparent)]
#[derive(Debug, PartialEq, PartialOrd, serde::Serialize, serde::Deserialize)]
#[serde(transparent)]
pub struct Point<Q: LinearQuantity, Space = ()>(Q, PhantomData<fn() -> Space>);

impl<Q: LinearQuantity, Space> Copy for Point<Q, Space> {}

impl<Q: LinearQuantity, Space> Clone for Point<Q, Space> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<Q: LinearQuantity> Point<Q> {
    /// A position at displacement `value` from the origin.
    pub fn new(value: Q) -> Self {
        Self::new_in_space(value)
    }

    /// Build a position straight from a raw coordinate.
    pub fn from_raw(raw: f64) -> Self {
        Self::from_raw_in_space(raw)
    }
}

impl<Q: LinearQuantity, Space> Point<Q, Space> {
    /// A position at displacement `value` from the origin in `Space`.
    pub fn new_in_space(value: Q) -> Self {
        Self(value, PhantomData)
    }

    /// Build a position in `Space` straight from a raw coordinate.
    pub fn from_raw_in_space(raw: f64) -> Self {
        Self(Q::from_raw(raw), PhantomData)
    }

    /// The position expressed as a displacement from the origin.
    pub fn value(self) -> Q {
        self.0
    }

    /// The raw coordinate.
    pub fn raw(self) -> f64 {
        self.0.raw()
    }

    /// Linearly interpolate between two positions: `t = 0` yields `self`,
    /// `t = 1` yields `other`. This is an affine combination, so the result is a
    /// position even though its definition mixes two of them.
    pub fn lerp(self, other: Self, t: f64) -> Self {
        Self::from_raw_in_space(self.raw() * (1.0 - t) + other.raw() * t)
    }

    /// The position halfway between two positions.
    pub fn midpoint(self, other: Self) -> Self {
        self.lerp(other, 0.5)
    }
}

impl<Q: LinearQuantity, Space> Sub for Point<Q, Space> {
    type Output = Q;
    /// The displacement between two positions.
    fn sub(self, rhs: Self) -> Q {
        Q::from_raw(self.raw() - rhs.raw())
    }
}

impl<Q: LinearQuantity, Space> Add<Q> for Point<Q, Space> {
    type Output = Self;
    /// Translate a position by a displacement.
    fn add(self, rhs: Q) -> Self {
        Self::from_raw_in_space(self.raw() + rhs.raw())
    }
}

impl<Q: LinearQuantity, Space> Sub<Q> for Point<Q, Space> {
    type Output = Self;
    /// Translate a position back by a displacement.
    fn sub(self, rhs: Q) -> Self {
        Self::from_raw_in_space(self.raw() - rhs.raw())
    }
}

impl<Q: LinearQuantity, Space> AddAssign<Q> for Point<Q, Space> {
    fn add_assign(&mut self, rhs: Q) {
        *self = *self + rhs;
    }
}

impl<Q: LinearQuantity, Space> SubAssign<Q> for Point<Q, Space> {
    fn sub_assign(&mut self, rhs: Q) {
        *self = *self - rhs;
    }
}

#[cfg(test)]
mod tests;
