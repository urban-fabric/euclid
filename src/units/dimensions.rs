use super::LinearQuantity;

/// A length-dimensioned quantity. Carries the cross-dimensional algebra
/// `length × length = area`, `length / time = speed`, and `length / speed = time` through its
/// associated [`Area`], [`Speed`], and [`Time`] partner quantities.
pub trait Length: LinearQuantity {
    /// The area produced by `length × length`.
    type Area: Area<Repr = Self::Repr>;
    /// The time that pairs with this length and speed.
    type Time: Time<Repr = Self::Repr>;
    /// The speed produced by `length / time`.
    type Speed: Speed<Repr = Self::Repr>;

    /// `length × length = area`.
    fn times(self, width: Self) -> Self::Area;
    /// `length / time = speed`.
    fn per(self, elapsed: Self::Time) -> Self::Speed;
    /// `length / speed = time`.
    fn over(self, speed: Self::Speed) -> Self::Time;

    /// `length²` — [`times`](Self::times) with a copy of `self`.
    fn squared(self) -> Self::Area;
}

/// A time-dimensioned quantity. `time × speed = length`.
pub trait Time: LinearQuantity {
    /// The length produced by `time × speed`.
    type Length: Length<Repr = Self::Repr>;
    /// The speed that pairs with this time.
    type Speed: Speed<Repr = Self::Repr>;

    /// `time × speed = length`.
    fn times(self, speed: Self::Speed) -> Self::Length;
}

/// A speed-dimensioned quantity. `speed × time = length`.
pub trait Speed: LinearQuantity {
    /// The length produced by `speed × time`.
    type Length: Length<Repr = Self::Repr>;
    /// The time that pairs with this speed.
    type Time: Time<Repr = Self::Repr>;

    /// `speed × time = length`.
    fn times(self, elapsed: Self::Time) -> Self::Length;
}

/// An area-dimensioned quantity. `area / length = length`.
pub trait Area: LinearQuantity {
    /// The length produced by `area / length`.
    type Length: Length<Repr = Self::Repr>;

    /// `area / length = length`.
    fn over(self, side: Self::Length) -> Self::Length;
}
