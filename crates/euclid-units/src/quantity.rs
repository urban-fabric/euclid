use euclid_numeric::{Compare, Element, Real};

use super::Dimension;

/// The mask a comparison over `Q`'s carrier yields — `bool` for host reals and a carrier mask
/// for expression/tensor implementations.
pub type Mask<Q> = <<Q as Measure>::Repr as Compare>::Mask;

/// A dimensioned scalar over any numeric carrier — the carrier-generic base of the units system.
///
/// `Measure` names a value's runtime [`Dimension`] and unit symbol and exposes its raw carrier,
/// without requiring real-number algebra. That admits integer-backed quantities — byte sizes,
/// counts, centisecond/decimeter storage, epoch timestamps — whose carrier is a [`Natural`] or
/// [`Integer`] rather than a [`Real`], so they cannot be a [`Quantity`]. Every `Real`-backed
/// `Measure` *is* a [`Quantity`] through the blanket impl below, which is what makes it eligible
/// for the scalar, linear, and dimension contracts — those stay opt-in per unit, each an empty
/// impl block over defaults, so a unit claims only the behaviour it actually has.
///
/// [`Natural`]: euclid_numeric::Natural
/// [`Integer`]: euclid_numeric::Integer
pub trait Measure: Clone {
    /// The numeric carrier, such as `f64`, `u64`, or `i64`.
    type Repr: Element;
    /// The physical dimension, available at runtime for error messages and dynamic checks.
    ///
    /// The *type* already encodes it — that is what makes a dimension error a compile error —
    /// so this is for the cases a type cannot reach: formatting, serialisation, and a planner
    /// deciding what it has been handed.
    const DIMENSION: Dimension;
    /// The unit symbol, as it should appear to a reader: `"m"`, `"km/h"`, `"°C"`.
    const UNIT_NAME: &'static str;
    /// The underlying number, with the dimension stripped.
    ///
    /// The escape hatch out of the units system, and the one place a dimension error becomes
    /// possible again — so prefer the dimensional operations and reach for this at the edges,
    /// where a number has to be handed to something that does not speak units.
    fn raw(&self) -> Self::Repr;
    /// Attach this unit to a bare number, which is **asserted rather than checked**: the caller
    /// is promising the value is already in [`UNIT_NAME`](Self::UNIT_NAME).
    fn from_raw(v: Self::Repr) -> Self;
}

/// A [`Measure`] over a [`Real`] carrier — the root of the scalar, linear, and dimension
/// contracts. Implemented automatically for every real-backed measure; implement [`Measure`]
/// rather than this trait directly.
pub trait Quantity: Measure<Repr: Real> {}

impl<T> Quantity for T
where
    T: Measure,
    T::Repr: Real,
{
}
