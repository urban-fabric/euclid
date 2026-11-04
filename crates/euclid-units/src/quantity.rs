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
/// `Measure` *is* a [`Quantity`] (and gains the scalar/linear/dimension contracts) through the
/// blanket impl below.
///
/// [`Natural`]: euclid_numeric::Natural
/// [`Integer`]: euclid_numeric::Integer
pub trait Measure: Clone {
    /// The numeric carrier, such as `f64`, `u64`, or `i64`.
    type Repr: Element;
    const DIMENSION: Dimension;
    const UNIT_NAME: &'static str;
    fn raw(&self) -> Self::Repr;
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
