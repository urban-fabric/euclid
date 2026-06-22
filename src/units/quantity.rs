use crate::{Compare, Real};

use super::Dimension;

/// The mask a comparison over `Q`'s carrier yields — `bool` for host reals and a carrier mask
/// for expression/tensor implementations.
pub type Mask<Q> = <<Q as Quantity>::Repr as Compare>::Mask;

/// A scalar quantity backed by a [`Real`] carrier. `Clone`, not `Copy`: expression and graph
/// handles may not be cheaply copyable.
pub trait Quantity: Clone {
    /// The numeric carrier, such as `f64` or an expression/tensor scalar.
    type Repr: Real;
    const DIMENSION: Dimension;
    const UNIT_NAME: &'static str;
    fn raw(&self) -> Self::Repr;
    fn from_raw(v: Self::Repr) -> Self;
}
