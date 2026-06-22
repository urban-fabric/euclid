//! Physical-dimension contracts over any [`Real`](crate::Real) carrier.
//!
//! Like the rest of euclid, `units` ships *contracts*, not concrete types. A quantity is any
//! [`Real`](crate::Real)-backed carrier that implements [`Quantity`]; the dimension traits
//! ([`Length`], [`Time`], [`Speed`], [`Area`], [`Angle`]) refine it with the operations that
//! dimension supports, including the cross-dimensional algebra (`length × length = area`,
//! `length / time = speed`, …) expressed through associated quantity types. euclid does not
//! ship a `Meters`/`Seconds` struct; a consumer defines its own unit carriers and implements
//! these traits for them.
//!
//! Quantities are scalar-valued: a length carries one `T`, where `T` can be a host scalar,
//! expression, or graph scalar. Values are `Clone`, comparisons return masks through
//! [`Compare`](crate::Compare), and host literals are lifted through
//! [`Real::scalar`](crate::Real::scalar)/[`scalar_like`](crate::Real::scalar_like).

mod angle;
mod dimension;
mod dimensions;
mod linear;
mod quantity;
mod scalar;

pub use angle::Angle;
pub use dimension::Dimension;
pub use dimensions::{Area, Length, Speed, Time};
pub use linear::LinearQuantity;
pub use quantity::{Mask, Quantity};
pub use scalar::ScalarQuantity;

#[cfg(test)]
mod tests;
