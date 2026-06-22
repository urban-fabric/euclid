//! Unit newtypes over any [`Real`] carrier.
//!
//! Each unit is a thin newtype over a [`Real`] carrier `T` with the same shape: the
//! struct, its `Quantity`/`LinearQuantity` impls, and the additive operator set
//! (`Add`/`Sub`/`Neg`, plus carrier scaling `Mul<T>`). The operator impls are written
//! out per unit because Rust's orphan rule forbids a blanket
//! `impl<Q: LinearQuantity> Add for Q`. Cross-unit products (length × length = area, …)
//! live in [`dimensional`](super::dimensional).

mod angle;
mod area;
mod length;
mod speed;
mod time;

pub use angle::Angle;
pub use area::SquareMeters;
pub use length::{Km, Meters};
pub use speed::SpeedMs;
pub use time::Seconds;
