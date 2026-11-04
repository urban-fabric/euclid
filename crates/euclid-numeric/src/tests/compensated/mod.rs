//! The compensated-arithmetic primitives and the cancellation-prone elementary forms added
//! alongside them.
//!
//! [`two_sum`](Real::two_sum) and [`two_product`](Real::two_product) are exactness claims, not
//! approximation claims, so most of these tests assert *bit-exact* identities rather than a
//! tolerance. The pay-off tests show compensated summation and an accurate dot product beating
//! naive accumulation by orders of magnitude — which is the whole reason the primitives exist.
//!
//! Split into the files below to keep each one inside this crate's 400-line budget.

mod derived;
mod primitives;
mod trigonometric;
