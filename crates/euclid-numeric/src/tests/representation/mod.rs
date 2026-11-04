//! Floating-point representation (`frexp`, `ldexp`, `next_after`, `ulp`, `signbit`, `modf`)
//! and the further analysis added with it (`exprel`, `rgamma`, `polygamma`).
//!
//! The representation group is mostly *exactness* claims — a decomposition that round-trips
//! bit-for-bit, a step that lands on precisely the adjacent representable value — so these
//! assert equality rather than a tolerance wherever that is the actual contract.
//!
//! Split into the files below to keep each one inside this crate's 400-line budget.

mod analysis;
mod float_bits;
