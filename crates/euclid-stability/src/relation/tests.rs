//! Each relation must accept a law that holds and reject one that does not.
//!
//! Both halves matter equally. A relation that returned `Ok` unconditionally would pass every
//! positive test in this file, so every relation below is also handed a function that violates
//! it, and required to complain.

use super::{
    Absolute, Compare, Exact, Relative, Ulps, agree, associative, bounded, commutative,
    distributes, even, f32_ulps_between, fixed_point, forward_stable, forward_stable_margin,
    idempotent, insensitive_to, involution, monotone, odd, order_independent, precision_agrees,
    preserves, round_trip,
};

// ===== The comparators ====================================================

mod comparisons;
mod metamorphic;
mod numerical;
