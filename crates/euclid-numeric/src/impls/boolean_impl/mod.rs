//! Concrete impls of the boolean/comparison tier. `bool` implements [`Element`] and [`Boolean`];
//! the ordered host carriers (`f64`/`f32`, `i32`/`i64`/`i128`, `u32`/`u64`/`u128`) implement
//! [`Compare`] with a `bool` [`Mask`](Compare::Mask). The trait *definitions* live in
//! [`boolean`](crate::boolean). (Complex is unordered, so it carries no comparison.)

use crate::{Boolean, Element};

mod float_compare;
mod signed_compare;
mod unsigned_compare;

impl Element for bool {
    /// The additive identity of the Boolean semiring (`∨`).
    #[inline]
    fn zero() -> Self {
        false
    }
    /// The multiplicative identity of the Boolean semiring (`∧`).
    #[inline]
    fn one() -> Self {
        true
    }
}

impl Boolean for bool {
    #[inline]
    fn and(&self, other: &Self) -> Self {
        *self && *other
    }
    #[inline]
    fn or(&self, other: &Self) -> Self {
        *self || *other
    }
    #[inline]
    fn xor(&self, other: &Self) -> Self {
        *self ^ *other
    }
    #[inline]
    fn not(&self) -> Self {
        !*self
    }
}

/// The eager host loop behind [`crate::Compare::while_loop`] for `bool`-masked carriers.
fn host_while_loop<C, P, B>(init: C, cond: P, body: B) -> C
where
    P: Fn(&C) -> bool,
    B: Fn(C) -> C,
{
    let mut state = init;
    while cond(&state) {
        state = body(state);
    }
    state
}
