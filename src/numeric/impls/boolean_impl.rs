//! Concrete impls of the boolean/comparison tier. `bool` implements [`Element`] and [`Boolean`];
//! the ordered host carriers (`f64`/`f32`, `i32`/`i64`/`i128`, `u32`/`u64`/`u128`) implement
//! [`Compare`] with a `bool` [`Mask`](Compare::Mask). The trait *definitions* live in
//! [`boolean`](crate::numeric::boolean). (Complex is unordered, so it carries no comparison.)

use crate::numeric::{Boolean, Compare, Element};

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

/// The eager host loop behind [`Compare::while_loop`] for `bool`-masked carriers.
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

impl Compare for f64 {
    type Mask = bool;

    #[inline]
    fn lt(&self, other: &Self) -> bool {
        *self < *other
    }
    #[inline]
    fn le(&self, other: &Self) -> bool {
        *self <= *other
    }
    #[inline]
    fn gt(&self, other: &Self) -> bool {
        *self > *other
    }
    #[inline]
    fn ge(&self, other: &Self) -> bool {
        *self >= *other
    }
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        *self == *other
    }
    #[inline]
    fn ne(&self, other: &Self) -> bool {
        *self != *other
    }
    #[inline]
    fn select(mask: &bool, on_true: Self, on_false: Self) -> Self {
        if *mask {
            on_true
        } else {
            on_false
        }
    }
    #[inline]
    fn while_loop<C, P, B>(init: C, cond: P, body: B) -> C
    where
        P: Fn(&C) -> bool,
        B: Fn(C) -> C,
    {
        host_while_loop(init, cond, body)
    }
}

impl Compare for f32 {
    type Mask = bool;

    #[inline]
    fn lt(&self, other: &Self) -> bool {
        *self < *other
    }
    #[inline]
    fn le(&self, other: &Self) -> bool {
        *self <= *other
    }
    #[inline]
    fn gt(&self, other: &Self) -> bool {
        *self > *other
    }
    #[inline]
    fn ge(&self, other: &Self) -> bool {
        *self >= *other
    }
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        *self == *other
    }
    #[inline]
    fn ne(&self, other: &Self) -> bool {
        *self != *other
    }
    #[inline]
    fn select(mask: &bool, on_true: Self, on_false: Self) -> Self {
        if *mask {
            on_true
        } else {
            on_false
        }
    }
    #[inline]
    fn while_loop<C, P, B>(init: C, cond: P, body: B) -> C
    where
        P: Fn(&C) -> bool,
        B: Fn(C) -> C,
    {
        host_while_loop(init, cond, body)
    }
}

impl Compare for i32 {
    type Mask = bool;

    #[inline]
    fn lt(&self, other: &Self) -> bool {
        *self < *other
    }
    #[inline]
    fn le(&self, other: &Self) -> bool {
        *self <= *other
    }
    #[inline]
    fn gt(&self, other: &Self) -> bool {
        *self > *other
    }
    #[inline]
    fn ge(&self, other: &Self) -> bool {
        *self >= *other
    }
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        *self == *other
    }
    #[inline]
    fn ne(&self, other: &Self) -> bool {
        *self != *other
    }
    #[inline]
    fn select(mask: &bool, on_true: Self, on_false: Self) -> Self {
        if *mask {
            on_true
        } else {
            on_false
        }
    }
    #[inline]
    fn while_loop<C, P, B>(init: C, cond: P, body: B) -> C
    where
        P: Fn(&C) -> bool,
        B: Fn(C) -> C,
    {
        host_while_loop(init, cond, body)
    }
}

impl Compare for i64 {
    type Mask = bool;

    #[inline]
    fn lt(&self, other: &Self) -> bool {
        *self < *other
    }
    #[inline]
    fn le(&self, other: &Self) -> bool {
        *self <= *other
    }
    #[inline]
    fn gt(&self, other: &Self) -> bool {
        *self > *other
    }
    #[inline]
    fn ge(&self, other: &Self) -> bool {
        *self >= *other
    }
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        *self == *other
    }
    #[inline]
    fn ne(&self, other: &Self) -> bool {
        *self != *other
    }
    #[inline]
    fn select(mask: &bool, on_true: Self, on_false: Self) -> Self {
        if *mask {
            on_true
        } else {
            on_false
        }
    }
    #[inline]
    fn while_loop<C, P, B>(init: C, cond: P, body: B) -> C
    where
        P: Fn(&C) -> bool,
        B: Fn(C) -> C,
    {
        host_while_loop(init, cond, body)
    }
}

impl Compare for i128 {
    type Mask = bool;

    #[inline]
    fn lt(&self, other: &Self) -> bool {
        *self < *other
    }
    #[inline]
    fn le(&self, other: &Self) -> bool {
        *self <= *other
    }
    #[inline]
    fn gt(&self, other: &Self) -> bool {
        *self > *other
    }
    #[inline]
    fn ge(&self, other: &Self) -> bool {
        *self >= *other
    }
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        *self == *other
    }
    #[inline]
    fn ne(&self, other: &Self) -> bool {
        *self != *other
    }
    #[inline]
    fn select(mask: &bool, on_true: Self, on_false: Self) -> Self {
        if *mask {
            on_true
        } else {
            on_false
        }
    }
    #[inline]
    fn while_loop<C, P, B>(init: C, cond: P, body: B) -> C
    where
        P: Fn(&C) -> bool,
        B: Fn(C) -> C,
    {
        host_while_loop(init, cond, body)
    }
}

impl Compare for u32 {
    type Mask = bool;

    #[inline]
    fn lt(&self, other: &Self) -> bool {
        *self < *other
    }
    #[inline]
    fn le(&self, other: &Self) -> bool {
        *self <= *other
    }
    #[inline]
    fn gt(&self, other: &Self) -> bool {
        *self > *other
    }
    #[inline]
    fn ge(&self, other: &Self) -> bool {
        *self >= *other
    }
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        *self == *other
    }
    #[inline]
    fn ne(&self, other: &Self) -> bool {
        *self != *other
    }
    #[inline]
    fn select(mask: &bool, on_true: Self, on_false: Self) -> Self {
        if *mask {
            on_true
        } else {
            on_false
        }
    }
    #[inline]
    fn while_loop<C, P, B>(init: C, cond: P, body: B) -> C
    where
        P: Fn(&C) -> bool,
        B: Fn(C) -> C,
    {
        host_while_loop(init, cond, body)
    }
}

impl Compare for u64 {
    type Mask = bool;

    #[inline]
    fn lt(&self, other: &Self) -> bool {
        *self < *other
    }
    #[inline]
    fn le(&self, other: &Self) -> bool {
        *self <= *other
    }
    #[inline]
    fn gt(&self, other: &Self) -> bool {
        *self > *other
    }
    #[inline]
    fn ge(&self, other: &Self) -> bool {
        *self >= *other
    }
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        *self == *other
    }
    #[inline]
    fn ne(&self, other: &Self) -> bool {
        *self != *other
    }
    #[inline]
    fn select(mask: &bool, on_true: Self, on_false: Self) -> Self {
        if *mask {
            on_true
        } else {
            on_false
        }
    }
    #[inline]
    fn while_loop<C, P, B>(init: C, cond: P, body: B) -> C
    where
        P: Fn(&C) -> bool,
        B: Fn(C) -> C,
    {
        host_while_loop(init, cond, body)
    }
}

impl Compare for u128 {
    type Mask = bool;

    #[inline]
    fn lt(&self, other: &Self) -> bool {
        *self < *other
    }
    #[inline]
    fn le(&self, other: &Self) -> bool {
        *self <= *other
    }
    #[inline]
    fn gt(&self, other: &Self) -> bool {
        *self > *other
    }
    #[inline]
    fn ge(&self, other: &Self) -> bool {
        *self >= *other
    }
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        *self == *other
    }
    #[inline]
    fn ne(&self, other: &Self) -> bool {
        *self != *other
    }
    #[inline]
    fn select(mask: &bool, on_true: Self, on_false: Self) -> Self {
        if *mask {
            on_true
        } else {
            on_false
        }
    }
    #[inline]
    fn while_loop<C, P, B>(init: C, cond: P, body: B) -> C
    where
        P: Fn(&C) -> bool,
        B: Fn(C) -> C,
    {
        host_while_loop(init, cond, body)
    }
}
