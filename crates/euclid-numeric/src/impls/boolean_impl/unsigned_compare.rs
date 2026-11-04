use crate::Compare;

use super::host_while_loop;

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
        if *mask { on_true } else { on_false }
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
        if *mask { on_true } else { on_false }
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
        if *mask { on_true } else { on_false }
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
