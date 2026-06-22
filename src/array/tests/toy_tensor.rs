mod matrix;
mod tensor;
mod vector;

use crate::numeric::{Compare, Element};
use core::ops::{Add, Mul, Neg, Sub};

#[derive(Clone, Debug, PartialEq)]
pub(super) struct ToyTensor(pub(super) f64);

impl Add for ToyTensor {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl Mul for ToyTensor {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self(self.0 * rhs.0)
    }
}

impl Sub for ToyTensor {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}

impl Neg for ToyTensor {
    type Output = Self;

    fn neg(self) -> Self {
        Self(-self.0)
    }
}

impl Element for ToyTensor {
    fn zero() -> Self {
        Self(0.0)
    }

    fn one() -> Self {
        Self(1.0)
    }
}

impl Compare for ToyTensor {
    type Mask = bool;

    fn lt(&self, other: &Self) -> bool {
        self.0 < other.0
    }

    fn le(&self, other: &Self) -> bool {
        self.0 <= other.0
    }

    fn gt(&self, other: &Self) -> bool {
        self.0 > other.0
    }

    fn ge(&self, other: &Self) -> bool {
        self.0 >= other.0
    }

    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }

    fn ne(&self, other: &Self) -> bool {
        self.0 != other.0
    }

    fn select(mask: &bool, on_true: Self, on_false: Self) -> Self {
        if *mask {
            on_true
        } else {
            on_false
        }
    }

    fn while_loop<C, P, B>(init: C, cond: P, body: B) -> C
    where
        P: Fn(&C) -> Self::Mask,
        B: Fn(C) -> C,
    {
        let mut state = init;
        while cond(&state) {
            state = body(state);
        }
        state
    }
}
