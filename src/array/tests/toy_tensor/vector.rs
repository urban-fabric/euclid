use crate::array::{Vector, VectorControlFlow};
use crate::numeric::Compare;

use super::ToyTensor;

impl Vector for ToyTensor {
    type Scalar = f64;

    fn full(_len: usize, value: Self::Scalar) -> Self {
        Self(value)
    }

    fn len(&self) -> usize {
        1
    }

    fn add(self, other: Self) -> Self {
        self + other
    }

    fn sub(self, other: Self) -> Self {
        self - other
    }

    fn scale(self, factor: Self::Scalar) -> Self {
        Self(self.0 * factor)
    }

    fn hadamard(self, other: Self) -> Self {
        self * other
    }

    fn dot(&self, other: &Self) -> Self::Scalar {
        self.0 * other.0
    }

    fn sum(&self) -> Self::Scalar {
        self.0
    }

    fn norm_squared(&self) -> Self::Scalar {
        self.0 * self.0
    }

    fn norm(&self) -> Self::Scalar {
        self.0.abs()
    }
}

impl VectorControlFlow for ToyTensor {
    fn cond<T, F>(
        pred: &Self::Mask,
        true_arg: Self,
        true_branch: T,
        false_arg: Self,
        false_branch: F,
    ) -> Self
    where
        T: FnOnce(Self) -> Self,
        F: FnOnce(Self) -> Self,
    {
        let on_true = true_branch(true_arg);
        let on_false = false_branch(false_arg);
        Self::select(pred, on_true, on_false)
    }

    fn while_loop<C, B>(init: Self, cond: C, body: B) -> Self
    where
        C: Fn(&Self) -> Self::Mask,
        B: Fn(Self) -> Self,
    {
        let mut state = init;
        while cond(&state) {
            state = body(state);
        }
        state
    }
}
