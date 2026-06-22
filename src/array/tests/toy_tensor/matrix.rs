use crate::array::{Matrix, MatrixControlFlow};
use crate::numeric::Compare;

use super::ToyTensor;

impl Matrix for ToyTensor {
    type Scalar = f64;
    type Vector = ToyTensor;

    fn shape(&self) -> (usize, usize) {
        (1, 1)
    }

    fn identity(_n: usize) -> Self {
        Self(1.0)
    }

    fn full(_rows: usize, _cols: usize, value: Self::Scalar) -> Self {
        Self(value)
    }

    fn add(self, other: Self) -> Self {
        self + other
    }

    fn scale(self, factor: Self::Scalar) -> Self {
        Self(self.0 * factor)
    }

    fn transpose(self) -> Self {
        self
    }

    fn adjoint(self) -> Self {
        self
    }

    fn matmul(self, rhs: Self) -> Self {
        self * rhs
    }

    fn matvec(self, v: Self::Vector) -> Self::Vector {
        self * v
    }

    fn trace(&self) -> Self::Scalar {
        self.0
    }

    fn diagonal(&self) -> Self::Vector {
        self.clone()
    }

    fn cholesky(self) -> Self {
        Self(self.0.sqrt())
    }

    fn solve_spd(self, rhs: Self) -> Self {
        Self(rhs.0 / self.0)
    }

    fn solve_spd_vec(self, rhs: Self::Vector) -> Self::Vector {
        Self(rhs.0 / self.0)
    }

    fn logdet_spd(self) -> Self::Scalar {
        self.0.ln()
    }

    fn mahalanobis_squared(&self, delta: &Self::Vector) -> Self::Scalar {
        delta.0 * delta.0 / self.0
    }
}

impl MatrixControlFlow for ToyTensor {
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
