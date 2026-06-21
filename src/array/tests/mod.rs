use super::*;
use crate::numeric::LazyCompare;
use core::ops::{Add, Mul, Neg, Sub};

mod boolean_tensor;
mod control_flow;

#[derive(Clone, Debug, PartialEq)]
struct ToyTensor(f64);

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

impl LazyCompare for ToyTensor {
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
        if *mask { on_true } else { on_false }
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

impl LazyVector for ToyTensor {
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

impl LazyVectorControlFlow for ToyTensor {
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

impl LazyMatrix for ToyTensor {
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

impl LazyMatrixControlFlow for ToyTensor {
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

impl LazyTensor for ToyTensor {
    type Index = ToyTensor;

    fn full(_shape: &[usize], value: Self) -> Self {
        value
    }

    fn shape(&self) -> Vec<usize> {
        vec![]
    }

    fn reshape(self, _shape: &[usize]) -> Self {
        self
    }

    fn broadcast_in_dim(&self, _out_dims: &[usize], _broadcast_dims: &[usize]) -> Self {
        self.clone()
    }

    fn transpose_axes(&self, _axes: &[usize]) -> Self {
        self.clone()
    }

    fn squeeze_axis(&self, _axis: isize) -> Self {
        self.clone()
    }

    fn slice_in_dim(&self, _start: usize, _limit: usize, _stride: usize, _axis: usize) -> Self {
        self.clone()
    }

    fn concatenate(inputs: &[Self], _axis: usize) -> Self {
        inputs.first().cloned().unwrap_or(Self(0.0))
    }

    fn stack(inputs: &[Self], _axis: usize) -> Self {
        inputs.first().cloned().unwrap_or(Self(0.0))
    }

    fn pad(
        &self,
        _value: Self,
        _low_padding: &[usize],
        _high_padding: &[usize],
        _interior_padding: &[usize],
    ) -> Self {
        self.clone()
    }

    fn reverse(&self, _axes: &[usize]) -> Self {
        self.clone()
    }

    fn gather(
        &self,
        _indices: &Self::Index,
        _offset_dims: &[usize],
        _collapsed_slice_dims: &[usize],
        _start_index_map: &[usize],
        _index_vector_dim: Option<usize>,
        _slice_sizes: &[usize],
    ) -> Self {
        self.clone()
    }

    fn take(&self, _indices: &Self::Index, _axis: usize) -> Self {
        self.clone()
    }

    fn dynamic_slice(&self, _start_indices: &[Self::Index], _slice_sizes: &[usize]) -> Self {
        self.clone()
    }

    fn dynamic_update_slice(&self, update: &Self, _start_indices: &[Self::Index]) -> Self {
        update.clone()
    }

    fn scatter_update_1d(&self, _indices: &Self::Index, updates: &Self) -> Self {
        updates.clone()
    }

    fn scatter_add_1d(&self, _indices: &Self::Index, updates: &Self) -> Self {
        self.clone() + updates.clone()
    }

    fn sum(self, _axes: &[usize]) -> Self {
        self
    }

    fn prod(self, _axes: &[usize]) -> Self {
        self
    }

    fn max(self, _axes: &[usize]) -> Self {
        self
    }

    fn min(self, _axes: &[usize]) -> Self {
        self
    }

    fn mean(self, _axes: &[usize]) -> Self {
        self
    }

    fn any(self, _axes: &[usize]) -> Self {
        self
    }

    fn all(self, _axes: &[usize]) -> Self {
        self
    }

    fn matmul(self, rhs: Self) -> Self {
        self * rhs
    }
}

impl LazyTensorControlFlow for ToyTensor {
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

#[test]
fn vector_and_matrix_like_constructors_use_shape_context() {
    let vector = ToyTensor(3.0);
    assert_eq!(<ToyTensor as LazyVector>::zeros(4), ToyTensor(0.0));
    assert_eq!(
        <ToyTensor as LazyVector>::full_like(&vector, 7.0),
        ToyTensor(7.0)
    );
    assert_eq!(
        <ToyTensor as LazyVector>::zeros_like(&vector),
        ToyTensor(0.0)
    );

    let matrix = ToyTensor(5.0);
    assert_eq!(<ToyTensor as LazyMatrix>::zeros(2, 3), ToyTensor(0.0));
    assert_eq!(
        <ToyTensor as LazyMatrix>::full_like(&matrix, 9.0),
        ToyTensor(9.0)
    );
    assert_eq!(
        <ToyTensor as LazyMatrix>::zeros_like(&matrix),
        ToyTensor(0.0)
    );
}

#[test]
fn tensor_constructors_structural_ops_and_reductions_are_available() {
    let tensor = ToyTensor(4.0);
    assert_eq!(
        <ToyTensor as LazyTensor>::full(&[2, 2], ToyTensor(8.0)),
        ToyTensor(8.0)
    );
    assert_eq!(
        <ToyTensor as LazyTensor>::full_like(&tensor, ToyTensor(8.0)),
        ToyTensor(8.0)
    );
    assert_eq!(tensor.slice_in_dim(0, 1, 1, 0), ToyTensor(4.0));
    assert_eq!(
        ToyTensor::concatenate(&[ToyTensor(1.0), ToyTensor(2.0)], 0),
        ToyTensor(1.0)
    );
    assert_eq!(
        ToyTensor::stack(&[ToyTensor(1.0), ToyTensor(2.0)], 0),
        ToyTensor(1.0)
    );
    assert_eq!(tensor.pad(ToyTensor(0.0), &[1], &[1], &[0]), ToyTensor(4.0));
    assert_eq!(tensor.reverse(&[0]), ToyTensor(4.0));
    assert_eq!(tensor.prod(&[]), ToyTensor(4.0));
}
