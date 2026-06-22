use crate::numeric::{Compare, Element};
use crate::tensor::{Tensor, TensorControlFlow};
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

impl Tensor for ToyTensor {
    type Index = ToyTensor;

    fn full(_shape: &[usize], value: Self) -> Self {
        value
    }

    fn from_host(data: &[f64], _shape: &[usize]) -> Self {
        Self(data.first().copied().unwrap_or(0.0))
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

    fn scatter_update(
        &self,
        _scatter_indices: &Self::Index,
        updates: &Self,
        _update_window_dims: &[usize],
        _inserted_window_dims: &[usize],
        _scatter_dims_to_operand_dims: &[usize],
        _index_vector_dim: Option<usize>,
        _indices_are_sorted: bool,
        _unique_indices: bool,
    ) -> Self {
        updates.clone()
    }

    fn scatter_add(
        &self,
        _scatter_indices: &Self::Index,
        updates: &Self,
        _update_window_dims: &[usize],
        _inserted_window_dims: &[usize],
        _scatter_dims_to_operand_dims: &[usize],
        _index_vector_dim: Option<usize>,
        _indices_are_sorted: bool,
        _unique_indices: bool,
    ) -> Self {
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

impl TensorControlFlow for ToyTensor {
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
