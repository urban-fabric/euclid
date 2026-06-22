use super::ToyTensor;
use crate::array::Tensor;
use crate::numeric::{Boolean, Element};

#[derive(Clone, Debug, PartialEq)]
struct ToyBooleanTensor {
    value: bool,
    dims: Vec<usize>,
}

impl Element for ToyBooleanTensor {
    fn zero() -> Self {
        Self {
            value: false,
            dims: vec![],
        }
    }

    fn one() -> Self {
        Self {
            value: true,
            dims: vec![],
        }
    }
}

impl Boolean for ToyBooleanTensor {
    fn and(&self, other: &Self) -> Self {
        Self {
            value: self.value && other.value,
            dims: self.dims.clone(),
        }
    }

    fn or(&self, other: &Self) -> Self {
        Self {
            value: self.value || other.value,
            dims: self.dims.clone(),
        }
    }

    fn xor(&self, other: &Self) -> Self {
        Self {
            value: self.value ^ other.value,
            dims: self.dims.clone(),
        }
    }

    fn not(&self) -> Self {
        Self {
            value: !self.value,
            dims: self.dims.clone(),
        }
    }
}

impl Tensor for ToyBooleanTensor {
    type Index = ToyTensor;

    fn full(shape: &[usize], value: Self) -> Self {
        Self {
            value: value.value,
            dims: shape.to_vec(),
        }
    }

    fn from_host(_data: &[f64], shape: &[usize]) -> Self {
        Self {
            value: false,
            dims: shape.to_vec(),
        }
    }

    fn shape(&self) -> Vec<usize> {
        self.dims.clone()
    }

    fn reshape(mut self, shape: &[usize]) -> Self {
        self.dims = shape.to_vec();
        self
    }

    fn broadcast_in_dim(&self, out_dims: &[usize], _broadcast_dims: &[usize]) -> Self {
        Self {
            value: self.value,
            dims: out_dims.to_vec(),
        }
    }

    fn transpose_axes(&self, axes: &[usize]) -> Self {
        Self {
            value: self.value,
            dims: axes.iter().map(|&axis| self.dims[axis]).collect(),
        }
    }

    fn squeeze_axis(&self, axis: isize) -> Self {
        let rank = self.dims.len() as isize;
        let axis = if axis < 0 { rank + axis } else { axis } as usize;
        let mut dims = self.dims.clone();
        dims.remove(axis);
        Self {
            value: self.value,
            dims,
        }
    }

    fn slice_in_dim(&self, _start: usize, limit: usize, _stride: usize, axis: usize) -> Self {
        let mut dims = self.dims.clone();
        dims[axis] = limit;
        Self {
            value: self.value,
            dims,
        }
    }

    fn concatenate(inputs: &[Self], axis: usize) -> Self {
        let Some(first) = inputs.first() else {
            return Self {
                value: false,
                dims: Vec::new(),
            };
        };
        let mut dims = first.dims.clone();
        dims[axis] = inputs.iter().map(|input| input.dims[axis]).sum();
        Self {
            value: inputs.iter().any(|input| input.value),
            dims,
        }
    }

    fn stack(inputs: &[Self], axis: usize) -> Self {
        let Some(first) = inputs.first() else {
            return Self {
                value: false,
                dims: Vec::new(),
            };
        };
        let mut dims = first.dims.clone();
        dims.insert(axis, inputs.len());
        Self {
            value: inputs.iter().any(|input| input.value),
            dims,
        }
    }

    fn pad(
        &self,
        value: Self,
        low_padding: &[usize],
        high_padding: &[usize],
        _interior_padding: &[usize],
    ) -> Self {
        let dims = self
            .dims
            .iter()
            .enumerate()
            .map(|(axis, dim)| dim + low_padding[axis] + high_padding[axis])
            .collect();
        Self {
            value: self.value || value.value,
            dims,
        }
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
        slice_sizes: &[usize],
    ) -> Self {
        Self {
            value: self.value,
            dims: slice_sizes.to_vec(),
        }
    }

    fn take(&self, _indices: &Self::Index, axis: usize) -> Self {
        let mut dims = self.dims.clone();
        dims[axis] = 1;
        Self {
            value: self.value,
            dims,
        }
    }

    fn dynamic_slice(&self, _start_indices: &[Self::Index], slice_sizes: &[usize]) -> Self {
        Self {
            value: self.value,
            dims: slice_sizes.to_vec(),
        }
    }

    fn dynamic_update_slice(&self, update: &Self, _start_indices: &[Self::Index]) -> Self {
        Self {
            value: update.value,
            dims: self.dims.clone(),
        }
    }

    fn scatter_update_1d(&self, _indices: &Self::Index, updates: &Self) -> Self {
        Self {
            value: updates.value,
            dims: self.dims.clone(),
        }
    }

    fn scatter_add_1d(&self, _indices: &Self::Index, updates: &Self) -> Self {
        Self {
            value: self.value || updates.value,
            dims: self.dims.clone(),
        }
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
        Self {
            value: updates.value,
            dims: self.dims.clone(),
        }
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
        Self {
            value: self.value || updates.value,
            dims: self.dims.clone(),
        }
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

    fn matmul(self, _rhs: Self) -> Self {
        self
    }
}

#[test]
fn lazy_tensor_can_be_boolean_without_being_a_ring() {
    fn accepts_bool_tensor<T: Tensor + Boolean>(tensor: T) -> T {
        tensor.reshape(&[3])
    }

    let tensor = ToyBooleanTensor {
        value: true,
        dims: vec![1, 3],
    };
    let reshaped = accepts_bool_tensor(tensor);

    assert_eq!(reshaped.shape(), vec![3]);
    assert_eq!(reshaped.rank(), 1);
    assert_eq!(
        reshaped.and(&ToyBooleanTensor {
            value: false,
            dims: vec![3]
        }),
        ToyBooleanTensor {
            value: false,
            dims: vec![3]
        }
    );

    assert_eq!(
        reshaped.dynamic_slice(&[ToyTensor(1.0)], &[1]).shape(),
        vec![1]
    );
}
