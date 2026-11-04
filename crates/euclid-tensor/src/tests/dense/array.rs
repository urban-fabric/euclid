//! `Dense`'s rank-n [`Tensor`](crate::Tensor) implementation: shape, reshape, broadcast,
//! transpose, slice, concatenate, pad, gather/scatter, the reductions, and `matmul`.

use super::Dense;

// --- the rank-n array core ----------------------------------------------

impl crate::Tensor for Dense {
    /// Single-dtype carrier: indices are tensors holding integral `f64` values.
    type Index = Self;

    fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    fn reshape(self, shape: &[usize]) -> Self {
        if shape.iter().product::<usize>() == self.count() {
            Self {
                shape: shape.to_vec(),
                data: self.data,
            }
        } else {
            Self::empty()
        }
    }

    fn full(shape: &[usize], value: Self) -> Self {
        Self {
            shape: shape.to_vec(),
            data: vec![value.item(); shape.iter().product()],
        }
    }

    fn from_host(data: &[f64], shape: &[usize]) -> Self {
        if shape.iter().product::<usize>() == data.len() {
            Self::new(shape, data.to_vec())
        } else {
            Self::empty()
        }
    }

    fn broadcast_in_dim(&self, out_dims: &[usize], broadcast_dims: &[usize]) -> Self {
        let out_len: usize = out_dims.iter().product();
        let data = (0..out_len)
            .map(|flat| {
                let out_coords = Self::unravel(flat, out_dims);
                // Each input axis maps to one output axis; size-1 input axes broadcast.
                let in_coords: Vec<usize> = broadcast_dims
                    .iter()
                    .enumerate()
                    .map(|(in_axis, &out_axis)| {
                        if self.shape.get(in_axis) == Some(&1) {
                            0
                        } else {
                            out_coords.get(out_axis).copied().unwrap_or(0)
                        }
                    })
                    .collect();
                self.at(&in_coords)
            })
            .collect();
        Self {
            shape: out_dims.to_vec(),
            data,
        }
    }

    fn transpose_axes(&self, axes: &[usize]) -> Self {
        let out_shape: Vec<usize> = axes.iter().map(|&a| self.shape[a]).collect();
        let out_len: usize = out_shape.iter().product();
        let data = (0..out_len)
            .map(|flat| {
                let out_coords = Self::unravel(flat, &out_shape);
                let mut in_coords = vec![0usize; self.shape.len()];
                for (i, &a) in axes.iter().enumerate() {
                    in_coords[a] = out_coords[i];
                }
                self.at(&in_coords)
            })
            .collect();
        Self {
            shape: out_shape,
            data,
        }
    }

    fn squeeze_axis(&self, axis: isize) -> Self {
        let rank = self.shape.len() as isize;
        let axis = if axis < 0 { rank + axis } else { axis };
        let Ok(axis) = usize::try_from(axis) else {
            return self.clone();
        };
        if self.shape.get(axis) != Some(&1) {
            return self.clone();
        }
        let mut shape = self.shape.clone();
        shape.remove(axis);
        Self {
            shape,
            data: self.data.clone(),
        }
    }

    fn slice_in_dim(&self, start: usize, limit: usize, stride: usize, axis: usize) -> Self {
        if stride == 0 || axis >= self.shape.len() {
            return Self::empty();
        }
        let limit = limit.min(self.shape[axis]);
        let taken = if limit > start {
            (limit - start).div_ceil(stride)
        } else {
            0
        };
        let mut out_shape = self.shape.clone();
        out_shape[axis] = taken;
        let out_len: usize = out_shape.iter().product();
        let data = (0..out_len)
            .map(|flat| {
                let mut coords = Self::unravel(flat, &out_shape);
                coords[axis] = start + coords[axis] * stride;
                self.at(&coords)
            })
            .collect();
        Self {
            shape: out_shape,
            data,
        }
    }

    fn concatenate(inputs: &[Self], axis: usize) -> Self {
        let Some(first) = inputs.first() else {
            return Self::empty();
        };
        if axis >= first.shape.len() {
            return Self::empty();
        }
        let mut out_shape = first.shape.clone();
        out_shape[axis] = inputs.iter().map(|t| t.shape[axis]).sum();
        let out_len: usize = out_shape.iter().product();
        let data = (0..out_len)
            .map(|flat| {
                let mut coords = Self::unravel(flat, &out_shape);
                let mut remaining = coords[axis];
                for part in inputs {
                    if remaining < part.shape[axis] {
                        coords[axis] = remaining;
                        return part.at(&coords);
                    }
                    remaining -= part.shape[axis];
                }
                f64::NAN
            })
            .collect();
        Self {
            shape: out_shape,
            data,
        }
    }

    fn stack(inputs: &[Self], axis: usize) -> Self {
        // Insert a length-1 axis in each input, then concatenate along it.
        let lifted: Vec<Self> = inputs
            .iter()
            .map(|t| {
                let mut shape = t.shape.clone();
                shape.insert(axis.min(shape.len()), 1);
                Self {
                    shape,
                    data: t.data.clone(),
                }
            })
            .collect();
        Self::concatenate(&lifted, axis)
    }

    fn pad(
        &self,
        value: Self,
        low_padding: &[usize],
        high_padding: &[usize],
        interior_padding: &[usize],
    ) -> Self {
        let fill = value.item();
        let out_shape: Vec<usize> = self
            .shape
            .iter()
            .enumerate()
            .map(|(a, &d)| {
                let interior = interior_padding.get(a).copied().unwrap_or(0);
                let spread = d + d.saturating_sub(1) * interior;
                low_padding.get(a).copied().unwrap_or(0)
                    + spread
                    + high_padding.get(a).copied().unwrap_or(0)
            })
            .collect();
        let out_len: usize = out_shape.iter().product();
        let data = (0..out_len)
            .map(|flat| {
                let coords = Self::unravel(flat, &out_shape);
                let mut in_coords = vec![0usize; self.shape.len()];
                for (a, &c) in coords.iter().enumerate() {
                    let low = low_padding.get(a).copied().unwrap_or(0);
                    let interior = interior_padding.get(a).copied().unwrap_or(0);
                    if c < low {
                        return fill;
                    }
                    let offset = c - low;
                    if offset % (interior + 1) != 0 {
                        return fill; // lands in an interior gap
                    }
                    let idx = offset / (interior + 1);
                    if idx >= self.shape[a] {
                        return fill;
                    }
                    in_coords[a] = idx;
                }
                self.at(&in_coords)
            })
            .collect();
        Self {
            shape: out_shape,
            data,
        }
    }

    fn reverse(&self, axes: &[usize]) -> Self {
        let data = (0..self.count())
            .map(|flat| {
                let mut coords = Self::unravel(flat, &self.shape);
                for &a in axes {
                    if a < self.shape.len() {
                        coords[a] = self.shape[a] - 1 - coords[a];
                    }
                }
                self.at(&coords)
            })
            .collect();
        Self {
            shape: self.shape.clone(),
            data,
        }
    }

    /// Restricted to the `take`-shaped case this reference exercises: gathering whole slices
    /// along axis 0 with a rank-1 index vector. The full XLA dimension-numbering surface is
    /// beyond a reference carrier; see `take` for the ergonomic form.
    fn gather(
        &self,
        indices: &Self::Index,
        _offset_dims: &[usize],
        _collapsed_slice_dims: &[usize],
        _start_index_map: &[usize],
        _index_vector_dim: Option<usize>,
        _slice_sizes: &[usize],
    ) -> Self {
        self.take(indices, 0)
    }

    fn take(&self, indices: &Self::Index, axis: usize) -> Self {
        if axis >= self.shape.len() {
            return Self::empty();
        }
        let picks: Vec<usize> = indices.data.iter().map(|&i| i as usize).collect();
        let mut out_shape = self.shape.clone();
        out_shape[axis] = picks.len();
        let out_len: usize = out_shape.iter().product();
        let data = (0..out_len)
            .map(|flat| {
                let mut coords = Self::unravel(flat, &out_shape);
                let pick = picks.get(coords[axis]).copied().unwrap_or(0);
                if pick >= self.shape[axis] {
                    return f64::NAN;
                }
                coords[axis] = pick;
                self.at(&coords)
            })
            .collect();
        Self {
            shape: out_shape,
            data,
        }
    }

    fn dynamic_slice(&self, start_indices: &[Self::Index], slice_sizes: &[usize]) -> Self {
        let starts: Vec<usize> = start_indices.iter().map(|s| s.item() as usize).collect();
        let out_len: usize = slice_sizes.iter().product();
        let data = (0..out_len)
            .map(|flat| {
                let coords = Self::unravel(flat, slice_sizes);
                let shifted: Vec<usize> = coords
                    .iter()
                    .enumerate()
                    .map(|(a, &c)| {
                        // Clamp the window into bounds, as XLA's dynamic_slice does.
                        let start = starts.get(a).copied().unwrap_or(0);
                        let max_start = self.shape[a].saturating_sub(slice_sizes[a]);
                        start.min(max_start) + c
                    })
                    .collect();
                self.at(&shifted)
            })
            .collect();
        Self {
            shape: slice_sizes.to_vec(),
            data,
        }
    }

    fn dynamic_update_slice(&self, update: &Self, start_indices: &[Self::Index]) -> Self {
        let starts: Vec<usize> = start_indices.iter().map(|s| s.item() as usize).collect();
        let mut out = self.data.clone();
        for flat in 0..update.count() {
            let coords = Self::unravel(flat, &update.shape);
            let target: Vec<usize> = coords
                .iter()
                .enumerate()
                .map(|(a, &c)| {
                    let start = starts.get(a).copied().unwrap_or(0);
                    let max_start = self.shape[a].saturating_sub(update.shape[a]);
                    start.min(max_start) + c
                })
                .collect();
            out[Self::ravel(&target, &self.shape)] = update.data[flat];
        }
        Self {
            shape: self.shape.clone(),
            data: out,
        }
    }

    fn scatter_update_1d(&self, indices: &Self::Index, updates: &Self) -> Self {
        let mut out = self.data.clone();
        for (slot, &idx) in indices.data.iter().enumerate() {
            let idx = idx as usize;
            if idx < out.len() {
                out[idx] = updates.lane(slot);
            }
        }
        Self {
            shape: self.shape.clone(),
            data: out,
        }
    }

    fn scatter_add_1d(&self, indices: &Self::Index, updates: &Self) -> Self {
        let mut out = self.data.clone();
        for (slot, &idx) in indices.data.iter().enumerate() {
            let idx = idx as usize;
            if idx < out.len() {
                out[idx] += updates.lane(slot);
            }
        }
        Self {
            shape: self.shape.clone(),
            data: out,
        }
    }

    /// Restricted to the 1-D case, as with `gather`.
    fn scatter_update(
        &self,
        scatter_indices: &Self::Index,
        updates: &Self,
        _update_window_dims: &[usize],
        _inserted_window_dims: &[usize],
        _scatter_dims_to_operand_dims: &[usize],
        _index_vector_dim: Option<usize>,
        _indices_are_sorted: bool,
        _unique_indices: bool,
    ) -> Self {
        self.scatter_update_1d(scatter_indices, updates)
    }

    /// Restricted to the 1-D case, as with `gather`.
    fn scatter_add(
        &self,
        scatter_indices: &Self::Index,
        updates: &Self,
        _update_window_dims: &[usize],
        _inserted_window_dims: &[usize],
        _scatter_dims_to_operand_dims: &[usize],
        _index_vector_dim: Option<usize>,
        _indices_are_sorted: bool,
        _unique_indices: bool,
    ) -> Self {
        self.scatter_add_1d(scatter_indices, updates)
    }

    fn sum(self, axes: &[usize]) -> Self {
        self.reduce(axes, 0.0, |a, b| a + b)
    }

    fn prod(self, axes: &[usize]) -> Self {
        self.reduce(axes, 1.0, |a, b| a * b)
    }

    fn max(self, axes: &[usize]) -> Self {
        self.reduce(axes, f64::NEG_INFINITY, f64::max)
    }

    fn min(self, axes: &[usize]) -> Self {
        self.reduce(axes, f64::INFINITY, f64::min)
    }

    fn mean(self, axes: &[usize]) -> Self {
        let resolved = self.resolve_axes(axes);
        let divisor: usize = resolved.iter().map(|&a| self.shape[a]).product();
        let summed = self.reduce(axes, 0.0, |a, b| a + b);
        summed.map(|x| x / divisor as f64)
    }

    fn any(self, axes: &[usize]) -> Self {
        self.reduce(axes, 0.0, |a, b| f64::from(a != 0.0 || b != 0.0))
    }

    fn all(self, axes: &[usize]) -> Self {
        self.reduce(axes, 1.0, |a, b| f64::from(a != 0.0 && b != 0.0))
    }

    /// Rank-2 `(m,k)·(k,n)` matrix product; rank-1 operands contract as a dot product.
    fn matmul(self, rhs: Self) -> Self {
        match (self.shape.as_slice(), rhs.shape.as_slice()) {
            ([k_l], [k_r]) if k_l == k_r => Self::point(
                self.data
                    .iter()
                    .zip(&rhs.data)
                    .map(|(&a, &b)| a * b)
                    .sum::<f64>(),
            ),
            ([m, k_l], [k_r, n]) if k_l == k_r => {
                let (m, k, n) = (*m, *k_l, *n);
                let mut data = vec![0.0; m * n];
                for i in 0..m {
                    for j in 0..n {
                        data[i * n + j] = (0..k)
                            .map(|p| self.data[i * k + p] * rhs.data[p * n + j])
                            .sum();
                    }
                }
                Self::new(&[m, n], data)
            }
            _ => Self::empty(),
        }
    }
}
