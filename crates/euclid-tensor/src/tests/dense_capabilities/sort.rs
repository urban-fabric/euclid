//! `Dense`'s [`TensorSort`](crate::TensorSort) implementation.

use super::super::dense::Dense;
use super::*;
use crate::TensorSort;

impl TensorSort for Dense {
    fn sort(self, axis: usize, descending: bool) -> Self {
        self.map_lanes(axis, |lane| {
            ordered_indices(lane, descending)
                .into_iter()
                .map(|i| lane[i])
                .collect()
        })
    }

    fn argsort(&self, axis: usize, descending: bool) -> Self::Index {
        self.map_lanes(axis, |lane| {
            ordered_indices(lane, descending)
                .into_iter()
                .map(|i| i as f64)
                .collect()
        })
    }

    fn argmax(&self, axis: usize) -> Self::Index {
        self.reduce_lanes(axis, |lane| {
            ordered_indices(lane, true).first().copied().unwrap_or(0) as f64
        })
    }

    fn argmin(&self, axis: usize) -> Self::Index {
        self.reduce_lanes(axis, |lane| {
            ordered_indices(lane, false).first().copied().unwrap_or(0) as f64
        })
    }

    fn top_k(&self, k: usize, largest: bool) -> (Self, Self::Index) {
        let rank = self.dims().len();
        if rank == 0 {
            return (self.clone(), self.clone());
        }
        let axis = rank - 1;
        let len = self.dims()[axis];
        let k = k.min(len);

        let mut out_shape = self.dims().to_vec();
        out_shape[axis] = k;

        let mut values = Vec::new();
        let mut indices = Vec::new();
        // The last axis is contiguous, so lanes are simple chunks.
        for chunk in self.values().chunks(len) {
            let order = ordered_indices(chunk, largest);
            for &i in order.iter().take(k) {
                values.push(chunk[i]);
                indices.push(i as f64);
            }
        }

        (
            Dense::new(&out_shape, values),
            Dense::new(&out_shape, indices),
        )
    }
}
