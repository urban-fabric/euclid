//! `Dense`'s [`TensorIndexing`](crate::TensorIndexing) implementation.

use super::super::dense::Dense;
use crate::TensorIndexing;

impl TensorIndexing for Dense {
    fn iota(len: usize) -> Self::Index {
        Dense::vector(&(0..len).map(|i| i as f64).collect::<Vec<_>>())
    }

    fn one_hot(indices: &Self::Index, depth: usize) -> Self {
        let mut out_shape = indices.dims().to_vec();
        out_shape.push(depth);
        let mut data = vec![0.0; indices.values().len() * depth];
        for (row, &idx) in indices.values().iter().enumerate() {
            let col = idx as usize;
            if idx >= 0.0 && col < depth {
                data[row * depth + col] = 1.0;
            }
        }
        Dense::new(&out_shape, data)
    }

    fn bincount(indices: &Self::Index, length: usize) -> Self {
        let mut counts = vec![0.0; length];
        for &idx in indices.values() {
            let slot = idx as usize;
            if idx >= 0.0 && slot < length {
                counts[slot] += 1.0;
            }
        }
        Dense::new(&[length], counts)
    }
}
