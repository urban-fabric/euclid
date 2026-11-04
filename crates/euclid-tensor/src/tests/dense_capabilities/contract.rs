//! `Dense`'s [`TensorContract`](crate::TensorContract) implementation.

use super::super::dense::Dense;
use super::*;
use crate::TensorContract;

impl TensorContract for Dense {
    fn dot_general(
        &self,
        rhs: &Self,
        lhs_contracting: &[usize],
        rhs_contracting: &[usize],
        lhs_batch: &[usize],
        rhs_batch: &[usize],
    ) -> Self {
        let lhs_shape = self.dims();
        let rhs_shape = rhs.dims();

        let lhs_free: Vec<usize> = (0..lhs_shape.len())
            .filter(|a| !lhs_contracting.contains(a) && !lhs_batch.contains(a))
            .collect();
        let rhs_free: Vec<usize> = (0..rhs_shape.len())
            .filter(|a| !rhs_contracting.contains(a) && !rhs_batch.contains(a))
            .collect();

        // Output layout is batch dims, then lhs free dims, then rhs free dims.
        let batch_dims: Vec<usize> = lhs_batch.iter().map(|&a| lhs_shape[a]).collect();
        let lhs_free_dims: Vec<usize> = lhs_free.iter().map(|&a| lhs_shape[a]).collect();
        let rhs_free_dims: Vec<usize> = rhs_free.iter().map(|&a| rhs_shape[a]).collect();
        let contract_dims: Vec<usize> = lhs_contracting.iter().map(|&a| lhs_shape[a]).collect();

        let out_shape: Vec<usize> = batch_dims
            .iter()
            .chain(&lhs_free_dims)
            .chain(&rhs_free_dims)
            .copied()
            .collect();
        let out_len: usize = out_shape.iter().product::<usize>().max(1);
        let contract_len: usize = contract_dims.iter().product::<usize>().max(1);

        let mut data = vec![0.0; out_len];
        for (slot, cell) in data.iter_mut().enumerate() {
            let out_coords = Dense::unravel(slot, &out_shape);
            let (batch_c, rest) = out_coords.split_at(batch_dims.len());
            let (lhs_free_c, rhs_free_c) = rest.split_at(lhs_free_dims.len());

            let mut total = 0.0;
            for c in 0..contract_len {
                let contract_c = Dense::unravel(c, &contract_dims);

                let mut lhs_coords = vec![0usize; lhs_shape.len()];
                for (i, &a) in lhs_batch.iter().enumerate() {
                    lhs_coords[a] = batch_c[i];
                }
                for (i, &a) in lhs_free.iter().enumerate() {
                    lhs_coords[a] = lhs_free_c[i];
                }
                for (i, &a) in lhs_contracting.iter().enumerate() {
                    lhs_coords[a] = contract_c[i];
                }

                let mut rhs_coords = vec![0usize; rhs_shape.len()];
                for (i, &a) in rhs_batch.iter().enumerate() {
                    rhs_coords[a] = batch_c[i];
                }
                for (i, &a) in rhs_free.iter().enumerate() {
                    rhs_coords[a] = rhs_free_c[i];
                }
                for (i, &a) in rhs_contracting.iter().enumerate() {
                    rhs_coords[a] = contract_c[i];
                }

                let l = self.values()[Dense::ravel(&lhs_coords, lhs_shape)];
                let r = rhs.values()[Dense::ravel(&rhs_coords, rhs_shape)];
                total += l * r;
            }
            *cell = total;
        }

        Dense::new(&out_shape, data)
    }

    fn einsum1(&self, config: &str) -> Self {
        einsum(&[self], config)
    }

    fn einsum2(&self, rhs: &Self, config: &str) -> Self {
        einsum(&[self, rhs], config)
    }
}
