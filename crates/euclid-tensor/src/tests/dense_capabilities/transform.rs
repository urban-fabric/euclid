//! `Dense`'s [`TensorTransform`](crate::TensorTransform) implementation.

use super::super::dense::Dense;
use super::*;
use crate::{Tensor, TensorTransform};

impl TensorTransform for Dense {
    fn roll(&self, shift: isize, axis: usize) -> Self {
        let Some(&len) = self.dims().get(axis) else {
            return self.clone();
        };
        if len == 0 {
            return self.clone();
        }
        let len_i = len as isize;
        self.map_lanes(axis, |lane| {
            (0..len)
                .map(|i| {
                    // Positive shift moves elements toward higher indices.
                    let src = ((i as isize - shift) % len_i + len_i) % len_i;
                    lane[src as usize]
                })
                .collect()
        })
    }

    fn tile(&self, reps: usize, axis: usize) -> Self {
        let copies: Vec<Self> = (0..reps).map(|_| self.clone()).collect();
        Tensor::concatenate(&copies, axis)
    }

    fn clip(self, lo: Self, hi: Self) -> Self {
        Real::clamp(self, lo, hi)
    }
}
