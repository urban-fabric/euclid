//! `Dense`'s [`TensorReduce`](crate::TensorReduce) implementation.

use super::super::dense::Dense;
use super::*;
use crate::TensorReduce;

impl TensorReduce for Dense {
    fn cumsum(self, axis: usize) -> Self {
        self.map_lanes(axis, |lane| {
            lane.iter()
                .scan(0.0, |acc, &x| {
                    *acc += x;
                    Some(*acc)
                })
                .collect()
        })
    }

    fn var(self, axes: &[usize]) -> Self {
        self.reduce_groups(axes, |g| {
            let m = mean_of(g);
            mean_of(&g.iter().map(|&x| (x - m) * (x - m)).collect::<Vec<_>>())
        })
    }

    fn std(self, axes: &[usize]) -> Self {
        TensorReduce::var(self, axes).map_values(f64::sqrt)
    }

    fn logsumexp(self, axes: &[usize]) -> Self {
        self.reduce_groups(axes, logsumexp_of)
    }

    fn softmax(self, axis: usize) -> Self {
        self.map_lanes(axis, |lane| {
            let lse = logsumexp_of(lane);
            lane.iter().map(|&x| (x - lse).exp()).collect()
        })
    }

    fn log_softmax(self, axis: usize) -> Self {
        self.map_lanes(axis, |lane| {
            let lse = logsumexp_of(lane);
            lane.iter().map(|&x| x - lse).collect()
        })
    }
}
