use crate::numeric::{Boolean, Integer, Real};

use super::RandomKey;

/// Basic explicit-key random samplers.
///
/// `shape` is the host-known output shape. Scalar implementations such as an `f64` backend use
/// `&[]`; array, graph, and dataframe backends can use non-empty shapes for tensor/column draws.
pub trait RandomSampler: RandomKey {
    /// Real-valued draw result: `f64` for scalar host sampling, or a backend value such as a JAX
    /// array / Polars expression wrapper.
    type Real: Real;
    /// Boolean-valued draw result for Bernoulli sampling.
    type Bool: Boolean;
    /// Signed integer draw result for integer sampling.
    type Int: Integer<Real = Self::Real>;

    /// Draw from `Uniform(low, high)`.
    fn uniform(key: &Self::Key, shape: &[usize], low: Self::Real, high: Self::Real) -> Self::Real;

    /// Draw from `Normal(mean, stddev)`.
    fn normal(key: &Self::Key, shape: &[usize], mean: Self::Real, stddev: Self::Real)
    -> Self::Real;

    /// Draw Bernoulli values with success probability `p`.
    fn bernoulli(key: &Self::Key, shape: &[usize], p: Self::Real) -> Self::Bool;

    /// Draw uniformly from the half-open integer interval `[low, high)`.
    fn randint(key: &Self::Key, shape: &[usize], low: Self::Int, high: Self::Int) -> Self::Int;
}
