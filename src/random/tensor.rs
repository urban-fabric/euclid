use crate::numeric::{Boolean, Integer, Natural, Real};
use crate::tensor::Tensor;

use super::{GraphRandomBackend, KeyedRandomScalar, RandomDType};

/// Explicit-key tensor random draws.
///
/// Implement this for arbitrary-rank backends. It is the only array random surface — rank-1 and
/// rank-2 draws are just tensor draws with the corresponding shape.
pub trait KeyedRandomTensor: KeyedRandomScalar {
    /// An arbitrary-rank real draw result.
    type Tensor: Tensor + Real;
    /// An arbitrary-rank boolean draw result.
    type BoolTensor: Tensor + Boolean;
    /// An arbitrary-rank signed-integer draw result.
    type IntTensor: Tensor + Integer<Real = Self::Tensor>;
    /// An arbitrary-rank unsigned-integer draw result.
    type UIntTensor: Tensor + Natural<Real = Self::Tensor, Integer = Self::IntTensor>;

    /// Draw an arbitrary-rank tensor of independent `Uniform(0, 1)` values.
    fn uniform01_tensor(key: &Self::Key, shape: &[usize]) -> Self::Tensor;

    /// Draw raw uniformly-random `u32` words.
    fn bits_u32_tensor(key: &Self::Key, shape: &[usize]) -> Self::UIntTensor;

    /// Draw raw uniformly-random `u64` words.
    fn bits_u64_tensor(key: &Self::Key, shape: &[usize]) -> Self::UIntTensor;

    /// Draw raw uniformly-random words with an explicit bit width.
    ///
    /// The provided default dispatches the two common widths to
    /// [`bits_u32_tensor`](Self::bits_u32_tensor)/[`bits_u64_tensor`](Self::bits_u64_tensor);
    /// a backend that needs other widths overrides this method.
    #[allow(clippy::panic)] // default-only guard: backends with other widths override this.
    fn random_bits_tensor(key: &Self::Key, bit_width: u32, shape: &[usize]) -> Self::UIntTensor {
        match bit_width {
            32 => Self::bits_u32_tensor(key, shape),
            64 => Self::bits_u64_tensor(key, shape),
            _ => panic!("random_bits_tensor only has default support for 32- or 64-bit words"),
        }
    }

    /// Draw an arbitrary-rank tensor of independent standard normals.
    fn standard_normal_tensor(key: &Self::Key, shape: &[usize]) -> Self::Tensor;

    /// Draw an arbitrary-rank tensor of independent standard normals with an explicit dtype tag.
    fn standard_normal_tensor_dtype(
        key: &Self::Key,
        shape: &[usize],
        _dtype: RandomDType,
    ) -> Self::Tensor {
        Self::standard_normal_tensor(key, shape)
    }

    /// Draw an arbitrary-rank tensor of independent `Normal(mean, stddev)` values.
    fn normal_tensor_dtype(
        key: &Self::Key,
        shape: &[usize],
        mean: f64,
        stddev: f64,
        dtype: RandomDType,
    ) -> Self::Tensor;

    /// Draw an arbitrary-rank tensor of independent `Uniform(min, max)` values.
    fn uniform_tensor_dtype(
        key: &Self::Key,
        shape: &[usize],
        min: f64,
        max: f64,
        dtype: RandomDType,
    ) -> Self::Tensor;

    /// Draw an arbitrary-rank tensor of independent `Gamma(shape, 1)` variates.
    fn standard_gamma_tensor(
        key: &Self::Key,
        shape_parameter: Self::Scalar,
        shape: &[usize],
    ) -> Self::Tensor;

    /// Draw Bernoulli masks with success probabilities `p`, broadcast to `shape`.
    fn bernoulli_tensor(key: &Self::Key, p: Self::Tensor, shape: &[usize]) -> Self::BoolTensor;

    /// Draw uniformly from the half-open integer interval `[low, high)`, broadcast to `shape`.
    fn randint_tensor(
        key: &Self::Key,
        low: Self::IntTensor,
        high: Self::IntTensor,
        shape: &[usize],
    ) -> Self::IntTensor;

    /// Draw integer values from `[low, high)` with an explicit dtype tag.
    fn randint_tensor_dtype(
        key: &Self::Key,
        low: i64,
        high: i64,
        shape: &[usize],
        dtype: RandomDType,
    ) -> Self::IntTensor;

    /// Draw categorical class indices from unnormalized `logits` along `axis`.
    ///
    /// `sample_shape` names leading independent sample axes, matching the JAX convention. The
    /// result shape is `sample_shape` followed by `logits.shape()` with `axis` removed, modulo any
    /// backend-specific broadcasting constraints.
    fn categorical_tensor(
        key: &Self::Key,
        logits: Self::Tensor,
        axis: usize,
        sample_shape: &[usize],
    ) -> Self::IntTensor;
}

/// Explicit-key tensor random draws from graph-native keys.
pub trait GraphRandomTensor: GraphRandomBackend + KeyedRandomTensor {
    /// Draw an arbitrary-rank tensor of independent `Uniform(0, 1)` values from a graph key.
    fn uniform01_graph_tensor(key: &Self::GraphKey, shape: &[usize]) -> Self::Tensor;

    /// Draw raw uniformly-random words from a graph key.
    fn random_bits_graph_tensor(
        key: &Self::GraphKey,
        bit_width: u32,
        shape: &[usize],
    ) -> Self::UIntTensor;

    /// Draw standard normals from a graph key.
    fn standard_normal_graph_tensor(key: &Self::GraphKey, shape: &[usize]) -> Self::Tensor;

    /// Draw standard normals from a graph key with an explicit dtype tag.
    fn standard_normal_graph_tensor_dtype(
        key: &Self::GraphKey,
        shape: &[usize],
        _dtype: RandomDType,
    ) -> Self::Tensor {
        Self::standard_normal_graph_tensor(key, shape)
    }

    /// Draw `Uniform(min, max)` values from a graph key with an explicit dtype tag.
    fn uniform_graph_tensor_dtype(
        key: &Self::GraphKey,
        shape: &[usize],
        min: f64,
        max: f64,
        dtype: RandomDType,
    ) -> Self::Tensor;

    /// Draw `Normal(mean, stddev)` values from a graph key with an explicit dtype tag.
    fn normal_graph_tensor_dtype(
        key: &Self::GraphKey,
        shape: &[usize],
        mean: f64,
        stddev: f64,
        dtype: RandomDType,
    ) -> Self::Tensor;
}

/// Optional tensor distribution samplers supplied by a backend.
pub trait KeyedRandomTensorDistributions: KeyedRandomTensor {
    /// Draw an arbitrary-rank tensor of independent `Exponential(1)` values.
    fn standard_exponential_tensor(key: &Self::Key, shape: &[usize]) -> Self::Tensor;

    /// Draw an arbitrary-rank tensor of independent standard Gumbel values.
    fn standard_gumbel_tensor(key: &Self::Key, shape: &[usize]) -> Self::Tensor;

    /// Draw an arbitrary-rank tensor of independent standard logistic values.
    fn standard_logistic_tensor(key: &Self::Key, shape: &[usize]) -> Self::Tensor;

    /// Draw an arbitrary-rank tensor of independent standard Cauchy values.
    fn standard_cauchy_tensor(key: &Self::Key, shape: &[usize]) -> Self::Tensor;

    /// Draw `Beta(alpha, beta)` values, with parameters broadcast to `shape`.
    fn beta_tensor(
        key: &Self::Key,
        alpha: Self::Tensor,
        beta: Self::Tensor,
        shape: &[usize],
    ) -> Self::Tensor;

    /// Draw `Dirichlet(alpha)` values over `axis`.
    ///
    /// `sample_shape` names leading independent sample axes. The event axis is preserved in the
    /// result.
    fn dirichlet_tensor(
        key: &Self::Key,
        alpha: Self::Tensor,
        axis: usize,
        sample_shape: &[usize],
    ) -> Self::Tensor;

    /// Draw `Poisson(rate)` values, with rates broadcast to `shape`.
    fn poisson_tensor(key: &Self::Key, rate: Self::Tensor, shape: &[usize]) -> Self::IntTensor;

    /// Draw a random permutation of `0..n`.
    fn permutation(key: &Self::Key, n: usize) -> Self::IntTensor;

    /// Shuffle tensor values along `axis`.
    fn shuffle_tensor(key: &Self::Key, values: Self::Tensor, axis: usize) -> Self::Tensor;
}
