//! JAX-style random-source contracts over the numeric and array traits.
//!
//! Randomness is modeled with explicit immutable keys. A draw is a pure function of its key,
//! distribution parameters, and result shape: using the same key twice produces the same value.
//! New independent randomness is obtained by [`KeyedRandomBackend::split`] or
//! [`KeyedRandomBackend::fold_in`], which keeps the contract reproducible and friendly to
//! graph/XLA backends.
//!
//! Batched key semantics should be map-like: drawing from a batch of keys is equivalent to drawing
//! once per key and stacking the results. Backends should not expose implementation details that
//! make the first key drive an entire batch.

use crate::array::{Matrix, Tensor, Vector};
use crate::numeric::{Boolean, Integer, Natural, Real};
use crate::DType;

/// Random-bit-generator algorithm identifiers when an implementation exposes one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RandomAlgorithm {
    /// Implementation-defined algorithm and state shape.
    Default,
    /// Counter-based ThreeFry family.
    ThreeFry,
    /// Counter-based Philox family.
    Philox,
}

/// Backend-neutral dtype tags for random samplers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RandomDType {
    Bool,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F16,
    F32,
    Bf16,
    F64,
}

impl RandomDType {
    pub const fn is_bool(self) -> bool {
        matches!(self, Self::Bool)
    }

    pub const fn is_signed_integer(self) -> bool {
        matches!(self, Self::I8 | Self::I16 | Self::I32 | Self::I64)
    }

    pub const fn is_unsigned_integer(self) -> bool {
        matches!(self, Self::U8 | Self::U16 | Self::U32 | Self::U64)
    }

    pub const fn is_integer(self) -> bool {
        self.is_signed_integer() || self.is_unsigned_integer()
    }

    pub const fn is_float(self) -> bool {
        matches!(self, Self::F16 | Self::F32 | Self::Bf16 | Self::F64)
    }

    pub const fn bit_width(self) -> u32 {
        match self {
            Self::Bool | Self::I8 | Self::U8 => 8,
            Self::I16 | Self::U16 | Self::F16 | Self::Bf16 => 16,
            Self::I32 | Self::U32 | Self::F32 => 32,
            Self::I64 | Self::U64 | Self::F64 => 64,
        }
    }
}

impl From<RandomDType> for DType {
    fn from(value: RandomDType) -> Self {
        match value {
            RandomDType::Bool => Self::Bool,
            RandomDType::I8 => Self::I8,
            RandomDType::I16 => Self::I16,
            RandomDType::I32 => Self::I32,
            RandomDType::I64 => Self::I64,
            RandomDType::U8 => Self::U8,
            RandomDType::U16 => Self::U16,
            RandomDType::U32 => Self::U32,
            RandomDType::U64 => Self::U64,
            RandomDType::F16 => Self::F16,
            RandomDType::F32 => Self::F32,
            RandomDType::Bf16 => Self::Bf16,
            RandomDType::F64 => Self::F64,
        }
    }
}

impl TryFrom<DType> for RandomDType {
    type Error = DType;

    fn try_from(value: DType) -> Result<Self, Self::Error> {
        match value {
            DType::Bool => Ok(Self::Bool),
            DType::I8 => Ok(Self::I8),
            DType::I16 => Ok(Self::I16),
            DType::I32 => Ok(Self::I32),
            DType::I64 => Ok(Self::I64),
            DType::U8 => Ok(Self::U8),
            DType::U16 => Ok(Self::U16),
            DType::U32 => Ok(Self::U32),
            DType::U64 => Ok(Self::U64),
            DType::F16 => Ok(Self::F16),
            DType::Bf16 => Ok(Self::Bf16),
            DType::F32 => Ok(Self::F32),
            DType::F64 => Ok(Self::F64),
            DType::C64 | DType::C128 => Err(value),
        }
    }
}

/// Explicit random-key management.
///
/// Keys are immutable values. Draw methods in the capability traits below do not advance or
/// consume the key; callers split keys before independent draws, or fold deterministic data such
/// as loop steps / chain ids into a key.
/// This mirrors JAX's PRNG discipline and is the shape compiled backends can represent directly.
pub trait KeyedRandomBackend {
    /// The backend's key value. For graph backends this is usually an integer tensor.
    type Key: Clone;
    /// Serializable/interoperable key payload.
    ///
    /// Typed-key implementations should include enough information here to reconstruct the same
    /// key and RNG algorithm. Legacy backends may use raw words such as `[u32; 2]`.
    type KeyData: Clone;
    /// The random-bit algorithm, when the implementation exposes one.
    fn algorithm() -> RandomAlgorithm {
        RandomAlgorithm::Default
    }

    /// Build a reproducible key from a host seed.
    fn key(seed: u64) -> Self::Key;

    /// Extract an interoperable key payload.
    fn key_data(key: &Self::Key) -> Self::KeyData;

    /// Rebuild a key from an interoperable payload.
    fn wrap_key_data(data: Self::KeyData) -> Self::Key;

    /// Deterministically split `key` into `count` independent child keys.
    fn split(key: &Self::Key, count: usize) -> Vec<Self::Key>;

    /// Deterministically split `key` into a row-major array of child keys.
    fn split_shape(key: &Self::Key, shape: &[usize]) -> Vec<Self::Key> {
        let count = shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .expect("random split shape element count overflow");
        Self::split(key, count)
    }

    /// Deterministically mix host data into a key.
    ///
    /// Use this for loop indices, replica ids, chain ids, epoch numbers, or any other structured
    /// source of independent randomness.
    fn fold_in(key: &Self::Key, data: u64) -> Self::Key;
}

/// Graph-native explicit random-key management.
///
/// Backends with a tracing/JIT layer can represent keys as graph values, accept
/// them as parameters, split/fold them inside a compiled computation, and return
/// key data as graph outputs. Eager backends can ignore this trait.
pub trait GraphRandomBackend: KeyedRandomBackend {
    /// A key value living in the backend's graph/tracing substrate.
    type GraphKey: Clone;
    /// Graph value carrying key data, usually unsigned integer words.
    type GraphKeyData: Clone;
    /// Scalar graph integer accepted by `graph_fold_in_value`.
    type GraphFoldData: Clone;

    /// Lift a host key into the active graph.
    fn graph_key(key: &Self::Key) -> Self::GraphKey;

    /// Create a graph parameter/input representing a key.
    fn graph_key_parameter(index: usize) -> Self::GraphKey;

    /// Return graph key data suitable for returning from compiled computations.
    fn graph_key_data(key: &Self::GraphKey) -> Self::GraphKeyData;

    /// Rebuild a graph key from graph key data.
    fn graph_wrap_key_data(data: Self::GraphKeyData) -> Self::GraphKey;

    /// Split a graph key into `count` scalar graph keys.
    fn graph_split(key: &Self::GraphKey, count: usize) -> Vec<Self::GraphKey>;

    /// Split a graph key into a row-major array of graph keys.
    fn graph_split_shape(key: &Self::GraphKey, shape: &[usize]) -> Vec<Self::GraphKey> {
        let count = shape
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .expect("graph random split shape element count overflow");
        Self::graph_split(key, count)
    }

    /// Mix host data into a graph key.
    fn graph_fold_in(key: &Self::GraphKey, data: u64) -> Self::GraphKey;

    /// Mix scalar graph data into a graph key.
    fn graph_fold_in_value(key: &Self::GraphKey, data: Self::GraphFoldData) -> Self::GraphKey;
}

/// Explicit-key scalar random draws.
///
/// Implement this for backends that can draw scalar real values without committing to any
/// vector, matrix, or tensor representation.
pub trait KeyedRandomScalar: KeyedRandomBackend {
    /// The backend's scalar real value carrier.
    type Scalar: Real;
    /// The backend's scalar boolean/mask carrier.
    type Bool: Boolean;
    /// The backend's scalar signed-integer carrier.
    type Int: Integer<Real = Self::Scalar>;
    /// The backend's scalar unsigned-integer carrier.
    type UInt: Natural<Real = Self::Scalar, Integer = Self::Int>;

    /// Draw a scalar from `Uniform(0, 1)`.
    fn uniform01_scalar(key: &Self::Key) -> Self::Scalar;

    /// Draw one raw uniformly-random `u32` word, lifted into the backend's unsigned carrier.
    fn bits_u32_scalar(key: &Self::Key) -> Self::UInt;

    /// Draw one raw uniformly-random `u64` word, lifted into the backend's unsigned carrier.
    fn bits_u64_scalar(key: &Self::Key) -> Self::UInt;

    /// Draw one raw uniformly-random word with an explicit bit width.
    fn random_bits_scalar(key: &Self::Key, bit_width: u32) -> Self::UInt {
        match bit_width {
            32 => Self::bits_u32_scalar(key),
            64 => Self::bits_u64_scalar(key),
            _ => panic!("random_bits_scalar only has default support for 32- or 64-bit words"),
        }
    }

    /// Draw a scalar standard normal.
    fn standard_normal_scalar(key: &Self::Key) -> Self::Scalar;

    /// Draw a scalar normal with explicit mean and standard deviation.
    fn normal_scalar(key: &Self::Key, mean: Self::Scalar, stddev: Self::Scalar) -> Self::Scalar {
        Self::standard_normal_scalar(key) * stddev + mean
    }

    /// Draw a scalar `Gamma(shape, 1)` variate.
    fn standard_gamma_scalar(key: &Self::Key, shape_parameter: Self::Scalar) -> Self::Scalar;

    /// Draw a scalar Bernoulli mask with success probability `p`.
    fn bernoulli_scalar(key: &Self::Key, p: Self::Scalar) -> Self::Bool;

    /// Draw uniformly from the half-open integer interval `[low, high)`.
    fn randint_scalar(key: &Self::Key, low: Self::Int, high: Self::Int) -> Self::Int;
}

/// Optional scalar distribution samplers supplied by a backend.
///
/// Implement this when the backend has native or carefully tuned algorithms for these
/// distributions. Generic code that cares about sampler quality can require this trait instead of
/// constructing samples from lower-level uniforms/normals/gammas.
pub trait KeyedRandomScalarDistributions: KeyedRandomScalar {
    /// Draw a scalar `Exponential(1)` variate.
    fn standard_exponential_scalar(key: &Self::Key) -> Self::Scalar;

    /// Draw a scalar standard Gumbel variate.
    fn standard_gumbel_scalar(key: &Self::Key) -> Self::Scalar;

    /// Draw a scalar standard logistic variate.
    fn standard_logistic_scalar(key: &Self::Key) -> Self::Scalar;

    /// Draw a scalar standard Cauchy variate.
    fn standard_cauchy_scalar(key: &Self::Key) -> Self::Scalar;

    /// Draw a scalar `Beta(alpha, beta)` variate.
    fn beta_scalar(key: &Self::Key, alpha: Self::Scalar, beta: Self::Scalar) -> Self::Scalar;

    /// Draw a scalar `Poisson(rate)` variate.
    fn poisson_scalar(key: &Self::Key, rate: Self::Scalar) -> Self::Int;
}

/// Explicit-key matrix/vector random draws.
///
/// This is the rank-1/rank-2 real-array surface. Integer and boolean array draws live on
/// [`KeyedRandomTensor`], because euclid's [`Vector`] and [`Matrix`] traits are field-oriented.
pub trait KeyedRandomMatVec: KeyedRandomScalar {
    /// A one-dimensional real draw result.
    type Vector: Vector<Scalar = Self::Scalar>;
    /// A two-dimensional real draw result.
    type Matrix: Matrix<Scalar = Self::Scalar, Vector = Self::Vector>;

    /// Draw `len` independent `Uniform(0, 1)` values.
    fn uniform01_vector(key: &Self::Key, len: usize) -> Self::Vector;
    /// Draw a `rows x cols` matrix of independent `Uniform(0, 1)` values.
    fn uniform01_matrix(key: &Self::Key, rows: usize, cols: usize) -> Self::Matrix;

    /// Draw `len` independent standard normals.
    fn standard_normal_vector(key: &Self::Key, len: usize) -> Self::Vector;
    /// Draw a `rows x cols` matrix of independent standard normals.
    fn standard_normal_matrix(key: &Self::Key, rows: usize, cols: usize) -> Self::Matrix;

    /// Draw `len` independent `Gamma(shape, 1)` variates.
    fn standard_gamma_vector(
        key: &Self::Key,
        shape_parameter: Self::Scalar,
        len: usize,
    ) -> Self::Vector;
    /// Draw a `rows x cols` matrix of independent `Gamma(shape, 1)` variates.
    fn standard_gamma_matrix(
        key: &Self::Key,
        shape_parameter: Self::Scalar,
        rows: usize,
        cols: usize,
    ) -> Self::Matrix;
}

/// Optional matrix/vector distribution samplers supplied by a backend.
pub trait KeyedRandomMatVecDistributions: KeyedRandomMatVec {
    /// Draw `len` independent `Exponential(1)` values.
    fn standard_exponential_vector(key: &Self::Key, len: usize) -> Self::Vector;
    /// Draw a matrix of independent `Exponential(1)` values.
    fn standard_exponential_matrix(key: &Self::Key, rows: usize, cols: usize) -> Self::Matrix;

    /// Draw `len` independent standard Gumbel values.
    fn standard_gumbel_vector(key: &Self::Key, len: usize) -> Self::Vector;
    /// Draw a matrix of independent standard Gumbel values.
    fn standard_gumbel_matrix(key: &Self::Key, rows: usize, cols: usize) -> Self::Matrix;

    /// Draw `len` independent standard logistic values.
    fn standard_logistic_vector(key: &Self::Key, len: usize) -> Self::Vector;
    /// Draw a matrix of independent standard logistic values.
    fn standard_logistic_matrix(key: &Self::Key, rows: usize, cols: usize) -> Self::Matrix;

    /// Draw `len` independent standard Cauchy values.
    fn standard_cauchy_vector(key: &Self::Key, len: usize) -> Self::Vector;
    /// Draw a matrix of independent standard Cauchy values.
    fn standard_cauchy_matrix(key: &Self::Key, rows: usize, cols: usize) -> Self::Matrix;
}

/// Explicit-key tensor random draws.
///
/// Implement this for arbitrary-rank backends. It is independent of
/// [`KeyedRandomMatVec`], so tensor-only systems do not need to expose separate vector or
/// matrix carriers.
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

/// A distribution sampled from an explicit-key backend.
pub trait SampleWithKey<B: KeyedRandomBackend + ?Sized> {
    type Output;

    fn sample_with_key(&self, key: &B::Key) -> Self::Output;
}

/// A random process with explicit non-random state and explicit keys.
pub trait KeyedRandomProcess<B: KeyedRandomBackend + ?Sized> {
    type State;
    type Output;

    fn sample_step(&self, state: Self::State, key: &B::Key) -> (Self::State, Self::Output);
}
