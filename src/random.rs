//! Backend-neutral random-source contracts over the lazy numeric/array tier.
//!
//! The default random-number primitives that work over euclid's *lazy* numbers:
//! traits for drawing `Uniform(0, 1)`, standard-normal, and gamma variates as lazy
//! scalars / vectors / matrices / tensors. They are intentionally independent of
//! host `rand`: an eager adapter wraps a `rand::Rng`, while an XLA adapter maps the
//! same requests to keyed random tensor ops. The associated scalar/vector/matrix/
//! tensor types are euclid's own lazy-array contracts, so nothing here names dense
//! storage or a concrete backend.

use crate::array::{LazyMatrix, LazyTensor, LazyVector};
use crate::numeric::LazyReal;

/// XLA/StableHLO random-bit-generator algorithms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RandomAlgorithm {
    /// Backend-defined algorithm and state shape.
    Default,
    /// Counter-based ThreeFry family.
    ThreeFry,
    /// Counter-based Philox family.
    Philox,
}

/// A lazy backend random stream for scalar, vector, and matrix real draws.
///
/// Eager implementations usually mutate a host PRNG. XLA-style implementations can model the
/// mutation as advancing/splitting an owned key inside the stream object while returning tensor
/// values for [`Vector`](Self::Vector) and [`Matrix`](Self::Matrix). The associated vector and
/// matrix types are euclid lazy-array contracts, so this trait does not mention dense storage.
pub trait RandomStream {
    /// The backend real value.
    type Scalar: LazyReal;
    /// A one-dimensional draw result in the backend's vector representation.
    type Vector: LazyVector<Scalar = Self::Scalar>;
    /// A two-dimensional draw result in the backend's matrix representation.
    type Matrix: LazyMatrix<Scalar = Self::Scalar, Vector = Self::Vector>;

    /// The random-bit algorithm used by this stream when the backend exposes one.
    fn algorithm(&self) -> RandomAlgorithm {
        RandomAlgorithm::Default
    }

    /// Draw a scalar from `Uniform(0, 1)`.
    fn uniform01(&mut self) -> Self::Scalar;
    /// Draw `len` independent `Uniform(0, 1)` values.
    fn uniform01_vector(&mut self, len: usize) -> Self::Vector;
    /// Draw a `rows x cols` matrix of independent `Uniform(0, 1)` values.
    fn uniform01_matrix(&mut self, rows: usize, cols: usize) -> Self::Matrix;

    /// Draw a scalar standard normal.
    fn standard_normal(&mut self) -> Self::Scalar;
    /// Draw `len` independent standard normals.
    fn standard_normal_vector(&mut self, len: usize) -> Self::Vector;
    /// Draw a `rows x cols` matrix of independent standard normals.
    fn standard_normal_matrix(&mut self, rows: usize, cols: usize) -> Self::Matrix;

    /// Draw a scalar `Gamma(shape, 1)` variate.
    fn standard_gamma(&mut self, shape: Self::Scalar) -> Self::Scalar;
}

/// A lazy random stream that can draw arbitrary-rank tensors.
///
/// `maxwell::Tensor` can implement this directly. Eager dense linear algebra usually implements
/// [`RandomStream`] instead because it has distinct scalar/vector/matrix concrete types.
pub trait TensorRandomStream: RandomStream {
    type Tensor: LazyTensor + LazyReal;

    fn uniform01_tensor(&mut self, shape: &[usize]) -> Self::Tensor;
    fn standard_normal_tensor(&mut self, shape: &[usize]) -> Self::Tensor;
}

/// A distribution sampled from a backend random stream.
pub trait SampleWith<R: RandomStream + ?Sized> {
    type Output;

    fn sample_with(&self, rng: &mut R) -> Self::Output;
}

/// Stateless/keyed lazy random backend surface for graph-oriented implementations.
///
/// This is the XLA-shaped sibling of [`RandomStream`]: each operation takes and returns an
/// explicit key so callers can model reproducible graph state without host mutation.
pub trait KeyedRandomBackend {
    type Key;
    type Scalar: LazyReal;
    type Vector: LazyVector<Scalar = Self::Scalar>;
    type Matrix: LazyMatrix<Scalar = Self::Scalar, Vector = Self::Vector>;
    type Tensor: LazyTensor + LazyReal;

    fn split(key: Self::Key, count: usize) -> Vec<Self::Key>;
    fn uniform01_scalar(key: Self::Key) -> (Self::Key, Self::Scalar);
    fn standard_normal_scalar(key: Self::Key) -> (Self::Key, Self::Scalar);
    fn uniform01_vector(key: Self::Key, len: usize) -> (Self::Key, Self::Vector);
    fn uniform01_matrix(key: Self::Key, rows: usize, cols: usize) -> (Self::Key, Self::Matrix);
    fn standard_normal_vector(key: Self::Key, len: usize) -> (Self::Key, Self::Vector);
    fn standard_normal_matrix(
        key: Self::Key,
        rows: usize,
        cols: usize,
    ) -> (Self::Key, Self::Matrix);
    fn standard_gamma(
        key: Self::Key,
        shape_parameter: Self::Scalar,
        len: usize,
    ) -> (Self::Key, Self::Vector);
    fn uniform01_tensor(key: Self::Key, shape: &[usize]) -> (Self::Key, Self::Tensor);
    fn standard_normal_tensor(key: Self::Key, shape: &[usize]) -> (Self::Key, Self::Tensor);
}

/// A lazy random process with explicit stream state.
pub trait RandomProcess<R: RandomStream + ?Sized> {
    type State;
    type Output;

    fn sample_step(&self, state: Self::State, rng: &mut R) -> (Self::State, Self::Output);
}
