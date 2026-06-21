//! Backend-neutral random-source contracts over the lazy numeric/array tier.
//!
//! Traits for drawing `Uniform(0, 1)`, standard-normal, and gamma variates as lazy
//! scalars/vectors/matrices/tensors, independent of host `rand`: an eager adapter wraps a
//! `rand::Rng`, an XLA adapter maps the same requests to keyed random tensor ops. The
//! associated types are euclid's own lazy-array contracts, so nothing here names dense
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
/// Eager implementations usually mutate a host PRNG; XLA-style ones advance/split an owned
/// key inside the stream object. The associated types are euclid lazy-array contracts.
pub trait RandomStream {
    /// The backend real value.
    type Scalar: LazyReal;
    /// A one-dimensional draw result.
    type Vector: LazyVector<Scalar = Self::Scalar>;
    /// A two-dimensional draw result.
    type Matrix: LazyMatrix<Scalar = Self::Scalar, Vector = Self::Vector>;

    /// The random-bit algorithm, when the backend exposes one.
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

/// A lazy random stream that can draw arbitrary-rank tensors. `maxwell::Tensor` implements
/// this directly; eager dense backends implement [`RandomStream`] instead (distinct types).
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

/// The XLA-shaped sibling of [`RandomStream`]: each operation takes and returns an explicit
/// key, modeling reproducible graph state without host mutation.
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
