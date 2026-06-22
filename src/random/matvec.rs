use crate::array::{Matrix, Vector};

use super::KeyedRandomScalar;

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
