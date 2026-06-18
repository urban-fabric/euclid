//! **DRAFT** — the array-level lazy contracts: `LazyVector`, `LazyMatrix`,
//! `LazyTensor`. The rank-graded peers of the scalar [`LazyReal`](crate::numeric::LazyReal).
//!
//! These are **abstract mathematical contracts** — the operations a kernel is written
//! in, *not* any backend's parameterization of them. Implementation-specific concerns
//! stay with the implementor:
//!
//! - *how* a stable Cholesky / solve is computed (blocked CPU kernels vs. an XLA
//!   `Cholesky` op), the `uplo`/`trans` flags, memory layout;
//! - *how* a contraction names its axes (XLA `DotGeneral`'s dimension-number tuples);
//! - *how* a comparison/selection or precision policy is represented.
//!
//! The contract exposes only the math (`dot`, `matmul`, `cholesky`, `solve_spd`,
//! `logdet_spd`, `mahalanobis_squared`, reductions). The concrete implementors are the
//! backends: `ramanujan::{DenseVector, DenseMatrix}` (the `f64` reference) and
//! `lebesgue::Tensor` (the lazy/XLA engine — one type satisfies all three ranks, since
//! a tensor *can also be a 1-D tensor*).
//!
//! ## The four ranks
//!
//! | rank | trait | adds |
//! |------|-------|------|
//! | 0 | [`LazyReal`](crate::numeric::LazyReal) | elementwise analysis + scalar lift |
//! | 1 | [`LazyVector`] | inner product, norms, vector-space ops |
//! | 2 | [`LazyMatrix`] | matmul + the SPD linear-algebra quantities |
//! | n | [`LazyTensor`] | shape + reductions over an elementwise-analytic array |
//!
//! `LazyVector`/`LazyMatrix` carry only vector-space / linear-algebra structure, so a
//! `DenseMatrix` (not elementwise-analytic) can implement them. `LazyTensor: LazyReal`
//! *is* elementwise-analytic, so the scalar kernels lift to batched tensor kernels by
//! monomorphizing at `T = Tensor`.

use crate::numeric::LazyReal;

/// **Rank 1.** A vector over a lazy-real scalar — the vector-space and inner-product
/// surface (no elementwise transcendentals; those belong to [`LazyTensor`]).
pub trait LazyVector: Clone {
    /// The element scalar.
    type Scalar: LazyReal;

    /// Element count.
    fn len(&self) -> usize;
    /// Whether the vector is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    // --- vector space ---
    fn add(self, other: Self) -> Self;
    fn sub(self, other: Self) -> Self;
    fn scale(self, factor: Self::Scalar) -> Self;
    /// Elementwise (Hadamard) product.
    fn hadamard(self, other: Self) -> Self;

    // --- inner product / norms ---
    /// Inner product `Σ aᵢ·bᵢ`.
    fn dot(&self, other: &Self) -> Self::Scalar;
    fn sum(&self) -> Self::Scalar;
    /// `Σ aᵢ²`.
    fn norm_squared(&self) -> Self::Scalar;
    /// `√(Σ aᵢ²)`.
    fn norm(&self) -> Self::Scalar;
}

/// **Rank 2.** Dense linear algebra over a scalar field — the symmetric-positive-
/// definite quantities the multivariate Gaussian families are written in. Stated as
/// *mathematical results* (`solve_spd`, `logdet_spd`, `mahalanobis_squared`); the
/// backend owns the stable algorithm. There is deliberately **no `inverse`**.
pub trait LazyMatrix: Clone {
    /// The element scalar.
    type Scalar: LazyReal;
    /// The companion vector type (`DenseVector` for ramanujan, `Self` for a tensor).
    type Vector: LazyVector<Scalar = Self::Scalar>;

    /// `(rows, cols)`.
    fn shape(&self) -> (usize, usize);

    // --- vector space + products ---
    fn add(self, other: Self) -> Self;
    fn scale(self, factor: Self::Scalar) -> Self;
    fn transpose(self) -> Self;
    fn matmul(self, rhs: Self) -> Self;
    fn matvec(self, v: Self::Vector) -> Self::Vector;
    fn trace(&self) -> Self::Scalar;
    fn diagonal(&self) -> Self::Vector;

    // --- SPD quantities (the backend picks the stable factorization) ---
    /// The lower Cholesky factor `L` with `Σ = L Lᵀ`, for SPD `Σ`.
    fn cholesky(self) -> Self;
    /// `Σ⁻¹ B` for SPD `Σ` — no explicit inverse is formed.
    fn solve_spd(self, rhs: Self) -> Self;
    /// `ln det Σ` for SPD `Σ`.
    fn logdet_spd(self) -> Self::Scalar;
    /// The squared Mahalanobis distance `δᵀ Σ⁻¹ δ` — the multivariate-Gaussian kernel.
    fn mahalanobis_squared(&self, delta: &Self::Vector) -> Self::Scalar;
}

/// **Rank n.** An n-dimensional, elementwise-analytic array. `LazyTensor: LazyReal`, so
/// every scalar kernel applies elementwise/batched for free; this trait adds only the
/// abstract array notions — shape and reductions. A rank-1 `LazyTensor` is also a
/// [`LazyVector`] and a rank-2 one a [`LazyMatrix`] (for a unified tensor type,
/// `Scalar = Vector = Self`).
///
/// Backend-specific machinery — how axes are named in a contraction, how a
/// comparison/selection mask is represented, broadcast/gather/scatter, precision — is
/// **not** here; it lives with the tensor backend.
pub trait LazyTensor: LazyReal {
    fn shape(&self) -> &[usize];
    fn rank(&self) -> usize {
        self.shape().len()
    }

    /// Reshape to a compatible shape.
    fn reshape(self, shape: &[usize]) -> Self;

    // --- reductions over the given axes ---
    fn sum(self, axes: &[usize]) -> Self;
    /// Reduce-max — the stable kernel behind log-sum-exp.
    fn max(self, axes: &[usize]) -> Self;

    /// The tensor contraction (matmul / batched-matmul / einsum). How the contracted
    /// and batch axes are named is the backend's concern.
    fn matmul(self, rhs: Self) -> Self;
}
