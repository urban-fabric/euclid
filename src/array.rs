//! **DRAFT** — the array-level lazy contracts: `LazyVector`, `LazyMatrix`,
//! `LazyTensor`. The rank-graded peers of the scalar [`LazyReal`].
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
//! The contract exposes only the rank math (`dot`, `matmul`, `cholesky`, `solve_spd`,
//! `logdet_spd`, `mahalanobis_squared`, reductions). Richer matrix extension traits
//! (`PreparedSpd`, `SpdFactor`, general decompositions) live in `frobenius`.
//!
//! ## Real **and** complex
//!
//! Every rank is generic over a [`NormedDivisionAlgebra`] — real
//! (`f64`/`f32`) or complex — so one kernel covers both. The vector-space and product
//! operations stay in the scalar field; the quantities that are **intrinsically real
//! even over ℂ** — vector `norm`/`norm_squared`, the SPD `logdet_spd` and
//! `mahalanobis_squared` — return the scalar's
//! [`RealPart`](crate::numeric::NormedDivisionAlgebra::RealPart) field. `dot` is the **Hermitian** inner product `Σ conj(aᵢ)·bᵢ` (the ordinary dot
//! for real vectors, since `conjugate` is the identity there), and `cholesky` is the
//! Hermitian factorization `Σ = L·Lᴴ` (`L·Lᵀ` for real `Σ`).
//!
//! ## The four ranks
//!
//! | rank | trait | adds |
//! |------|-------|------|
//! | 0 | [`NormedDivisionAlgebra`] | conjugate + magnitude + the real field |
//! | 1 | [`LazyVector`] | inner product, norms, vector-space ops |
//! | 2 | [`LazyMatrix`] | matmul + the SPD linear-algebra quantities |
//! | n | [`LazyTensor`] | shape + reductions over an elementwise-analytic array |
//!
//! `LazyVector`/`LazyMatrix` carry only vector-space / linear-algebra structure, so a
//! `DenseMatrix` (not elementwise-analytic) can implement them. `LazyTensor: NormedDivisionAlgebra`
//! is the array itself *as* a lazy scalar; a **real** tensor additionally implements
//! [`LazyReal`] (so the scalar kernels lift to batched tensor
//! kernels by monomorphizing at `T = Tensor`), and a complex tensor
//! [`LazyComplex`](crate::numeric::LazyComplex).

use crate::numeric::{LazyReal, NormedDivisionAlgebra};

/// The real field of a vector/matrix's scalar — its norms, log-determinant, and
/// Mahalanobis distance. `S` for a real scalar, the real part for a complex one.
type RealPart<S> = <S as NormedDivisionAlgebra>::RealPart;

/// **Rank 1.** A vector over a lazy scalar (real or complex) — the vector-space and
/// inner-product surface (no elementwise transcendentals; those belong to
/// [`LazyTensor`]).
pub trait LazyVector: Clone {
    /// The element scalar — real or complex.
    type Scalar: NormedDivisionAlgebra;

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
    /// The **Hermitian** inner product `Σ conj(aᵢ)·bᵢ` — the ordinary `Σ aᵢ·bᵢ` for a
    /// real vector, since `conjugate` is the identity there.
    fn dot(&self, other: &Self) -> Self::Scalar;
    fn sum(&self) -> Self::Scalar;
    /// `Σ |aᵢ|²` — real even over ℂ.
    fn norm_squared(&self) -> RealPart<Self::Scalar>;
    /// `√(Σ |aᵢ|²)` — real even over ℂ.
    fn norm(&self) -> RealPart<Self::Scalar>;
}

/// **Rank 2.** Dense linear algebra over a scalar field (real or complex) — the
/// Hermitian-positive-definite quantities the multivariate Gaussian families are
/// written in. Stated as *mathematical results* (`solve_spd`, `logdet_spd`,
/// `mahalanobis_squared`); the backend owns the stable algorithm. There is deliberately
/// **no `inverse`**.
pub trait LazyMatrix: Clone {
    /// The element scalar — real or complex.
    type Scalar: NormedDivisionAlgebra;
    /// The companion vector type (`DenseVector` for ramanujan, `Self` for a tensor).
    type Vector: LazyVector<Scalar = Self::Scalar>;

    /// `(rows, cols)`.
    fn shape(&self) -> (usize, usize);

    // --- constructors ---
    // Context-free (like [`LazyReal::scalar`](crate::numeric::LazyReal::scalar)): an
    // eager backend allocates; a lazy tensor backend emits a constant node (`Broadcast` of
    // `0`, an `Eye`/`Iota` for the identity) in its default graph/device context. They
    // give the abstract algorithms (`A \ I`, `AᴴA + λI`, …) the values they need to
    // compose without dropping to a concrete type.
    /// The `n×n` identity matrix.
    fn identity(n: usize) -> Self;
    /// An all-zeros `rows×cols` matrix.
    fn zeros(rows: usize, cols: usize) -> Self;

    // --- vector space + products ---
    fn add(self, other: Self) -> Self;
    fn scale(self, factor: Self::Scalar) -> Self;
    /// The (plain) transpose `Aᵀ` — the structural axis swap (XLA `Transpose`).
    fn transpose(self) -> Self;
    /// The conjugate (Hermitian) transpose `Aᴴ` — `Transpose` then elementwise
    /// `conj` (XLA `Transpose`+`Conj`); the plain `transpose` for a real scalar. The
    /// building block of the Hermitian forms (`AᴴA`, the normal equations) a generic
    /// algorithm composes without dropping to a concrete type.
    fn adjoint(self) -> Self;
    fn matmul(self, rhs: Self) -> Self;
    fn matvec(self, v: Self::Vector) -> Self::Vector;
    fn trace(&self) -> Self::Scalar;
    fn diagonal(&self) -> Self::Vector;

    // --- HPD quantities (the backend picks the stable factorization) ---
    /// The lower Cholesky factor `L` with `Σ = L·Lᴴ`, for Hermitian-PD `Σ`
    /// (`Σ = L·Lᵀ` for real `Σ`).
    fn cholesky(self) -> Self;
    /// `Σ⁻¹ B` for Hermitian-PD `Σ` — no explicit inverse is formed.
    fn solve_spd(self, rhs: Self) -> Self;
    /// `Σ⁻¹ b` for Hermitian-PD `Σ` and a single right-hand-side **vector** `b` — the vector
    /// analogue of [`solve_spd`](Self::solve_spd). It is the missing primitive for the
    /// Gaussian *mean* family (`Σ⁻¹µ`, GP posterior mean, conditional mean), which cannot be
    /// written from `solve_spd` (matrix rhs) alone because `Self` and `Self::Vector` differ.
    fn solve_spd_vec(self, rhs: Self::Vector) -> Self::Vector;
    /// `ln det Σ` for Hermitian-PD `Σ` — real (the determinant of an HPD matrix is a
    /// positive real).
    fn logdet_spd(self) -> RealPart<Self::Scalar>;
    /// The squared Mahalanobis distance `δᴴ Σ⁻¹ δ` — the multivariate-Gaussian kernel,
    /// real even over ℂ (a Hermitian form).
    fn mahalanobis_squared(&self, delta: &Self::Vector) -> RealPart<Self::Scalar>;
}

/// **Rank n.** An n-dimensional array over a lazy scalar (real or complex), carrying
/// the abstract array notions — shape and reductions. `LazyTensor: NormedDivisionAlgebra`, so a
/// tensor is itself a lazy scalar (`Scalar = Vector = Self` for the unified backend); a
/// **real** tensor additionally implements [`LazyReal`], so
/// every scalar kernel applies elementwise/batched for free, and a complex tensor
/// [`LazyComplex`](crate::numeric::LazyComplex). A rank-1 `LazyTensor` is also a
/// [`LazyVector`] and a rank-2 one a [`LazyMatrix`].
///
/// Backend-specific machinery — how axes are named in a contraction, how a
/// comparison/selection mask is represented, broadcast/gather/scatter, precision — is
/// **not** here; it lives with the tensor backend.
pub trait LazyTensor: NormedDivisionAlgebra {
    /// The axis lengths. Returned **owned** — a tensor's shape is dynamic metadata a
    /// backend computes (a lazy graph node, a columnar expression), not necessarily a
    /// stored contiguous slice; consistent with [`LazyMatrix::shape`] /
    /// [`LazyVector::len`] returning owned shape info.
    fn shape(&self) -> Vec<usize>;
    fn rank(&self) -> usize {
        self.shape().len()
    }

    /// Reshape to a compatible shape.
    fn reshape(self, shape: &[usize]) -> Self;

    // --- reductions over the given axes ---
    fn sum(self, axes: &[usize]) -> Self;
    /// Reduce-max — the stable kernel behind log-sum-exp. Requires a **real** (ordered)
    /// element scalar; a max over ℂ is not defined.
    fn max(self, axes: &[usize]) -> Self
    where
        Self: LazyReal;

    /// The tensor contraction (matmul / batched-matmul / einsum). How the contracted
    /// and batch axes are named is the backend's concern.
    fn matmul(self, rhs: Self) -> Self;
}
