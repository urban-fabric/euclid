//! **DRAFT** — the rank-graded lazy array contracts ([`LazyVector`], [`LazyMatrix`],
//! [`LazyTensor`]), the peers of the scalar [`LazyReal`].
//!
//! These are abstract math contracts — the rank operations a kernel is written in
//! (`dot`, `matmul`, `cholesky`, `solve_spd`, `logdet_spd`, `mahalanobis_squared`,
//! reductions), not a backend's parameterization of them. *How* a Cholesky/solve runs,
//! how a contraction names its axes, how a mask is represented, layout, and precision
//! all stay with the implementor. Richer matrix traits (`PreparedSpd`, `SpdFactor`,
//! general decompositions) live in `frobenius`.
//!
//! Two bands, mirroring XLA/JAX's `lax` (`reshape`/`reduce`/`dot` are dtype-agnostic,
//! `cholesky`/`triangular_solve` want a field):
//!
//! - **Structural** ([`LazyTensor`] on [`LazyRing`]): shape, reshape, reductions, and
//!   contraction over real, complex, *or* integer elements. The element kind is a
//!   refinement — a real tensor also implements [`LazyReal`], a complex one
//!   [`LazyComplex`](crate::numeric::LazyComplex), an integer one
//!   [`LazyInteger`](crate::numeric::LazyInteger).
//! - **Linear algebra** ([`LazyVector`]/[`LazyMatrix`] over a [`NormedDivisionAlgebra`]):
//!   inner products, norms, and SPD quantities genuinely need a field. The intrinsically
//!   real quantities (`norm`/`norm_squared`, `logdet_spd`, `mahalanobis_squared`) return
//!   the scalar's [`RealPart`](crate::numeric::NormedDivisionAlgebra::RealPart) even over
//!   ℂ; `dot` is the Hermitian `Σ conj(aᵢ)·bᵢ` and `cholesky` the Hermitian `Σ = L·Lᴴ`
//!   (both collapse to the plain real form, since `conjugate` is the identity on ℝ).
//!
//! | rank | trait | bound | adds |
//! |------|-------|-------|------|
//! | 0 | [`NormedDivisionAlgebra`] | — | conjugate + magnitude + the real field |
//! | 1 | [`LazyVector`] | field | inner product, norms, vector-space ops |
//! | 2 | [`LazyMatrix`] | field | matmul + the SPD linear-algebra quantities |
//! | n | [`LazyTensor`] | [`LazyRing`] | shape + reductions + contraction |

use crate::numeric::{LazyReal, LazyRing, NormedDivisionAlgebra};

/// The real field of a scalar `S` — `S` itself for a real scalar, its real part for a
/// complex one. Where norms, log-determinant, and Mahalanobis distance live.
type RealPart<S> = <S as NormedDivisionAlgebra>::RealPart;

/// **Rank 1.** A vector over a lazy scalar (real or complex): the vector-space and
/// inner-product surface (elementwise transcendentals belong to [`LazyTensor`]).
pub trait LazyVector: Clone {
    /// The element scalar — real or complex.
    type Scalar: NormedDivisionAlgebra;

    fn len(&self) -> usize;
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
    /// The Hermitian inner product `Σ conj(aᵢ)·bᵢ` (the ordinary `Σ aᵢ·bᵢ` for real vectors).
    fn dot(&self, other: &Self) -> Self::Scalar;
    fn sum(&self) -> Self::Scalar;
    /// `Σ |aᵢ|²` — real even over ℂ.
    fn norm_squared(&self) -> RealPart<Self::Scalar>;
    /// `√(Σ |aᵢ|²)` — real even over ℂ.
    fn norm(&self) -> RealPart<Self::Scalar>;
}

/// **Rank 2.** Dense linear algebra over a scalar field (real or complex), centered on
/// the Hermitian-positive-definite quantities the multivariate Gaussians need. Stated as
/// mathematical results (`solve_spd`, `logdet_spd`, …); the backend owns the stable
/// algorithm. There is deliberately **no `inverse`**.
pub trait LazyMatrix: Clone {
    /// The element scalar — real or complex.
    type Scalar: NormedDivisionAlgebra;
    /// The companion vector type (`Self` for a tensor backend).
    type Vector: LazyVector<Scalar = Self::Scalar>;

    /// `(rows, cols)`.
    fn shape(&self) -> (usize, usize);

    // --- constructors ---
    // Context-free, like `LazyReal::scalar`: an eager backend allocates, a lazy one emits
    // a constant node. They let the abstract algorithms (`A \ I`, `AᴴA + λI`, …) compose
    // without dropping to a concrete type.
    /// The `n×n` identity matrix.
    fn identity(n: usize) -> Self;
    /// An all-zeros `rows×cols` matrix.
    fn zeros(rows: usize, cols: usize) -> Self;

    // --- vector space + products ---
    fn add(self, other: Self) -> Self;
    fn scale(self, factor: Self::Scalar) -> Self;
    /// The plain transpose `Aᵀ`.
    fn transpose(self) -> Self;
    /// The conjugate (Hermitian) transpose `Aᴴ` — `transpose` then `conj`; the plain
    /// `transpose` for a real scalar. The building block of the Hermitian forms (`AᴴA`).
    fn adjoint(self) -> Self;
    fn matmul(self, rhs: Self) -> Self;
    fn matvec(self, v: Self::Vector) -> Self::Vector;
    fn trace(&self) -> Self::Scalar;
    fn diagonal(&self) -> Self::Vector;

    // --- HPD quantities (the backend picks the stable factorization) ---
    /// The lower Cholesky factor `L` with `Σ = L·Lᴴ` (`L·Lᵀ` for real `Σ`).
    fn cholesky(self) -> Self;
    /// `Σ⁻¹ B` for Hermitian-PD `Σ` — no explicit inverse is formed.
    fn solve_spd(self, rhs: Self) -> Self;
    /// `Σ⁻¹ b`, the single-vector analogue of [`solve_spd`](Self::solve_spd). Its own
    /// primitive because `Self` and `Self::Vector` differ; the Gaussian *mean* family
    /// (`Σ⁻¹µ`, GP/conditional means) needs it.
    fn solve_spd_vec(self, rhs: Self::Vector) -> Self::Vector;
    /// `ln det Σ` for Hermitian-PD `Σ` — real (an HPD determinant is a positive real).
    fn logdet_spd(self) -> RealPart<Self::Scalar>;
    /// The squared Mahalanobis distance `δᴴ Σ⁻¹ δ` — real even over ℂ (a Hermitian form).
    fn mahalanobis_squared(&self, delta: &Self::Vector) -> RealPart<Self::Scalar>;
}

/// **Rank n.** An n-dimensional array over a lazy *ring* element (real, complex, or
/// integer): the dtype-agnostic array notions — shape, reshape, reductions, contraction.
/// `LazyTensor: LazyRing`, so a tensor is itself a lazy ring element, and the element kind
/// is a refinement: a real tensor also implements [`LazyReal`], a complex one
/// [`LazyComplex`](crate::numeric::LazyComplex), an integer one
/// [`LazyInteger`](crate::numeric::LazyInteger). A rank-1 tensor over a field is also a
/// [`LazyVector`], a rank-2 one a [`LazyMatrix`].
///
/// The bound is [`LazyRing`], not [`NormedDivisionAlgebra`], so the structural ops carry
/// no field demand; only [`max`](LazyTensor::max) and the rank-1/2 linear algebra ask for
/// more. Axis naming, mask representation, broadcast/gather/scatter, and precision live
/// with the backend.
pub trait LazyTensor: LazyRing {
    /// The axis lengths, returned owned — a tensor's shape is dynamic metadata a backend
    /// computes, not necessarily a stored slice.
    fn shape(&self) -> Vec<usize>;
    fn rank(&self) -> usize {
        self.shape().len()
    }

    /// Reshape to a compatible shape.
    fn reshape(self, shape: &[usize]) -> Self;

    // --- reductions over the given axes ---
    fn sum(self, axes: &[usize]) -> Self;
    /// Reduce-max (the kernel behind log-sum-exp). Requires a real, ordered scalar.
    fn max(self, axes: &[usize]) -> Self
    where
        Self: LazyReal;

    /// The tensor contraction (matmul / batched-matmul / einsum); the backend names the axes.
    fn matmul(self, rhs: Self) -> Self;
}
