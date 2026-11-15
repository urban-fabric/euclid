//! # euclid-matrix — backend matrix-extension contracts
//!
//! The **matrix extension traits** beyond euclid's core [`Tensor`] rank contract:
//! reusable SPD factors ([`SpdFactor`] / [`PreparedSpd`]), the Hermitian-positive-definite
//! solve ([`SpdSolve`]), triangular solve ([`TriangularSolve`]), and the heavyweight
//! decompositions ([`QrDecomposition`] / [`HermitianEigendecomposition`] /
//! [`SingularValueDecomposition`] / [`GeneralSolve`] / [`Determinant`]). A backend
//! (ramanujan eagerly, `maxwell`/XLA compiled) implements these for its concrete tensor type.
//! Depends on **only `euclid-tensor` and `euclid-numeric`**, and on nothing third-party.
//!
//! # Vocabulary only
//!
//! This crate names the operations and stops. It held algorithms written over them —
//! `least_squares`, `gram`, `inverse`, `dominant_eigenvector` and the rest — until 0.4 moved
//! them out, because they are **matrix kernels** and euclid holds only what the fleet must
//! agree on. A Cholesky or a QR has many valid implementations, and choosing between a blocked,
//! packed or GPU path is exactly what a backend is for, so there was nothing here to agree on.
//! What must be agreed is the *signature* a backend implements and a caller binds, which is
//! what remains.
//!
//! The consequence is deliberate: an algorithm over these traits now lives with a backend, so
//! two backends wanting the same algorithm write it twice. That is the cost of the boundary,
//! and it is the right trade — a shared algorithm that cannot be tested without euclid shipping
//! a tensor is worse than two tested ones.
//!
//! ## Rank, not separate matrix and vector types
//!
//! euclid 0.2 replaced `LazyMatrix`/`LazyVector` with a single rank-n [`Tensor`]: "a vector
//! is just a rank-1 tensor and a matrix a rank-2 tensor; there are no separate vector/matrix
//! contracts". euclid-matrix follows that. The consequences are deliberate and worth stating,
//! because they are the whole shape of this API:
//!
//! - [`SpdFactor`] carries **one** associated tensor type rather than a `Vector`/`Matrix`
//!   pair, and its old `solve_vec`/`solve` and `whiten_vec`/`whiten` pairs collapse into
//!   [`solve`](SpdFactor::solve) and [`whiten`](SpdFactor::whiten). Rank now travels with
//!   the value, so the two spellings had become the same signature.
//! - Element type is a **refinement bound on `Self`**, not an associated `Scalar`. A tensor
//!   *is* its numeric carrier, so [`NormedDivisionAlgebra`] methods such as `conjugate` are
//!   elementwise. [`Determinant::determinant`] therefore returns a rank-0 `Self` rather than
//!   a distinct scalar type.
//! - euclid deliberately shed linear-algebra *content*, and this crate now follows that the
//!   whole way: the HPD solve is **named** here as [`SpdSolve`] but not implemented, and the
//!   derived helpers that once accompanied it — `identity`, `adjoint`, `gram` — are gone, since
//!   a helper over a tensor needs a tensor to run on.
//!
//! Every square-matrix argument below is rank-2 and every vector argument rank-1 unless an
//! item says otherwise; the bound cannot express that, so it is documented per item.

#[cfg(feature = "conformance")]
pub mod conformance;

use euclid_numeric::{NormedDivisionAlgebra, Real};
use euclid_tensor::Tensor;

/// The real field of a tensor's elements — its norms, log-determinant, and Mahalanobis
/// distance. `Self` for a real carrier, the real part for a complex one.
pub type RealPart<M> = <M as NormedDivisionAlgebra>::RealPart;

/// The carrier this crate's algorithms are generic over: a rank-n tensor whose elements
/// form a normed division algebra.
///
/// [`NormedDivisionAlgebra`] already implies `Arithmetic: Ring: Semiring: Element`, so this
/// one bound supplies `matmul`, `zeros`/`one`, the four ring operators, division, and
/// `conjugate` together. It exists to keep the many `where` clauses below readable, not to
/// add a requirement.
pub trait MatrixTensor: Tensor + NormedDivisionAlgebra {}

impl<M> MatrixTensor for M where M: Tensor + NormedDivisionAlgebra {}

/// Which triangular half of a factor matrix is meaningful.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Triangle {
    /// Lower-triangular factor, as returned by a conventional Cholesky `L`.
    Lower,
    /// Upper-triangular factor, as returned by QR's `R`.
    Upper,
}

/// How a factor should be viewed while solving a triangular system.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TransposeMode {
    /// Solve with `A`.
    None,
    /// Solve with `Aᵀ`.
    Transpose,
    /// Solve with `Aᴴ`.
    Adjoint,
}

/// The Hermitian-positive-definite solve `Σ⁻¹B`, via the backend's Cholesky path.
///
/// This was `euclid::array::LazyMatrix::solve_spd` before euclid 0.2 shed linear-algebra
/// content. It is a backend primitive rather than a derived algorithm — it maps to a
/// `POTRF`/`POTRS` pair eagerly and to a compiled Cholesky-solve on XLA — so it is named
/// here and implemented there, exactly as the decomposition traits below are.
///
/// Hermitian-positive-definiteness of `self` is the caller's **precondition**; implementors
/// need not verify it.
pub trait SpdSolve: MatrixTensor {
    /// `Σ⁻¹B` for a rank-2 `Σ` (square) and a rank-1 or rank-2 right-hand side.
    fn solve_spd(self, rhs: Self) -> Self;
}

/// A reusable SPD (Cholesky) factorization — the **prepare once, solve/logdet many** seam.
///
/// The per-call density path re-factors `Σ` every call on every backend. This hoists the
/// O(n³) factorization out of that path so an eager backend can cache the factor and an XLA
/// backend can feed it as a graph input.
pub trait SpdFactor {
    /// The tensor type this factor solves against. One type serves both right-hand sides:
    /// a vector is a rank-1 tensor and a multi-column `B` a rank-2 one.
    type Matrix: MatrixTensor;

    /// `Σ⁻¹B`, reusing the stored factor. `B` may be rank-1 or rank-2.
    fn solve(&self, rhs: Self::Matrix) -> Self::Matrix;

    /// `L⁻¹B` for `Σ = LLᴴ` — the whitening transform. `B` may be rank-1 or rank-2.
    fn whiten(&self, rhs: Self::Matrix) -> Self::Matrix;

    /// `ln det Σ` from the stored factor.
    fn logdet(&self) -> RealPart<Self::Matrix>;
}

/// An HPD matrix that can produce a reusable [`SpdFactor`].
pub trait PreparedSpd: MatrixTensor {
    /// The factor type produced by [`prepare_spd`](Self::prepare_spd).
    type Factor: SpdFactor<Matrix = Self>;

    /// Factor a rank-2 HPD matrix once into a reusable [`SpdFactor`].
    fn prepare_spd(self) -> Self::Factor;
}

/// A backend primitive for `AX = B` where `A` is triangular.
///
/// This maps directly to eager BLAS/LAPACK-style triangular solves and to compiled tensor
/// primitives such as XLA's triangular solve. Algorithms should prefer this when they
/// already hold a triangular factor instead of re-entering a full matrix solve.
pub trait TriangularSolve: MatrixTensor {
    /// Solve `AX = B` with a triangular rank-2 `A`. `B` may be rank-1 or rank-2 — the old
    /// `solve_triangular_vec` spelling collapsed into this one when rank moved onto the
    /// value.
    fn solve_triangular(
        self,
        rhs: Self,
        triangle: Triangle,
        transpose: TransposeMode,
        unit_diagonal: bool,
    ) -> Self;
}

/// Matrix operations whose spectra are real-valued even for a complex matrix.
pub trait MatrixRealSpectrum: MatrixTensor {
    /// The rank-1 eigenvalue / singular-value tensor. Real by construction, which is why it
    /// is a distinct type from `Self` for a complex carrier.
    type RealSpectrum: Tensor + Real;
}

/// General nonsingular linear solve `A⁻¹B`.
pub trait GeneralSolve: MatrixTensor {
    /// Solve `AX = B` without assuming `A` is Hermitian positive definite.
    fn solve(self, rhs: Self) -> Self;
}

/// Determinant of a square matrix.
pub trait Determinant: MatrixTensor {
    /// The determinant `det A`, as a **rank-0** tensor. euclid's unified-dtype model has no
    /// separate scalar type, so a scalar result is a rank-0 value of the same carrier.
    fn determinant(&self) -> Self;
}

/// QR factorization.
pub trait QrDecomposition: MatrixTensor {
    /// Reduced/economy QR factorization `A = QR`.
    ///
    /// For an `m x n` full-column-rank matrix with `m >= n`, this returns `Q` as `m x n`
    /// with orthonormal columns and `R` as `n x n` upper-triangular.
    fn qr(self) -> (Self, Self);
}

/// Hermitian eigendecomposition.
pub trait HermitianEigendecomposition: MatrixRealSpectrum {
    /// Hermitian eigendecomposition `A = V diag(λ) Vᴴ`, eigenvalues ascending.
    fn eigh(self) -> (Self::RealSpectrum, Self);
}

/// Singular value decomposition.
pub trait SingularValueDecomposition: MatrixRealSpectrum {
    /// Singular value decomposition `A = U diag(σ) Vᴴ`, singular values descending.
    fn svd(self) -> (Self, Self::RealSpectrum, Self);
}

/// The heavyweight, non-Hermitian-PD decompositions and solves: general nonsingular solve,
/// determinant, QR, Hermitian eigendecomposition, and SVD.
///
/// These are deliberately outside euclid's core [`Tensor`] rank contract. Backends that can
/// provide them implement this extension trait here; portable kernels that need only the SPD
/// spine can stay on [`SpdSolve`].
pub trait LazyDecompositions:
    Determinant
    + GeneralSolve
    + QrDecomposition
    + HermitianEigendecomposition
    + SingularValueDecomposition
{
}

impl<M> LazyDecompositions for M where
    M: Determinant
        + GeneralSolve
        + QrDecomposition
        + HermitianEigendecomposition
        + SingularValueDecomposition
{
}
