//! # frobenius — backend matrix-extension contracts
//!
//! The **matrix extension traits** beyond euclid's core [`Tensor`] rank contract:
//! reusable SPD factors ([`SpdFactor`] / [`PreparedSpd`]), the Hermitian-positive-definite
//! solve ([`SpdSolve`]), triangular solve ([`TriangularSolve`]), and the heavyweight
//! decompositions ([`QrDecomposition`] / [`HermitianEigendecomposition`] /
//! [`SingularValueDecomposition`] / [`GeneralSolve`] / [`Determinant`]). A backend
//! (ramanujan eagerly, `maxwell`/XLA compiled) implements these for its concrete tensor
//! type; this crate names the vocabulary plus the algorithms written once over it.
//! Depends on **only `euclid`**.
//!
//! ```ignore
//! use euclid_tensor::Tensor;
//! use frobenius::least_squares;
//!
//! // One source — runs on any Tensor backend (e.g. the compiled maxwell::Tensor).
//! let x = least_squares(a, b);
//! ```
//!
//! ## Rank, not separate matrix and vector types
//!
//! euclid 0.2 replaced `LazyMatrix`/`LazyVector` with a single rank-n [`Tensor`]: "a vector
//! is just a rank-1 tensor and a matrix a rank-2 tensor; there are no separate vector/matrix
//! contracts". frobenius follows that. The consequences are deliberate and worth stating,
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
//! - euclid deliberately shed linear-algebra *content*, so `identity`, `adjoint`, and the
//!   HPD solve are no longer euclid's. They are named here — [`identity`], [`adjoint`],
//!   [`SpdSolve`] — which is the correct home for them: this crate is exactly "the matrix
//!   extensions beyond the rank contract".
//!
//! Every square-matrix argument below is rank-2 and every vector argument rank-1 unless a
//! function says otherwise; the bound cannot express that, so it is documented per item.

use euclid_numeric::{NormedDivisionAlgebra, Real};
use euclid_tensor::{Tensor, TensorIndexing};

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

/// The `n x n` identity matrix.
///
/// euclid dropped `LazyMatrix::identity` along with the rest of its linear-algebra content,
/// so it is derived here from the rank contract rather than asked of the backend:
/// one-hot-encoding `[0, 1, …, n−1]` to depth `n` sets exactly the diagonal. That keeps
/// every backend correct by construction instead of trusting `n²` host values.
pub fn identity<M>(n: usize) -> M
where
    M: TensorIndexing + MatrixTensor,
{
    M::one_hot(&M::iota(n), n)
}

/// The conjugate transpose `Aᴴ` of a rank-2 tensor.
///
/// Elementwise conjugation composed with an axis swap. For a real carrier `conjugate` is the
/// identity and this is the plain transpose.
pub fn adjoint<M>(a: &M) -> M
where
    M: MatrixTensor,
{
    a.conjugate().transpose_axes(&[1, 0])
}

/// The Gram matrix `AᴴA` — Hermitian positive-semidefinite.
///
/// This is the normal-equation matrix behind least squares and the unscaled covariance of
/// `A`'s columns. Forming it squares the condition number; prefer a QR-based path when the
/// backend provides one.
pub fn gram<M>(a: M) -> M
where
    M: MatrixTensor,
{
    adjoint(&a).matmul(a)
}

/// The inverse `Σ⁻¹` of a Hermitian-positive-definite matrix via the Cholesky solve
/// (`Σ \ I`) — no general factorization.
///
/// Prefer [`SpdSolve::solve_spd`] to *apply* the inverse; forming it is both slower and less
/// stable.
pub fn spd_inverse<M>(sigma: M) -> M
where
    M: SpdSolve + TensorIndexing,
{
    let n = leading_dim(&sigma);
    sigma.solve_spd(identity(n))
}

/// The inverse `A⁻¹` of a general nonsingular matrix, by solving `AX = I`.
///
/// As with [`spd_inverse`], prefer [`GeneralSolve::solve`] to apply the inverse rather than
/// forming it.
pub fn inverse<M>(a: M) -> M
where
    M: GeneralSolve + TensorIndexing,
{
    let n = leading_dim(&a);
    a.solve(identity(n))
}

/// The least-squares solution of the (over-determined, full-column-rank) system `A·x ≈ B`
/// via the normal equations `(AᴴA)·x = Aᴴ·B`, solved through the SPD (Cholesky) path.
///
/// Note the normal-equation path forms `AᴴA` and therefore squares the condition number; a
/// QR-based solve is preferable when the backend provides one.
pub fn least_squares<M>(a: M, b: M) -> M
where
    M: SpdSolve,
{
    let ah = adjoint(&a);
    let ata = ah.clone().matmul(a); // AᴴA  (Hermitian PD for full-rank A)
    let atb = ah.matmul(b); // AᴴB
    ata.solve_spd(atb)
}

/// The quadratic form `xᴴAx`, which is **real** for a Hermitian `A`.
///
/// `x` is a rank-2 column (`n x 1`), so the result is a `1 x 1` real tensor. The real part is
/// taken rather than asserted: for a Hermitian `A` the imaginary component is zero up to
/// rounding, and discarding it is the documented contract rather than an error case.
pub fn quadratic_form<M>(a: M, x: M) -> RealPart<M>
where
    M: MatrixTensor,
{
    adjoint(&x).matmul(a.matmul(x)).real_part()
}

/// The dominant eigenvector of `A` by **power iteration**, starting from `start` and running
/// a fixed `iterations` steps.
///
/// Each step forms `Ax` and renormalizes to unit 2-norm, so the iterate converges to the
/// eigenvector of largest `|λ|` at a rate set by the ratio of the two leading eigenvalue
/// magnitudes. The step count is the caller's, and fixed: there is no convergence test,
/// because a compiled backend cannot branch on a device value without a host synchronization
/// this contract has no room for. A caller needing a tolerance should iterate and check
/// itself.
///
/// `start` must be rank-2 `n x 1` and must not be orthogonal to the dominant eigenvector —
/// with an unlucky start the iteration converges to a subdominant one.
pub fn dominant_eigenvector<M>(a: M, start: M, iterations: usize) -> M
where
    M: MatrixTensor,
    RealPart<M>: Tensor,
{
    let mut x = normalize(start);
    for _ in 0..iterations {
        x = normalize(a.clone().matmul(x));
    }
    x
}

/// Scale a rank-2 column to unit 2-norm: `x / sqrt(Σ|xᵢ|²)`.
fn normalize<M>(x: M) -> M
where
    M: MatrixTensor,
    RealPart<M>: Tensor,
{
    let norm = x.norm_sqr().sum(&[0, 1]).sqrt();
    x / M::from_real(norm)
}

/// The leading axis length of a tensor — the `n` of an `n x n` matrix.
///
/// Returns `0` for a rank-0 tensor, which makes [`identity`] produce an empty matrix rather
/// than panicking on a shape this contract cannot otherwise reject.
fn leading_dim<M>(a: &M) -> usize
where
    M: Tensor,
{
    a.shape().first().copied().unwrap_or(0)
}
