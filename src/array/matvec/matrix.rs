use super::vector::Vector;
use crate::numeric::{Compare, NormedDivisionAlgebra};

/// **Rank 2.** Dense linear algebra over a scalar field (real or complex), centered on
/// Hermitian-positive-definite results. The trait exposes stable operations
/// (`solve_spd`, `logdet_spd`, `mahalanobis_squared`) and deliberately has no `inverse`.
pub trait Matrix: Clone {
    /// The element scalar — real or complex.
    type Scalar: NormedDivisionAlgebra;
    /// The companion vector type (`Self` for implementations where tensors carry all ranks).
    type Vector: Vector<Scalar = Self::Scalar>;

    /// `(rows, cols)`.
    fn shape(&self) -> (usize, usize);

    /// The `n×n` identity matrix.
    fn identity(n: usize) -> Self;
    /// A matrix filled with `value`.
    fn full(rows: usize, cols: usize, value: Self::Scalar) -> Self;
    /// An all-zeros `rows×cols` matrix.
    fn zeros(rows: usize, cols: usize) -> Self;
    /// A matrix with the same shape as `self`, filled with `value`.
    fn full_like(&self, value: Self::Scalar) -> Self {
        let (rows, cols) = self.shape();
        Self::full(rows, cols, value)
    }
    /// An all-zeros matrix with the same shape as `self`.
    fn zeros_like(&self) -> Self {
        let (rows, cols) = self.shape();
        Self::zeros(rows, cols)
    }

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

    // --- HPD quantities ---
    /// The lower Cholesky factor `L` with `Σ = L·Lᴴ` (`L·Lᵀ` for real `Σ`).
    fn cholesky(self) -> Self;
    /// `Σ⁻¹ B` for Hermitian-PD `Σ` — no explicit inverse is formed.
    fn solve_spd(self, rhs: Self) -> Self;
    /// `Σ⁻¹ b`, the single-vector analogue of [`solve_spd`](Self::solve_spd). Its own
    /// primitive because `Self` and `Self::Vector` differ; the Gaussian *mean* family
    /// (`Σ⁻¹µ`, GP/conditional means) needs it.
    fn solve_spd_vec(self, rhs: Self::Vector) -> Self::Vector;
    /// `ln det Σ` for Hermitian-PD `Σ` — real (an HPD determinant is a positive real).
    fn logdet_spd(self) -> <Self::Scalar as NormedDivisionAlgebra>::RealPart;
    /// The squared Mahalanobis distance `δᴴ Σ⁻¹ δ` — real even over ℂ (a Hermitian form).
    fn mahalanobis_squared(&self, delta: &Self::Vector) -> <Self::Scalar as NormedDivisionAlgebra>::RealPart;

    /// Alias for [`where_cond`](Self::where_cond), named like an ordinary conditional
    /// expression.
    fn if_then_else(mask: &<Self as Compare>::Mask, on_true: Self, on_false: Self) -> Self
    where
        Self: Compare,
    {
        <Self as Compare>::if_then_else(mask, on_true, on_false)
    }
}

/// Explicit graph control-flow contract for matrix carriers.
pub trait MatrixControlFlow: Matrix + Compare {
    /// Graph-level conditional over matrix values.
    fn cond<T, F>(
        pred: &Self::Mask,
        true_arg: Self,
        true_branch: T,
        false_arg: Self,
        false_branch: F,
    ) -> Self
    where
        T: FnOnce(Self) -> Self,
        F: FnOnce(Self) -> Self;

    /// Graph-level while loop over matrix state.
    fn while_loop<C, B>(init: Self, cond: C, body: B) -> Self
    where
        C: Fn(&Self) -> Self::Mask,
        B: Fn(Self) -> Self;
}

/// Dense matrix decompositions and the general (non-SPD) solve.
///
/// These return mathematical *factors* — QR, Hermitian eigen, singular-value, and LU — in the
/// same "expose the result, not an `inverse`" spirit as [`Matrix`]. They sit outside the
/// base trait because not every carrier supplies dense factorizations. euclid declares the
/// contract; implementations provide the numerics.
/// Real-valued spectra (eigenvalues, singular values) come back as a [`Vector`](Matrix::Vector).
pub trait MatrixDecompose: Matrix {
    /// Reduced QR factorization `A = Q·R`, with orthonormal `Q` and upper-triangular `R`.
    fn qr(self) -> (Self, Self);

    /// Hermitian eigendecomposition `A = V·diag(λ)·Vᴴ`, as `(eigenvalues, eigenvectors)` —
    /// the eigenvalues real and ascending, `V` orthonormal.
    fn eigh(self) -> (Self::Vector, Self);

    /// Singular-value decomposition `A = U·diag(σ)·Vᴴ`, as `(U, singular_values, V)`.
    fn svd(self) -> (Self, Self::Vector, Self);

    /// LU factorization with partial pivoting: `(L, U, P)` with `P·A = L·U`.
    fn lu(self) -> (Self, Self, Self);

    /// Solve the general linear system `A·X = rhs` — no positive-definiteness assumed (cf.
    /// [`solve_spd`](Matrix::solve_spd)).
    fn solve(self, rhs: Self) -> Self;

    /// Solve `A·x = b` for a single vector `b`, the [`solve`](Self::solve) analogue for
    /// [`Vector`](Matrix::Vector).
    fn solve_vec(self, rhs: Self::Vector) -> Self::Vector;
}
