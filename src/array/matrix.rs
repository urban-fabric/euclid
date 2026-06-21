use super::{RealPart, control_flow, vector::LazyVector};
use crate::numeric::{LazyCompare, NormedDivisionAlgebra};

/// **Rank 2.** Dense linear algebra over a scalar field (real or complex), centered on
/// Hermitian-positive-definite results. The trait exposes stable operations
/// (`solve_spd`, `logdet_spd`, `mahalanobis_squared`) and deliberately has no `inverse`.
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
    /// A matrix filled with `value`.
    fn full(rows: usize, cols: usize, value: Self::Scalar) -> Self;
    /// An all-zeros `rows×cols` matrix.
    fn zeros(rows: usize, cols: usize) -> Self {
        Self::full(rows, cols, Self::Scalar::field_zero())
    }
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
    fn logdet_spd(self) -> RealPart<Self::Scalar>;
    /// The squared Mahalanobis distance `δᴴ Σ⁻¹ δ` — real even over ℂ (a Hermitian form).
    fn mahalanobis_squared(&self, delta: &Self::Vector) -> RealPart<Self::Scalar>;

    // --- matrix control flow ------------------------------------------------

    /// Elementwise matrix conditional: `mask ? on_true : on_false`.
    ///
    /// This is the matrix-level spelling of [`LazyCompare::select`]. Backends
    /// with matrix masks lower this to an elementwise select/where operation.
    fn where_cond(mask: &<Self as LazyCompare>::Mask, on_true: Self, on_false: Self) -> Self
    where
        Self: LazyCompare,
    {
        control_flow::where_cond(mask, on_true, on_false)
    }

    /// Alias for [`where_cond`](Self::where_cond), named like an ordinary conditional
    /// expression.
    fn if_then_else(mask: &<Self as LazyCompare>::Mask, on_true: Self, on_false: Self) -> Self
    where
        Self: LazyCompare,
    {
        Self::where_cond(mask, on_true, on_false)
    }
}

/// Explicit graph control-flow contract for matrix carriers.
pub trait LazyMatrixControlFlow: LazyMatrix + LazyCompare {
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
