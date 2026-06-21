use super::{RealPart, control_flow};
use crate::numeric::{LazyCompare, NormedDivisionAlgebra};

/// **Rank 1.** A vector over a lazy scalar (real or complex): the vector-space and
/// inner-product surface (elementwise transcendentals belong to [`LazyTensor`](super::LazyTensor)).
pub trait LazyVector: Clone {
    /// The element scalar — real or complex.
    type Scalar: NormedDivisionAlgebra;

    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    // --- constructors ---
    /// A vector filled with `value`.
    fn full(len: usize, value: Self::Scalar) -> Self;
    /// The all-zeros vector.
    fn zeros(len: usize) -> Self {
        Self::full(len, Self::Scalar::field_zero())
    }
    /// A vector with the same length as `self`, filled with `value`.
    fn full_like(&self, value: Self::Scalar) -> Self {
        Self::full(self.len(), value)
    }
    /// An all-zeros vector with the same length as `self`.
    fn zeros_like(&self) -> Self {
        Self::zeros(self.len())
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

    // --- vector control flow ------------------------------------------------

    /// Elementwise vector conditional: `mask ? on_true : on_false`.
    ///
    /// This is the vector-level spelling of [`LazyCompare::select`]. Backends
    /// with vector masks lower this to an elementwise select/where operation.
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

/// Explicit graph control-flow contract for vector carriers.
pub trait LazyVectorControlFlow: LazyVector + LazyCompare {
    /// Graph-level conditional over vector values.
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

    /// Graph-level while loop over vector state.
    fn while_loop<C, B>(init: Self, cond: C, body: B) -> Self
    where
        C: Fn(&Self) -> Self::Mask,
        B: Fn(Self) -> Self;
}
