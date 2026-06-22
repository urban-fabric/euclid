use super::RealPart;
use crate::numeric::{Compare, Element, NormedDivisionAlgebra};

/// **Rank 1.** A vector over a scalar carrier (real or complex): the vector-space and
/// inner-product surface (elementwise transcendentals belong to [`Tensor`](crate::array::Tensor)).
pub trait Vector: Clone {
    /// The element scalar — real or complex.
    type Scalar: NormedDivisionAlgebra;

    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// A vector filled with `value`.
    fn full(len: usize, value: Self::Scalar) -> Self;
    /// The all-zeros vector.
    fn zeros(len: usize) -> Self;
    /// A vector with the same length as `self`, filled with `value`.
    fn full_like(&self, value: Self::Scalar) -> Self;
    /// An all-zeros vector with the same length as `self`.
    fn zeros_like(&self) -> Self;
    fn add(self, other: Self) -> Self;
    fn sub(self, other: Self) -> Self;
    fn scale(self, factor: Self::Scalar) -> Self;
    /// Elementwise (Hadamard) product.
    fn hadamard(self, other: Self) -> Self;

    /// The Hermitian inner product `Σ conj(aᵢ)·bᵢ` (the ordinary `Σ aᵢ·bᵢ` for real vectors).
    fn dot(&self, other: &Self) -> Self::Scalar;
    fn sum(&self) -> Self::Scalar;
    /// `Σ |aᵢ|²` — real even over ℂ.
    fn norm_squared(&self) -> RealPart<Self::Scalar>;
    /// `√(Σ |aᵢ|²)` — real even over ℂ.
    fn norm(&self) -> RealPart<Self::Scalar>;

    /// Elementwise vector conditional: `mask ? on_true : on_false`.
    ///
    /// This is the vector-level spelling of [`Compare::if_then_else`]. Implementations
    /// with vector masks lower this to an elementwise select operation.
    fn if_then_else(mask: &<Self as Compare>::Mask, on_true: Self, on_false: Self) -> Self
    where
        Self: Compare,
    {
        <Self as Compare>::if_then_else(mask, on_true, on_false)
    }
}

/// Explicit graph control-flow contract for vector carriers.
pub trait VectorControlFlow: Vector + Compare {
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
