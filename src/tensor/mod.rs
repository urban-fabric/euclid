//! The rank-n tensor contract: [`Tensor`] and its opt-in capability extensions.
//!
//! euclid's array surface is a single rank-graded type, the [`Tensor`]. A vector is a rank-1
//! tensor and a matrix a rank-2 tensor; there are no separate `Vector`/`Matrix` contracts. A
//! tensor interacts only with other tensors and with the scalar [`numeric`](crate::numeric)
//! surface beneath it.
//!
//! The contract is layered:
//!
//! - **Structural core**: [`Tensor`] starts at [`Clone`], so shape, reshape, indexing/gather,
//!   and structural ops work for real, complex, integer, natural, and boolean tensors alike.
//!   Arithmetic methods add the bounds they need: [`sum`](Tensor::sum) and
//!   [`matmul`](Tensor::matmul) require [`Ring`](crate::numeric::Ring),
//!   [`prod`](Tensor::prod) requires [`Semiring`](crate::numeric::Semiring),
//!   [`max`](Tensor::max)/[`min`](Tensor::min) require [`Real`](crate::numeric::Real),
//!   [`mean`](Tensor::mean) requires
//!   [`NormedDivisionAlgebra`](crate::numeric::NormedDivisionAlgebra), and
//!   [`any`](Tensor::any)/[`all`](Tensor::all) require [`Boolean`](crate::numeric::Boolean).
//! - **Graph control flow**: [`TensorControlFlow`] is the explicit contract for carriers that
//!   can represent conditionals and loops.
//! - **Capability extensions**: opt-in traits that grow the surface an implementation can offer
//!   without widening the core. [`TensorReduce`] adds statistical/scan reductions (variance,
//!   log-sum-exp, softmax, cumulative sum); [`TensorSort`] adds sorting and the
//!   argmin/argmax/top-k index family; [`TensorContract`] adds general dot/einsum contraction;
//!   [`TensorIndexing`] adds range/one-hot/histogram encoding; and [`TensorTransform`] adds
//!   roll/tile/clip.
//!
//! The `base` module holds the structural core; the `extensions` module holds the capability
//! traits. Axis naming, mask representation, broadcast/gather/scatter details, and precision
//! policy are outside this contract.

mod base;
mod extensions;

#[cfg(test)]
mod tests;

pub use base::{Tensor, TensorControlFlow};
pub use extensions::{
    FftKind, TensorContract, TensorConv, TensorFft, TensorIndexing, TensorReduce, TensorSort,
    TensorTransform, TensorWindow,
};
