//! The rank-n tensor interface: [`Tensor`] and its opt-in capability extensions.
//!
//! A tensor interacts only with other tensors and with the scalar [`numeric`](crate::numeric)
//! surface — never with the rank-1/2 `matvec` contracts — so it stands on its own here. The
//! `base` module holds the structural core (shape, indexing, reductions, contraction, control
//! flow); the `extensions` module holds the statistical, sorting, contraction, indexing, and
//! structural capability traits an implementation may add on top.

mod base;
mod extensions;

pub use base::{Tensor, TensorControlFlow};
pub use extensions::{
    FftKind, TensorContract, TensorConv, TensorFft, TensorIndexing, TensorReduce, TensorSort,
    TensorTransform, TensorWindow,
};
