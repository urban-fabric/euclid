//! JAX-style random-source contracts over the numeric and array traits.
//!
//! Randomness is modeled with explicit immutable keys. A draw is a pure function of its key,
//! distribution parameters, and result shape: using the same key twice produces the same value.
//! New independent randomness is obtained by [`KeyedRandomBackend::split`] or
//! [`KeyedRandomBackend::fold_in`], which keeps the contract reproducible and friendly to
//! graph/XLA backends.
//!
//! Batched key semantics should be map-like: drawing from a batch of keys is equivalent to drawing
//! once per key and stacking the results. Backends should not expose implementation details that
//! make the first key drive an entire batch.

mod dtype;
mod key;
mod matvec;
mod process;
mod scalar;
mod tensor;

pub use dtype::{RandomAlgorithm, RandomDType};
pub use key::{GraphRandomBackend, KeyedRandomBackend};
pub use matvec::{KeyedRandomMatVec, KeyedRandomMatVecDistributions};
pub use process::{KeyedRandomProcess, SampleWithKey};
pub use scalar::{KeyedRandomScalar, KeyedRandomScalarDistributions};
pub use tensor::{GraphRandomTensor, KeyedRandomTensor, KeyedRandomTensorDistributions};
