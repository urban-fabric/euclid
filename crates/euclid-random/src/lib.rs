//! # euclid-random
//!
//! Small explicit-key random sampler contracts.
//!
//! Randomness is modeled with explicit immutable keys. A draw is a pure function of its key,
//! distribution parameters, and result shape: using the same key twice produces the same value.
//! New independent randomness is obtained by [`RandomKey::split`] or [`RandomKey::fold_in`],
//! which keeps the contract reproducible and friendly to eager, dataframe, and graph backends.
//!
//! The sampler surface is intentionally small: uniform, normal, Bernoulli, and integer draws over
//! a host-known `shape`. Scalar backends use `&[]`; array/dataframe backends can interpret
//! non-empty shapes in their native way.

mod key;
mod sampler;

#[cfg(test)]
mod tests;

pub use key::RandomKey;
pub use sampler::{CountSampler, GammaSampler, RandomSampler};
