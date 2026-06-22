//! # euclid
//!
//! Dependency-free numeric contracts for carrier-generic Rust math.
//!
//! A formula written against these traits can run on ordinary `f64`/`f32` values and on
//! consumer-defined expression, graph, tensor, or unit-aware carriers. euclid defines the
//! operations a formula may use; it does not define storage layout, execution runtime, array
//! allocation, or a random-number engine.
//!
//! The crate includes `f64`/`f32`, integer, natural, and `bool` implementations, plus
//! physical-unit wrappers over any [`Real`] carrier.
//!
//! ## Usage Rules
//!
//! - Choose the narrowest trait bound that contains the operations a formula needs.
//! - Treat generic values as `Clone`, not `Copy`.
//! - Lift host literals with methods such as [`Real::scalar`] and
//!   [`Real::scalar_like`].
//! - Comparisons return masks, consumed by [`Compare::select`] or
//!   [`Compare::if_then_else`].
//! - Use matrix operations such as `solve_spd`, `logdet_spd`, and
//!   `mahalanobis_squared` instead of forming an inverse.
//!
//! ## Modules
//!
//! - [`numeric`] defines scalar algebra, real analysis, interpolation, masks, and special
//!   functions.
//! - [`array`](mod@array) defines [`Vector`], [`Matrix`], and [`Tensor`].
//! - [`random`] defines JAX-style keyed random-source contracts.
//! - [`units`] defines physical quantities over any [`Real`] carrier.
//!
//! ## Example
//!
//! ```
//! use euclid::Real;
//!
//! // Standard-normal log-density over any real carrier.
//! fn standard_normal_logpdf<T: Real>(x: T) -> T {
//!     let x2 = x.square();
//!     -(T::half() * x2) - T::half_ln_tau()
//! }
//!
//! let lp = standard_normal_logpdf(0.0_f64);
//! assert!((lp - (-0.5 * (2.0 * std::f64::consts::PI).ln())).abs() < 1e-12);
//! ```
//!
//! In this crate, `f64`/`f32` implement the real and interpolation traits;
//! `i32`/`i64`/`i128` implement [`Integer`]; `u32`/`u64`/`u128` implement
//! [`Natural`]; and `bool` implements [`Boolean`].

pub mod array;
pub mod control_flow;
pub mod execute;
pub mod numeric;
pub mod random;
pub mod spec;
pub mod units;

pub use array::{Matrix, Tensor, Vector};
pub use control_flow::ControlFlow;
pub use execute::{Compile, Compiled, Parameter, Traced, TypedParameter};
pub use numeric::{
    Arithmetic, Boolean, Compare, Complex, Element, InnerProductSpace, Integer, Interpolate,
    Natural, NormedDivisionAlgebra, NormedVectorSpace, Quaternion, Real, Ring, Semiring,
    VectorSpace, sample, sample_at,
};
pub use random::{
    GraphRandomBackend, GraphRandomTensor, KeyedRandomBackend, KeyedRandomMatVec,
    KeyedRandomMatVecDistributions, KeyedRandomProcess, KeyedRandomScalar,
    KeyedRandomScalarDistributions, KeyedRandomTensor, KeyedRandomTensorDistributions,
    RandomAlgorithm, RandomDType, SampleWithKey,
};
pub use spec::{Cast, DType, HasTensorSpec, TensorSpec};
pub use units::{
    Angle, Dimension, Km, LinearQuantity, Meters, Quantity, ScalarQuantity, Seconds, SpeedMs,
    SquareMeters,
};
