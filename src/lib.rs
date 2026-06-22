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
//!
//! ## Modules
//!
//! - [`numeric`] defines scalar algebra, real analysis, masks, and special functions.
//! - [`interpolate`] defines scalar-parameterized interpolation (`lerp`/`midpoint`).
//! - [`tensor`](mod@tensor) defines the rank-graded [`Tensor`] (a vector is a rank-1 tensor, a
//!   matrix a rank-2 tensor).
//! - [`random`] defines explicit-key random sampler contracts.
//! - [`units`] defines physical-dimension contracts over any [`Real`] carrier — the seven SI
//!   base dimensions (`Length`, `Mass`, `Time`, `ElectricCurrent`, `Temperature`,
//!   `AmountOfSubstance`, `LuminousIntensity`) and the standard derived set (`Force`, `Energy`,
//!   `Power`, `Pressure`, `Voltage`, `Resistance`, …), wired together by their cross-dimensional
//!   algebra.
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

pub mod control_flow;
pub mod interpolate;
pub mod numeric;
pub mod random;
pub mod spec;
pub mod tensor;
pub mod units;

pub use control_flow::ControlFlow;
pub use interpolate::Interpolate;
pub use numeric::{
    Arithmetic, Boolean, Compare, Complex, Element, Integer, Natural, NormedDivisionAlgebra,
    Quaternion, Real, Ring, Semiring,
};
pub use random::{RandomKey, RandomSampler};
pub use spec::{Cast, DType, HasTensorSpec, TensorSpec};
pub use tensor::Tensor;
pub use units::{
    Angle, Area, Dimension, Length, LinearQuantity, Quantity, ScalarQuantity, Speed, Time,
};
