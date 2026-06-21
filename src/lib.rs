//! # euclid
//!
//! Domain-neutral numeric foundations for the urban-fabric math stack.
//!
//! euclid defines lazy trait contracts for scalar, array, random, and unit-aware math.
//! It has no external dependencies. The crate includes the contracts, the `f64`/`f32`
//! reference scalar implementations, integer/natural/bool scalar carriers, and lazy
//! physical-unit wrappers.
//!
//! The contracts are intentionally small:
//!
//! - Values are `Clone`, not `Copy`.
//! - Constants are lifted with methods such as [`LazyReal::scalar`].
//! - Comparisons return masks, consumed by [`LazyCompare::select`] or
//!   [`LazyCompare::if_then_else`].
//! - Matrix traits expose stable mathematical results such as `solve_spd` and
//!   `logdet_spd`, not `inverse`.
//! - Each trait names the structure it requires; code should use the narrowest bound that
//!   contains the operations it needs.
//!
//! ## Modules
//!
//! - [`numeric`] defines scalar algebra, real analysis, masks, and special functions.
//! - [`array`](mod@array) defines [`LazyVector`], [`LazyMatrix`], and [`LazyTensor`].
//! - [`random`] defines stream-shaped and keyed random-source contracts.
//! - [`units`] defines lazy physical quantities over any [`LazyReal`] carrier.
//!
//! ## Example
//!
//! ```
//! use euclid::LazyReal;
//!
//! // Standard-normal log-density, generic over the lazy real surface.
//! fn standard_normal_logpdf<T: LazyReal>(x: T) -> T {
//!     let x2 = x.clone() * x;
//!     -(T::half() * x2) - T::half_ln_tau()
//! }
//!
//! let lp = standard_normal_logpdf(0.0_f64);
//! assert!((lp - (-0.5 * (2.0 * std::f64::consts::PI).ln())).abs() < 1e-12);
//! ```
//!
//! In this crate, `f64`/`f32` implement the real traits; `i32`/`i64`/`i128`
//! implement [`LazyInteger`]; `u32`/`u64`/`u128` implement [`LazyNatural`]; and
//! `bool` implements [`LazyBool`].

pub mod array;
pub mod numeric;
pub mod random;
pub mod units;

pub use array::{LazyMatrix, LazyTensor, LazyVector};
pub use numeric::{
    Arithmetic, LazyBool, LazyCompare, LazyComplex, LazyInteger, LazyNatural, LazyQuaternion,
    LazyReal, LazyRing, LazySemiring, NormedDivisionAlgebra,
};
pub use random::{
    KeyedRandomBackend, RandomAlgorithm, RandomProcess, RandomStream, SampleWith,
    TensorRandomStream,
};
pub use units::{
    Dimension, LazyKm, LazyLinearQuantity, LazyMeters, LazyQuantity, LazyScalarQuantity,
    LazySeconds, LazySpeedMs, LazySquareMeters,
};
