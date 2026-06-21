//! # euclid — domain-neutral numeric foundations
//!
//! The backend-neutral contract layer for the urban-fabric math stack: trait contracts
//! that let a numeric formula be written **once** and run on many backends. The same
//! Gaussian log-density or Cholesky solve evaluates eagerly on `f64`/`f32` today and
//! lowers to a tensor compiler (XLA, via `maxwell`) tomorrow, unchanged.
//!
//! The crate is pure trait contracts plus the `f64`/`f32` reference scalars, with no
//! external dependencies. The concrete `Copy` number tier, the abstract-algebra tower,
//! and physical units live in sibling crates — see [Ecosystem](#ecosystem).
//!
//! Every contract is *lazy* — `Clone`, never `Copy`, and free of anything a tensor
//! backend cannot produce: a comparison returns a **mask**, not a `bool`; a constant is
//! lifted with [`LazyReal::scalar`], not read from a `const`; piecewise logic uses the
//! comparison-free [`min`](LazyReal::min)/[`max`](LazyReal::max)/[`clamp`](LazyReal::clamp)
//! selectors. A formula staying inside this surface lowers op-for-op (`exp` → `Exp`,
//! `dot` → `DotGeneral`, …).
//!
//! ## The two tiers
//!
//! - **Scalar surface** ([`numeric`]) — the analysis and algebra of one number, graded by
//!   structure: [`LazySemiring`] (`+ *`, the naturals via [`LazyNatural`]) refined by
//!   [`LazyRing`] (`+ - *`), in turn refined by [`Arithmetic`] (adds `/`) and
//!   [`LazyInteger`]; [`LazyReal`] (transcendentals, constants, selectors, special
//!   functions, and stable scalar log-domain helpers); [`NormedDivisionAlgebra`] over
//!   ℝ/ℂ/ℍ, refined by [`LazyComplex`] and [`LazyQuaternion`]; and [`LazyBool`] +
//!   [`LazyCompare`], the mask tier.
//! - **Array contracts** ([`array`](mod@array)) — the rank-graded peers: [`LazyVector`]
//!   (rank 1), [`LazyMatrix`] (rank 2: matmul + the SPD quantities), [`LazyTensor`]
//!   (rank n: shape, reshape, reductions, contraction). [`LazyTensor`] sits on
//!   [`LazyRing`]; [`LazyVector`]/[`LazyMatrix`] stay over a [`NormedDivisionAlgebra`].
//!
//! [`random`] adds backend-neutral random-source contracts ([`RandomStream`],
//! [`KeyedRandomBackend`]), covering both host-PRNG and keyed/counter-based styles.
//!
//! ## Example — one formula, written against the contract
//!
//! ```
//! use euclid::LazyReal;
//!
//! // Standard-normal log-density, generic over `T: LazyReal` — the same code runs on a
//! // concrete `f64` today and on a tensor scalar tomorrow.
//! fn standard_normal_logpdf<T: LazyReal>(x: T) -> T {
//!     let x2 = x.clone() * x;
//!     -(T::half() * x2) - T::half_ln_tau()
//! }
//!
//! let lp = standard_normal_logpdf(0.0_f64);
//! assert!((lp - (-0.5 * (2.0 * std::f64::consts::PI).ln())).abs() < 1e-12);
//! ```
//!
//! In-crate, `f64`/`f32` implement the real traits,
//! `i64`/`i128` implement [`LazyInteger`],
//! `u32`/`u64`/`u128` implement [`LazyNatural`],
//! and `bool` implements [`LazyBool`].

pub mod array;
pub mod numeric;
pub mod random;

pub use array::{LazyMatrix, LazyTensor, LazyVector};
pub use numeric::{
    Arithmetic, LazyBool, LazyCompare, LazyComplex, LazyInteger, LazyNatural, LazyQuaternion,
    LazyReal, LazyRing, LazySemiring, NormedDivisionAlgebra,
};
pub use random::{
    KeyedRandomBackend, RandomAlgorithm, RandomProcess, RandomStream, SampleWith,
    TensorRandomStream,
};
