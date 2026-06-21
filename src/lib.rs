//! # euclid — domain-neutral numeric foundations
//!
//! euclid is the **backend-neutral contract layer** for the urban-fabric math stack:
//! a set of trait *contracts* — and nothing more — that let a numeric formula be
//! written **once** and run on many backends. The same Gaussian log-density,
//! Cholesky solve, or Mahalanobis distance you write against these traits evaluates
//! eagerly on `f64`/`f32` today and lowers to a tensor compiler (XLA, via `maxwell`)
//! tomorrow, with no change to the formula.
//!
//! The crate is **pure trait contracts plus the `f64`/`f32` reference scalars**, and
//! it has **no external dependencies**. The concrete `Copy` number tier (`Real` /
//! `Complex`), the abstract-algebra tower (semiring / dioid / ring / field + the exact
//! carriers `Rational` / `Zn` / `GaussianRational`), and physical units live in sibling
//! crates — see [Ecosystem](#ecosystem).
//!
//! ## The idea: lazy, write-once numerics
//!
//! Every contract here is *lazy* — `Clone`, never `Copy`, and free of any operation a
//! tensor backend cannot produce. A comparison returns a **mask**, not a `bool`; a
//! constant is *lifted* with [`LazyReal::scalar`] rather than read from an associated
//! `const`; piecewise logic is expressed with comparison-free [`min`](LazyReal::min) /
//! [`max`](LazyReal::max) / [`clamp`](LazyReal::clamp) selectors. A formula that stays
//! inside this surface is therefore one a tensor compiler can lower op-for-op
//! (`exp` → `Exp`, `erf` → `Erf`, `dot` → `DotGeneral`, …).
//!
//! ## The two tiers
//!
//! - **Scalar surface** ([`numeric`]) — the elementwise *analysis* and *algebra* of a
//!   single number, graded by structure:
//!   - [`LazyRing`] (`+ - *`, no division) refined by [`Arithmetic`] (adds `/`) and
//!     [`LazyInteger`] (the `s64`-style integer surface);
//!   - [`LazyReal`] — the real transcendentals, constants, selectors, and special
//!     functions (`exp`/`ln`/`sqrt`, trig, `erf`/`ln_gamma`/`beta_reg`/`bessel_i0e`, …);
//!   - [`NormedDivisionAlgebra`] over ℝ — ℝ/ℂ/ℍ, refined by [`LazyComplex`] and
//!     [`LazyQuaternion`], naming `conjugate` / `magnitude` / the real part;
//!   - [`LazyBool`] + [`LazyCompare`] — the mask tier a comparison produces and
//!     `select` consumes.
//! - **Array contracts** ([`array`](mod@array)) — the rank-graded peers of the scalar surface:
//!   [`LazyVector`] (rank 1), [`LazyMatrix`] (rank 2: matmul + the SPD quantities
//!   `solve_spd` / `logdet_spd` / `mahalanobis_squared`), and [`LazyTensor`] (rank n:
//!   shape, reshape, reductions, contraction). The structural [`LazyTensor`] sits on
//!   [`LazyRing`] (real/complex/**integer** alike); the linear-algebra
//!   [`LazyVector`]/[`LazyMatrix`] stay over a [`NormedDivisionAlgebra`] field.
//!
//! [`random`] adds backend-neutral random-source contracts ([`RandomStream`],
//! [`KeyedRandomBackend`]) over those numeric/array types, covering both host-PRNG and
//! keyed/counter-based (ThreeFry/Philox) styles.
//!
//! ## Example — one formula, written against the contract
//!
//! ```
//! use euclid::LazyReal;
//!
//! /// Standard-normal log-density `−½x² − ½·ln(2π)`, written once over the lazy
//! /// real surface. It is generic over `T: LazyReal`, so the *same* code runs on a
//! /// concrete `f64` today and on a tensor scalar tomorrow.
//! fn standard_normal_logpdf<T: LazyReal>(x: T) -> T {
//!     let x2 = x.clone() * x;
//!     -(T::half() * x2) - T::half_ln_tau()
//! }
//!
//! // Evaluated eagerly here because `f64: LazyReal`.
//! let lp = standard_normal_logpdf(0.0_f64);
//! assert!((lp - (-0.5 * (2.0 * std::f64::consts::PI).ln())).abs() < 1e-12);
//! ```
//!
//! ## Ecosystem
//!
//! euclid owns the contracts; the implementations live downstream:
//!
//! - **`napier`** — the concrete `Copy` number tier (`Real` / `Complex`) and the
//!   abstract-algebra tower, formed by fusing this lazy surface with the algebraic field.
//! - **`frobenius`** — backend-generic linear-algebra algorithms and the richer matrix
//!   extension traits (`SpdFactor`, general decompositions) over [`LazyMatrix`].
//! - **`maxwell`** — the XLA / StableHLO backend: a euclid-contract implementation that
//!   compiles formulas to HLO so the write-once kernels run on accelerators.
//! - **`oresme`** — physical units, quantities, and conversions.
//! - **`bayes`** — closed-form probability and statistics written against this surface.
//!
//! In-crate, `f64`/`f32` implement the real traits, `i64`/`i128` implement
//! [`LazyInteger`], and `bool` implements [`LazyBool`]; the array and random contracts
//! are implemented by the backends above.

pub mod array;
pub mod numeric;
pub mod random;

pub use array::{LazyMatrix, LazyTensor, LazyVector};
pub use numeric::{
    Arithmetic, LazyBool, LazyCompare, LazyComplex, LazyInteger, LazyQuaternion, LazyReal,
    LazyRing, NormedDivisionAlgebra,
};
// The special functions live under `numeric::special`; this root alias keeps the
// historical `euclid::special` path working for existing consumers.
pub use numeric::special;
pub use random::{
    KeyedRandomBackend, RandomAlgorithm, RandomProcess, RandomStream, SampleWith,
    TensorRandomStream,
};
