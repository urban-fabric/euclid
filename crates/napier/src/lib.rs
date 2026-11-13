//! # napier — the abstract-algebra tower + the concrete `Copy` numeric tier.
//!
//! napier owns the two abstract foundations that meet on the concrete number line:
//! the **algebra** tower (`Field` → `OrderedField` / `AlgebraicallyClosedField`,
//! plus the semiring/dioid stack and the exact carriers `Rational`/`Zn`/
//! `GaussianRational`; see [`algebra`]) and, fused with euclid's **numeric**
//! analysis surface (`Real` / `Complex` / `NormedDivisionAlgebra`, imported here
//! under their former `LazyReal` / `LazyComplex` names so they do not collide
//! with this crate's own concrete traits), the
//! concrete `Copy` *number* on the real / complex line:
//!
//! - [`Real`] = [`algebra::OrderedField`] + `LazyReal` + `Copy`, plus the associated
//!   constants (`PI`/`EPSILON`/`INFINITY`/…) and the `bool`-returning predicates
//!   (`is_finite`/`is_nan`/`is_sign_negative`). Implemented by `f64`/`f32`.
//! - [`Complex`] = [`algebra::AlgebraicallyClosedField`] + `LazyComplex<RealPart: Real>` +
//!   `Copy`, plus the complex transcendentals. Implemented by
//!   `jacobi::Complex{32,64}`.
//!
//! A `Tensor` / columnar `Expr` backend stays purely on euclid's lazy tier (no
//! `Copy`, no scalar `bool`); the eager geometry / matrix / physics kernels that
//! genuinely need the real number line bound on napier's concrete tier.

pub mod algebra;
mod complex;
mod real;

pub use algebra::{GaussianRational, Integer, Rational, Rational64, Rational128, Zn, Zp};
pub use complex::Complex;
pub use real::Real;
