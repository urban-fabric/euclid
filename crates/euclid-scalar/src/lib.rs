//! # euclid-scalar — the concrete `Copy` number line.
//!
//! Where euclid's two abstract foundations meet on an actual machine number:
//! `euclid-algebra`'s **structure** (which tier a carrier reaches) and
//! `euclid-numeric`'s **analysis** (what operations a formula may use). Fusing them and adding
//! `Copy` yields the ordinary real and complex number — implemented by `f64`/`f32`.
//!
//! - [`Real`] = [`euclid_algebra::OrderedField`] + [`euclid_numeric::Real`] + `Copy`, plus the
//!   associated constants (`PI`/`EPSILON`/`INFINITY`/…) and the `bool`-returning predicates
//!   (`is_finite`/`is_nan`/`is_sign_negative`).
//! - [`Complex`] = [`euclid_algebra::AlgebraicallyClosedField`] + [`euclid_numeric::Complex`]
//!   with a [`Real`] real part + `Copy`, plus the complex transcendentals. Implemented by
//!   `jacobi::Complex{32,64}`.
//!
//! ## When to bind this instead of `euclid-numeric`
//!
//! Prefer [`euclid_numeric::Real`] — it is the wider bound, and a formula written against it
//! runs on expression handles, graph nodes and tensor elements as well as on `f64`.
//!
//! Bind *this* crate only when a kernel genuinely needs the real number line itself: `Copy`,
//! host `bool` from a predicate, and associated constants rather than lifted literals. Eager
//! geometry, matrix and physics kernels qualify. A `Tensor` or columnar `Expr` backend does not
//! — it stays entirely on euclid-numeric's contract tier and never depends on this crate.
//!
//! ## Why it is a separate crate
//!
//! Until 0.4 this tier lived inside `euclid-algebra`, which therefore owned two goals at once:
//! the abstract tower and the concrete number. The symptom was a name collision — it had to
//! import euclid's `Real`/`Complex` as `LazyReal`/`LazyComplex` to keep them apart from its own
//! same-named traits, so the crate documented the workaround instead of the boundary.
//!
//! Splitting them removes the aliases. `euclid_numeric::Real` is the contract,
//! `euclid_scalar::Real` is the concrete number, and the distinction is now carried by the
//! crate name rather than by an import rename. It also leaves `euclid-algebra` with no
//! dependencies at all: algebraic structure needs nothing from analysis, and this crate is the
//! only one that needs both.

mod complex;
mod real;

pub use complex::Complex;
pub use real::Real;

// The README's examples, compiled.
//
// `cfg(doctest)` means this item exists only while rustdoc is collecting doctests: it is absent
// from every real build, adds nothing to the public surface, and never appears in the rendered
// docs. What it buys is that `cargo test --doc` compiles the code blocks in README.md itself.
//
// A crate whose deliverable is its trait surface has a README whose example *is* part of that
// surface, and a README example that no longer compiles is worse than none — it documents a
// signature the crate no longer has. Two crates used to check this by hand-copying the example
// into `tests/readme_check.rs`, which compiles the copy and lets the original drift. This checks
// the original.
//
// Illustrative fragments in the README are marked ```rust,ignore, which is the honest marking:
// they reference names that are deliberately undefined.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeExamples;
