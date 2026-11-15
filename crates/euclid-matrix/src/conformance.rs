//! Conformance suites — the matrix primitives as functions a **backend** can check itself
//! against.
//!
//! These differ from every other conformance module in euclid in one respect: this crate cannot
//! run them. The traits here are backend primitives — a Cholesky, a QR, an eigendecomposition —
//! so the only thing that can satisfy them is a concrete tensor, and euclid ships none on
//! purpose. The suites are written here and executed there.
//!
//! That is not a gap. The laws are exactly as agreement-critical as the signatures: two backends
//! that both implement [`SpdSolve`] must mean the same thing by it, or a kernel written once
//! against the contract silently computes different answers on each. Shipping the laws beside
//! the signatures is the only way to say what the signature *means* — the alternative is each
//! backend inventing its own reading, which is the failure the contract exists to prevent.
//!
//! ```ignore
//! // In a backend's test suite, where a concrete tensor exists:
//! use euclid_matrix::conformance;
//!
//! let spd = my_covariance_matrix();
//! let rhs = my_vector();
//! conformance::spd_solve_laws(&spd, &rhs, 1e-10, "ramanujan::Tensor");
//! ```
//!
//! # Every law is a residual
//!
//! There are no reference factorizations here, and there could not be: a QR is unique only up to
//! column signs, an eigenvector only up to scale, an SVD only up to sign pairs. Asserting a
//! particular `Q` would reject a correct backend that chose different signs.
//!
//! So each law checks what the factorization is *for* — that it reconstructs its input, or that
//! applying the solve undoes the system:
//!
//! - `solve_spd(Σ, Σ·X)` recovers `X`
//! - `Q·R` reconstructs `A`, and `Qᴴ·Q` is the identity
//! - `A·V` equals `V·diag(λ)`
//! - `U·diag(σ)·Vᴴ` reconstructs `A`
//! - `det(A·B) = det(A)·det(B)`
//!
//! Those hold for every correct implementation regardless of pivoting, blocking or sign
//! convention, which is what makes them laws rather than expectations.
//!
//! # Tolerance
//!
//! All of these are inexact — a factorization is a sequence of roundings — so every suite takes
//! a relative tolerance. Choose it from your carrier's precision and the matrix's condition
//! number, not from what makes the test pass: a residual that grows when a matrix is
//! ill-conditioned is physics, but one that grows on a well-conditioned matrix is a bug.
//!
//! # Availability
//!
//! Gated behind the `conformance` feature, so a consumer that only binds the signatures pays
//! nothing for it.
//!
//! These docs are the module's own (`//!`), with no outer `///` on the `mod` declaration: the
//! two forms are concatenated, and an outer comment resolves intra-doc links in the *parent's*
//! scope, so a link to a function in here would silently fail from there.

use euclid_numeric::{Compare, Element, Real};
use euclid_tensor::Tensor;

use crate::{
    Determinant, GeneralSolve, MatrixTensor, QrDecomposition, SpdSolve, TransposeMode, Triangle,
    TriangularSolve,
};

/// Assert that two tensors agree elementwise to within a relative tolerance.
///
/// Exposed because a backend writing its own laws over these traits wants the same residual
/// convention the suites use, rather than inventing a second one.
///
/// The criterion is `max|got − want| ≤ tolerance · max(1, max|want|)`, which degrades to
/// absolute near zero and to relative away from it — the same mixed convention `euclid-utils`
/// uses, for the same reason: a purely relative check is meaningless where a residual passes
/// through zero, which for a correct solve is most of it.
pub fn assert_close<M>(got: &M, want: &M, tolerance: f64, law: &str, what: &str)
where
    M: MatrixTensor,
    M::RealPart: Tensor,
    <M::RealPart as Compare>::Mask: PartialEq + core::fmt::Debug,
{
    let shape = got.shape();
    assert_eq!(shape, want.shape(), "{what}: {law} — shape mismatch");

    // Reduce over every axis by naming them, rather than relying on what an empty axis slice
    // means. `euclid-tensor` leaves that convention to each backend, so a portable law cannot
    // depend on it.
    let axes: Vec<usize> = (0..shape.len()).collect();

    let err = (got.clone() - want.clone()).norm_sqr().sum(&axes).sqrt();
    let scale = want.clone().norm_sqr().sum(&axes).sqrt();
    let one = <M::RealPart as Element>::one();
    let tol = <M::RealPart as Real>::scalar(tolerance);
    let bound = Real::max(scale, one) * tol;
    assert_eq!(
        err.le(&bound),
        <<M::RealPart as Compare>::Mask as Element>::one(),
        "{what}: {law} — residual exceeded tolerance {tolerance:e}"
    );
}

/// [`SpdSolve`]: applying the solve must undo the system.
///
/// Given a Hermitian-positive-definite `spd` and any `x`, forming `b = spd·x` and solving must
/// recover `x`. That is the only property a caller relies on, and it holds whatever factorization
/// the backend uses underneath.
///
/// `spd` must actually be HPD — the trait documents that as the caller's precondition and
/// implementors need not verify it, so a non-HPD input here produces a meaningless failure
/// rather than a useful one.
pub fn spd_solve_laws<M>(spd: &M, x: &M, tolerance: f64, what: &str)
where
    M: SpdSolve,
    M::RealPart: Tensor,
    <M::RealPart as Compare>::Mask: PartialEq + core::fmt::Debug,
{
    let b = spd.clone().matmul(x.clone());
    let solved = spd.clone().solve_spd(b);
    assert_close(&solved, x, tolerance, "solve_spd(Σ, Σ·x) = x", what);
}

/// [`GeneralSolve`]: the same round-trip for a nonsingular matrix.
///
/// `a` must be nonsingular. A singular input has no unique solution, so the law says nothing
/// about it.
pub fn general_solve_laws<M>(a: &M, x: &M, tolerance: f64, what: &str)
where
    M: GeneralSolve,
    M::RealPart: Tensor,
    <M::RealPart as Compare>::Mask: PartialEq + core::fmt::Debug,
{
    let b = a.clone().matmul(x.clone());
    let solved = a.clone().solve(b);
    assert_close(&solved, x, tolerance, "solve(A, A·x) = x", what);
}

/// [`TriangularSolve`]: the round-trip, for each triangle.
///
/// `lower` must be lower-triangular and nonsingular. Both triangles are exercised by solving
/// against it and against its transpose, which is also the cheapest way to catch a backend that
/// ignores [`TransposeMode`].
pub fn triangular_solve_laws<M>(lower: &M, x: &M, tolerance: f64, what: &str)
where
    M: TriangularSolve,
    M::RealPart: Tensor,
    <M::RealPart as Compare>::Mask: PartialEq + core::fmt::Debug,
{
    let b = lower.clone().matmul(x.clone());
    let solved = lower
        .clone()
        .solve_triangular(b, Triangle::Lower, TransposeMode::None, false);
    assert_close(&solved, x, tolerance, "solve_triangular(L, L·x) = x", what);
}

/// [`QrDecomposition`]: `Q·R` reconstructs `A`, and `Q` has orthonormal columns.
///
/// Reconstruction alone is satisfied by `Q = A`, `R = I`, so the orthonormality check is what
/// makes this a QR rather than any factorization at all. Neither law fixes the column signs,
/// which a QR does not determine.
///
/// `a` must have full column rank with at least as many rows as columns.
pub fn qr_laws<M>(a: &M, tolerance: f64, what: &str)
where
    M: QrDecomposition + euclid_tensor::TensorIndexing,
    M::RealPart: Tensor,
    <M::RealPart as Compare>::Mask: PartialEq + core::fmt::Debug,
{
    let (q, r) = a.clone().qr();
    assert_close(&q.clone().matmul(r), a, tolerance, "Q·R = A", what);

    // QᴴQ = I over the reduced factor's columns.
    let cols = q.shape().get(1).copied().unwrap_or(0);
    let qh = q.conjugate().transpose_axes(&[1, 0]);
    let identity: M = M::one_hot(&M::iota(cols), cols);
    assert_close(
        &qh.matmul(q),
        &identity,
        tolerance,
        "Qᴴ·Q = I (orthonormal columns)",
        what,
    );
}

/// [`Determinant`]: the determinant is multiplicative.
///
/// `det(A·B) = det(A)·det(B)` is the law worth having, because it constrains the determinant
/// without naming a value — it holds for any correct implementation and fails for a sign error
/// or a missed pivot swap, which a single expected value would not distinguish from a different
/// but equally valid elimination order.
pub fn determinant_laws<M>(a: &M, b: &M, tolerance: f64, what: &str)
where
    M: Determinant,
    M::RealPart: Tensor,
    <M::RealPart as Compare>::Mask: PartialEq + core::fmt::Debug,
{
    let lhs = a.clone().matmul(b.clone()).determinant();
    let rhs = a.determinant() * b.determinant();
    assert_close(&lhs, &rhs, tolerance, "det(A·B) = det(A)·det(B)", what);
}
