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
//! | trait | law | here |
//! |---|---|---|
//! | [`SpdSolve`] | `solve_spd(Σ, Σ·x)` recovers `x` | [`spd_solve_laws`] |
//! | [`GeneralSolve`] | `solve(A, A·x)` recovers `x` | [`general_solve_laws`] |
//! | [`TriangularSolve`] | the same, for `L` and for `Lᴴ` | [`triangular_solve_laws`] |
//! | [`PreparedSpd`] / [`SpdFactor`] | a cached factor solves what the direct path solves | [`prepared_spd_laws`] |
//! | [`QrDecomposition`] | `Q·R = A`, and `Qᴴ·Q = I` | [`qr_laws`] |
//! | [`Determinant`] | `det(A·B) = det(A)·det(B)` | [`determinant_laws`] |
//! | [`HermitianEigendecomposition`](crate::HermitianEigendecomposition) | `A·V = V·diag(λ)` | **not expressible — see below** |
//! | [`SingularValueDecomposition`](crate::SingularValueDecomposition) | `U·diag(σ)·Vᴴ = A` | **not expressible — see below** |
//!
//! Those hold for every correct implementation regardless of pivoting, blocking or sign
//! convention, which is what makes them laws rather than expectations.
//!
//! # The two laws that cannot be written against this contract
//!
//! This table used to list the eigen and SVD laws without the caveat, and no function
//! implemented either — so the module advertised two laws a backend author would have gone
//! looking for. The reason they were never written is structural rather than an oversight, and
//! worth stating because it is a **gap in the trait surface**, not in the test suite.
//!
//! [`HermitianEigendecomposition::eigh`](crate::HermitianEigendecomposition) returns `(Self::RealSpectrum, Self)` and
//! [`SingularValueDecomposition::svd`](crate::SingularValueDecomposition) returns `(Self, Self::RealSpectrum, Self)`. Both laws
//! need to form `V·diag(λ)`, which means combining a `RealSpectrum` with a `Self` — and
//! [`MatrixRealSpectrum`](crate::MatrixRealSpectrum) provides **no conversion between them**. For a real carrier they are
//! usually the same type and it would work by accident; for a complex carrier they are different
//! types with nothing bridging them, so a law written generically does not compile.
//!
//! Closing it needs one additive method on [`MatrixRealSpectrum`](crate::MatrixRealSpectrum) — something of the shape
//! `fn from_spectrum(s: Self::RealSpectrum) -> Self`, lifting a real spectrum into the carrier.
//! That is a new required method, so it breaks every implementor, which puts it in the next
//! breaking release rather than this one. Until then these two traits are the only part of
//! euclid's contract surface with **no law at all**, and a backend implementing them is on its
//! own — which is exactly the position the laws exist to avoid.
//!
//! # Nothing here has ever been executed
//!
//! Said plainly, because it bears on how much the suites above are worth. euclid ships no
//! concrete tensor, so no carrier in this repository can satisfy these traits, so no law body in
//! this file has ever run. `--all-features` type-checks them, which for generic trait code is a
//! real check and catches a law that does not compose, and it is not the same as a law that is
//! *correct*.
//!
//! The precedent is not hypothetical: `euclid-tensor`'s own 3-cycle law was malformed at rank ≥ 4
//! for as long as its reference sample stopped at rank 3, and it was found the first time a
//! generated sample reached rank 4. Every law in this file is in that state permanently. Treat a
//! failure here as a hypothesis about your backend *or* about the law, and read the law before
//! changing your code.
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
    Determinant, GeneralSolve, MatrixTensor, PreparedSpd, QrDecomposition, SpdFactor, SpdSolve,
    TransposeMode, Triangle, TriangularSolve,
};

/// Assert that two tensors agree elementwise to within a relative tolerance.
///
/// Exposed because a backend writing its own laws over these traits wants the same residual
/// convention the suites use, rather than inventing a second one.
///
/// The criterion is **Frobenius**: `‖got − want‖_F ≤ tolerance · max(1, ‖want‖_F)`, which
/// degrades to absolute near zero and to relative away from it — a purely relative check is
/// meaningless where a residual passes through zero, which for a correct solve is most of it.
///
/// This doc comment used to describe a *max* norm — `max|got − want| ≤ tolerance · max(1,
/// max|want|)` — while the body computed the Frobenius norm it always has. The two differ by up
/// to a factor of `√(elements)`, so a backend choosing a tolerance from the old wording was
/// choosing it for the wrong norm and getting a stricter check than it bargained for. Frobenius
/// is the right choice and the code was right; only the description was wrong.
pub fn assert_close<M>(got: &M, want: &M, tolerance: f64, law: &str, what: &str)
where
    M: MatrixTensor,
    M::RealPart: Tensor,
    <M::RealPart as Compare>::Mask: PartialEq + core::fmt::Debug,
{
    let shape = got.shape();
    assert_eq!(shape, want.shape(), "{what}: {law} — shape mismatch");

    // Name every axis rather than passing an empty slice. `Tensor::sum` does define the empty
    // case as "reduce over every axis", so this is no longer load-bearing for correctness — it
    // is kept because being explicit costs nothing and because the rank-0 case (which
    // `determinant_laws` hits) then goes through the same path as every other rank instead of
    // relying on reduce-all being the identity there.
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

/// [`TriangularSolve`]: the round-trip, forward and transposed.
///
/// `lower` must be lower-triangular and nonsingular, and rank-2.
///
/// Two solves, and the second is the one that earns its place. Solving `L·x = b` exercises the
/// forward path; solving `Lᴴ·x = b` with [`TransposeMode::Transpose`] exercises the flag — and a
/// backend that **ignores `TransposeMode` entirely** passes the first and fails the second. That
/// is the cheapest available check on a parameter nothing else constrains.
///
/// This function's documentation claimed to do that before it did: it said "both triangles are
/// exercised by solving against it and against its transpose" while the body performed a single
/// forward solve. A law that overstates what it checks is worse than a missing one, because a
/// backend reads the claim and stops looking.
pub fn triangular_solve_laws<M>(lower: &M, x: &M, tolerance: f64, what: &str)
where
    M: TriangularSolve,
    M::RealPart: Tensor,
    <M::RealPart as Compare>::Mask: PartialEq + core::fmt::Debug,
{
    // Forward: L·x = b must recover x.
    let b = lower.clone().matmul(x.clone());
    let solved = lower
        .clone()
        .solve_triangular(b, Triangle::Lower, TransposeMode::None, false);
    assert_close(&solved, x, tolerance, "solve_triangular(L, L·x) = x", what);

    // Transposed: form the right-hand side with Lᴴ explicitly, then ask the backend to solve
    // against L *with the transpose flag set*. The two must agree, which they cannot if the flag
    // is being dropped.
    let lower_h = lower.clone().conjugate().transpose_axes(&[1, 0]);
    let b_h = lower_h.matmul(x.clone());
    let solved_h =
        lower
            .clone()
            .solve_triangular(b_h, Triangle::Lower, TransposeMode::Adjoint, false);
    assert_close(
        &solved_h,
        x,
        tolerance,
        "solve_triangular(L, Lᴴ·x, transpose) = x",
        what,
    );
}

/// [`PreparedSpd`] / [`SpdFactor`]: a cached factor must solve what the direct path solves.
///
/// The seam exists to hoist an O(n³) factorization out of a per-call path, so the thing that can
/// go wrong is the cache: a factor stored before some scaling step, reused after the matrix it
/// came from was consumed, or built from the wrong triangle. None of that is visible in a single
/// `solve_spd` call, and all of it shows here.
///
/// So the law is **agreement between the two paths**, not a residual: `prepare_spd().solve(b)`
/// and `solve_spd(b)` are two routes to `Σ⁻¹b` and must land in the same place. Written as an
/// agreement rather than against `x` deliberately — if both paths are wrong in the same way this
/// passes, and [`spd_solve_laws`] is what catches that, so run both.
///
/// `spd` must be rank-2 and HPD, per [`PreparedSpd::prepare_spd`].
pub fn prepared_spd_laws<M>(spd: &M, rhs: &M, tolerance: f64, what: &str)
where
    M: SpdSolve + PreparedSpd,
    M::RealPart: Tensor,
    <M::RealPart as Compare>::Mask: PartialEq + core::fmt::Debug,
{
    let direct = spd.clone().solve_spd(rhs.clone());
    let factored = spd.clone().prepare_spd().solve(rhs.clone());
    assert_close(
        &factored,
        &direct,
        tolerance,
        "prepare_spd(Σ).solve(b) = solve_spd(Σ, b)",
        what,
    );

    // The factor must be reusable, which is the entire point of the seam — a backend that
    // invalidates or consumes its factor on first use passes the check above and fails this.
    let factor = spd.clone().prepare_spd();
    let first = factor.solve(rhs.clone());
    let second = factor.solve(rhs.clone());
    assert_close(
        &second,
        &first,
        tolerance,
        "a prepared factor solves the same way twice",
        what,
    );
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
