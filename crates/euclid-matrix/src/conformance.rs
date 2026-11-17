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
//! | [`HermitianEigendecomposition`](crate::HermitianEigendecomposition) | `A·V = V·diag(λ)`, `Vᴴ·V = I` | [`eigh_laws`] |
//! | [`SingularValueDecomposition`](crate::SingularValueDecomposition) | `U·diag(σ)·Vᴴ = A` | [`svd_laws`] |
//!
//! Those hold for every correct implementation regardless of pivoting, blocking or sign
//! convention, which is what makes them laws rather than expectations.
//!
//! # The two laws that needed a contract change first
//!
//! The table above listed the eigen and SVD laws before either existed, and the reason neither
//! was written is worth keeping: it was **a gap in the trait surface, not an oversight**.
//!
//! [`eigh`](crate::HermitianEigendecomposition::eigh) returns `(Self::RealSpectrum, Self)` and
//! [`svd`](crate::SingularValueDecomposition::svd) returns `(Self, Self::RealSpectrum, Self)`.
//! Both laws must form `V·diag(λ)`, which means combining a `RealSpectrum` with a `Self` — and
//! [`MatrixRealSpectrum`](crate::MatrixRealSpectrum) related the two types not at all. For a real
//! carrier they are usually the same type and it would have worked by accident; for a complex one
//! nothing bridged them and no generic expression for `V·diag(λ)` existed.
//!
//! [`MatrixRealSpectrum::from_spectrum`](crate::MatrixRealSpectrum::from_spectrum) closes it, and
//! both laws are now written. It is a new required method, so it breaks every implementor — done
//! before 0.5.0 was tagged rather than deferred, precisely so it does not have to break anyone
//! later. Neither law materialises a diagonal matrix: scaling columns is a broadcast and an
//! elementwise multiply, which is portable and avoids an `n × n` of mostly zeros.
//!
//! # Backend coverage
//!
//! A rank-zero eager fixture exercises the residual helper. The solve and factorization
//! laws need a backend implementing the corresponding matrix primitives.
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

use euclid_scalar::{Compare, Element, Real};
use euclid_tensor::Tensor;

use crate::{
    Determinant, GeneralSolve, MatrixTensor, PreparedSpd, QrDecomposition, SpdFactor, SpdSolve,
    TransposeMode, Triangle, TriangularSolve,
};

/// Assert that two tensors agree to within a scaled Frobenius tolerance.
///
/// Exposed because a backend writing its own laws over these traits wants the same residual
/// convention the suites use, rather than inventing a second one.
///
/// The criterion is **Frobenius**: `‖got − want‖_F ≤ tolerance · max(1, ‖want‖_F)`, which
/// degrades to absolute near zero and to relative away from it — a purely relative check is
/// meaningless where a residual passes through zero, which for a correct solve is most of it.
///
/// # Panics
///
/// Panics on a shape mismatch, an invalid tolerance, a non-finite residual or reference,
/// or a residual outside the allowed bound.
pub fn assert_close<M>(got: &M, want: &M, tolerance: f64, law: &str, what: &str)
where
    M: MatrixTensor,
    M::RealPart: Tensor,
    <M::RealPart as Compare>::Mask: PartialEq + core::fmt::Debug,
{
    let shape = got.shape();
    assert_eq!(shape, want.shape(), "{what}: {law} — shape mismatch");
    assert!(
        tolerance.is_finite() && tolerance >= 0.0,
        "{what}: {law} — tolerance must be finite and nonnegative"
    );
    if shape.contains(&0) {
        return;
    }

    let axes: Vec<usize> = (0..shape.len()).collect();
    let yes = <<M::RealPart as Compare>::Mask as Element>::one();
    let mut error = (got.clone() - want.clone()).magnitude();
    let mut reference = want.magnitude();
    let mut absolute_floor = <M::RealPart as Element>::one();
    let mut error_max = Tensor::max(error.clone(), &axes);
    let mut reference_max = Tensor::max(reference.clone(), &axes);
    if tolerance == 0.0 {
        // A zero tolerance needs exact equality even when scaling would erase a subnormal.
        let error_sum = error.sum(&axes);
        assert_eq!(
            Compare::eq(&error_sum, &M::RealPart::zero()),
            yes,
            "{what}: {law} — nonzero residual"
        );
        return;
    }
    if Real::max(error_max.clone(), reference_max.clone()).is_finite() != yes {
        // For finite inputs, quarter scaling leaves room for subtraction and
        // the magnitude even with four components at the format boundary.
        absolute_floor = M::RealPart::scalar(0.25);
        let factor = M::from_real(absolute_floor.clone());
        let scaled_want = want.clone() * factor.clone();
        error = (got.clone() * factor - scaled_want.clone()).magnitude();
        reference = scaled_want.magnitude();
        error_max = Tensor::max(error.clone(), &axes);
        reference_max = Tensor::max(reference.clone(), &axes);
    }

    let error_norm = normalized_frobenius(error, &error_max, &shape, &axes);
    let reference_norm = normalized_frobenius(reference, &reference_max, &shape, &axes);
    assert_eq!(
        error_norm.is_finite(),
        yes,
        "{what}: {law} — non-finite residual"
    );
    assert_eq!(
        reference_norm.is_finite(),
        yes,
        "{what}: {law} — non-finite reference"
    );
    assert_eq!(
        scaled_residual_within(
            (error_max, error_norm),
            (reference_max, reference_norm),
            absolute_floor,
            M::RealPart::scalar(tolerance),
        ),
        yes,
        "{what}: {law} — residual exceeded tolerance {tolerance:e}"
    );
}

fn scaled_residual_within<T: Real>(
    (error_max, error_norm): (T, T),
    (reference_max, reference_norm): (T, T),
    absolute_floor: T,
    tolerance: T,
) -> T::Mask {
    let common = Real::max(
        Real::max(error_max.clone(), reference_max.clone()),
        absolute_floor.clone(),
    );
    let error = (error_max / common.clone()) * error_norm;
    let reference = (reference_max / common.clone()) * reference_norm;
    error.le(&(Real::max(reference, absolute_floor / common) * tolerance))
}

#[cfg(test)]
mod tests;

/// Norm after division by the largest magnitude. This keeps the squares in range.
fn normalized_frobenius<T: Tensor + Real>(
    magnitude: T,
    max: &T,
    shape: &[usize],
    axes: &[usize],
) -> T {
    let divisor = T::select(&Compare::eq(max, &T::zero()), T::one(), max.clone());
    (magnitude / divisor.broadcast_in_dim(shape, &[]))
        .square()
        .sum(axes)
        .sqrt()
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

/// [`HermitianEigendecomposition`](crate::HermitianEigendecomposition): `A·V = V·diag(λ)`, and
/// `V` is unitary.
///
/// The defining property, and it needs no reference eigenvalues: whatever `λ` and `V` a backend
/// produces, applying `A` to the eigenvectors must equal scaling them by their eigenvalues. That
/// holds for any correct implementation and for any phase convention, which asserting a
/// particular `V` would not.
///
/// Unitarity is checked alongside, because the eigenvector relation alone is satisfied by a
/// degenerate `V` — for `λ = 0` any `V` with `A·V = 0` passes — and `Vᴴ·V = I` is what makes it
/// a decomposition rather than a collection of vectors.
///
/// Ascending order is **not** checked here. It is part of the contract, but checking it needs an
/// ordering predicate over the spectrum that the rank-1 tensor surface does not offer portably;
/// a backend should assert it in its own suite where it has concrete values to compare.
///
/// `a` must be rank-2, square and Hermitian.
pub fn eigh_laws<M>(a: &M, tolerance: f64, what: &str)
where
    M: crate::HermitianEigendecomposition + euclid_tensor::TensorIndexing,
    M::RealPart: Tensor,
    <M::RealPart as Compare>::Mask: PartialEq + core::fmt::Debug,
{
    let n = a.shape().first().copied().unwrap_or(0);
    let (lambda, v) = a.clone().eigh();

    // V·diag(λ) as a broadcast multiply rather than a materialised diagonal: mapping the
    // spectrum's only axis onto output axis 1 makes λ vary along the columns, so element [i, j]
    // becomes V[i, j]·λ[j].
    let scaled = v.clone() * M::from_spectrum(lambda).broadcast_in_dim(&[n, n], &[1]);
    assert_close(
        &a.clone().matmul(v.clone()),
        &scaled,
        tolerance,
        "A·V = V·diag(λ)",
        what,
    );

    let vh = v.clone().conjugate().transpose_axes(&[1, 0]);
    let identity: M = M::one_hot(&M::iota(n), n);
    assert_close(
        &vh.matmul(v),
        &identity,
        tolerance,
        "Vᴴ·V = I (eigenvectors are unitary)",
        what,
    );
}

/// [`SingularValueDecomposition`](crate::SingularValueDecomposition):
/// `U·diag(σ)·Vᴴ` reconstructs `A`.
///
/// Reconstruction is the whole claim, and it is insensitive to the matched phase pair an SVD
/// leaves free — negate a column of `U` and the corresponding column of `V` and the product is
/// unchanged, which is why this is a law and an expected `U` would not be.
///
/// Note the convention this depends on: [`svd`](crate::SingularValueDecomposition::svd) returns
/// `V`, not `Vᴴ`. A backend that returns the conjugate transpose instead fails here, which is
/// the intended outcome — that mismatch otherwise produces a correctly-shaped, silently wrong
/// answer for every square input.
///
/// `a` must be rank-2. For a non-square `a` the reduced form is assumed, per the trait.
pub fn svd_laws<M>(a: &M, tolerance: f64, what: &str)
where
    M: crate::SingularValueDecomposition,
    M::RealPart: Tensor,
    <M::RealPart as Compare>::Mask: PartialEq + core::fmt::Debug,
{
    let (u, sigma, v) = a.clone().svd();

    // Scale U's columns by σ, then apply Vᴴ. `k` comes from U's column count rather than from
    // `a`'s shape, so this is correct for the reduced form on a non-square input.
    let rows = u.shape().first().copied().unwrap_or(0);
    let k = u.shape().get(1).copied().unwrap_or(0);
    let scaled = u * M::from_spectrum(sigma).broadcast_in_dim(&[rows, k], &[1]);
    let vh = v.conjugate().transpose_axes(&[1, 0]);

    assert_close(&scaled.matmul(vh), a, tolerance, "U·diag(σ)·Vᴴ = A", what);
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
