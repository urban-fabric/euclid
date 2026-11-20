//! The decomposition contracts: QR, symmetric eigen, SVD, and the determinant.
//!
//! Each of these is checked by *reassembly* — `QR == A`, `VΛVᵀ == A`, `UΣVᵀ == A` — plus the
//! structural claims the factors make about themselves, orthogonality and ordering. A backend
//! that returns plausible factors which do not multiply back is the failure these catch, and no
//! reference table is needed for any of it.

use super::*;

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
