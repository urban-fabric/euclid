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

//! # The split
//!
//! This module passed the crate's 400-line budget. The residual machinery stays here, since every
//! suite leans on it, and the laws divide by what they ask a backend for: a solution, or a
//! factorization.

mod factorization;
mod solve;

// Re-exported so every public path this module has ever had still resolves: the
// split is an internal rearrangement and no carrier should have to notice it.
pub use factorization::{determinant_laws, eigh_laws, qr_laws, svd_laws};
pub use solve::{general_solve_laws, prepared_spd_laws, spd_solve_laws, triangular_solve_laws};

use crate::{
    Determinant, GeneralSolve, MatrixTensor, PreparedSpd, QrDecomposition, SpdFactor, SpdSolve,
    TransposeMode, Triangle, TriangularSolve,
};
use euclid_scalar::{Compare, Element, Real};
use euclid_tensor::Tensor;

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
