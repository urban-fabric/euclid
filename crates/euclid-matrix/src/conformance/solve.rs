//! The solve contracts: that `Ax = b` is satisfied by the `x` a backend returns.
//!
//! Every law here is a *residual* bound rather than a comparison against a reference solution,
//! which is the only honest instrument: a well-conditioned system has one answer and an
//! ill-conditioned one has many, and `‖Ax − b‖` is small for the right reasons in both cases.
//! See [`super::assert_close`] for the scaling that makes the bound condition-aware.

use super::*;

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
