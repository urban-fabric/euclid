//! Rank-graded lazy array contracts: [`LazyVector`], [`LazyMatrix`], and [`LazyTensor`].
//!
//! These traits describe mathematical array operations: shape changes, indexing, gather,
//! reductions, contraction, vector products, matrix products, SPD solves, log determinants,
//! Mahalanobis distance, and elementwise conditionals. They do not define storage layout, axis metadata,
//! precision policy, or concrete factorization APIs.
//!
//! The surface has two parts:
//!
//! - **Structural tensors**: [`LazyTensor`] starts at [`Clone`], so shape and structural
//!   operations work for real, complex, integer, and boolean tensors. Arithmetic methods add
//!   the bounds they need.
//! - **Graph control flow**: [`LazyTensorControlFlow`], [`LazyVectorControlFlow`], and
//!   [`LazyMatrixControlFlow`] are explicit contracts for carriers that can represent
//!   conditionals and loops.
//! - **Linear algebra**: [`LazyVector`] and [`LazyMatrix`] use
//!   [`NormedDivisionAlgebra`]. `dot` is Hermitian,
//!   `cholesky` is Hermitian positive-definite, and real-valued quantities return the
//!   scalar's [`RealPart`](crate::numeric::NormedDivisionAlgebra::RealPart).
//!
//! | rank | trait | bound | adds |
//! |------|-------|-------|------|
//! | 0 | [`NormedDivisionAlgebra`] | — | conjugate + magnitude + the real field |
//! | 1 | [`LazyVector`] | field | constructors + inner product + norms + vector-space ops |
//! | 2 | [`LazyMatrix`] | field | constructors + matmul + the SPD linear-algebra quantities |
//! | n | [`LazyTensor`] | [`Clone`] | shape + constructors + structural ops + reductions + contraction |

use crate::numeric::NormedDivisionAlgebra;

mod control_flow;
mod matrix;
mod tensor;
#[cfg(test)]
mod tests;
mod vector;

pub use matrix::{LazyMatrix, LazyMatrixControlFlow};
pub use tensor::{LazyTensor, LazyTensorControlFlow};
pub use vector::{LazyVector, LazyVectorControlFlow};

/// The real field of a scalar `S` — `S` itself for a real scalar, its real part for a
/// complex one. Where norms, log-determinant, and Mahalanobis distance live.
type RealPart<S> = <S as NormedDivisionAlgebra>::RealPart;
