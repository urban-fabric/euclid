//! Rank-graded array contracts: [`Vector`], [`Matrix`], and [`Tensor`].
//!
//! These traits describe mathematical array operations: shape changes, indexing, gather,
//! reductions, contraction, vector products, matrix products, SPD solves, log determinants,
//! Mahalanobis distance, and elementwise conditionals. They do not define storage layout, axis metadata,
//! precision policy, or concrete factorization APIs.
//!
//! The array surface splits into two self-contained interfaces, mirroring how the contracts
//! actually interact (the scalar [`numeric`](crate::numeric) surface is the third, standalone
//! interface beneath both):
//!
//! - `tensor` — the rank-n [`Tensor`] interface. A tensor interacts only with other
//!   tensors and with scalars, so it stands alone.
//! - `matvec` — the rank-1/2 [`Vector`] and [`Matrix`] interface. Vectors and
//!   matrices interact with *each other* (matvec products, diagonals, SPD vector solves) as
//!   well as with scalars, so they are grouped together.
//!
//! Within each interface the surface is layered:
//!
//! - **Structural tensors**: [`Tensor`] starts at [`Clone`], so shape and structural
//!   operations work for real, complex, integer, and boolean tensors. Arithmetic methods add
//!   the bounds they need.
//! - **Graph control flow**: [`TensorControlFlow`], [`VectorControlFlow`], and
//!   [`MatrixControlFlow`] are explicit contracts for carriers that can represent
//!   conditionals and loops.
//! - **Capability extensions**: opt-in traits that grow the surface an implementation can offer
//!   without widening the core. [`TensorReduce`] adds statistical/scan reductions
//!   (variance, log-sum-exp, softmax, cumulative sum); [`TensorSort`] adds sorting and
//!   the argmin/argmax/top-k index family; [`TensorContract`] adds general dot/einsum
//!   contraction; [`TensorIndexing`] adds range/one-hot/histogram encoding; and
//!   [`TensorTransform`] adds roll/tile/clip. [`MatrixDecompose`] adds the dense QR,
//!   eigen, SVD, and LU factorizations plus the general (non-SPD) solve.
//! - **Linear algebra**: [`Vector`] and [`Matrix`] use
//!   [`NormedDivisionAlgebra`](crate::numeric::NormedDivisionAlgebra). `dot` is Hermitian,
//!   `cholesky` is Hermitian positive-definite, and real-valued quantities return the
//!   scalar's [`RealPart`](crate::numeric::NormedDivisionAlgebra::RealPart).
//!
//! | rank | trait | bound | adds |
//! |------|-------|-------|------|
//! | 0 | [`NormedDivisionAlgebra`](crate::numeric::NormedDivisionAlgebra) | — | conjugate + magnitude + the real field |
//! | 1 | [`Vector`] | field | constructors + inner product + norms + vector-space ops |
//! | 2 | [`Matrix`] | field | constructors + matmul + the SPD linear-algebra quantities |
//! | n | [`Tensor`] | [`Clone`] | shape + constructors + structural ops + reductions + contraction |

mod matvec;
mod tensor;
#[cfg(test)]
mod tests;

pub use matvec::{Matrix, MatrixControlFlow, MatrixDecompose, Vector, VectorControlFlow};
pub use tensor::{
    FftKind, Tensor, TensorContract, TensorControlFlow, TensorConv, TensorFft, TensorIndexing,
    TensorReduce, TensorSort, TensorTransform, TensorWindow,
};
