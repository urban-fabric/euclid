//! The rank-1/2 linear-algebra interface: [`Vector`] and [`Matrix`].
//!
//! Vectors and matrices are grouped together because they interact with *each other* as well
//! as with the scalar [`numeric`](crate::numeric) surface: [`Matrix::matvec`] consumes a
//! [`Matrix::Vector`], [`Matrix::diagonal`] produces one, and the SPD quantities
//! (`solve_spd_vec`, `mahalanobis_squared`) mix the two ranks. Both use
//! [`NormedDivisionAlgebra`] as their scalar, so real and complex linear algebra share one
//! contract and real-valued results come back as a `RealPart`.

use crate::numeric::NormedDivisionAlgebra;

mod matrix;
mod vector;

pub use matrix::{Matrix, MatrixControlFlow, MatrixDecompose};
pub use vector::{Vector, VectorControlFlow};

/// The real field of a scalar `S` — `S` itself for a real scalar, its real part for a
/// complex one. Where norms, log-determinant, and Mahalanobis distance live.
type RealPart<S> = <S as NormedDivisionAlgebra>::RealPart;
