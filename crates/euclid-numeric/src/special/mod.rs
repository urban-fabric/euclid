//! Crate-internal special-function implementations.
//!
//! The public surface is the numeric trait layer (`Real` for real-valued functions and
//! `Natural` for count combinatorics). This module is deliberately private to the crate:
//! it houses the concrete `f64`/`f32` algorithms used by the scalar trait implementations.

pub(crate) mod bessel;
pub(crate) mod beta;
pub(crate) mod erf;
pub(crate) mod gamma;
pub(crate) mod host;

/// Native single-precision peers of the elementwise functions — same names, `f32` throughout.
pub(crate) mod f32;
