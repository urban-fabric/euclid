//! Private numerical steps shared by scalar implementations and kernels.
//!
//! Callers supply their own precision, convergence tolerance, and domain policy.
//! Algebraic steps use backend-neutral traits where possible so they can also
//! build expressions. [`crate::HostReal`] supplies the comparisons for choosing
//! a floating-point evaluation regime.

pub(crate) mod arithmetic;
pub(crate) mod continued_fraction;
pub(crate) mod discrete;
pub(crate) mod gamma;
pub(crate) mod log_domain;
pub(crate) mod normal;
pub(crate) mod polynomial;
pub(crate) mod representation;
pub(crate) mod roots;
pub(crate) mod trigonometry;
