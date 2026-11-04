//! **Does it satisfy the relations it must?**
//!
//! Identity tests are the strongest tool available here, because they need no external
//! reference: a recurrence, complement, reflection, or round-trip validates itself. They
//! cannot be transcribed wrongly, they sweep whole domains cheaply, and they catch the class
//! of bug that sampled reference values miss — a formula that is right at the points someone
//! happened to tabulate and wrong between them.
//!
//! Each test states the mathematical relation it checks, so a failure says which law broke.
//!
//! Split by function family.

mod beta_family;
mod cross_precision;
mod erf_family;
mod exp_log;
mod gamma_family;
mod logistic;
mod trigonometry;
