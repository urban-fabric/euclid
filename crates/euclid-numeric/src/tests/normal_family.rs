//! The standard normal family (`ndtr`, `log_ndtr`, `ndtri`, `erfc_inv`) and the elementary
//! additions alongside it (`log10`, `exp2`, and the exact integer operations).
//!
//! The normal CDF is the most-used function in applied statistics and the one whose naive
//! composition fails soonest, so it gets the same four-way treatment as the rest: reference
//! values, tail stability, round-trips, and edges into the relationship web.
//!
//! Split into the files below to keep each one inside this crate's 400-line budget.

mod distributions;
mod elementary;
mod relationships;
