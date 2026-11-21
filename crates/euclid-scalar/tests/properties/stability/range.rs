//! The range contract: what every kernel owes across the whole format.

use super::*;

// ===== Range contract =====================================================

/// A kernel, its domain, and where `NaN` is the correct answer rather than a defect.
type Ranged = (
    &'static str,
    fn(f64) -> f64,
    fn(f64) -> bool,
    fn(f64) -> bool,
);

/// `x` is a pole of `Γ` and `ψ`: zero or a negative integer.
fn is_nonpositive_integer(x: f64) -> bool {
    x <= 0.0 && x.fract() == 0.0
}

// Shared row shape and threshold, used by both halves below.

/// A kernel and its closed-form bounds, inclusive, with whether both signs are in scope.
type Bounded = (&'static str, fn(f64) -> f64, f64, f64, bool);

/// The point below which `K₁(x) ≈ 1/x` exceeds the format, so an infinity is the right answer.
const K1_OVERFLOW_POINT: f64 = 5.562_684_646_268_004e-309;

#[path = "range/bounds.rs"]
mod bounds;
#[path = "range/finiteness.rs"]
mod finiteness;
