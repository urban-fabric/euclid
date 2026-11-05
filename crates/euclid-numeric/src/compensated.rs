//! Accurate reductions: summation and dot products that carry their own
//! round-off correction.
//!
//! A naive `fold(+)` over `n` terms admits an error that grows with the
//! *condition number* of the sum — the ratio of `Σ|xᵢ|` to `|Σxᵢ|` — so a
//! sequence that cancels loses precision in proportion to how much it cancels.
//! The classic demonstration is one large value followed by many small ones:
//! each small term rounds away individually against the running total, and the
//! sum of the terms that rounded away is not small.
//!
//! These reductions fix that by keeping the part of every addition that did not
//! fit. [`Real::two_sum`] returns a rounded sum *and* its exact remainder, so the
//! remainder can be accumulated separately and added back at the end. The result
//! behaves as though the accumulator had roughly twice the working precision,
//! for one extra add per term and no allocation.
//!
//! The variant here is Kahan–Babuška–Neumaier: the correction is accumulated
//! unconditionally rather than being folded back into the running total, which is
//! what makes it correct when a later term is *larger* than the total so far —
//! the case plain Kahan gets wrong.
//!
//! ## Why this is in the contracts crate
//!
//! [`two_sum`](Real::two_sum) and [`two_product`](Real::two_product) have been here
//! all along, and this module is the reduction everyone writes on top of them.
//! Three separate hand-rolled copies existed across Urban Fabric before it did,
//! and the surface they were reaching for is small enough to state once.
//!
//! ## Non-finite input
//!
//! `two_sum` reports a **zero** remainder whenever the rounded sum is not finite,
//! precisely so that an overflow saturates to `±∞` instead of poisoning the
//! correction with a `NaN` that outlives the term that caused it. These
//! reductions inherit that: a sequence that overflows returns `±∞`, and one
//! containing a `NaN` returns `NaN`, rather than returning a finite number that
//! silently depends on summation order.

use crate::Real;

/// The compensated sum of `xs`, or zero for an empty slice.
///
/// Accurate to within one rounding of the exact sum for any sequence the
/// accumulator does not overflow, where a naive fold loses digits in proportion
/// to how much the sequence cancels.
///
/// ```
/// use euclid_numeric::compensated;
///
/// // A large value and many small ones: each small term is below the ULP of the
/// // running total, so a naive fold drops all of them.
/// let mut xs = vec![1.0e16_f64];
/// xs.extend(std::iter::repeat_n(1.0, 1000));
///
/// let naive: f64 = xs.iter().sum();
/// let exact = compensated::sum(&xs);
///
/// assert_eq!(naive, 1.0e16);          // every `1.0` rounded away
/// assert_eq!(exact, 1.0e16 + 1000.0); // ...and is recovered here
/// ```
#[must_use]
pub fn sum<T: Real>(xs: &[T]) -> T {
    let mut total = T::zero();
    let mut correction = T::zero();
    for x in xs {
        let (s, e) = total.two_sum(x.clone());
        total = s;
        correction = correction + e;
    }
    total + correction
}

/// The compensated sum of an iterator, for a caller that does not have a slice.
///
/// Identical to [`sum`] in accuracy; separate because materialising an iterator
/// into a `Vec` purely to sum it is the thing this avoids.
#[must_use]
pub fn sum_iter<T: Real, I: IntoIterator<Item = T>>(xs: I) -> T {
    let mut total = T::zero();
    let mut correction = T::zero();
    for x in xs {
        let (s, e) = total.two_sum(x);
        total = s;
        correction = correction + e;
    }
    total + correction
}

/// The compensated dot product of `a` and `b`, over `min(a.len(), b.len())` terms.
///
/// Both the products *and* their sum are compensated:
/// [`two_product`](Real::two_product) recovers the part of each `aᵢ·bᵢ` that the
/// multiply rounded away, which a compensated sum over pre-rounded products
/// cannot. That matters for the case this exists for — a nearly-orthogonal pair,
/// where the answer is the small residue of large cancelling products.
///
/// ```
/// use euclid_numeric::compensated;
///
/// // Two vectors whose products cancel to a value far below their magnitudes.
/// let a = [1.0e8_f64, 1.0];
/// let b = [1.0e8_f64, -1.0e16];
/// assert_eq!(compensated::dot(&a, &b), 0.0);
/// ```
#[must_use]
pub fn dot<T: Real>(a: &[T], b: &[T]) -> T {
    let mut total = T::zero();
    let mut correction = T::zero();
    for (x, y) in a.iter().zip(b.iter()) {
        let (p, pe) = x.clone().two_product(y.clone());
        let (s, se) = total.two_sum(p);
        total = s;
        correction = correction + se + pe;
    }
    total + correction
}
