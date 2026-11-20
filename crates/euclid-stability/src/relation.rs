//! Metamorphic relations, named.
//!
//! A metamorphic relation says that two executions must agree, without either needing a known
//! answer. That is what makes it checkable over generated input: `f(f(x)) = x` can be verified
//! for a random `x` even when nobody knows what `f(x)` should be. It is the only kind of
//! property available for code with no oracle — a compiler, a path solver, a special function —
//! and it is most of what the fleet actually tests.
//!
//! The lineage is separate from property testing proper. T.Y. Chen named metamorphic testing in
//! 1998; the engineering landmarks are EMI for compilers (mutate dead code, demand identical
//! behaviour — hundreds of GCC and LLVM bugs) and SQLancer's TLP and NoREC for databases. What
//! those have in common is that the relation is *designed*, not stumbled on, and the same dozen
//! shapes recur. This module is those shapes.
//!
//! # Why they are here rather than written out each time
//!
//! Three reasons, in increasing order of importance.
//!
//! The relation gets a name, so a reader sees "this is an involution check" rather than
//! reconstructing it from two calls and an assertion. The failure message is written once and
//! says which law broke and on what. And — the real reason — **the degenerate cases are handled
//! in one place.** A hand-written commutativity check compares with `==`, which is wrong for a
//! float and silently wrong for a float that happens to be `NaN`; a hand-written round-trip
//! check forgets that the backward direction may be lossy. Each of those is a mistake that
//! looks like a passing test.
//!
//! # Comparison is a parameter
//!
//! Every relation takes a [`Compare`], because the right notion of "the same" is not a property
//! of the relation. `f(f(x)) = x` over a graph means structural equality; over an `f64` it means
//! within a couple of ulps, and asserting `==` there is the single most common way to write a
//! numerical test that cannot pass.
//!
//! ```
//! use euclid_stability::relation::{Exact, Ulps, involution};
//! use euclid_stability::{generate, property};
//!
//! property! {
//!     fn negation_is_an_involution(x in generate::int(-1000, 1000)) {
//!         involution("negate", x, |v| -v, &Exact)?;
//!     }
//!
//!     fn reciprocal_is_an_involution(x in generate::float::positive()) {
//!         // Two round trips through a division: a couple of ulps, not equality.
//!         involution("reciprocal", x, |v| 1.0 / v, &Ulps(2))?;
//!     }
//! }
//! ```
//!
//! Each returns [`Outcome`], so `?` inside a `property!` body reports the failure through the
//! same machinery as [`check!`](macro@crate::check).

use core::fmt::Debug;

use crate::runner::{Failure, Outcome};
use crate::tolerance;

/// What it means for two results to be "the same".
///
/// Implement this for a domain with its own notion of closeness. The four provided cover
/// everything in the fleet so far.
pub trait Compare<T> {
    /// Whether `a` and `b` count as the same result.
    fn same(&self, a: &T, b: &T) -> bool;

    /// How to describe this comparison in a failure message.
    fn describe(&self) -> String;
}

/// Bit-for-bit (well, [`PartialEq`]) equality.
///
/// Correct for integers, graphs, and anything discrete. **Wrong for a computed float**, where
/// it is the comparison that makes a true property look false.
pub struct Exact;

impl<T: PartialEq> Compare<T> for Exact {
    fn same(&self, a: &T, b: &T) -> bool {
        a == b
    }
    fn describe(&self) -> String {
        "exactly equal".to_string()
    }
}

/// Equal to within `n` representable steps. The right default for floats.
pub struct Ulps(pub u64);

impl Compare<f64> for Ulps {
    fn same(&self, a: &f64, b: &f64) -> bool {
        tolerance::within_ulps(*a, *b, self.0)
    }
    fn describe(&self) -> String {
        format!("within {} ulp", self.0)
    }
}

/// Equal to within a relative error.
pub struct Relative(pub f64);

impl Compare<f64> for Relative {
    fn same(&self, a: &f64, b: &f64) -> bool {
        tolerance::within_relative(*a, *b, self.0)
    }
    fn describe(&self) -> String {
        format!("within {:e} relative", self.0)
    }
}

/// Equal to within an absolute difference. Only for quantities with a known scale.
pub struct Absolute(pub f64);

impl Compare<f64> for Absolute {
    fn same(&self, a: &f64, b: &f64) -> bool {
        tolerance::within_absolute(*a, *b, self.0)
    }
    fn describe(&self) -> String {
        format!("within {:e} absolute", self.0)
    }
}

fn broke<T: Debug>(law: &str, name: &str, comparison: &str, left: &T, right: &T) -> Failure {
    Failure::new(
        format!(
            "{name}: {law} does not hold ({comparison})\n    left:  {left:?}\n    right: {right:?}"
        ),
        "a metamorphic relation",
    )
    .with_identity(format!("{name}: {law}"))
}

mod laws;
mod stability;

// Re-exported flat, so every caller keeps the path it had before the split:
// `euclid_stability::relation::involution`, not `::relation::laws::involution`.
pub use laws::*;
pub use stability::*;
// Reached by `relation::tests`, which asserts the `f32` ulp distance directly.
#[cfg(test)]
pub(crate) use stability::f32_ulps_between;

#[cfg(test)]
mod tests;
