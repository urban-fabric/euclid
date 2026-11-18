//! **Does every carrier of a discrete tier honour the contract, or only the one the suites
//! instantiate?**
//!
//! [`super::discrete`] asks the *portability* question: does a formula written once against
//! [`Natural`] or [`Integer`] compile and run at all three widths. This asks the *conformance*
//! one: is each of the six carriers individually correct across the whole of its trait.
//!
//! The two are easy to conflate and are not the same test. A formula that runs at `u32`, `u64`
//! and `u128` demonstrates the tiers are genuinely generic; it says nothing about the sixty-odd
//! methods those three `impl` blocks define separately. The widths still supply their own
//! `div_rem`, `signum`, `abs` and casts. The GCD and LCM implementations delegate to shared
//! Euclidean arithmetic.
//!
//! # Why this module exists
//!
//! Mutation testing, which measures the question directly. `impls/natural_impl.rs` had **69
//! surviving mutants** and `impls/integer_impl.rs` **34**, and the distribution told the story:
//! almost every one was at `u32`, `u128`, `i32` or `i128`, while `u64` and `i64` — the widths
//! [`super::discrete`] and [`super::conformance_suites`] happened to sample — were well pinned.
//!
//! One of those twins had drifted into a genuine defect rather than merely an untested one —
//! `u128`'s hand-written `binomial`, recorded on
//! [`binomial_exact`](crate::kernels::host::factorial) and pinned in that module's tests. It is
//! the first real bug this mutation work turned up, against a great many missing assertions,
//! and it is the argument for measuring rather than reasoning about which twin is covered.
//!
//! # Where the expectations come from
//!
//! Every expectation comes from an independent source. Using the kernel's
//! multiplicative recurrence as a reference would have repeated the binomial off-by-one.
//!
//! | Operation | Independent source |
//! |---|---|
//! | `gcd` | Euclid's own worked example, `gcd(1071, 462) = 21` |
//! | `lcm` | `gcd(a, b) · lcm(a, b) = \|a · b\|`, a theorem about both at once |
//! | `isqrt` | its definition, `r² ≤ n < (r+1)²`, checked rather than recomputed |
//! | `div_rem` | its definition, `q · rhs + r = self` with `0 ≤ r < \|rhs\|` |
//! | `binomial` | Pascal's rule — addition, where the kernel multiplies and divides |
//! | `ln_factorial` | `ln(n!) − ln((n−1)!) = ln n`, which crosses the `n = 21` threshold |
//! | `signum`/`abs` | their definitions, against sign-separated samples |
//! | the casts | integer literals |
//!
//! # The zero identity
//!
//! `Element::zero() -> Default::default()` is equivalent on all six carriers because `Default`
//! for a machine integer is zero. The `one()` tests check the multiplicative identity.

use super::agrees;

mod integer;
mod natural;

/// Pascal's triangle up to and including row `rows`, in exact `u128` integers.
///
/// The reference for every binomial assertion here, and deliberately a *different algorithm*
/// from the kernel's: Pascal's rule is addition only, where
/// [`binomial`](crate::kernels::host::factorial) runs a multiplicative recurrence with a
/// division in it.
fn pascal(rows: usize) -> Vec<Vec<u128>> {
    let mut triangle = vec![vec![1_u128]];
    for n in 1..=rows {
        let previous = &triangle[n - 1];
        let mut row = vec![1_u128];
        for k in 1..n {
            row.push(previous[k - 1] + previous[k]);
        }
        row.push(1);
        triangle.push(row);
    }
    triangle
}
