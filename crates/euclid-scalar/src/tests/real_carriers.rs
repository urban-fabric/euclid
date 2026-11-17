//! **Do the hand-written [`Real`] composites return the right values on *both* carriers?**
//!
//! [`super::real_surface`] asks whether a formula written once against [`Real`] runs on `f64`
//! and `f32`. This asks whether the two carriers compute the same functions.
//!
//! # Why this module exists
//!
//! Most of [`Real`] is delegation — `f32::sqrt`, `f64::sin` — and those are covered by
//! [`super::identities::cross_precision`], which pins each one bit-exact or to a measured ulp
//! budget. But around thirty methods are **required trait methods with no default**, written out
//! twice: once in `impls/real_f64.rs` and once in `impls/real_f32.rs`. `sinc`, `ln_sub_exp`,
//! `hypot3`, `two_product`, `sin_cos_degrees`, `logit`, `xlogy` and the rest of the log-domain
//! family are each two independent bodies that happen to look alike.
//!
//! Mutation testing showed the `f32` half of every one of those pairs was unpinned.
//! `impls/real_f32.rs` was the largest single survivor file in the crate, and the residue after
//! the cross-precision pass was concentrated exactly here: `sinc` 22, `ln_sub_exp` 15,
//! `sin_cos_degrees` 11, `hypot3` 6, `two_product` 5, and a tail across the log-domain
//! composites. That is MUTANTS.md's cause 1 — a covered body beside an uncovered twin — with the
//! `f64` body being the covered one.
//!
//! # Why these need values rather than budgets
//!
//! A cross-precision budget compares `f32` against `f64` and asks whether they agree. That is
//! the wrong instrument for a twin pair, for two reasons:
//!
//! - **It cannot see a shared error.** Both bodies were written from the same derivation, so a
//!   mistake in the derivation appears in both and the comparison passes.
//! - **It is blind where the branch is.** `sinc` switches to a Taylor series below `|x| < 1e-2`,
//!   and `sin(x)/x` in `f64` is accurate enough there that the `f32` series can be wrong in its
//!   third coefficient and still land inside any plausible ulp budget.
//!
//! So every suite here states the *defining property* and checks it on each carrier
//! independently. Each test is generic over [`Real`] and instantiated at both.
//!
//! # Where the expectations come from
//!
//! Round trips and defining identities wherever one exists, per the house rule in
//! [`crate::tests`] — "prefer an identity to a transcribed digit":
//!
//! | Composite | Independent source |
//! |---|---|
//! | `ln_sub_exp` | `ln_add_exp(ln_sub_exp(a, b), b) = a`, inverting it with its own sibling |
//! | `logit` | `logit(sigmoid(x)) = x`, the pair being inverses by definition |
//! | `ln_exp_m1` | `ln_1p_exp(ln_exp_m1(x)) = x`, likewise |
//! | `two_product` | `p + err = a · b` *exactly*, checked in the next precision up |
//! | `hypot3` | Pythagorean quadruples — `(1,2,2;3)`, `(1,4,8;9)` — which are exact integers |
//! | `sinc` | continuity across its own branch boundary, plus `sin(x)/x` away from zero |
//! | `sin_cos_degrees` | exactness at the quadrant multiples, which is the algorithm's purpose |
//! | `xlogy` | `0 · ln 0 = 0`, the absorbing-zero convention it exists to implement |

use super::agrees;

mod delegations;
mod elementary;
mod log_domain;
mod normed_algebra;
mod reduction;
mod special_functions;

/// The tolerance each carrier is held to, as a relative bound.
///
/// `f32` carries about `1.2e-7` of resolution and `f64` about `2.2e-16`; these are roughly an
/// order of magnitude above each, which is the room a two-or-three-operation composite needs
/// without being loose enough to hide a wrong coefficient. Where a composite is *exact* — the
/// quadrant values, the absorbing zeros, the error-free transforms — the test asserts equality
/// instead and does not use these at all.
const F32_TOL: f64 = 1e-6;
const F64_TOL: f64 = 1e-14;
