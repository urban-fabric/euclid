//! **How accurate is it?** Reference-value and dense-sweep accuracy checks for the special
//! functions on the public [`Real`] surface.
//!
//! Every tolerance here is set from *measured* error, and annotated with what was measured.
//! A tolerance is a claim about this implementation, so loosening one without explaining why
//! hides exactly the regression it exists to catch.
//!
//! References come from two independent sources, never from euclid itself:
//!
//! - **Exact closed forms**, computed here in the test (factorials, harmonic sums, the
//!   binomial sum for an integer-parameter incomplete beta, power series for `I₀`/`I₁`).
//! - **60-digit `Decimal` values**, generated out-of-band and pasted in. Where the natural
//!   closed form would cancel catastrophically, the *tail* form was used instead — see
//!   [`gamma_lr_deep_tail_is_accurate`], which is the case that proves it matters.
//!
//! Split into the files below to keep each one inside this crate's 400-line budget.

mod airy;
mod erf_and_gamma;
mod incomplete_and_bessel;
