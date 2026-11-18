//! The `Real` surface's identities, as named metamorphic relations.
//!
//! Where `conditioning.rs` hunts for the worst input, this suite states the laws that must hold
//! for *every* input: a round trip returns, an odd function is odd, a bounded function stays in
//! range, a monotone function preserves order. None of them needs a known answer, which is why
//! they can be checked over generated floats at all.
//!
//! # Written through `relation`, deliberately
//!
//! Nearly every property here is one call into [`euclid_stability::relation`] rather than a
//! hand-written pair of evaluations and a comparison. That is the standardization this suite
//! exists to demonstrate, and the reason is not brevity:
//!
//! - The comparison is a **parameter**, so the tolerance appears once and visibly. A
//!   hand-written identity check reaches for `==`, which is wrong for a computed float, or for
//!   an absolute epsilon, which is wrong at unknown magnitude.
//! - The degenerate cases are handled once. `relation::monotone` sorts its own arguments;
//!   `Ulps` refuses `NaN` rather than silently comparing it equal to itself.
//! - The relation is **named**, so a failure says "f(-x) = -f(x) does not hold" instead of
//!   leaving a reader to reconstruct the law from two calls.
//!
//! # Tolerances, and why they differ
//!
//! A tolerance is a claim about the implementation, so each one below is justified where it is
//! written. The pattern: a single kernel evaluation should be within an ulp or two; a composed
//! identity accumulates each step's error, so `sin² + cos²` gets more room than `sin` alone;
//! and a round trip through two kernels gets the sum of both.
//!
//! Where a relation is checked against zero — `erf(x) + erfc(x) - 1`, `cosh² - sinh²` — the
//! comparison must be absolute rather than relative, because relative error against zero is
//! undefined and `within_relative` correctly refuses it.

use euclid_scalar::Real;
use euclid_stability::relation::{Absolute, Exact, Relative, Ulps};
use euclid_stability::{
    Config, check, check_relative, check_ulps, classify, generate, numeric, property, relation,
};

fn settings(suite: &str) -> Config {
    Config::default()
        .cases(400)
        .regressions(format!("tests/regressions/{suite}.txt"))
}

#[path = "identities/bounds.rs"]
mod bounds;
#[path = "identities/composed.rs"]
mod composed;
#[path = "identities/contracts.rs"]
mod contracts;
#[path = "identities/distribution_extremes.rs"]
mod distribution_extremes;
#[path = "identities/gamma_equations.rs"]
mod gamma_equations;
#[path = "identities/log_domain.rs"]
mod log_domain;
#[path = "identities/monotonicity.rs"]
mod monotonicity;
#[path = "identities/normal.rs"]
mod normal;
#[path = "identities/round_trips.rs"]
mod round_trips;
#[path = "identities/symmetry.rs"]
mod symmetry;
#[path = "identities/transcendental.rs"]
mod transcendental;
#[path = "identities/wronskians.rs"]
mod wronskians;
