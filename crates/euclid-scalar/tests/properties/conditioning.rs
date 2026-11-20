//! Searching for numerical instability, rather than asserting its absence at fixed points.
//!
//! `src/tests/accuracy` already checks this crate's kernels against known values at chosen
//! arguments. That is the right way to pin a published constant, and it has a blind spot: the
//! arguments were chosen by whoever wrote the test, so the worst argument is found only if
//! someone thought of it. This suite is the complement — it *searches*, over a float
//! distribution built to reach the awkward regions, and it hill-climbs on observed error.
//!
//! # Every test here has two halves
//!
//! Each cancellation-prone form in this crate exists because a naive expression of the same
//! quantity is catastrophically wrong somewhere. So the tests assert both directions:
//!
//! 1. **The safe form is accurate**, checked against an independent reference.
//! 2. **The naive form really is bad**, enforced by
//!    [`Config::require`](euclid_stability::Config::require): the property *requires* that some
//!    fraction of generated cases exhibit severe naive error, so the test fails if the
//!    generator ever stops reaching the ill-conditioned region.
//!
//! The second half is what makes this a demonstration rather than a hope. Without it, a change
//! that narrowed the generator would leave a green suite checking nothing interesting — the
//! safe form would agree with the naive one everywhere, because nowhere hard was ever tried.
//!
//! # Choosing the reference, and the range that makes it valid
//!
//! Near zero the reference is a truncated Taylor series: it converges in a few terms, has no
//! cancellation, and is not this crate's own answer, so the check is not circular. But a
//! truncated series is only accurate over the range where its next term is negligible, and that
//! range is **much smaller than it looks**. The first draft of this file generated `|x|` up to
//! `2^-2` and compared against a four-term series at a `1e-13` tolerance; the truncation error
//! there is around `1e-5`, so the suite failed immediately — correctly, and against the test
//! rather than the code.
//!
//! Each generator below is therefore bounded by the exponent at which its reference is good to
//! about `1e-15` relative, and the arithmetic is recorded next to it. That is a real constraint
//! and not a workaround: outside the stated range the naive form is well conditioned anyway, so
//! nothing interesting is being skipped.
//!
//! Set `EUCLID_STABILITY_STATS=1` to see the distribution each property reached and the worst
//! error its climb found.

use euclid_scalar::Real;
use euclid_scalar::compensated;
use euclid_stability::relation::Relative;
use euclid_stability::{
    Config, check, check_relative, check_ulps, classify, generate, property, relation, target,
    tolerance,
};

/// How many of an `f64`'s ~16 significant digits `approximate` lost against `reference`.
///
/// Returned as `f64` so it can be handed to [`target`] directly: the search then climbs towards
/// the input that destroys the most digits, which is exactly the question "how bad can this
/// get".
fn digits_destroyed(approximate: f64, reference: f64) -> f64 {
    if reference == 0.0 || !reference.is_finite() || !approximate.is_finite() {
        return 0.0;
    }
    let relative = ((approximate - reference) / reference).abs();
    if relative <= 0.0 {
        return 0.0;
    }
    (16.0 + relative.log10()).max(0.0)
}

fn settings(suite: &str) -> Config {
    Config::default()
        .cases(400)
        .regressions(format!("tests/regressions/{suite}.txt"))
}

/// [`settings`] plus a budget, for the properties here that hill-climb on `digits_destroyed`.
///
/// Those five climbs were doing their job — finding the input that destroys the most digits — and
/// then **threw the number away**: the score was compared against a threshold and dropped, so the
/// suite could see a safe form stop being safe and could not see it losing eleven digits where it
/// used to lose six. Same omission the `erf` round trip had, same fix.
///
/// Separate from [`settings`] rather than folded into it, because a budget may only go on a
/// property that reports a score. The engine rejects the combination outright — "a budget over a
/// property with no score is the one configuration that looks like protection and provides none"
/// — which is how the first attempt at this, putting `.budget` in `settings`, failed loudly
/// against the seven properties here that assert exactness and have nothing to climb.
fn measured(suite: &str) -> Config {
    settings(suite).budget("tests/budgets.txt")
}

#[path = "conditioning/cancellation.rs"]
mod cancellation;
#[path = "conditioning/measurement.rs"]
mod measurement;
#[path = "conditioning/range.rs"]
mod range;
#[path = "conditioning/summation.rs"]
mod summation;
#[path = "conditioning/transforms.rs"]
mod transforms;
