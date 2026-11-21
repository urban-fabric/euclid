//! Running a property: search, shrink, persist, report.
//!
//! # The three outcomes that are not failures
//!
//! A property run has more ways to be inconclusive than to fail, and conflating them is how a
//! suite ends up reporting far less than it appears to:
//!
//! - **Rejection.** The generator produced a case the property does not claim to cover. Draw
//!   again. Tracked against a budget, because a generator rejecting most of what it makes is
//!   testing a fraction of its case count — and the only way to notice is to be told.
//! - **Overrun.** A tape or generation budget did not feed the generator to completion.
//!   It means "this candidate is incomplete", never "the property holds".
//! - **Cap reached.** Shrinking stopped on its probe budget rather than at a fixpoint, so the
//!   reported case may not be minimal. Said out loud, since a reader otherwise assumes minimal.
//!
//! # Panics are failures
//!
//! The property runs inside [`catch_unwind`](std::panic::catch_unwind), so an `assert!`, an
//! index out of bounds or an arithmetic overflow inside the code under test is a normal
//! failure that gets shrunk like any other. Without this, the first panicking case would abort
//! the process and report an unshrunk input — which is exactly the case you most want
//! minimised.
//!
//! The panic hook is silenced for the duration of the search, because shrinking deliberately
//! re-runs the failing property hundreds of times and the default hook would print hundreds of
//! backtraces. `run` returns the final failure and its location; `check` then panics with
//! that report outside the probe, so libtest receives one failure report.

use std::cell::{Cell, RefCell};
use std::panic::{self, AssertUnwindSafe};
use std::sync::Once;

use crate::generate::{DrawError, Gen};
use crate::rng::SplitMix64;
use crate::shrink::{self, Probe};
use crate::source::Source;
use crate::{budget, persist, stats, target};

mod execute;
mod failure;
mod result;

pub use execute::run;
pub use result::{
    Counterexample, Exploration, Failure, FailureKind, Outcome, RunError, RunErrorKind, RunReport,
};

/// How hard to search.
#[derive(Clone, Debug)]
pub struct Config {
    /// Cases that must actually **run** before the property is declared to hold.
    ///
    /// Run, not attempted. A rejected case does not spend one of these — see
    /// [`max_rejects`](Self::max_rejects) for why that distinction is the whole point.
    pub cases: usize,
    /// Rejections tolerated across the whole run before giving up and saying so.
    ///
    /// # This is a budget on *extra attempts*, not on cases
    ///
    /// It used to be neither, and the bug is worth recording because it made the diagnostic this
    /// field exists for unreachable. The search ran `for case in 0..cases` and counted a
    /// rejection as a spent case, so `rejects <= cases` held by construction — and with the
    /// default `cases: 256` against `max_rejects: 4096`, the budget could never be exceeded and
    /// the "gave up after N rejected cases" report could never fire.
    ///
    /// Worse, the thing it was meant to catch happened silently instead. A generator rejecting
    /// nine cases in ten ran **19 of 256 cases** and passed green, which is precisely the
    /// "testing far less than its case count suggests" failure the module docs promise to
    /// report.
    ///
    /// So the search now draws until `cases` have *run*, and this bounds the rejections it will
    /// tolerate on the way. QuickCheck's `tests` versus `max_tests` split, which is the one idea
    /// that crate has that this one was missing.
    pub max_rejects: usize,
    /// Longest tape a generator may consume for one case.
    pub max_tape: usize,
    /// Base seed. Fixed by default: a suite that passes today must pass tomorrow, and a
    /// property that only fails for one seed in ten is a property that fails.
    pub seed: u64,
    /// File to persist minimised failures to, replayed before the random search. `None`
    /// disables persistence.
    pub regressions: Option<String>,
    /// Labels that must reach a minimum share of cases, as `(label, percent)`. See
    /// [`crate::stats`] for why these are enforced rather than merely reported.
    pub requirements: Vec<(String, f64)>,
    /// Share of the case budget spent hill-climbing towards a higher
    /// [`target`](crate::target()) score, rather than drawing fresh. Ignored when the property
    /// never reports a score.
    pub targeted_fraction: f64,
    /// File recording the worst [`target`](crate::target()) score this property has reported, so
    /// that a later run scoring higher fails. `None` disables the check. See
    /// [`crate::budget`].
    pub budget: Option<String>,
    /// Opt-in edge-first or complete finite-domain exploration. Random preserves existing tapes.
    pub exploration: Exploration,
    /// Repeat each counterexample this many times before and after shrinking (at least two).
    pub replay_checks: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            cases: 256,
            max_rejects: 4_096,
            max_tape: 4_096,
            seed: 0x5EED_1234_5678_9ABC,
            regressions: None,
            requirements: Vec::new(),
            // Half the budget, when targeting is in use. Spending everything on the climb
            // would abandon exploration and get stuck in the first local maximum found; a
            // fifty-fifty split keeps a fresh draw coming while still concentrating effort.
            targeted_fraction: 0.5,
            budget: None,
            exploration: Exploration::Random,
            replay_checks: 2,
        }
    }
}

impl Config {
    /// Try known complete edge tapes before random cases.
    #[must_use]
    pub fn edges_first(mut self) -> Self {
        self.exploration = Exploration::EdgesFirst;
        self
    }

    /// Enumerate every known finite case; error if the domain is unknown or exceeds `cases`.
    #[must_use]
    pub fn exhaustive(mut self) -> Self {
        self.exploration = Exploration::Exhaustive;
        self
    }

    /// Set the stability replay count for counterexamples, with a minimum of two.
    #[must_use]
    pub fn replay_checks(mut self, count: usize) -> Self {
        self.replay_checks = count;
        self
    }

    /// Set the case count.
    #[must_use]
    pub fn cases(mut self, cases: usize) -> Self {
        self.cases = cases;
        self
    }

    /// Set the regression file.
    #[must_use]
    pub fn regressions(mut self, path: impl Into<String>) -> Self {
        self.regressions = Some(path.into());
        self
    }

    /// Set how many rejections to tolerate before giving up and reporting.
    ///
    /// Rarely worth setting. The default is generous because a rejection now costs an extra
    /// draw rather than a case, so the budget is a guard against a generator that can never
    /// satisfy its own filter rather than a tuning knob. See
    /// [`max_rejects`](Self::max_rejects).
    #[must_use]
    pub fn max_rejects(mut self, max_rejects: usize) -> Self {
        self.max_rejects = max_rejects;
        self
    }

    /// Require that at least `percent` of the cases that ran carried `label`.
    ///
    /// The run fails if they did not. This is an assertion about the *generator*, and it is
    /// what stops a property from passing green over input that never reached the interesting
    /// region — see [`crate::stats`].
    #[must_use]
    pub fn require(mut self, label: impl Into<String>, percent: f64) -> Self {
        self.requirements.push((label.into(), percent));
        self
    }

    /// Set the share of the case budget spent hill-climbing on the reported
    /// [`target`](crate::target()) score.
    #[must_use]
    pub fn targeted_fraction(mut self, fraction: f64) -> Self {
        self.targeted_fraction = fraction.clamp(0.0, 1.0);
        self
    }

    /// Record the worst [`target`](crate::target()) score to `path`, and fail a later run that
    /// scores higher.
    ///
    /// Requires the property to call [`target`](crate::target()): a budget over a property that
    /// reports no score is protection that does not exist, so it is an error rather than a
    /// no-op. See [`crate::budget`] for why the number belongs in a committed file.
    #[must_use]
    pub fn budget(mut self, path: impl Into<String>) -> Self {
        self.budget = Some(path.into());
        self
    }
}

mod climb;
mod environment;

// Re-exported so `runner::check` and the items `runner::execute` reaches stay where they
// were before the split.
pub(crate) use climb::{Climber, Stale, Verdict, quietly, report_stale, run_quietly};
#[cfg(test)]
pub(crate) use climb::{PROBING, forward_panic};
pub use environment::{SEED, SOAK, check};
pub(crate) use environment::{enforce_budget, with_overrides};
// Reached only by `runner::tests`, which drives the parsers and the summary directly.
#[cfg(test)]
pub(crate) use environment::{apply_overrides, parse_seed, rejection_summary};

#[cfg(test)]
mod tests;
