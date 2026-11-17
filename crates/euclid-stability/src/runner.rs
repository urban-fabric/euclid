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

/// Multiply every property's case count, for a deliberate soak run.
///
/// `EUCLID_STABILITY_SOAK=100 cargo test` searches a hundred times as hard.
pub const SOAK: &str = "EUCLID_STABILITY_SOAK";

/// Replace every property's base seed, for a soak down a different path.
pub const SEED: &str = "EUCLID_STABILITY_SEED";

/// Apply the soak overrides to a configuration.
///
/// # Why a multiplier rather than an absolute count
///
/// QuickCheck's `QUICKCHECK_TESTS` sets the case count outright, and that does not transfer
/// here: euclid's suites choose their own counts deliberately — 400 for the `Real` identities,
/// 600 where a climb needs room — and an absolute override would flatten considered numbers into
/// one. A multiplier composes with them instead, so `SOAK=100` means "a hundred times whatever
/// this property decided it needed".
///
/// # On determinism
///
/// [`Config::seed`] is fixed by default and the crate docs are emphatic about why: a property
/// that fails on one seed in ten is a property that fails, and letting the seed drift per run
/// turns that finding into a flake. These overrides do not weaken that. Unset — which is every
/// ordinary `cargo test` — they change nothing, and a suite that passes today still passes
/// tomorrow. Set, they are a soak someone asked for by name.
///
/// A soak that trips an accuracy [`budget`] has **found something**: searching
/// harder reached a worse input than the committed baseline, and the recorded number is now
/// known to be optimistic. That is a real finding about the kernel rather than an artefact of
/// the longer run, so it is reported as a failure like any other.
fn with_overrides(config: &Config) -> Config {
    // Bound before borrowing, so the `String`s outlive the `&str`s handed on.
    let soak = std::env::var(SOAK).ok();
    let seed = std::env::var(SEED).ok();
    apply_overrides(config, soak.as_deref(), seed.as_deref())
}

/// The override logic, separated from reading the environment.
///
/// Split out purely so it can be tested: a test that sets a process-wide variable races every
/// other test in the binary, and libtest gives no way to serialise them. Taking the raw strings
/// as arguments makes the interesting part — what a malformed, zero or missing value does — an
/// ordinary pure function.
///
/// That mattered immediately. Mutation testing found the `factor > 0` guard below unpinned by the
/// first version of these tests, which only checked the *unset* path. A relaxation to `>= 0`
/// accepts `SOAK=0`, which multiplies the case count to zero and turns every property in the
/// workspace into the "never exercised" failure.
fn apply_overrides(config: &Config, soak: Option<&str>, seed: Option<&str>) -> Config {
    let mut config = config.clone();

    if let Some(factor) = soak
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .filter(|factor| *factor > 0)
    {
        config.cases = config.cases.saturating_mul(factor);
        // The rejection budget has to scale with the case count or a soak turns every filtered
        // generator into a spurious "gave up" report.
        config.max_rejects = config.max_rejects.saturating_mul(factor);
    }

    if let Some(seed) = seed.and_then(|raw| parse_seed(raw.trim())) {
        config.seed = seed;
    }

    config
}

/// Parse a seed as decimal, or as hex behind an `0x` prefix.
fn parse_seed(raw: &str) -> Option<u64> {
    raw.strip_prefix("0x")
        .or_else(|| raw.strip_prefix("0X"))
        .map_or_else(
            || raw.parse::<u64>().ok(),
            |hex| u64::from_str_radix(hex, 16).ok(),
        )
}

/// Run `property` over cases drawn from `generator`.
///
/// Panics with a formatted report if the property fails, which is how a test harness learns
/// about it. On success it returns normally, having reported nothing.
///
/// Honours the [`SOAK`] and [`SEED`] environment overrides.
///
/// # Panics
/// If the property fails, or if every case was rejected.
#[allow(clippy::panic)]
pub fn check<T: 'static + core::fmt::Debug>(
    name: &str,
    config: &Config,
    generator: &Gen<T>,
    property: impl FnMut(T) -> Outcome,
) {
    match run(name, config, generator, property) {
        Ok(report) => {
            if std::env::var_os("EUCLID_STABILITY_STATS").is_some() {
                eprintln!(
                    "{name}: {}{}\n  best target score: {:?}",
                    report.statistics(),
                    rejection_summary(report.rejected, report.attempts),
                    report.best_score
                );
            }
        }
        Err(error) => panic!("{error}"),
    }
}

/// The rejection rate, for the stats dump.
///
/// A free function rather than an inline `format!`, for the same reason
/// [`apply_overrides`] is one: the block it belongs to only runs when
/// `EUCLID_STABILITY_STATS` is set, and no test here can set an environment variable —
/// `std::env::set_var` is `unsafe` as of the 2024 edition and the workspace denies
/// `unsafe_code`. Inline, the arithmetic was unreachable by any test and mutation testing found
/// five separate corruptions of it surviving. Extracted, it is an ordinary pure function.
///
/// Worth printing even when the run is well inside its budget. A generator rejecting 90% is now
/// *correct* — the search simply draws until `cases` have run — but it is paying ten draws per
/// case to get there, and that is a generator worth narrowing.
fn rejection_summary(rejects: usize, attempts: usize) -> String {
    if rejects == 0 {
        return String::new();
    }
    #[allow(clippy::cast_precision_loss)]
    let rate = (rejects as f64) * 100.0 / (attempts as f64);
    format!("\n  rejected: {rejects} of {attempts} draws ({rate:.1}%)")
}

/// Compare the run's worst score against its committed baseline.
///
/// Returns an error if the score rose above the baseline, or none was reported.
fn enforce_budget(name: &str, path: &str, best: Option<f64>) -> Result<(), String> {
    let Some(best) = best else {
        return Err(format!(
            "{name}: a budget file is configured but the property never called target(), so \
             there is no score to record and nothing is being protected.\n  \
             budget: {path}\n\n\
             Either report a score — target(observed_error) — or drop .budget() from the config. \
             A budget over a property with no score is the one configuration that looks like \
             protection and provides none."
        ));
    };

    let (verdict, previous) = budget::evaluate(path, name, best);
    match verdict {
        budget::Verdict::Recorded => eprintln!(
            "{name}: recorded a target() baseline of {best:e} in {path}. Commit it — a later run \
             scoring higher will fail."
        ),
        budget::Verdict::Held => {}
        budget::Verdict::Improved => eprintln!(
            "{name}: worst target() score is {best:e}, well under the recorded {}. The baseline \
             is now loose enough to admit a regression; re-record with {}=1 and commit it.",
            previous.unwrap_or(f64::NAN),
            budget::RECORD
        ),
        budget::Verdict::Regressed => {
            let baseline = previous.unwrap_or(f64::NAN);
            let change = if baseline == 0.0 {
                String::new()
            } else {
                format!(" ({:+.0}%)", (best / baseline - 1.0) * 100.0)
            };
            return Err(format!(
                "property regressed: {name}\n  \
                 worst target() score rose from {baseline:e} to {best:e}{change}\n  \
                 recorded in: {path}\n\n\
                 The property still holds — its own assertion has headroom for both numbers — so \
                 this is the only check that can see the change. Either the code under test got \
                 less accurate, or the search changed: editing a generator, `cases`, or \
                 `targeted_fraction` makes the search visit different tapes and find a different \
                 worst case, which is a legitimate new baseline. If that is what happened, \
                 re-record with {}=1 and commit the new number.",
                budget::RECORD
            ));
        }
    }
    Ok(())
}

/// Hill-climbs towards a higher [`target`](crate::target()) score.
///
/// Hypothesis's `target()` idea. The property reports a number it would like maximised — for a
/// numerical kernel, the observed ULP error — and the search stops being uniform and starts
/// looking for the worst case. That turns "is this accurate on 256 random inputs" into "how
/// inaccurate can this be made", which is a different and much better question.
///
/// The mutation is deliberately crude: perturb one tape entry of the best tape so far. Because
/// a value is a pure function of its tape, any perturbation yields a valid case, so the climber
/// needs no knowledge of the domain it is searching. A smarter mutation operator would help,
/// and is the obvious next improvement.
struct Climber {
    best_tape: Option<Vec<u64>>,
    best_score: Option<f64>,
    first_targeted_case: usize,
    enabled: bool,
}

impl Climber {
    fn new(config: &Config) -> Self {
        #[allow(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss
        )]
        let explore = ((config.cases as f64) * (1.0 - config.targeted_fraction)) as usize;
        Self {
            best_tape: None,
            best_score: None,
            first_targeted_case: explore.max(1),
            enabled: config.targeted_fraction > 0.0,
        }
    }

    fn best_score(&self) -> Option<f64> {
        self.best_score
    }

    /// A tape to try, or `None` to draw fresh.
    ///
    /// # The mutation has to span step sizes
    ///
    /// This is where a first attempt went wrong, and the lesson generalises. The obvious moves
    /// are "a fresh random value", "halve it" and "add one" — and all three are useless. Adding
    /// one to a `u64` entry moves the *drawn* value by about `2^-64` of its range, which is no
    /// move at all; halving it jumps to the other end; a fresh value is just uniform sampling
    /// again. Measured against a plain random search, that operator made the best score found
    /// *worse*, because it spent half the budget not climbing.
    ///
    /// A hill climb needs local moves at **every scale**, since it cannot know whether the peak
    /// is one step away or a million. Flipping a single random bit gives exactly that: the low
    /// bits are fine adjustments, the high bits are coarse jumps, and one uniform choice of bit
    /// position covers the whole spectrum. It is the classic mutation-fuzzer move, and it works
    /// here for the same reason it works there.
    fn propose(&self, case: usize, seed: u64) -> Option<Vec<u64>> {
        if !self.enabled || case < self.first_targeted_case {
            return None;
        }
        let best = self.best_tape.as_ref()?;
        if best.is_empty() {
            return None;
        }

        let mut mutated = best.clone();
        let mut rng = SplitMix64::new(seed);
        let position = (rng.next_u64() as usize) % mutated.len();
        let entry = mutated[position];

        mutated[position] = match rng.next_u64() % 8 {
            // Mostly a single bit flip, at a uniformly chosen position — a step of every
            // possible magnitude, which is the move the climb actually needs.
            0..=5 => entry ^ (1u64 << (rng.next_u64() % 64)),
            // A nudge at a random scale, which reaches values a bit flip cannot (a flip can
            // only ever toggle, so it cannot walk steadily in one direction).
            6 => {
                let step = rng.next_u64() >> (rng.next_u64() % 64);
                if rng.next_u64().is_multiple_of(2) {
                    entry.wrapping_add(step)
                } else {
                    entry.wrapping_sub(step)
                }
            }
            // And occasionally a fresh value, to escape a local maximum the flips are circling.
            _ => rng.next_u64(),
        };
        Some(mutated)
    }

    /// Whether `score` would become the new best, so the caller knows if the tape is worth
    /// materialising.
    ///
    fn would_improve(&self, score: Option<f64>) -> bool {
        self.improving_score(score).is_some()
    }

    fn improving_score(&self, score: Option<f64>) -> Option<f64> {
        score.filter(|s| s.is_finite() && self.best_score.is_none_or(|best| *s > best))
    }

    fn observe(&mut self, tape: &[u64], score: Option<f64>) {
        if let Some(score) = self.improving_score(score) {
            self.best_score = Some(score);
            self.best_tape = Some(tape.to_vec());
        }
    }
}

enum Verdict {
    Held,
    Generation(Failure),
    Failed(Failure),
    /// Carries no reason: during replay the reason is not reported anywhere, because a
    /// rejection while shrinking only means "discard this candidate". The search path tracks
    /// reasons separately, where they do reach the user.
    Rejected,
    Overrun,
}

/// A committed tape whose case has drifted out from under it.
struct Stale {
    tape: String,
    recorded: String,
    now: String,
}

/// Report tapes whose renderings no longer match, or refresh the file if asked to.
///
/// Returns an error if any tape has gone stale and this is not a recording run.
fn report_stale(
    name: &str,
    path: &str,
    stale: &[Stale],
    current: &[(Vec<u64>, Option<String>)],
) -> Result<(), String> {
    if budget::recording() {
        persist::refresh(path, name, current);
        eprintln!(
            "{name}: refreshed {} stale rendering(s) in {path}. The tapes are kept, but each now \
             pins a different case than the one originally found — delete a line instead if the \
             original case mattered. Recording is a read-modify-write, so use \
             `--test-threads=1` if a suite has several properties sharing this file.",
            stale.len()
        );
        return Ok(());
    }

    Err(format!(
        "stale regressions: {name}\n{}\n  in: {path}\n\n\
         A tape is a sequence of choices, not a value, so it only means anything for the \
         generator that consumed it. These tapes still decode, but they now produce different \
         cases than the ones that were committed — so the regressions they were saved to pin are \
         no longer being tested, and the suite would have stayed green about it.\n\n\
         If you changed a generator deliberately, re-record with {}=1 and commit the result. If a \
         particular case above still matters, delete its line and assert it directly, where no \
         generator edit can move it.",
        stale
            .iter()
            .map(|entry| format!(
                "  tape {}:\n    recorded: {}\n    now:      {}",
                &entry.tape[..entry.tape.len().min(16)],
                entry.recorded,
                entry.now
            ))
            .collect::<Vec<_>>()
            .join("\n"),
        budget::RECORD
    ))
}

/// Run one property case with panic reporting suppressed during probes.
fn run_quietly<T>(property: &mut impl FnMut(T) -> Outcome, value: T) -> Outcome {
    quietly("inside the property", || property(value)).and_then(core::convert::identity)
}

fn quietly<T>(location: &'static str, body: impl FnOnce() -> T) -> Result<T, Failure> {
    install_hook_once();
    let previous = PROBING.with(|flag| flag.replace(true));
    let previous_site = PANIC_SITE.with(|site| site.replace(None));
    let outcome = panic::catch_unwind(AssertUnwindSafe(body));
    let site = PANIC_SITE.with(|site| site.replace(previous_site));
    PROBING.with(|flag| flag.set(previous));
    outcome.map_err(|payload| {
        let message = payload
            .downcast_ref::<&str>()
            .map(|s| (*s).to_string())
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "panicked".into());
        Failure {
            kind: FailureKind::Panic,
            ..Failure::new(format!("panicked: {message}"), location)
                .with_identity(site.unwrap_or_else(|| location.into()))
        }
    })
}

fn install_hook_once() {
    static INSTALLED: Once = Once::new();
    INSTALLED.call_once(|| {
        let previous = panic::take_hook();
        panic::set_hook(Box::new(move |info| {
            let probing = PROBING.with(Cell::get);
            if probing {
                PANIC_SITE.with(|site| {
                    *site.borrow_mut() = info.location().map(|where_| {
                        format!("{}:{}:{}", where_.file(), where_.line(), where_.column())
                    })
                });
            }
            if forward_panic(probing, || {
                std::env::var_os("EUCLID_STABILITY_VERBOSE").is_some()
            }) {
                previous(info);
            }
        }));
    });
}

fn forward_panic(probing: bool, verbose: impl FnOnce() -> bool) -> bool {
    !probing || verbose()
}

thread_local! {
    static PROBING: Cell<bool> = const { Cell::new(false) };
    static PANIC_SITE: RefCell<Option<String>> = const { RefCell::new(None) };
}

#[cfg(test)]
mod tests;
