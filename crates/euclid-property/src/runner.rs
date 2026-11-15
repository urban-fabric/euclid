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
//! - **Overrun.** A tape did not feed the generator to completion. Only reachable while
//!   shrinking, where it means "this candidate is incomplete", never "the property holds".
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
//! backtraces. The final minimised case is re-run once with the hook restored, so the
//! backtrace a reader sees is the one for the small input.

use std::cell::Cell;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};

use crate::generate::{DrawError, Gen};
use crate::persist;
use crate::rng::SplitMix64;
use crate::shrink::{self, Probe};
use crate::source::Source;
use crate::{stats, target};

/// A property's verdict on one case.
pub type Outcome = Result<(), Failure>;

/// A property did not hold.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Failure {
    /// What went wrong, as the `check!` that fired describes it.
    pub message: String,
    /// Where it fired.
    pub location: &'static str,
}

impl Failure {
    /// A failure with a message and a source location.
    #[must_use]
    pub fn new(message: impl Into<String>, location: &'static str) -> Self {
        Self {
            message: message.into(),
            location,
        }
    }
}

/// How hard to search.
#[derive(Clone, Debug)]
pub struct Config {
    /// Cases to try before declaring the property held.
    pub cases: usize,
    /// Rejections tolerated across the whole run before giving up and saying so.
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
        }
    }
}

impl Config {
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
}

/// Run `property` over cases drawn from `generator`.
///
/// Panics with a formatted report if the property fails, which is how a test harness learns
/// about it. On success it returns normally, having reported nothing.
///
/// # Panics
/// If the property fails, or if every case was rejected.
pub fn check<T: 'static>(
    name: &str,
    config: &Config,
    generator: &Gen<T>,
    mut property: impl FnMut(T) -> Outcome,
) {
    // Persisted regressions first. A counterexample that cost a random search to find should
    // be the first thing tried on every subsequent run, not rediscovered by luck.
    let persisted = config
        .regressions
        .as_deref()
        .map(persist::load)
        .unwrap_or_default();
    for tape in persisted {
        if let Verdict::Failed(failure) = replay(generator, &mut property, tape.clone()) {
            report(
                name,
                config,
                &tape,
                &failure,
                generator,
                &mut property,
                true,
            );
        }
    }

    stats::reset();

    let mut rejects = 0usize;
    let mut reasons: Vec<&'static str> = Vec::new();
    let mut climber = Climber::new(config);

    for case in 0..config.cases {
        let seed = config
            .seed
            .wrapping_add(case as u64)
            .wrapping_mul(0x2545_F491);

        // Either a fresh draw, or a perturbation of the best-scoring tape found so far. The
        // climb only engages once the property has actually reported a score.
        let mut source = match climber.propose(case, seed) {
            Some(tape) => Source::replaying(tape),
            None => Source::generating(seed, config.max_tape),
        };

        match generator.draw(&mut source) {
            Err(DrawError::Overrun) => {
                rejects += 1;
                reasons.push("tape exhausted");
            }
            Err(DrawError::Rejected(rejected)) => {
                rejects += 1;
                reasons.push(rejected.reason);
            }
            Ok(value) => {
                let tape = source.consumed();
                target::clear();
                let outcome = run_quietly(&mut property, value);
                stats::count_case();
                climber.observe(&tape, target::reported());

                if let Err(failure) = outcome {
                    report(
                        name,
                        config,
                        &tape,
                        &failure,
                        generator,
                        &mut property,
                        false,
                    );
                }
            }
        }

        assert!(
            rejects <= config.max_rejects,
            "{name}: gave up after {rejects} rejected cases (budget {}). \
             The generator is producing mostly inputs the property excludes, so the case count \
             is misleading. Narrow the generator instead of assuming inside the property. \
             Reasons seen: {:?}",
            config.max_rejects,
            dedup(&reasons)
        );
    }

    assert!(
        rejects < config.cases,
        "{name}: every one of {} cases was rejected, so the property was never exercised. \
         Reasons seen: {:?}",
        config.cases,
        dedup(&reasons)
    );

    if std::env::var_os("EUCLID_PROPERTY_STATS").is_some() {
        let climbed = climber.best_score().map_or_else(String::new, |best| {
            format!("\n  best target score: {best:e}")
        });
        eprintln!("{name}: {}{climbed}", stats::report());
    }

    // Coverage last, so a genuine failure is reported in preference to a coverage shortfall —
    // a property that fails has more urgent news than one whose generator was narrow.
    let unmet = stats::unmet(&config.requirements);
    // A property that applied *no* labels at all almost never means "the generator is too
    // narrow". It means the requirement belongs to a sibling property that shares this one's
    // block-level `#![config(...)]`, which is the single most common mistake with this feature
    // and cost three debugging rounds before the hint was added.
    let shared_block_hint = if stats::collected().is_empty() {
        "\n\nThis property applied no labels at all. If the requirement came from a shared \
         `#![config(...)]` on a `property!` block, it is being imposed on a property that does \
         not use it — move this property into its own block, or give it the label."
    } else {
        ""
    };
    assert!(
        unmet.is_empty(),
        "{name}: the generator did not produce the required coverage, so this property passed \
         over input that never reached the cases it is meant to check.\n{}\n\n{}{shared_block_hint}",
        unmet
            .iter()
            .map(|(label, required, actual)| format!(
                "  {label:?}: required {required:.2}% of cases, got {actual:.2}%"
            ))
            .collect::<Vec<_>>()
            .join("\n"),
        stats::report()
    );
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

    fn observe(&mut self, tape: &[u64], score: Option<f64>) {
        let Some(score) = score.filter(|s| s.is_finite()) else {
            return;
        };
        if self.best_score.is_none_or(|best| score > best) {
            self.best_score = Some(score);
            self.best_tape = Some(tape.to_vec());
        }
    }
}

enum Verdict {
    Held,
    Failed(Failure),
    /// Carries no reason: during replay the reason is not reported anywhere, because a
    /// rejection while shrinking only means "discard this candidate". The search path tracks
    /// reasons separately, where they do reach the user.
    Rejected,
    Overrun,
}

/// Rebuild a value from `tape` and run the property on it.
///
/// A free function rather than a closure, so the shrinker and the search can both borrow
/// `property` without one of them holding it for the whole run.
fn replay<T: 'static>(
    generator: &Gen<T>,
    property: &mut impl FnMut(T) -> Outcome,
    tape: Vec<u64>,
) -> Verdict {
    let mut source = Source::replaying(tape);
    match generator.draw(&mut source) {
        Err(DrawError::Overrun) => Verdict::Overrun,
        Err(DrawError::Rejected(_)) => Verdict::Rejected,
        Ok(value) => match run_quietly(property, value) {
            Ok(()) => Verdict::Held,
            Err(failure) => Verdict::Failed(failure),
        },
    }
}

/// Shrink, persist, and panic with the report.
///
/// The `panic` lint is allowed here, and only here, because panicking is this function's
/// *purpose* rather than a failure to handle an error. A `#[test]` reports a failure by
/// unwinding; there is no other channel to libtest, and the return type `!` says so. The
/// workspace denies `panic` to stop a library from aborting a consumer's process over a
/// recoverable condition, which is not what this is.
#[allow(clippy::panic)]
fn report<T: 'static>(
    name: &str,
    config: &Config,
    tape: &[u64],
    original: &Failure,
    generator: &Gen<T>,
    property: &mut impl FnMut(T) -> Outcome,
    was_persisted: bool,
) -> ! {
    let (minimal, probes) = shrink::minimise(tape.to_vec(), |candidate| {
        match replay(generator, property, candidate.to_vec()) {
            Verdict::Failed(_) => Probe::StillFails,
            _ => Probe::Discard,
        }
    });

    // Re-run the minimal case so the message a reader sees describes the *small* input rather
    // than whatever the search happened to stumble on first.
    let final_failure = match replay(generator, property, minimal.clone()) {
        Verdict::Failed(failure) => failure,
        _ => original.clone(),
    };

    if !was_persisted && let Some(path) = config.regressions.as_deref() {
        persist::save(path, &minimal, name);
    }

    let saved = match (&config.regressions, was_persisted) {
        (Some(path), false) => format!("\n  saved to:   {path}"),
        (Some(path), true) => format!("\n  replayed:   {path}"),
        (None, _) => String::new(),
    };
    let truncated = if probes >= shrink::MAX_PROBES {
        "\n  note:       shrinking hit its probe budget, so this may not be minimal"
    } else {
        ""
    };

    panic!(
        "property failed: {name}\n  \
         {}\n  at:         {}\n  \
         shrank:     {} -> {} tape entries in {probes} probes\n  \
         tape:       {}{saved}{truncated}",
        final_failure.message,
        final_failure.location,
        tape.len(),
        minimal.len(),
        persist::encode(&minimal),
    );
}

/// Install a panic hook that stays quiet only while [`PROBING`] is set.
///
/// The flag is the whole design, and the first attempt at this got it wrong in a way worth
/// recording: installing a hook that suppressed *everything* also suppressed the final report,
/// so a failing property printed only `FAILED` with no message. The engine's own tests did not
/// catch it, because they read the panic payload through `catch_unwind` and the payload was
/// always correct — it was only the printing that was lost. A hook must therefore distinguish
/// "a probe is running" from "we are reporting", and that is what the flag does.
fn install_hook_once() {
    static INSTALLED: AtomicBool = AtomicBool::new(false);

    if !INSTALLED.swap(true, Ordering::SeqCst) {
        let previous = panic::take_hook();
        panic::set_hook(Box::new(move |info| {
            let probing = PROBING.with(Cell::get);
            if !probing || std::env::var_os("EUCLID_PROPERTY_VERBOSE").is_some() {
                previous(info);
            }
        }));
    }
}

thread_local! {
    /// Whether a candidate case is currently being probed, in which case a panic is expected
    /// and its backtrace is noise — shrinking re-runs a failing property hundreds of times.
    static PROBING: Cell<bool> = const { Cell::new(false) };
}

/// Run the property with probe panics suppressed, converting a panic into a failure.
fn run_quietly<T>(property: &mut impl FnMut(T) -> Outcome, value: T) -> Outcome {
    install_hook_once();
    PROBING.with(|probing| probing.set(true));
    let outcome = catch(property, value);
    PROBING.with(|probing| probing.set(false));
    outcome
}

fn catch<T>(property: &mut impl FnMut(T) -> Outcome, value: T) -> Outcome {
    match panic::catch_unwind(AssertUnwindSafe(|| property(value))) {
        Ok(outcome) => outcome,
        Err(payload) => {
            let message = payload
                .downcast_ref::<&str>()
                .map(|s| (*s).to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "panicked".to_string());
            Err(Failure::new(
                format!("panicked: {message}"),
                "inside the property",
            ))
        }
    }
}

fn dedup(reasons: &[&'static str]) -> Vec<&'static str> {
    let mut seen: Vec<&'static str> = Vec::new();
    for reason in reasons {
        if !seen.contains(reason) {
            seen.push(reason);
        }
    }
    seen.truncate(4);
    seen
}

#[cfg(test)]
mod tests;
