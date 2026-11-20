//! The hill climber, and the staleness report.
//!
//! A child module so the file stays inside the crate's 400-line budget. The climber is what
//! makes a recorded budget a number *at an effort* rather than a worst case: it searches for a
//! worse input, so more cases mean more climbing and the measured score rises with the
//! multiplier. The staleness report is how that is communicated when a baseline is beaten.

use super::*;

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
pub(crate) struct Climber {
    pub(crate) best_tape: Option<Vec<u64>>,
    pub(crate) best_score: Option<f64>,
    pub(crate) first_targeted_case: usize,
    pub(crate) enabled: bool,
}

impl Climber {
    pub(crate) fn new(config: &Config) -> Self {
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

    pub(crate) fn best_score(&self) -> Option<f64> {
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
    pub(crate) fn propose(&self, case: usize, seed: u64) -> Option<Vec<u64>> {
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
    pub(crate) fn would_improve(&self, score: Option<f64>) -> bool {
        self.improving_score(score).is_some()
    }

    pub(crate) fn improving_score(&self, score: Option<f64>) -> Option<f64> {
        score.filter(|s| s.is_finite() && self.best_score.is_none_or(|best| *s > best))
    }

    pub(crate) fn observe(&mut self, tape: &[u64], score: Option<f64>) {
        if let Some(score) = self.improving_score(score) {
            self.best_score = Some(score);
            self.best_tape = Some(tape.to_vec());
        }
    }
}

pub(crate) enum Verdict {
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
pub(crate) struct Stale {
    pub(crate) tape: String,
    pub(crate) recorded: String,
    pub(crate) now: String,
}

/// Report tapes whose renderings no longer match, or refresh the file if asked to.
///
/// Returns an error if any tape has gone stale and this is not a recording run.
pub(crate) fn report_stale(
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
pub(crate) fn run_quietly<T>(property: &mut impl FnMut(T) -> Outcome, value: T) -> Outcome {
    quietly("inside the property", || property(value)).and_then(core::convert::identity)
}

pub(crate) fn quietly<T>(location: &'static str, body: impl FnOnce() -> T) -> Result<T, Failure> {
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

pub(crate) fn forward_panic(probing: bool, verbose: impl FnOnce() -> bool) -> bool {
    !probing || verbose()
}

thread_local! {
    pub(crate) static PROBING: Cell<bool> = const { Cell::new(false) };
    static PANIC_SITE: RefCell<Option<String>> = const { RefCell::new(None) };
}
