//! Recording the worst error a search found, so that losing accuracy fails the suite.
//!
//! [`target`](crate::target()) makes a search hunt the worst case instead of sampling uniformly,
//! and then throws the answer away. The run prints `ok`, the worst observed error is gone, and
//! the next run starts over. So a refactor that makes `erf` two ulp less accurate passes every
//! check in the repository — the property still holds, because the assertion was written with
//! enough headroom to admit the old error *and* the new one.
//!
//! That is the gap this module closes. The worst score a property reports is written to a
//! committed file, and a later run that scores **higher** fails:
//!
//! ```text
//! property regressed: the_exp_ln_round_trip_is_accurate
//!   worst target() score rose from 3.0e0 to 7.0e0 (+133%)
//!   recorded in: tests/budgets.txt
//! ```
//!
//! The assertion inside the property is still the contract — "within 8 ulp" is what a caller may
//! rely on. The budget is a second, tighter statement: *and today it is actually 3*. One says
//! what is promised, the other notices the day the promise starts being used up.
//!
//! # This only works because the seed is fixed
//!
//! A baseline is worth having only if the number is reproducible, and it is: [`Config::seed`](crate::Config::seed) is
//! fixed by default for exactly this reason, so the search visits the same tapes in the same
//! order and arrives at the same worst case on every machine. The score is a property of the
//! code under test, not of the run.
//!
//! The corollary is that editing a generator, the case count, or `targeted_fraction` changes the
//! score legitimately — a different search finds a different worst case. That is a deliberate
//! edit, so re-record with `EUCLID_STABILITY_RECORD=1` and commit the new number. The failure
//! message says so, because the alternative is a reader concluding their refactor broke a kernel
//! when they had only widened a range.
//!
//! # What a good score is, again
//!
//! The same rule as [`target`](crate::target()): higher must mean worse, and it has to be
//! continuous in the input. A budget over "did it fail" is a budget over `0`, which records
//! nothing and protects nothing. Error in ulps, relative error, or the ratio of observed error
//! to an admissible bound are all good; the third is the best of them, because a score of `1.0`
//! then means "exactly at the limit" and the number is comparable across kernels.
//!
//! # Why a file rather than a constant in the test
//!
//! A constant in the source would do the same job, and the reason not to is the same reason
//! [`persist`](crate::persist) uses a file: the number is *evidence*, not a decision. It is
//! produced by a run, it changes when the code does, and it should arrive in a diff as one line
//! saying `3.0e0 -> 2.1e0` rather than buried in a rewritten assertion. Keeping it out of the
//! property also keeps the property readable — the contract stays a round number a human chose,
//! and the measurement lives where measurements live.

use std::fmt::Write as _;
use std::fs;
use std::path::Path;

/// The environment variable that rewrites baselines instead of checking them.
pub const RECORD: &str = "EUCLID_STABILITY_RECORD";

/// Whether this run should rewrite baselines rather than enforce them.
#[must_use]
pub fn recording() -> bool {
    std::env::var_os(RECORD).is_some()
}

/// One recorded worst score.
#[derive(Clone, Debug, PartialEq)]
pub struct Entry {
    /// The property the score belongs to.
    pub name: String,
    /// The worst score that property's search has reported.
    pub score: f64,
}

/// Every baseline recorded in `path`.
///
/// A missing or unreadable file means "nothing recorded yet", which is the first-run state and
/// not an error — the same degradation [`persist::load`](crate::persist::load) uses, and for the
/// same reason: a harness that panics on a missing file is a harness that cannot be adopted
/// incrementally.
#[must_use]
pub fn load(path: &str) -> Vec<Entry> {
    let Ok(text) = fs::read_to_string(path) else {
        return Vec::new();
    };
    text.lines().filter_map(parse_line).collect()
}

/// The baseline recorded for `name`, if there is one.
#[must_use]
pub fn recorded(path: &str, name: &str) -> Option<f64> {
    load(path)
        .into_iter()
        .find(|entry| entry.name == name)
        .map(|entry| entry.score)
}

/// `<name> <score>`, with `#` comments and blank lines ignored.
fn parse_line(line: &str) -> Option<Entry> {
    let line = line.split('#').next()?.trim();
    if line.is_empty() {
        return None;
    }
    // Score last, so a name containing whitespace still parses. Property names cannot, being
    // Rust identifiers, but a format that only works for well-behaved input is a format that
    // fails on the first surprise.
    let (name, score) = line.rsplit_once(char::is_whitespace)?;
    let score: f64 = score.trim().parse().ok()?;
    score.is_finite().then(|| Entry {
        name: name.trim().to_string(),
        score,
    })
}

/// Write `score` for `name`, replacing any existing entry and keeping the file sorted.
///
/// Sorted so the file is stable under re-recording: two people recording different properties
/// produce diffs that touch one line each rather than reordering the whole file. Failure to
/// write is reported to stderr and otherwise ignored, matching
/// [`persist::save`](crate::persist::save) — a read-only checkout should not turn a passing
/// suite into an I/O panic.
pub fn record(path: &str, name: &str, score: f64) {
    let mut entries = load(path);
    match entries.iter_mut().find(|entry| entry.name == name) {
        Some(entry) => entry.score = score,
        None => entries.push(Entry {
            name: name.to_string(),
            score,
        }),
    }
    entries.sort_by(|a, b| a.name.cmp(&b.name));

    if let Some(parent) = Path::new(path).parent()
        && !parent.as_os_str().is_empty()
        && let Err(error) = fs::create_dir_all(parent)
    {
        eprintln!(
            "euclid-stability: could not create {}: {error}",
            parent.display()
        );
        return;
    }

    let mut out = String::from(
        "# Worst target() score per property, recorded by euclid-stability.\n\
         # Commit this file: a rise here is an accuracy regression that every other check passes.\n\
         # Re-record deliberately with EUCLID_STABILITY_RECORD=1 after changing a generator.\n",
    );
    for entry in &entries {
        let _ = writeln!(out, "{} {:e}", entry.name, entry.score);
    }

    if let Err(error) = fs::write(path, out) {
        eprintln!("euclid-stability: could not write {path}: {error}");
    }
}

/// What a run's score means against its baseline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// No baseline yet; this run's score was recorded as the first one.
    Recorded,
    /// At or below the baseline, by a margin too small to be worth re-recording.
    Held,
    /// Meaningfully better than the baseline, so the baseline is loose and could be tightened.
    Improved,
    /// Worse than the baseline. An accuracy regression.
    Regressed,
}

/// A score is "meaningfully better" at half the baseline or below.
///
/// A looser threshold would nag on noise and a tighter one would never fire. The point of the
/// suggestion is to catch the case where a kernel was *improved* and the baseline was left
/// behind, which is a real and silent loss of protection: a budget of 7 over a kernel that now
/// errs by 0.5 admits a fourteen-fold regression without complaining.
///
/// It is **multiplicative, so it only means anything for a positive baseline.** Against `0` the
/// comparison is `0 <= 0`, which would report an exact kernel as improvable on every run; against
/// a negative baseline the multiplication moves the threshold the wrong way, so a score that is
/// genuinely better reads as further from the limit than it is. Both are excluded below rather
/// than papered over, because the alternative is a nag that fires forever on the one kind of
/// result nobody needs to act on. A regression is still caught by the plain `>` in either case —
/// only the suggestion is suppressed.
const IMPROVEMENT: f64 = 0.5;

/// Compare `score` against the baseline in `path` for `name`, recording where appropriate.
///
/// Returns the verdict and the previous baseline. [`Verdict::Regressed`] is the caller's cue to
/// fail the run; the others are informational.
pub fn evaluate(path: &str, name: &str, score: f64) -> (Verdict, Option<f64>) {
    let previous = recorded(path, name);

    if recording() {
        record(path, name, score);
        return (Verdict::Recorded, previous);
    }

    match previous {
        None => {
            record(path, name, score);
            (Verdict::Recorded, None)
        }
        Some(baseline) if score > baseline => (Verdict::Regressed, Some(baseline)),
        Some(baseline) if baseline > 0.0 && score <= baseline * IMPROVEMENT => {
            (Verdict::Improved, Some(baseline))
        }
        Some(baseline) => (Verdict::Held, Some(baseline)),
    }
}

#[cfg(test)]
mod tests;
