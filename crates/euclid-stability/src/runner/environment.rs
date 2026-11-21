//! Environment overrides, and the budget gate.
//!
//! A child module so the file stays inside the crate's 400-line budget. These are the parts that
//! read the *outside world* — `EUCLID_STABILITY_SOAK`, `EUCLID_STABILITY_SEED` and the recorded
//! budget file — which is why they are together and why they are the parts with their own
//! parsing tests.

use super::*;

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
pub(crate) fn with_overrides(config: &Config) -> Config {
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
pub(crate) fn apply_overrides(config: &Config, soak: Option<&str>, seed: Option<&str>) -> Config {
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
pub(crate) fn parse_seed(raw: &str) -> Option<u64> {
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
pub(crate) fn rejection_summary(rejects: usize, attempts: usize) -> String {
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
pub(crate) fn enforce_budget(name: &str, path: &str, best: Option<f64>) -> Result<(), String> {
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
