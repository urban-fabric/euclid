//! Replay health checks and minimisation that preserves the failing assertion.

use super::*;

pub(super) fn replay_rendered<T: 'static + core::fmt::Debug>(
    generator: &Gen<T>,
    property: &mut impl FnMut(T) -> Outcome,
    tape: &[u64],
) -> (Verdict, Option<String>) {
    let mut source = Source::replaying(tape.to_vec());
    match quietly("inside the generator", || generator.draw(&mut source)) {
        Err(failure) => (Verdict::Generation(failure), None),
        Ok(Err(DrawError::Overrun)) => (Verdict::Overrun, None),
        Ok(Err(DrawError::Rejected(_))) => (Verdict::Rejected, None),
        Ok(Ok(value)) => match quietly("inside case rendering", || persist::render(&value)) {
            Ok(rendering) => (verdict(run_quietly(property, value)), Some(rendering)),
            Err(failure) => (Verdict::Generation(failure), None),
        },
    }
}

fn verdict(outcome: Outcome) -> Verdict {
    match outcome {
        Ok(()) => Verdict::Held,
        Err(failure) if failure.kind == FailureKind::Discard => Verdict::Rejected,
        Err(failure) => Verdict::Failed(failure),
    }
}

fn stable<T: 'static + core::fmt::Debug>(
    report: &RunReport,
    generator: &Gen<T>,
    property: &mut impl FnMut(T) -> Outcome,
    tape: &[u64],
    original: &Failure,
) -> Result<(Failure, String), RunError> {
    let mut result = None;
    for _ in 0..report.config.replay_checks {
        let (verdict, rendering) = replay_rendered(generator, property, tape);
        let detail = match &verdict {
            Verdict::Generation(failure) => format!("; {}", failure.message),
            _ => String::new(),
        };
        if let (Verdict::Failed(failure), Some(case)) = (verdict, rendering)
            && failure.same_failure(original)
            && result
                .as_ref()
                .is_none_or(|(_, previous)| *previous == case)
        {
            result = Some((failure, case));
        } else {
            return Err(RunError::new(
                RunErrorKind::Unstable,
                report,
                format!(
                    "unstable property: {}: counterexample did not reproduce the same case and failure{detail}; tape {} was not saved",
                    report.name,
                    persist::encode(tape)
                ),
            ));
        }
    }
    result.ok_or_else(|| {
        RunError::new(
            RunErrorKind::InvalidConfig,
            report,
            "replay checks must be positive",
        )
    })
}

pub(super) fn counterexample<T: 'static + core::fmt::Debug>(
    report: &RunReport,
    tape: &[u64],
    original: &Failure,
    generator: &Gen<T>,
    property: &mut impl FnMut(T) -> Outcome,
    was_persisted: bool,
) -> RunError {
    if let Err(error) = stable(report, generator, property, tape, original) {
        return error;
    }
    let mut scratch = Source::replaying(Vec::new());
    let (minimal, cost) = shrink::minimise(tape.to_vec(), |candidate| {
        scratch.restart_replaying(candidate);
        match quietly("inside the generator", || generator.draw(&mut scratch)) {
            Ok(Ok(value)) => match run_quietly(property, value) {
                Err(failure)
                    if failure.kind != FailureKind::Discard && failure.same_failure(original) =>
                {
                    Probe::StillFails
                }
                _ => Probe::Discard,
            },
            _ => Probe::Discard,
        }
    });
    let (final_failure, case) = match stable(report, generator, property, &minimal, original) {
        Ok(result) => result,
        Err(error) => return error,
    };
    let saved_to = if !was_persisted {
        report.config.regressions.as_deref().map(|path| {
            persist::save(path, &minimal, &report.name, Some(&case));
            path.to_string()
        })
    } else {
        None
    };
    let saved = if was_persisted {
        format!("\n  replayed: {:?}", report.config.regressions)
    } else {
        saved_to
            .as_ref()
            .map_or(String::new(), |path| format!("\n  saved to: {path}"))
    };
    let capped = if cost.hit_the_cap() {
        "\n  note: shrinking hit its probe budget; this may not be minimal"
    } else {
        ""
    };
    let reused = reuse_suffix(cost.reused);
    let message = format!(
        "property failed: {}\n  {}\n  at: {}\n  case: {case}\n  shrank: {} -> {} tape entries in {} probes{reused}\n  tape: {}{saved}{capped}",
        report.name,
        final_failure.message,
        if final_failure.kind == FailureKind::Panic {
            &final_failure.identity
        } else {
            final_failure.location
        },
        tape.len(),
        minimal.len(),
        cost.probes,
        persist::encode(&minimal)
    );
    let mut error = RunError::new(RunErrorKind::Property, report, message);
    error.counterexample = Some(Box::new(Counterexample {
        original_tape: tape.to_vec(),
        tape: minimal,
        case,
        failure: final_failure,
        cost,
        replayed: was_persisted,
        saved_to,
    }));
    error
}

pub(super) fn reuse_suffix(reused: usize) -> String {
    if reused > 0 {
        format!(", {reused} reused")
    } else {
        String::new()
    }
}
