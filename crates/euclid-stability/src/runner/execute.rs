//! Search orchestration with structured exits and scoped per-run instrumentation.

use super::*;

struct Scope {
    stats: stats::State,
    target: Option<f64>,
}

impl Scope {
    fn enter() -> Self {
        let scope = Self {
            stats: stats::snapshot(),
            target: target::reported(),
        };
        stats::reset();
        target::clear();
        scope
    }
}

impl Drop for Scope {
    fn drop(&mut self) {
        stats::restore(self.stats.clone());
        target::restore(self.target);
    }
}

/// Search, shrink and record evidence without panicking on property or generator failures.
///
/// `Outcome` remains Result-based. `Failure::discard` (used by `assume!`) rejects a case.
/// Rejected cases and their labels/scores do not count as successful evidence. The returned
/// report includes effective environment overrides; nested runs preserve parent instrumentation.
pub fn run<T: 'static + core::fmt::Debug>(
    name: &str,
    config: &Config,
    generator: &Gen<T>,
    mut property: impl FnMut(T) -> Outcome,
) -> Result<RunReport, RunError> {
    let _scope = Scope::enter();
    let mut report = RunReport::new(name, with_overrides(config));
    let config = report.config.clone();
    if config.cases == 0
        || config.replay_checks < 2
        || !config.targeted_fraction.is_finite()
        || !(0.0..=1.0).contains(&config.targeted_fraction)
        || config
            .requirements
            .iter()
            .any(|(_, p)| !p.is_finite() || !(0.0..=100.0).contains(p))
    {
        return Err(RunError::new(
            RunErrorKind::InvalidConfig,
            &report,
            "cases must be positive, replay_checks at least two, and coverage/fraction bounds finite and valid",
        ));
    }
    let metadata = if config.exploration == Exploration::Random {
        None
    } else {
        Some(
            quietly("inside generator exploration", || generator.exploration()).map_err(
                |failure| RunError::new(RunErrorKind::Generation, &report, failure.message),
            )?,
        )
    };
    let planned = match (config.exploration, metadata) {
        (Exploration::Random, _) => None,
        (Exploration::EdgesFirst, Some((_, edges))) => Some(edges),
        (Exploration::Exhaustive, Some((Some(cases), _))) if cases.len() <= config.cases => {
            Some(cases)
        }
        _ => {
            return Err(RunError::new(
                RunErrorKind::UnsupportedExploration,
                &report,
                "exhaustive domain is unknown or exceeds the configured case budget",
            ));
        }
    };

    if planned
        .as_ref()
        .is_some_and(|cases| cases.iter().any(|tape| tape.len() > config.max_tape))
    {
        return Err(RunError::new(
            RunErrorKind::UnsupportedExploration,
            &report,
            "an exploration tape exceeds max_tape; raise the tape budget explicitly",
        ));
    }

    let persisted = config
        .regressions
        .as_deref()
        .map(persist::load)
        .unwrap_or_default();
    let mut stale = Vec::new();
    let mut current = Vec::new();
    for entry in persisted {
        report.replayed += 1;
        let (verdict, rendering) = failure::replay_rendered(generator, &mut property, &entry.tape);
        match verdict {
            Verdict::Failed(original) => {
                return Err(failure::counterexample(
                    &report,
                    &entry.tape,
                    &original,
                    generator,
                    &mut property,
                    true,
                ));
            }
            Verdict::Generation(failure) => {
                return Err(RunError::new(
                    RunErrorKind::Generation,
                    &report,
                    failure.message,
                ));
            }
            _ => {}
        }
        if !entry.belongs_to(name) {
            continue;
        }
        current.push((entry.tape.clone(), rendering.clone()));
        let changed = match (&entry.rendering, rendering) {
            (Some(recorded), Some(now)) if *recorded != now => Some(now),
            (Some(_), None) => Some("<no longer produces a case>".into()),
            _ => None,
        };
        if let Some(now) = changed {
            stale.push(Stale {
                tape: persist::encode(&entry.tape),
                recorded: entry.rendering.unwrap_or_default(),
                now,
            });
        }
    }
    if !stale.is_empty()
        && let Some(path) = config.regressions.as_deref()
    {
        report_stale(name, path, &stale, &current)
            .map_err(|message| RunError::new(RunErrorKind::StaleRegression, &report, message))?;
    }
    stats::reset();
    let mut source = Source::generating(config.seed, config.max_tape);
    let mut climber = Climber::new(&config);
    // At most cases - 1 passes and max_rejects rejections can precede
    // the final attempt. The iterator advances independently of the report.
    let max_attempts = config.cases.saturating_add(config.max_rejects);
    for index in 0..max_attempts {
        if report.passed >= config.cases || report.rejected > config.max_rejects {
            break;
        }
        if config.exploration == Exploration::Exhaustive
            && index >= planned.as_ref().map_or(0, |cases| cases.len())
        {
            report.exhaustive = true;
            break;
        }
        let seed = config
            .seed
            .wrapping_add(index as u64)
            .wrapping_mul(0x2545_F491);
        report.attempts = index + 1;
        if let Some(tape) = planned.as_ref().and_then(|cases| cases.get(index)) {
            source.restart_replaying(tape);
            if config.exploration == Exploration::EdgesFirst {
                report.edge_cases += 1;
            }
        } else {
            match climber.propose(report.passed, seed) {
                Some(tape) => source.restart_climbing(&tape, seed, config.max_tape),
                None => source.restart_generating(seed, config.max_tape),
            }
        }
        let draw = match quietly("inside the generator", || generator.draw(&mut source)) {
            Ok(draw) => draw,
            Err(failure) => {
                report.labels = stats::collected();
                report.best_score = climber.best_score();
                return Err(RunError::new(
                    RunErrorKind::Generation,
                    &report,
                    failure.message,
                ));
            }
        };
        let rejection = match draw {
            Err(DrawError::Overrun) => Some("tape exhausted".to_string()),
            Err(DrawError::Rejected(rejected)) => Some(rejected.reason.to_string()),
            Ok(value) => {
                target::clear();
                stats::begin_case();
                let outcome = run_quietly(&mut property, value);
                let rejected = outcome
                    .as_ref()
                    .err()
                    .is_some_and(|failure| failure.kind == FailureKind::Discard);
                stats::finish_case(!rejected);
                if rejected {
                    outcome.err().map(|failure| failure.message)
                } else {
                    let score = target::reported();
                    if outcome.is_err() || climber.would_improve(score) {
                        let tape = source.consumed();
                        climber.observe(&tape, score);
                        if let Err(original) = outcome {
                            report.failed += 1;
                            report.labels = stats::collected();
                            report.best_score = climber.best_score();
                            return Err(failure::counterexample(
                                &report,
                                &tape,
                                &original,
                                generator,
                                &mut property,
                                false,
                            ));
                        }
                    }
                    report.passed += 1;
                    None
                }
            }
        };
        if let Some(reason) = rejection {
            report.rejected += 1;
            *report.rejection_reasons.entry(reason).or_insert(0) += 1;
        }
    }
    report.labels = stats::collected();
    report.best_score = climber.best_score();
    if config.exploration == Exploration::Exhaustive && report.rejected <= config.max_rejects {
        report.exhaustive = report.attempts == planned.as_ref().map_or(0, |cases| cases.len());
    }
    if report.passed == 0 {
        return Err(RunError::new(
            RunErrorKind::Rejections,
            &report,
            format!(
                "{name}: every one of {} attempts was rejected; the property was never exercised. Reasons: {:?}",
                report.attempts, report.rejection_reasons
            ),
        ));
    }
    if report.rejected > config.max_rejects {
        return Err(RunError::new(
            RunErrorKind::Rejections,
            &report,
            format!(
                "{name}: gave up after {} rejected cases (budget {}), having run only {} of {} cases. Reasons: {:?}",
                report.rejected,
                config.max_rejects,
                report.passed,
                config.cases,
                report.rejection_reasons
            ),
        ));
    }
    let unmet = stats::unmet(&config.requirements);
    if !unmet.is_empty() {
        let hint = if report.labels.is_empty() {
            "\nThis property applied no labels. Check shared block-level configuration."
        } else {
            ""
        };
        let requirements = unmet
            .iter()
            .map(|(label, required, actual)| {
                format!("{label:?}: required {required:.2}% of cases, got {actual:.2}%")
            })
            .collect::<Vec<_>>()
            .join("\n");
        return Err(RunError::new(
            RunErrorKind::Coverage,
            &report,
            format!(
                "{name}: required coverage was not met\n{requirements}\n{}{hint}",
                report.statistics()
            ),
        ));
    }
    if let Some(path) = config.budget.as_deref() {
        enforce_budget(name, path, report.best_score)
            .map_err(|message| RunError::new(RunErrorKind::Budget, &report, message))?;
    }
    Ok(report)
}
