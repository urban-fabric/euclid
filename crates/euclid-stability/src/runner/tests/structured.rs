//! Structured results and honest accepted-case accounting.

use super::*;
use crate::{RunErrorKind, assume, classify, run, target};

#[test]
fn a_success_reports_effective_config_counts_labels_and_score() {
    let config = Config::default().cases(12).require("accepted", 100.0);
    let report = run("passing", &config, &int(0, 20), |_| {
        classify!("accepted");
        classify!("accepted");
        target(7.0);
        Ok(())
    })
    .expect("property holds");
    assert_eq!(
        (report.attempts, report.passed, report.rejected),
        (12, 12, 0)
    );
    assert_eq!(report.labels.get("accepted"), Some(&12));
    assert_eq!(report.best_score, Some(7.0));
    assert_eq!(report.config.seed, config.seed);
    assert!(report.statistics().contains("100.00%"));
}

#[test]
fn assumptions_cannot_pass_a_property_that_never_checks_anything() {
    let error = run(
        "vacuous",
        &Config::default().cases(8).max_rejects(3),
        &int(0, 1),
        |_| {
            assume!(false, "outside domain");
            Ok(())
        },
    )
    .expect_err("discard is not a success");
    assert_eq!(error.kind, RunErrorKind::Rejections);
    assert_eq!(
        (
            error.report.passed,
            error.report.rejected,
            error.report.attempts
        ),
        (0, 4, 4)
    );
    assert_eq!(
        error.report.rejection_reasons.get("outside domain"),
        Some(&4)
    );
}

#[test]
fn rejected_property_cases_do_not_contribute_coverage_or_target_scores() {
    let mut reached = 0;
    let report = run(
        "filtered",
        &Config::default().cases(40).require("accepted", 100.0),
        &int(0, 1),
        |value| {
            if value == 0 {
                classify!("discarded");
                target(1e9);
            }
            assume!(value != 0);
            classify!("accepted");
            target(1.0);
            reached += 1;
            Ok(())
        },
    )
    .expect("enough accepted draws exist");
    assert_eq!(reached, 40);
    assert_eq!(report.passed, 40);
    assert!(report.rejected > 0);
    assert_eq!(report.attempts, report.passed + report.rejected);
    assert!(!report.labels.contains_key("discarded"));
    assert_eq!(report.labels.get("accepted"), Some(&40));
    assert_eq!(report.best_score, Some(1.0));
}

#[test]
fn a_stable_failure_returns_a_replayable_minimal_case() {
    let generator = int(0, 100);
    let error = run("bounded", &Config::default(), &generator, |value| {
        if value >= 50 {
            Err(Failure::new(format!("too large: {value}"), "boundary"))
        } else {
            Ok(())
        }
    })
    .expect_err("the boundary is reachable");
    assert_eq!(error.kind, RunErrorKind::Property);
    assert_eq!(error.report.failed, 1);
    assert_eq!(error.report.attempts, error.report.passed + 1);
    let counterexample = error.counterexample.expect("a stable failure has evidence");
    let replay = generator
        .draw(&mut crate::Source::replaying(counterexample.tape))
        .expect("complete tape");
    assert_eq!(replay, 50);
    assert_eq!(counterexample.case, "50");
    assert_eq!(counterexample.failure.identity, "boundary");
    assert!(counterexample.cost.probes > 0);
}

#[test]
fn generator_panics_are_structured_failures() {
    let generator = Gen::<i64>::new(|_| panic!("bad generator"));
    let error = run("broken generator", &Config::default(), &generator, |_| {
        Ok(())
    })
    .expect_err("generator panics");
    assert_eq!(error.kind, RunErrorKind::Generation);
    assert!(error.message.contains("bad generator"));
    assert_eq!(error.report.attempts, 1);
}

#[test]
fn generator_exploration_panics_also_return_structured_errors() {
    let generator = crate::generate::weighted(vec![(0, int(0, 1))]);
    for config in [
        Config::default().edges_first(),
        Config::default().exhaustive(),
    ] {
        let error = run("invalid exploration", &config, &generator, |_| Ok(())).unwrap_err();
        assert_eq!(error.kind, RunErrorKind::Generation);
        assert_eq!(error.report.attempts, 0);
        assert!(
            error
                .message
                .contains("weights must sum to a positive value")
        );
    }
}

#[test]
fn invalid_configuration_is_reported_before_running_the_property() {
    for config in [
        Config::default().cases(0),
        Config::default().replay_checks(1),
        Config::default().require("unmeasurable", f64::NAN),
        Config::default().require("impossible", 101.0),
        Config::default().targeted_fraction(f64::NAN),
        Config {
            targeted_fraction: -0.1,
            ..Config::default()
        },
        Config {
            targeted_fraction: 1.1,
            ..Config::default()
        },
    ] {
        let error = run("invalid", &config, &int(0, 1), |_| {
            panic!("must not execute")
        })
        .expect_err("invalid configuration");
        assert_eq!(error.kind, RunErrorKind::InvalidConfig);
        assert_eq!(error.report.attempts, 0);
    }
}

#[test]
fn nested_runs_preserve_parent_labels_scores_and_probe_state() {
    let report = run("outer", &Config::default().cases(2), &int(0, 1), |_| {
        classify!("outer");
        target(3.0);
        let inner = run("inner", &Config::default().cases(3), &int(0, 1), |_| {
            classify!("inner");
            target(99.0);
            Ok(())
        })
        .expect("inner holds");
        assert_eq!(inner.labels.get("inner"), Some(&3));
        assert_eq!(crate::target::reported(), Some(3.0));
        assert!(crate::runner::PROBING.with(std::cell::Cell::get));
        classify!("after inner");
        Ok(())
    })
    .expect("outer holds");
    assert_eq!(report.labels.get("outer"), Some(&2));
    assert_eq!(report.labels.get("after inner"), Some(&2));
    assert!(!report.labels.contains_key("inner"));
    assert_eq!(report.best_score, Some(3.0));
    assert!(!crate::runner::PROBING.with(std::cell::Cell::get));
}

#[test]
fn reports_distinguish_coverage_budget_and_stale_evidence_errors() {
    let coverage = run(
        "coverage",
        &Config::default().cases(3).require("absent", 1.0),
        &int(0, 1),
        |_| Ok(()),
    )
    .expect_err("coverage is absent");
    assert_eq!(coverage.kind, RunErrorKind::Coverage);
    assert_eq!(coverage.report.passed, 3);

    let path = scratch("structured-budget");
    std::fs::write(&path, "budget 1e0\n").expect("writable");
    let budget = run(
        "budget",
        &Config::default().cases(3).budget(&path),
        &int(0, 1),
        |_| {
            target(2.0);
            Ok(())
        },
    )
    .expect_err("accuracy regressed");
    assert_eq!(budget.kind, RunErrorKind::Budget);
    assert_eq!(budget.report.best_score, Some(2.0));

    let path = scratch("structured-stale");
    crate::persist::save(&path, &[0], "stale", Some("outdated"));
    let stale = run(
        "stale",
        &Config::default().regressions(&path),
        &int(0, 1),
        |_| Ok(()),
    )
    .expect_err("a recorded rendering changed");
    assert_eq!(stale.kind, RunErrorKind::StaleRegression);
    assert_eq!(stale.report.replayed, 1);
}

#[test]
fn failure_statistics_exclude_replay_and_shrink_probes() {
    let error = run("labeled failure", &Config::default(), &int(0, 10), |_| {
        classify!("reached");
        target(5.0);
        Err(Failure::new("always fails", "fixed"))
    })
    .unwrap_err();
    assert_eq!((error.report.passed, error.report.failed), (0, 1));
    assert_eq!(error.report.labels.get("reached"), Some(&1));
    assert!(error.report.statistics().contains("100.00%"));
    assert_eq!(error.report.best_score, Some(5.0));
}

#[test]
fn shrinking_cannot_use_a_case_excluded_by_an_assumption() {
    let error = run(
        "positive domain",
        &Config::default().edges_first(),
        &int(0, 20),
        |value| {
            assume!(value >= 7);
            Err(Failure::new("fails in the domain", "one assertion"))
        },
    )
    .unwrap_err();
    assert_eq!(error.report.rejected, 1);
    assert_eq!(error.counterexample.unwrap().case, "7");
}

#[test]
fn exactly_the_rejection_budget_is_allowed() {
    let report = run(
        "one rejection",
        &Config::default().cases(2).max_rejects(1).edges_first(),
        &int(0, 2),
        |value| {
            assume!(value != 0);
            Ok(())
        },
    )
    .unwrap();
    assert_eq!((report.passed, report.rejected), (2, 1));
    let report = run(
        "no rejections allowed",
        &Config::default().cases(2).max_rejects(0),
        &int(0, 2),
        |_| Ok(()),
    )
    .unwrap();
    assert_eq!((report.passed, report.rejected), (2, 0));
}

#[test]
fn a_replayed_assumption_discards_instead_of_becoming_a_counterexample() {
    let path = scratch("replayed-assumption");
    crate::persist::save(&path, &[0], "domain", Some("0"));
    let report = run(
        "domain",
        &Config::default().cases(3).regressions(&path),
        &int(0, 1),
        |value| {
            assume!(value == 1);
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(report.replayed, 1);
    assert_eq!(report.passed, 3);
}

#[test]
fn a_generator_panic_during_regression_replay_stops_before_fresh_search() {
    let path = scratch("replayed-generator-panic");
    crate::persist::save(&path, &[0], "generator", None);
    let generator = Gen::<i64>::new(|_| panic!("replay generator failed"));
    let error = run(
        "generator",
        &Config::default().regressions(&path),
        &generator,
        |_| Ok(()),
    )
    .unwrap_err();
    assert_eq!(error.kind, RunErrorKind::Generation);
    assert!(error.message.contains("replay generator failed"));
    assert_eq!(error.report.replayed, 1);
    assert_eq!(error.report.attempts, 0);
}
