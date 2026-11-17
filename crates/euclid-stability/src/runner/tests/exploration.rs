//! Explicit deterministic edges and complete small domains.

use super::*;
use crate::{RunErrorKind, generate, run};

#[test]
fn edge_first_tries_both_range_ends_before_the_interior() {
    let mut seen = Vec::new();
    let report = run(
        "edges",
        &Config::default().cases(3).edges_first(),
        &int(10, 20),
        |value| {
            seen.push(value);
            Ok(())
        },
    )
    .expect("holds");
    assert_eq!(seen, [10, 20, 15]);
    assert_eq!(report.edge_cases, 3);
    assert!(!report.exhaustive);
}

#[test]
fn exhaustive_pairing_visits_all_four_boolean_combinations() {
    let mut seen = std::collections::BTreeSet::new();
    let generator = generate::pair(generate::boolean(), generate::boolean());
    let report = run(
        "bool pairs",
        &Config::default().cases(4).exhaustive(),
        &generator,
        |case| {
            assert!(seen.insert(case), "no duplicate input in this domain");
            Ok(())
        },
    )
    .expect("all finite cases hold");
    assert!(report.exhaustive);
    assert_eq!(report.passed, 4);
    assert_eq!(seen.len(), 4);
}

#[test]
fn exhaustive_map_filter_and_fixed_arrays_compose() {
    let generator = generate::array::<_, 3>(generate::boolean());
    let mut seen = std::collections::BTreeSet::new();
    let report = run(
        "arrays",
        &Config::default().exhaustive(),
        &generator,
        |value| {
            seen.insert(value);
            Ok(())
        },
    )
    .expect("finite boolean arrays");
    assert!(report.exhaustive);
    assert_eq!((report.passed, seen.len()), (8, 8));

    let report = run(
        "filter",
        &Config::default().exhaustive(),
        &int(-2, 2).filter("nonzero", |v| *v != 0).map(|v| v * v),
        |value| {
            assert!([1, 4].contains(&value));
            Ok(())
        },
    )
    .expect("all valid finite cases hold");
    assert!(report.exhaustive);
    assert_eq!((report.passed, report.rejected), (4, 1));
}

#[test]
fn unknown_or_over_budget_domains_are_not_called_exhaustive() {
    for generator in [
        int(0, 1000),
        int(0, 2).and_then(|n| int(0, n)),
        generate::uint(0, u64::MAX).map(|v| v as i64),
    ] {
        let error = run(
            "unknown",
            &Config::default().cases(5).exhaustive(),
            &generator,
            |_| Ok(()),
        )
        .expect_err("cannot establish complete enumeration");
        assert_eq!(error.kind, RunErrorKind::UnsupportedExploration);
        assert!(!error.report.exhaustive);
    }
}

#[test]
fn optional_results_and_bounded_sequences_enumerate_every_branch() {
    let generator = generate::option(generate::result(
        generate::boolean(),
        generate::just("error"),
    ));
    let mut seen = std::collections::BTreeSet::new();
    let report = run(
        "sum types",
        &Config::default().exhaustive(),
        &generator,
        |value| {
            seen.insert(value);
            Ok(())
        },
    )
    .expect("all variants");
    assert!(report.exhaustive);
    assert_eq!(seen.len(), 4);
    let generator = generate::vec(generate::boolean(), 0, 2);
    let report = run("lists", &Config::default().exhaustive(), &generator, |_| {
        Ok(())
    })
    .unwrap();
    assert!(report.exhaustive);
    assert_eq!(report.passed, 7);
}

#[test]
fn failure_from_edge_exploration_is_persisted_as_a_normal_replayable_tape() {
    let path = scratch("edge-persistence");
    let generator = int(0, 9);
    let fail = |value| {
        if value == 9 {
            Err(Failure::new("upper end", "edge"))
        } else {
            Ok(())
        }
    };
    let error = run(
        "edge failure",
        &Config::default().edges_first().regressions(&path),
        &generator,
        fail,
    )
    .unwrap_err();
    let case = error.counterexample.unwrap();
    assert_eq!(case.case, "9");
    let again = run(
        "edge failure",
        &Config::default().regressions(&path),
        &generator,
        fail,
    )
    .unwrap_err();
    assert_eq!(again.report.replayed, 1);
    assert!(again.counterexample.unwrap().replayed);
}

#[test]
fn explicit_examples_all_run_before_random_cases() {
    let generator = int(100, 200).with_examples(vec![2, 3, 5, 7, 11]);
    let mut seen = Vec::new();
    run(
        "examples",
        &Config::default().cases(5).edges_first(),
        &generator,
        |value| {
            seen.push(value);
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(seen, [2, 3, 5, 7, 11]);
}

#[test]
fn one_of_preserves_complete_finite_branches() {
    let generator = generate::one_of(vec![int(0, 1), int(10, 12)]);
    let mut seen = std::collections::BTreeSet::new();
    let report = run(
        "branches",
        &Config::default().exhaustive(),
        &generator,
        |value| {
            seen.insert(value);
            Ok(())
        },
    )
    .unwrap();
    assert!(report.exhaustive);
    assert_eq!(seen, [0, 1, 10, 11, 12].into_iter().collect());
}

#[test]
fn exploration_cannot_silently_exceed_the_tape_budget() {
    let generator = generate::pair(generate::boolean(), generate::boolean());
    for config in [
        Config::default().edges_first(),
        Config::default().exhaustive(),
    ] {
        let config = Config {
            max_tape: 1,
            ..config
        };
        let error = run("short tape", &config, &generator, |_| Ok(())).unwrap_err();
        assert_eq!(error.kind, RunErrorKind::UnsupportedExploration);
        assert_eq!(error.report.attempts, 0);
    }
    let config = Config {
        max_tape: 2,
        ..Config::default().cases(4).exhaustive()
    };
    let report = run("exact tape budget", &config, &generator, |_| Ok(())).unwrap();
    assert!(report.exhaustive);
    assert_eq!(report.passed, 4);
}

#[test]
fn unsigned_domains_enumerate_the_inclusive_upper_bound() {
    for (low, high) in [(7u64, 11u64), (9, 9), (u64::MAX - 2, u64::MAX)] {
        let mut seen = std::collections::BTreeSet::new();
        let report = run(
            "unsigned domain",
            &Config::default().exhaustive(),
            &generate::uint(low, high),
            |value| {
                seen.insert(value);
                Ok(())
            },
        )
        .unwrap();
        assert!(report.exhaustive);
        assert_eq!(seen, (low..=high).collect());
    }
}

#[test]
fn finite_ranges_and_weighted_alternatives_cover_their_declared_domains() {
    for width in 1..=32 {
        let low = -17;
        let high = low + width - 1;
        let mut seen = std::collections::BTreeSet::new();
        let report = run(
            "range",
            &Config::default().exhaustive(),
            &int(low, high),
            |value| {
                seen.insert(value);
                Ok(())
            },
        )
        .unwrap();
        assert!(report.exhaustive);
        assert_eq!(seen, (low..=high).collect());
    }
    let generator = generate::weighted(vec![(0, int(-5, -1)), (7, int(2, 4)), (3, int(10, 11))]);
    let mut seen = std::collections::BTreeSet::new();
    let report = run(
        "weights",
        &Config::default().exhaustive(),
        &generator,
        |value| {
            seen.insert(value);
            Ok(())
        },
    )
    .unwrap();
    assert!(report.exhaustive);
    assert_eq!(seen, [2, 3, 4, 10, 11].into_iter().collect());
}

#[test]
fn large_composed_domains_refuse_an_incomplete_exhaustive_run() {
    for generator in [
        generate::vec(generate::boolean(), 0, 12),
        generate::vec(generate::just(false), 0, 65),
    ] {
        let error = run(
            "too large",
            &Config::default().cases(20_000).exhaustive(),
            &generator,
            |_| Ok(()),
        )
        .unwrap_err();
        assert_eq!(error.kind, RunErrorKind::UnsupportedExploration);
        assert_eq!(error.report.attempts, 0);
    }
}

#[test]
fn small_primitive_edges_do_not_waste_the_budget_on_duplicates() {
    let mut pairs = std::collections::BTreeSet::new();
    run(
        "boolean edges",
        &Config::default().cases(4).edges_first(),
        &generate::pair(generate::boolean(), generate::boolean()),
        |value| {
            pairs.insert(value);
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(pairs.len(), 4);

    let mut seen = std::collections::BTreeSet::new();
    run(
        "singleton component",
        &Config::default().cases(3).edges_first(),
        &generate::pair(int(0, 2), int(7, 7)),
        |value| {
            seen.insert(value);
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(seen, [(0, 7), (1, 7), (2, 7)].into_iter().collect());
}

#[test]
fn the_largest_supported_boolean_array_domain_can_be_completed() {
    let mut seen = std::collections::BTreeSet::new();
    let report = run(
        "twelve flags",
        &Config::default().cases(4096).exhaustive(),
        &generate::array::<_, 12>(generate::boolean()),
        |value| {
            seen.insert(value);
            Ok(())
        },
    )
    .unwrap();
    assert!(report.exhaustive);
    assert_eq!((report.passed, seen.len()), (4096, 4096));
}

#[test]
fn list_edges_visit_the_empty_and_maximum_structures() {
    let mut seen = Vec::new();
    run(
        "list edges",
        &Config::default().cases(3).edges_first(),
        &generate::vec(generate::boolean(), 0, 2),
        |value| {
            seen.push(value);
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(seen, vec![vec![], vec![false, false], vec![true, true]]);
}

#[test]
fn alternatives_share_the_enumeration_limit_instead_of_each_spending_it() {
    let generator = generate::one_of(vec![int(0, 3000), int(4000, 7000)]);
    let error = run(
        "combined budget",
        &Config::default().cases(10_000).exhaustive(),
        &generator,
        |_| Ok(()),
    )
    .unwrap_err();
    assert_eq!(error.kind, RunErrorKind::UnsupportedExploration);
    assert_eq!(error.report.attempts, 0);
}
