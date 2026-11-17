//! Failures must reproduce and shrinking must preserve their identity.

use super::*;
use crate::{RunErrorKind, run};

#[test]
fn a_flaky_failure_is_not_minimized_or_persisted() {
    let path = scratch("flaky");
    let mut calls = 0;
    let error = run(
        "flaky",
        &Config::default().regressions(&path),
        &int(0, 10),
        |_| {
            calls += 1;
            if calls == 1 {
                Err(Failure::new("first call only", "flaky"))
            } else {
                Ok(())
            }
        },
    )
    .expect_err("unreproducible failure");
    assert_eq!(error.kind, RunErrorKind::Unstable);
    assert!(error.counterexample.is_none());
    assert!(!std::path::Path::new(&path).exists());
}

#[test]
fn a_generator_that_changes_under_replay_is_reported() {
    let state = std::rc::Rc::new(std::cell::Cell::new(0));
    let generator = Gen::new(move |_| {
        let value = state.get();
        state.set(1 - value);
        Ok(value)
    });
    let error = run("generator changed", &Config::default(), &generator, |_| {
        Err(Failure::new("always false", "fixed assertion"))
    })
    .expect_err("the same tape must render identically");
    assert_eq!(error.kind, RunErrorKind::Unstable);
    assert!(error.counterexample.is_none());
}

#[test]
fn shrinking_cannot_switch_to_a_different_assertion() {
    let error = run(
        "two failures",
        &Config::default().edges_first(),
        &int(0, 20),
        |value| match value {
            0 => Ok(()),
            1..=9 => Err(Failure::new("a different bug", "small")),
            _ => Err(Failure::new("the original bug", "large")),
        },
    )
    .expect_err("edge 20 fails first");
    let case = error.counterexample.expect("stable counterexample");
    assert_eq!(case.case, "10");
    assert_eq!(case.failure.identity, "large");
}

#[test]
fn panic_locations_distinguish_failures_even_with_changing_messages() {
    let error = run(
        "two panics",
        &Config::default().edges_first(),
        &int(0, 20),
        |value| {
            if value >= 10 {
                panic!("large {value}");
            }
            if value > 0 {
                panic!("small {value}");
            }
            Ok(())
        },
    )
    .expect_err("edge 20 panics");
    assert_eq!(error.kind, RunErrorKind::Property);
    let case = error.counterexample.expect("stable panic");
    assert_eq!(case.case, "10");
    assert!(case.failure.message.contains("large 10"));
    assert_eq!(case.failure.kind, crate::FailureKind::Panic);
}

#[test]
fn the_final_minimal_case_must_still_fail_consistently() {
    let mut saw_one = 0;
    let error = run(
        "unstable minimum",
        &Config::default().edges_first(),
        &int(0, 20),
        |value| {
            if value == 0 {
                return Ok(());
            }
            if value == 1 {
                saw_one += 1;
                if saw_one > 1 {
                    return Ok(());
                }
            }
            Err(Failure::new("shared assertion", "one site"))
        },
    )
    .expect_err("cached shrink verdict cannot establish stability");
    assert_eq!(error.kind, RunErrorKind::Unstable);
    assert!(error.counterexample.is_none());
}

#[test]
fn metamorphic_helpers_give_different_laws_distinct_failure_identities() {
    let left = crate::relation::involution("increment", 1, |v| v + 1, &crate::relation::Exact)
        .unwrap_err();
    let right = crate::relation::idempotent("increment", 1, |v| v + 1, &crate::relation::Exact)
        .unwrap_err();
    assert_ne!(left.identity, right.identity);
}

#[test]
fn a_panicking_case_formatter_cannot_escape_the_structured_runner() {
    #[derive(Clone)]
    struct BrokenDebug;
    impl core::fmt::Debug for BrokenDebug {
        fn fmt(&self, _: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            panic!("broken formatter")
        }
    }
    let error = run(
        "rendering",
        &Config::default(),
        &crate::generate::just(BrokenDebug),
        |_| Err(Failure::new("failed", "assertion")),
    )
    .unwrap_err();
    assert_eq!(error.kind, RunErrorKind::Unstable);
    assert!(error.message.contains("broken formatter"));
    assert!(error.counterexample.is_none());
}

#[test]
fn assertions_on_one_source_line_remain_distinct_while_shrinking() {
    #[rustfmt::skip]
    fn assertions(value: i64) -> crate::Outcome { if value < 10 { crate::check!(value == 0, "small"); } crate::check!(value < 10, "large"); Ok(()) }

    let error = run(
        "same-line assertions",
        &Config::default().edges_first(),
        &int(0, 20),
        assertions,
    )
    .unwrap_err();
    let case = error.counterexample.unwrap();
    assert_eq!(case.case, "10");
    assert_eq!(case.failure.message, "large");
}
