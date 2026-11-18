//! Evidence checks for the runner module.

use super::*;

/// And the budget that bounds those extra draws has to be **reachable**, which it previously was
/// not: the search counted a rejection as a spent case, so `rejects <= cases` held by
/// construction and `max_rejects` could never be exceeded at any sane config. The diagnostic it
/// guards was dead code.
///
/// The generator here rejects half its draws and the budget is far below what 256 cases need, so
/// the run must stop short and say so. Note this is distinct from
/// [`rejecting_every_case_is_an_error_not_a_pass`]: there *nothing* ran, which is its own
/// message. Here the property ran and simply could not be fed enough.
#[test]
fn an_exhausted_rejection_budget_is_reported() {
    let generator = int(0, 1).filter("zero only", |v| *v == 0);
    let config = Config::default().cases(256).max_rejects(8);
    let ran = std::cell::Cell::new(0usize);
    let mut property = |_: i64| {
        ran.set(ran.get() + 1);
        Ok(())
    };

    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        check("budget", &config, &generator, &mut property);
    }));
    let message = result
        .expect_err("a run that cannot reach its case count must be reported, not pass quietly")
        .downcast_ref::<String>()
        .cloned()
        .unwrap_or_default();

    assert!(ran.get() > 0, "precondition: some cases must have run");
    assert!(
        message.contains("gave up") && message.contains("256"),
        "the report should say how many cases of how many asked for ran: {message}"
    );
    assert!(
        message.contains("zero only"),
        "the report should name the assumption doing the rejecting: {message}"
    );
}

/// A budget regression must fail the run, **through `check`**.
///
/// `budget::evaluate` has thorough direct tests and `enforce_budget` had none, so mutation
/// testing could replace that whole function with `()` and leave every test green. This is the
/// crate's headline feature — the check that notices a kernel getting less accurate while every
/// assertion still holds — and it was deletable.
///
/// The same test pins `Climber::best_score`, which could be made to return `Some(0.0)` or
/// `Some(-1.0)`: either reads as a dramatic *improvement* against the baseline, so the run prints
/// a "baseline is now loose" note and passes.
#[test]
fn a_budget_regression_fails_the_run() {
    let path = scratch("budget-regress");
    // A baseline far below what the property will score.
    std::fs::write(&path, "regressing 1e0\n").expect("scratch file is writable");

    let config = Config::default().cases(16).budget(&path);
    let mut property = |_: i64| {
        crate::target(500.0);
        Ok(())
    };

    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        check("regressing", &config, &int(0, 10), &mut property);
    }));
    let message = result
        .expect_err("a score above the recorded baseline must fail the run")
        .downcast_ref::<String>()
        .cloned()
        .unwrap_or_default();

    assert!(
        message.contains("property regressed"),
        "the report must name the regression: {message}"
    );
    assert!(
        message.contains("5e2"),
        "the report must carry the score that was reached: {message}"
    );
    // The percentage is computed against a non-zero baseline, so it must appear.
    assert!(
        message.contains('%'),
        "the report must quantify the rise: {message}"
    );
}

/// A score *within* the baseline passes, so the check is not simply always-fail.
#[test]
fn a_score_inside_its_budget_holds() {
    let path = scratch("budget-hold");
    std::fs::write(&path, "holding 1e3\n").expect("scratch file is writable");

    let config = Config::default().cases(16).budget(&path);
    let mut property = |_: i64| {
        crate::target(1.0);
        Ok(())
    };
    check("holding", &config, &int(0, 10), &mut property);
}

/// A budget over a property that never scores is protection that does not exist, so it is an
/// error rather than a no-op.
#[test]
fn a_budget_without_a_score_is_an_error() {
    let path = scratch("budget-noscore");
    let config = Config::default().cases(16).budget(&path);
    let mut property = |_: i64| Ok(());

    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        check("unscored", &config, &int(0, 10), &mut property);
    }));
    let message = result
        .expect_err("a budget with no target() must be reported")
        .downcast_ref::<String>()
        .cloned()
        .unwrap_or_default();
    assert!(message.contains("never called target()"), "{message}");
}

/// A committed tape whose case has drifted must be reported, **through `check`**.
///
/// `report_stale` was in the same position as `enforce_budget`: replaceable with `()` against a
/// green suite. Both arms of the drift detection were unpinned too — the guard comparing the
/// recorded rendering against the current one, and the arm for a tape that no longer produces a
/// case at all.
///
/// This is the quieter of the two failures and the more expensive. A stale tape still decodes, so
/// the file still looks like a regression suite; it simply pins a different case than the one
/// that was committed, and nothing says so.
#[test]
fn a_stale_regression_rendering_is_reported() {
    let path = scratch("stale");
    // A tape committed as having rendered to something this generator will never produce.
    std::fs::write(&path, "0100000000000000 # stale :: 999\n").expect("writable");

    let config = Config::default().cases(8).regressions(&path);
    let mut property = |_: i64| Ok(());

    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        check("stale", &config, &int(0, 10), &mut property);
    }));
    let message = result
        .expect_err("a tape whose rendering has drifted must be reported")
        .downcast_ref::<String>()
        .cloned()
        .unwrap_or_default();

    assert!(
        message.contains("stale regressions"),
        "the report must name the problem: {message}"
    );
    assert!(
        message.contains("999"),
        "the report must show what was recorded: {message}"
    );
}

/// And a tape that no longer produces a case at all is the other arm of the same check.
#[test]
fn a_regression_tape_that_no_longer_produces_a_case_is_reported() {
    let path = scratch("stale-nocase");
    std::fs::write(&path, "0100000000000000 # gone :: 7\n").expect("writable");

    // A generator that rejects everything, so the committed tape renders to nothing.
    let generator = int(0, 10).filter("never", |_| false);
    let config = Config::default().cases(8).max_rejects(4).regressions(&path);
    let mut property = |_: i64| Ok(());

    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        check("gone", &config, &generator, &mut property);
    }));
    let message = result
        .expect_err("the run cannot pass")
        .downcast_ref::<String>()
        .cloned()
        .unwrap_or_default();

    assert!(
        message.contains("stale regressions") && message.contains("no longer produces a case"),
        "a tape that produces no case must be reported as stale: {message}"
    );
}

/// A newly found counterexample is **saved**, and a replayed one is not saved again.
///
/// The condition is `if !was_persisted && ...`, and dropping that negation inverts both halves:
/// a fresh failure is never recorded, so the search that found it has to find it again next time,
/// and a replayed one is appended on every run. Mutation testing found it unpinned because no
/// test checked the file afterwards.
#[test]
fn a_found_failure_is_persisted_for_the_next_run() {
    let path = scratch("persist-found");
    let config = Config::default().cases(256).regressions(&path);
    let mut property = |v: i64| {
        if v < 50 {
            Ok(())
        } else {
            Err(Failure::new(format!("value was {v}"), "here"))
        }
    };

    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        check("persisted", &config, &int(0, 1_000), &mut property);
    }));
    let message = result
        .expect_err("the property is false")
        .downcast_ref::<String>()
        .cloned()
        .unwrap_or_default();
    assert!(message.contains("saved to"), "{message}");

    let saved = crate::persist::load(&path);
    assert_eq!(
        saved.len(),
        1,
        "the minimised counterexample must be written to {path}"
    );
    assert!(
        saved[0].belongs_to("persisted"),
        "the line must name the property that found it"
    );
    assert_eq!(
        saved[0].rendering.as_deref(),
        Some("50"),
        "the saved line must carry the minimised case, not the one first found"
    );
}
