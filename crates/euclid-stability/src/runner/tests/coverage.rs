//! Coverage checks for the runner module.

use super::*;

/// The headline claim of [`crate::stats`]: a property whose generator never reached the
/// interesting region must **fail**, not pass green. Reporting the distribution is not enough,
/// because nobody reads a passing test's output.
#[test]
fn an_unmet_coverage_requirement_fails_the_run() {
    let config = Config::default().cases(64).require("impossible", 1.0);
    let mut property = |v: i64| {
        crate::classify!("ordinary");
        if (0..=10).contains(&v) {
            Ok(())
        } else {
            Err(Failure::new("out of range", "here"))
        }
    };

    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        check("coverage", &config, &int(0, 10), &mut property);
    }));
    let message = result
        .expect_err("an unmet requirement must fail the run")
        .downcast_ref::<String>()
        .cloned()
        .unwrap_or_default();

    assert!(message.contains("required coverage"), "{message}");
    assert!(message.contains("impossible"), "{message}");
    assert!(
        message.contains("ordinary"),
        "the report should show what WAS generated: {message}"
    );
}

#[test]
fn a_met_coverage_requirement_passes() {
    let config = Config::default().cases(64).require("seen", 90.0);
    let mut property = |_: i64| {
        crate::classify!("seen");
        Ok(())
    };
    check("coverage", &config, &int(0, 10), &mut property);
}

/// A real failure must win over a coverage shortfall, because a property that is false has more
/// urgent news than one whose generator was narrow.
#[test]
fn a_real_failure_is_reported_in_preference_to_a_coverage_shortfall() {
    let config = Config::default().cases(64).require("never", 50.0);
    let message = failure_message(int(0, 100), |v| {
        if v < 50 {
            Ok(())
        } else {
            Err(Failure::new(format!("value was {v}"), "here"))
        }
    })
    .expect("the property is false");
    assert!(message.contains("property failed"), "{message}");
    assert!(
        !message.contains("required coverage"),
        "the coverage shortfall should not have masked the real failure: {message}"
    );
    drop(config);
}
