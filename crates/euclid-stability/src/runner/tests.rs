//! The runner: that it finds failures, shrinks them, and does not mistake inconclusive for
//! passing.
//!
//! These use [`check`] directly and catch its panic, because a failing property is reported by
//! panicking and that is exactly what needs testing. `check` is the function a `#[test]`
//! normally calls and never expects to return.

use std::panic::{self, AssertUnwindSafe};

use super::{Config, check};
use crate::Failure;
use crate::generate::{Gen, int, vec as gen_vec};

/// Run `check` and capture its panic message, if any.
///
/// `Debug` is required because the runner renders each case to put it on the failure report and
/// into the regression file — see [`persist`](crate::persist) for why that rendering is checked
/// rather than decorative.
fn failure_message<T: 'static + core::fmt::Debug>(
    generator: Gen<T>,
    property: impl FnMut(T) -> Result<(), Failure>,
) -> Option<String> {
    let config = Config::default().cases(512);
    let mut property = property;
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        check("probe", &config, &generator, &mut property);
    }));
    match result {
        Ok(()) => None,
        Err(payload) => Some(
            payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_string()))
                .unwrap_or_else(|| "non-string panic".to_string()),
        ),
    }
}

#[test]
fn a_true_property_passes_quietly() {
    let message = failure_message(int(0, 100), |v| {
        if (0..=100).contains(&v) {
            Ok(())
        } else {
            Err(Failure::new("out of range", "here"))
        }
    });
    assert!(message.is_none(), "a true property reported: {message:?}");
}

#[test]
fn a_false_property_is_found_and_reported() {
    let message = failure_message(int(0, 100), |v| {
        if v < 50 {
            Ok(())
        } else {
            Err(Failure::new(format!("{v} is too big"), "here"))
        }
    })
    .expect("a false property must be reported");

    assert!(message.contains("property failed: probe"), "{message}");
    assert!(message.contains("tape:"), "report must carry the tape");
}

/// The payoff: the reported case must be the *boundary*, not whatever the search found first.
#[test]
fn a_found_failure_is_shrunk_to_the_boundary() {
    let message = failure_message(int(0, 1_000), |v| {
        if v < 50 {
            Ok(())
        } else {
            Err(Failure::new(format!("value was {v}"), "here"))
        }
    })
    .expect("a false property must be reported");

    assert!(
        message.contains("value was 50"),
        "expected the minimal failing value 50, got:\n{message}"
    );
}

/// A list property must shrink to the shortest failing list, which is the case a reader can
/// actually reason about.
#[test]
fn a_list_failure_shrinks_to_the_shortest_list() {
    let message = failure_message(gen_vec(int(0, 100), 0, 20), |xs: Vec<i64>| {
        if xs.iter().filter(|v| **v >= 10).count() < 2 {
            Ok(())
        } else {
            Err(Failure::new(format!("list was {xs:?}"), "here"))
        }
    })
    .expect("a false property must be reported");

    // Two elements, each exactly at the threshold, is the minimum that still fails.
    assert!(
        message.contains("list was [10, 10]"),
        "expected the minimal list [10, 10], got:\n{message}"
    );
}

/// The dependent-generator case, end to end: the shrunk counterexample must still respect the
/// bound the first draw established. This is the property proptest's `max_flat_map_regens`
/// limit makes awkward, so it is worth pinning.
#[test]
fn shrinking_preserves_a_dependent_bound() {
    let generator = int(1, 8).and_then(|n| gen_vec(int(0, n - 1), 1, 6).map(move |xs| (n, xs)));

    let message = failure_message(generator, |(n, xs): (i64, Vec<i64>)| {
        // Sanity: the generator's own invariant. If shrinking ever broke it, this would be the
        // failure reported instead of the intended one.
        if xs.iter().any(|x| *x >= n) {
            return Err(Failure::new(
                format!("GENERATOR INVARIANT BROKEN: n={n}, xs={xs:?}"),
                "here",
            ));
        }
        if xs.len() < 3 {
            Ok(())
        } else {
            Err(Failure::new(format!("n={n}, xs={xs:?}"), "here"))
        }
    })
    .expect("a false property must be reported");

    assert!(
        !message.contains("GENERATOR INVARIANT BROKEN"),
        "shrinking produced a value outside its dependent bound:\n{message}"
    );
    assert!(
        message.contains("xs=[0, 0, 0]"),
        "expected the minimal list of three zeroes, got:\n{message}"
    );
}

/// A panicking property is a failure, shrunk like any other — not an aborted test run reporting
/// an unshrunk input.
#[test]
fn a_panicking_property_is_caught_and_shrunk() {
    let message = failure_message(int(0, 1_000), |v| {
        assert!(v < 50, "panicked at {v}");
        Ok(())
    })
    .expect("a panicking property must be reported");

    assert!(message.contains("panicked"), "{message}");
    assert!(
        message.contains("panicked at 50"),
        "the panic should have been shrunk to the boundary, got:\n{message}"
    );
}

/// Rejecting everything must be reported, never mistaken for the property holding. A generator
/// filtered down to nothing is the quiet way a suite stops testing anything.
#[test]
fn rejecting_every_case_is_an_error_not_a_pass() {
    let generator = int(0, 10).filter("impossible", |_| false);
    let message = failure_message(generator, |_: i64| Ok(())).expect(
        "a generator that rejects everything must be reported, not treated as a passing property",
    );
    assert!(
        message.contains("rejected"),
        "the report should say why: {message}"
    );
}

/// A heavy but not total rejection rate is tolerated, since that is a legitimate (if wasteful)
/// way to express a precondition.
#[test]
fn partial_rejection_still_runs_the_property() {
    let generator = int(0, 100).filter("evens only", |v| v % 2 == 0);
    let mut seen = 0usize;
    let message = failure_message(generator, |v: i64| {
        seen += 1;
        if v % 2 == 0 {
            Ok(())
        } else {
            Err(Failure::new("odd slipped through the filter", "here"))
        }
    });
    assert!(message.is_none(), "{message:?}");
    assert!(seen > 0, "the property never ran");
}

#[test]
fn the_default_seed_is_deterministic() {
    let run = || {
        failure_message(int(0, 10_000), |v| {
            if v < 9_000 {
                Ok(())
            } else {
                Err(Failure::new(format!("saw {v}"), "here"))
            }
        })
    };
    assert_eq!(run(), run(), "two runs disagreed, so the seed is not fixed");
}

// ===== Coverage requirements ==============================================

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

// ===== Targeted search ====================================================

/// Targeting must actually steer. The property scores how far the input is from a hidden point
/// that uniform sampling is very unlikely to approach, and the climb must get closer than a
/// pure random search over the same budget.
///
/// Stated as a comparison against `targeted_fraction(0.0)` rather than against an absolute
/// threshold, because the claim is "the climb helps", and that is the only honest way to say it.
#[test]
fn targeting_finds_a_higher_score_than_uniform_search() {
    fn best_score_over(fraction: f64) -> f64 {
        let config = Config::default().cases(600).targeted_fraction(fraction);
        let observed = std::cell::Cell::new(f64::NEG_INFINITY);
        let mut property = |v: i64| {
            // A smooth score peaking at a specific large value. Smooth is the requirement:
            // a step function gives the climber no gradient to follow.
            #[allow(clippy::cast_precision_loss)]
            let score = -((v - 900_000) as f64).abs();
            crate::target(score);
            observed.set(observed.get().max(score));
            Ok(())
        };
        check("targeting", &config, &int(0, 1_000_000), &mut property);
        observed.get()
    }

    let uniform = best_score_over(0.0);
    let targeted = best_score_over(0.5);

    assert!(
        targeted > uniform,
        "targeting did not improve on uniform search: targeted {targeted:e} vs uniform {uniform:e}"
    );
}

/// A property that reports no score must behave exactly as if targeting were off, so adding the
/// feature cannot change an existing suite.
#[test]
fn a_property_with_no_score_is_unaffected_by_targeting() {
    let run = |fraction: f64| {
        let config = Config::default().cases(128).targeted_fraction(fraction);
        let seen = std::cell::RefCell::new(Vec::new());
        let mut property = |v: i64| {
            seen.borrow_mut().push(v);
            Ok(())
        };
        check("untargeted", &config, &int(0, 1000), &mut property);
        seen.into_inner()
    };
    assert_eq!(
        run(0.0),
        run(1.0),
        "targeting changed the cases of a property that reports no score"
    );
}

/// Targeting must never make a property more permissive — it changes which cases are tried, not
/// which ones pass.
#[test]
fn targeting_never_hides_a_failure() {
    let config = Config::default().cases(256).targeted_fraction(1.0);
    let mut property = |v: i64| {
        crate::target(f64::from(i32::try_from(v).unwrap_or(0)));
        if v < 50 {
            Ok(())
        } else {
            Err(Failure::new(format!("value was {v}"), "here"))
        }
    };
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        check("targeted", &config, &int(0, 100), &mut property);
    }));
    assert!(
        result.is_err(),
        "a false property passed because targeting was enabled"
    );
}
