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

/// `cases` means cases that **ran**, not draws attempted.
///
/// This is the test the suite was missing, and its absence hid a real defect for three releases.
/// The assertion above it — `seen > 0` — is true of a run that executed one case in ten and said
/// nothing about it, which is exactly what the engine used to do: a generator rejecting nine in
/// ten ran **19 of 256 cases** and passed green. Counting is the whole point, so the count is
/// what gets asserted.
#[test]
fn a_rejecting_generator_still_runs_every_case_asked_for() {
    // Rejects nine draws in ten.
    let generator = int(0, 99).filter("one in ten", |v| *v < 10);
    let ran = std::cell::Cell::new(0usize);
    let config = Config::default().cases(256);
    let mut property = |_: i64| {
        ran.set(ran.get() + 1);
        Ok(())
    };
    check("rejection", &config, &generator, &mut property);

    assert_eq!(
        ran.get(),
        256,
        "a 90%-rejecting generator ran {} of 256 cases; a rejection must cost a draw, not a case",
        ran.get()
    );
}

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

/// Climbing must not cost cases.
///
/// The second half of the accounting defect above, and the subtler half. A climbing step
/// perturbs one entry of the best tape, and a perturbation that *raises* a length makes the
/// generator read past the end of that tape — which used to overrun, because the candidate was
/// replayed under `Mode::Fixed`. Those overruns were counted as rejections and, before the fix
/// above, silently consumed cases: measured at **228 of 256** on this very generator.
///
/// The score here deliberately rewards the *shortest* list, so the best tape is one entry long
/// and almost every mutation of it asks for more tape than it has. A generator whose score
/// happens to reward length does not reproduce this at all, which is why the first attempt to
/// pin it came back green.
#[test]
fn climbing_does_not_discard_cases() {
    let generator = gen_vec(int(0, 9), 0, 60);

    for fraction in [0.0, 0.5, 1.0] {
        let ran = std::cell::Cell::new(0usize);
        let config = Config::default().cases(256).targeted_fraction(fraction);
        let mut property = |xs: Vec<i64>| {
            ran.set(ran.get() + 1);
            #[allow(clippy::cast_precision_loss)]
            crate::target(-(xs.len() as f64));
            Ok(())
        };
        check("climbing", &config, &generator, &mut property);

        assert_eq!(
            ran.get(),
            256,
            "targeted_fraction({fraction}) ran {} of 256 cases; a climbing step must extend the \
             tape it perturbs rather than overrun it",
            ran.get()
        );
    }
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

// ===== Soak overrides =====================================================

/// The overrides are off unless asked for, which is what keeps the fixed-seed guarantee.
#[test]
fn an_unset_override_changes_nothing() {
    let config = Config::default().cases(400);
    let applied = super::apply_overrides(&config, None, None);
    assert_eq!(applied.cases, 400);
    assert_eq!(applied.seed, config.seed);
    assert_eq!(applied.max_rejects, config.max_rejects);
}

/// A soak factor multiplies the case count **and** the rejection budget.
///
/// Both, or a soak turns every filtered generator into a spurious "gave up" report: ten times the
/// cases need ten times the draws to feed them.
#[test]
fn a_soak_factor_scales_the_search() {
    let config = Config::default().cases(400).max_rejects(100);

    let applied = super::apply_overrides(&config, Some("10"), None);
    assert_eq!(applied.cases, 4_000);
    assert_eq!(applied.max_rejects, 1_000);
    assert_eq!(applied.seed, config.seed, "a soak must not move the seed");

    // Whitespace is tolerated, since these arrive from a shell.
    assert_eq!(
        super::apply_overrides(&config, Some(" 2 "), None).cases,
        800
    );
}

/// A factor of zero, or nonsense, leaves the configuration alone.
///
/// Zero is the dangerous one and the reason for the guard: multiplying the case count by it
/// would run no cases at all, which every property then reports as "never exercised". An
/// environment variable must not be able to turn the whole suite into a failure that looks like
/// a generator bug.
#[test]
fn a_degenerate_soak_factor_is_ignored_rather_than_applied() {
    let config = Config::default().cases(400).max_rejects(100);
    for raw in [
        "0",
        "",
        "nonsense",
        "-1",
        "1.5",
        "99999999999999999999999999",
    ] {
        let applied = super::apply_overrides(&config, Some(raw), None);
        assert_eq!(
            applied.cases, 400,
            "SOAK={raw:?} must leave the case count alone"
        );
        assert_eq!(applied.max_rejects, 100, "SOAK={raw:?}");
    }

    // And the smallest meaningful factor is honoured rather than swept up with the degenerate
    // ones, which is what distinguishes this guard from one that rejects everything.
    assert_eq!(super::apply_overrides(&config, Some("1"), None).cases, 400);
    assert_eq!(
        super::apply_overrides(&config, Some("3"), None).cases,
        1_200
    );
}

/// A seed override replaces the seed and touches nothing else.
#[test]
fn a_seed_override_replaces_only_the_seed() {
    let config = Config::default().cases(400);

    let applied = super::apply_overrides(&config, None, Some("0xFF"));
    assert_eq!(applied.seed, 255);
    assert_eq!(
        applied.cases, 400,
        "a seed override must not move the cases"
    );

    assert_eq!(super::apply_overrides(&config, None, Some("42")).seed, 42);
    // Nonsense leaves the fixed default in place rather than seeding from zero.
    assert_eq!(
        super::apply_overrides(&config, None, Some("nonsense")).seed,
        config.seed
    );

    // Both at once compose.
    let both = super::apply_overrides(&config, Some("5"), Some("7"));
    assert_eq!((both.cases, both.seed), (2_000, 7));
}

/// A seed is accepted in decimal or hex, since the default is written in hex and a reader
/// copying it back in should not have to convert.
#[test]
fn a_seed_parses_in_both_bases() {
    assert_eq!(super::parse_seed("42"), Some(42));
    assert_eq!(super::parse_seed("0xFF"), Some(255));
    assert_eq!(super::parse_seed("0Xff"), Some(255));
    assert_eq!(super::parse_seed(""), None);
    assert_eq!(super::parse_seed("nonsense"), None);
    // The default seed, round-tripped through its own notation.
    assert_eq!(
        super::parse_seed("0x5EED123456789ABC"),
        Some(0x5EED_1234_5678_9ABC)
    );
}

// ===== The two silent-protection features =================================

/// A temp path no other test is using.
fn scratch(name: &str) -> String {
    let dir = std::env::temp_dir().join(format!("euclid-stability-runner-{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::create_dir_all(&dir);
    dir.join("file.txt").to_string_lossy().to_string()
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

// ===== The builders =======================================================

/// A builder sets its own field and **keeps** everything set before it.
///
/// Mutation testing replaced `Config::regressions` and `Config::budget` with
/// `Default::default()`, which silently discards every earlier call in the chain *and* turns the
/// feature off. `settings("round_trips").cases(400)` would quietly become a 256-case run
/// persisting nothing, and no test noticed.
#[test]
fn the_builders_compose_rather_than_reset() {
    let config = Config::default()
        .cases(400)
        .max_rejects(99)
        .targeted_fraction(0.25)
        .require("label", 5.0)
        .regressions("regressions.txt")
        .budget("budgets.txt");

    assert_eq!(
        config.cases, 400,
        "a later builder discarded the case count"
    );
    assert_eq!(config.max_rejects, 99);
    assert!((config.targeted_fraction - 0.25).abs() < 1e-12);
    assert_eq!(config.requirements.len(), 1);
    assert_eq!(config.regressions.as_deref(), Some("regressions.txt"));
    assert_eq!(config.budget.as_deref(), Some("budgets.txt"));
}

/// A tape that runs out mid-draw is a rejection with its own reason, and must be reported.
///
/// The search's `Overrun` arm was unexercised: every rejection test here uses a `filter`, which
/// produces `Rejected`. Mutation testing found the overrun counter could be made to never
/// increment, which turns a generator that can never fit inside `max_tape` from a loud failure
/// into an infinite loop.
#[test]
fn a_generator_that_cannot_fit_the_tape_is_reported() {
    // Twenty elements cannot be drawn from a two-entry tape.
    let generator = gen_vec(int(0, 100), 20, 20);
    let config = Config {
        max_tape: 2,
        ..Config::default().cases(8).max_rejects(16)
    };
    let mut property = |_: Vec<i64>| Ok(());

    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        check("overrun", &config, &generator, &mut property);
    }));
    let message = result
        .expect_err("a generator that always overruns must be reported")
        .downcast_ref::<String>()
        .cloned()
        .unwrap_or_default();

    assert!(
        message.contains("tape exhausted"),
        "the report must name the overrun as the reason: {message}"
    );
}

// ===== The mutation operator ==============================================

/// The climb's mutation operator, tested directly rather than through its outcome.
///
/// [`targeting_finds_a_higher_score_than_uniform_search`] asserts that climbing beats uniform
/// sampling, and that is too weak to pin the operator: mutation testing broke the bit flip four
/// different ways — never flipping at all, only ever *setting* bits, masking the entry down to a
/// single bit, and reducing the shift so only bit zero can move — and the climb still beat random
/// search each time, because the other two arms carried it.
///
/// So this asserts the property the docstring on [`Climber::propose`] actually argues for: **local
/// moves at every scale**. That claim has three measurable parts, and each one is a different
/// mutant:
///
/// - *Local*: most proposals differ from the best tape in exactly one bit. Masking the entry
///   instead of flipping a bit destroys this.
/// - *Every scale*: the flipped bit ranges over the whole word, low and high. Shifting the mask
///   right instead of left confines it to bit zero.
/// - *Both directions*: a bit that is set can be cleared. `|` in place of `^` can only ever set.
///
/// The failing first attempt is worth recording: measuring only "does the proposal differ from
/// the original" passes against all four mutants, because all four still produce *different*
/// tapes. It is the shape of the difference that matters.
#[test]
fn the_mutation_operator_makes_local_moves_at_every_scale() {
    let config = Config::default().cases(2).targeted_fraction(1.0);
    let mut climber = super::Climber::new(&config);
    let original = 0x0F0F_0F0F_0F0F_0F0Fu64;
    climber.observe(&[original], Some(1.0));

    let mut single_bit = 0usize;
    let mut cleared_a_bit = 0usize;
    let mut high_bit_moved = 0usize;
    let mut total = 0usize;

    for case in 0..4_000usize {
        let Some(proposed) = climber.propose(config.cases, case as u64 * 0x9E37_79B9) else {
            continue;
        };
        total += 1;
        let delta = proposed[0] ^ original;
        if delta.count_ones() == 1 {
            single_bit += 1;
        }
        // A bit that was set in the original is now clear: the move went downward.
        if original & delta != 0 {
            cleared_a_bit += 1;
        }
        if delta >= 1u64 << 40 {
            high_bit_moved += 1;
        }
    }

    assert!(total > 3_000, "only {total} of 4000 proposals were made");
    assert!(
        single_bit * 2 > total,
        "only {single_bit} of {total} proposals were single-bit flips; the operator is not \
         making local moves"
    );
    assert!(
        cleared_a_bit * 5 > total,
        "only {cleared_a_bit} of {total} proposals cleared an existing bit; the operator can \
         only move in one direction"
    );
    assert!(
        high_bit_moved * 8 > total,
        "only {high_bit_moved} of {total} proposals moved a bit above 2^40; the operator is \
         confined to small steps"
    );
}

/// Climbing engages only after the exploration budget, and the boundary is exact.
///
/// `targeted_fraction` splits the case budget, so the case at which proposals begin *is* the
/// split. Mutation testing inverted and relaxed that comparison; either way the explore/climb
/// ratio stops being what was configured, which changes every targeted search in the fleet
/// without changing any assertion.
#[test]
fn climbing_begins_exactly_at_the_end_of_exploration() {
    // 100 cases, half explored: the climb starts at case 50.
    let config = Config::default().cases(100).targeted_fraction(0.5);
    let mut climber = super::Climber::new(&config);
    climber.observe(&[1, 2, 3], Some(1.0));

    assert!(
        climber.propose(0, 1).is_none(),
        "case 0 is exploration, not climbing"
    );
    assert!(
        climber.propose(49, 1).is_none(),
        "case 49 is the last exploration case"
    );
    assert!(
        climber.propose(50, 1).is_some(),
        "case 50 is the first climbing case"
    );
    assert!(climber.propose(99, 1).is_some());

    // With targeting off, no case ever climbs.
    let off = Config::default().cases(100).targeted_fraction(0.0);
    let mut never = super::Climber::new(&off);
    never.observe(&[1, 2, 3], Some(1.0));
    assert!(never.propose(99, 1).is_none());
}

/// Every position on the tape must be reachable, or entries past the first never get climbed.
///
/// The position is `rng % tape.len()`. Mutation testing replaced that remainder with a division
/// and an addition, both of which index past the end — and nothing failed, because no test ever
/// asked the climber for a proposal over a multi-entry tape.
#[test]
fn every_tape_position_is_reachable_by_a_proposal() {
    let config = Config::default().cases(2).targeted_fraction(1.0);
    let mut climber = super::Climber::new(&config);
    let original = vec![10u64, 20, 30, 40, 50];
    climber.observe(&original, Some(1.0));

    let mut touched = [false; 5];
    for case in 0..2_000usize {
        let Some(proposed) = climber.propose(config.cases, case as u64 * 0x2545_F491) else {
            continue;
        };
        assert_eq!(proposed.len(), original.len(), "the tape changed length");
        for (i, (before, after)) in original.iter().zip(&proposed).enumerate() {
            if before != after {
                touched[i] = true;
            }
        }
    }

    assert_eq!(
        touched, [true; 5],
        "some tape positions were never proposed for mutation: {touched:?}"
    );
}

/// The rejection rate reports the share of **draws**, not of cases.
///
/// Extracted from the stats block so it can be tested at all — see
/// [`rejection_summary`](super::rejection_summary). The denominator is the one thing worth
/// getting right: a run that made 2,560 draws to get 256 cases rejected 90% of its *draws*, and
/// expressing that against the case count would report 1000%.
#[test]
fn the_rejection_summary_is_a_share_of_draws() {
    // Silent when there is nothing to report.
    assert_eq!(super::rejection_summary(0, 256), "");

    let summary = super::rejection_summary(2_304, 2_560);
    assert!(
        summary.contains("2304 of 2560 draws"),
        "both counts must appear: {summary}"
    );
    assert!(
        summary.contains("90.0%"),
        "2304 of 2560 is 90.0% of draws: {summary}"
    );

    // Half of four is fifty percent, not two hundred.
    assert!(super::rejection_summary(2, 4).contains("50.0%"));
    // And a single rejection out of many is a small share, not a large one.
    assert!(super::rejection_summary(1, 1_000).contains("0.1%"));
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

/// The shrink report distinguishes probes spent from probes reused.
///
/// `reused` is reported only when it is non-zero, and inverting that test makes a report say
/// "0 reused" where there were none and stay silent where there were some. The distinction is the
/// whole point of the memo — "900 probes, 400 reused" explored 900 distinct candidates on a
/// budget that used to stop at 1,300 iterations — so a report that hides it hides the measurement
/// the memo exists to provide.
#[test]
fn the_shrink_report_names_reused_probes_when_there_are_any() {
    let message = failure_message(gen_vec(int(0, 100), 0, 20), |xs: Vec<i64>| {
        if xs.iter().filter(|v| **v >= 10).count() < 2 {
            Ok(())
        } else {
            Err(Failure::new(format!("list was {xs:?}"), "here"))
        }
    })
    .expect("a false property must be reported");

    assert!(
        message.contains("probes"),
        "the report must state the shrink cost: {message}"
    );
    assert!(
        message.contains("reused"),
        "a structured shrink re-probes its fixpoint round, so reuse must be reported: {message}"
    );
}

/// The nudge arm walks **steadily** in one direction, which a bit flip cannot.
///
/// The operator's third component, and the reason the docstring gives for it: "a flip can only
/// ever toggle, so it cannot walk steadily in one direction". That makes its signature a *small
/// change in magnitude spread across several bits* — a bit flip at position `k` always moves the
/// value by exactly `2^k` and so differs in exactly one bit, and a fresh random value moves it
/// enormously.
///
/// Mutation testing broke this two ways: deleting the arm outright, so the value falls through to
/// a fresh draw, and shifting the step left instead of right, so every nudge is huge. Both leave
/// [`the_mutation_operator_makes_local_moves_at_every_scale`] green, because the bit-flip arm
/// still does its job.
#[test]
fn the_nudge_arm_takes_small_multi_bit_steps() {
    let config = Config::default().cases(2).targeted_fraction(1.0);
    let mut climber = super::Climber::new(&config);
    let original = 0x0F0F_0F0F_0F0F_0F0Fu64;
    climber.observe(&[original], Some(1.0));

    let mut small_walks = 0usize;
    for case in 0..8_000usize {
        let Some(proposed) = climber.propose(config.cases, case as u64 * 0x9E37_79B9) else {
            continue;
        };
        let delta_bits = (proposed[0] ^ original).count_ones();
        let magnitude = i128::from(proposed[0]).abs_diff(i128::from(original));
        // Small in value but spread across bits: neither a single flip nor a fresh draw.
        if magnitude > 0 && magnitude < (1 << 20) && delta_bits > 1 {
            small_walks += 1;
        }
    }

    assert!(
        small_walks > 10,
        "only {small_walks} of 8000 proposals were small multi-bit steps; the climb cannot walk \
         steadily towards a nearby peak"
    );
}
