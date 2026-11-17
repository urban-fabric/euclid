//! Search checks for the runner module.

use super::*;

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

/// The rejection rate reports the share of **draws**, not of cases.
///
/// Extracted from the stats block so it can be tested at all — see
/// [`rejection_summary`](crate::runner::rejection_summary). The denominator is the one thing worth
/// getting right: a run that made 2,560 draws to get 256 cases rejected 90% of its *draws*, and
/// expressing that against the case count would report 1000%.
#[test]
fn the_rejection_summary_is_a_share_of_draws() {
    // Silent when there is nothing to report.
    assert_eq!(crate::runner::rejection_summary(0, 256), "");

    let summary = crate::runner::rejection_summary(2_304, 2_560);
    assert!(
        summary.contains("2304 of 2560 draws"),
        "both counts must appear: {summary}"
    );
    assert!(
        summary.contains("90.0%"),
        "2304 of 2560 is 90.0% of draws: {summary}"
    );

    // Half of four is fifty percent, not two hundred.
    assert!(crate::runner::rejection_summary(2, 4).contains("50.0%"));
    // And a single rejection out of many is a small share, not a large one.
    assert!(crate::runner::rejection_summary(1, 1_000).contains("0.1%"));
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
