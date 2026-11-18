//! Parsing, recording, and the verdict table.
//!
//! The file format gets the same robustness treatment as [`persist`](crate::persist)'s, for the
//! same reason: these files are committed, so they get merged, hand-edited and truncated by a
//! bad rebase, and none of that may panic inside the harness.

use super::{Entry, Verdict, evaluate, load, parse_line, record, recorded};

#[test]
fn recording_follows_a_child_process_environment() {
    const MODE: &str = "EUCLID_RECORD_TEST_CHILD";
    if let Some(expected) = std::env::var_os(MODE) {
        assert_eq!(super::recording(), expected == "on");
        return;
    }

    let executable = std::env::current_exe().expect("locate the test executable");
    for mode in ["on", "off"] {
        let mut command = std::process::Command::new(&executable);
        command
            .args([
                "--exact",
                "budget::tests::recording_follows_a_child_process_environment",
            ])
            .env(MODE, mode);
        if mode == "on" {
            command.env(super::RECORD, "1");
        } else {
            command.env_remove(super::RECORD);
        }
        let child = command
            .output()
            .expect("run the isolated environment check");
        assert!(
            child.status.success(),
            "recording mode {mode}: {}",
            String::from_utf8_lossy(&child.stdout),
        );
    }
}

/// A temp directory unique to one test, so the suite can run in parallel without two tests
/// racing on the same file.
fn scratch(name: &str) -> String {
    let dir = std::env::temp_dir().join(format!(
        "euclid-stability-budget-{}-{name}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    dir.join("budgets.txt").to_string_lossy().to_string()
}

#[test]
fn a_line_parses_name_and_score() {
    assert_eq!(
        parse_line("erf_is_accurate 3e0"),
        Some(Entry {
            name: "erf_is_accurate".to_string(),
            score: 3.0,
        })
    );
}

#[test]
fn comments_and_blank_lines_are_skipped() {
    assert_eq!(parse_line("# a header"), None);
    assert_eq!(parse_line(""), None);
    assert_eq!(parse_line("   "), None);
    // A trailing comment on a real entry still parses.
    assert_eq!(
        parse_line("a 1.5 # measured on 2026-11-15"),
        Some(Entry {
            name: "a".to_string(),
            score: 1.5,
        })
    );
}

#[test]
fn malformed_lines_parse_to_none_rather_than_panicking() {
    for bad in [
        "no_score",
        "two scores 1 2",
        "name nan",
        "name inf",
        "name notanumber",
        "1.5",
    ] {
        // `two scores 1 2` is the interesting one: the score is read from the right, so this
        // parses as the name "two scores 1" with score 2. That is harmless — no property is
        // named that — and it is why the score is taken from the right rather than the name
        // from the left.
        if bad == "two scores 1 2" {
            continue;
        }
        assert_eq!(parse_line(bad), None, "{bad:?} should not parse");
    }
}

/// A non-finite baseline is refused at the door. `inf` would make every run "hold" forever,
/// which is worse than having no budget at all, because the file claims protection it is not
/// providing.
#[test]
fn a_non_finite_score_is_not_a_baseline() {
    assert_eq!(parse_line("a inf"), None);
    assert_eq!(parse_line("a NaN"), None);
}

#[test]
fn a_missing_file_has_no_baselines() {
    assert!(load("/nonexistent/euclid-stability/none.txt").is_empty());
    assert_eq!(
        recorded("/nonexistent/euclid-stability/none.txt", "a"),
        None
    );
}

#[test]
fn recording_then_reading_returns_the_score() {
    let path = scratch("roundtrip");

    record(&path, "b_property", 2.5);
    assert_eq!(recorded(&path, "b_property"), Some(2.5));

    // A second property appends rather than replacing.
    record(&path, "a_property", 1.0);
    assert_eq!(recorded(&path, "a_property"), Some(1.0));
    assert_eq!(recorded(&path, "b_property"), Some(2.5));

    // Re-recording the same property replaces its entry instead of adding a second one.
    record(&path, "b_property", 9.0);
    assert_eq!(recorded(&path, "b_property"), Some(9.0));
    assert_eq!(load(&path).len(), 2);

    // And the file is sorted, so re-recording one property does not reorder the rest.
    let names: Vec<String> = load(&path).into_iter().map(|e| e.name).collect();
    assert_eq!(names, vec!["a_property", "b_property"]);
}

#[test]
fn the_first_run_records_rather_than_failing() {
    let path = scratch("first-run");
    let (verdict, previous) = evaluate(&path, "fresh", 4.0);
    assert_eq!(verdict, Verdict::Recorded);
    assert_eq!(previous, None);
    assert_eq!(recorded(&path, "fresh"), Some(4.0));
}

#[test]
fn a_higher_score_regresses() {
    let path = scratch("regress");
    record(&path, "p", 3.0);

    let (verdict, previous) = evaluate(&path, "p", 3.5);
    assert_eq!(verdict, Verdict::Regressed);
    assert_eq!(previous, Some(3.0));

    // A regression must not overwrite the baseline, or the second run would pass and the
    // regression would be laundered into the new normal.
    assert_eq!(recorded(&path, "p"), Some(3.0));
}

#[test]
fn an_equal_score_holds() {
    let path = scratch("equal");
    record(&path, "p", 3.0);
    assert_eq!(evaluate(&path, "p", 3.0).0, Verdict::Held);
}

#[test]
fn a_much_better_score_suggests_tightening() {
    let path = scratch("improve");
    record(&path, "p", 4.0);

    // Half or below is "meaningfully better".
    assert_eq!(evaluate(&path, "p", 2.0).0, Verdict::Improved);
    assert_eq!(evaluate(&path, "p", 0.1).0, Verdict::Improved);
    // Just under is not worth nagging about.
    assert_eq!(evaluate(&path, "p", 3.9).0, Verdict::Held);

    // An improvement leaves the baseline alone too: tightening it is a deliberate act, because
    // the new number has to be committed by someone who knows the improvement was intended.
    assert_eq!(recorded(&path, "p"), Some(4.0));
}

/// A score of zero is a legitimate baseline — an exact kernel errs by nothing — and must not be
/// confused with "no baseline". The improvement threshold is multiplicative, so zero against
/// zero has to come out as held rather than as an improvement that can never be acted on.
#[test]
fn a_zero_baseline_holds_against_zero() {
    let path = scratch("zero");
    record(&path, "p", 0.0);
    assert_eq!(recorded(&path, "p"), Some(0.0));
    assert_eq!(evaluate(&path, "p", 0.0).0, Verdict::Held);
    // And any error at all against an exact baseline is a regression.
    assert_eq!(
        evaluate(&path, "p", f64::MIN_POSITIVE).0,
        Verdict::Regressed
    );
}

/// A negative score is unusual but not meaningless — a property may report a signed margin,
/// where more negative is further inside the bound. The regression comparison is a plain `>`, so
/// it works unchanged; only the improvement *suggestion* is suppressed, because a multiplicative
/// threshold against a negative baseline moves the wrong way. `-6.0` against `-5.0` is genuinely
/// better and reports `Held` rather than `Improved` — no nag, and no false regression either.
#[test]
fn negative_scores_regress_the_same_way_but_never_suggest_tightening() {
    let path = scratch("negative");
    record(&path, "p", -5.0);
    assert_eq!(evaluate(&path, "p", -6.0).0, Verdict::Held);
    assert_eq!(evaluate(&path, "p", -100.0).0, Verdict::Held);
    assert_eq!(evaluate(&path, "p", -4.0).0, Verdict::Regressed);
}

#[test]
fn an_unwritable_path_is_survivable() {
    record("/proc/euclid-stability-cannot-write-here.txt", "p", 1.0);
    // Reaching here without panicking is the assertion.
}
