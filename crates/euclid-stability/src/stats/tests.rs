//! Classification and the coverage requirement.
//!
//! These manipulate the thread-local collector directly, so they must run on one thread each
//! and must reset first — libtest gives each test its own thread, which is what makes a
//! thread-local collector workable here at all.

use super::{cases_counted, collected, count_case, label, report, reset, share, unmet};

#[test]
fn labels_accumulate_counts() {
    reset();
    label("alpha");
    label("alpha");
    label("beta");

    let counts = collected();
    assert_eq!(counts.get("alpha"), Some(&2));
    assert_eq!(counts.get("beta"), Some(&1));
    assert_eq!(counts.get("gamma"), None);
}

#[test]
fn reset_clears_everything() {
    reset();
    label("alpha");
    count_case();
    reset();
    assert!(collected().is_empty());
    assert_eq!(cases_counted(), 0);
}

/// The denominator is the cases that *ran*, not the cases configured. A rejected case never
/// reached the body, so counting it would dilute coverage and let a requirement pass while the
/// generator was mostly producing rejects.
#[test]
fn share_is_over_cases_that_ran() {
    reset();
    for _ in 0..4 {
        count_case();
    }
    label("hit");
    assert!((share("hit") - 25.0).abs() < 1e-9, "got {}", share("hit"));
    assert_eq!(share("never"), 0.0);
}

#[test]
fn share_of_nothing_is_zero_rather_than_a_division_by_zero() {
    reset();
    assert_eq!(share("anything"), 0.0);
}

#[test]
fn an_unmet_requirement_is_reported_with_both_numbers() {
    reset();
    for _ in 0..10 {
        count_case();
    }
    label("rare");

    let shortfalls = unmet(&[("rare".to_string(), 50.0)]);
    assert_eq!(shortfalls.len(), 1);
    let (name, required, actual) = &shortfalls[0];
    assert_eq!(name, "rare");
    assert!((required - 50.0).abs() < 1e-9);
    assert!((actual - 10.0).abs() < 1e-9);
}

#[test]
fn a_met_requirement_is_not_reported() {
    reset();
    for _ in 0..10 {
        count_case();
        label("common");
    }
    assert!(unmet(&[("common".to_string(), 50.0)]).is_empty());
}

/// A label that never appeared is the case this whole module exists for: the generator produced
/// nothing in the region the property claims to check, and the run must not pass silently.
#[test]
fn a_label_that_never_appeared_is_a_shortfall() {
    reset();
    for _ in 0..100 {
        count_case();
    }
    let shortfalls = unmet(&[("subnormal".to_string(), 1.0)]);
    assert_eq!(shortfalls.len(), 1);
    assert!((shortfalls[0].2 - 0.0).abs() < 1e-9);
}

#[test]
fn the_report_names_every_label_with_its_share() {
    reset();
    for _ in 0..4 {
        count_case();
    }
    label("alpha");
    label("alpha");
    label("beta");

    let text = report();
    assert!(text.contains("alpha"), "{text}");
    assert!(text.contains("beta"), "{text}");
    assert!(text.contains("50.00%"), "{text}");
    assert!(text.contains("25.00%"), "{text}");
}

#[test]
fn an_unlabelled_run_still_reports_its_case_count() {
    reset();
    for _ in 0..7 {
        count_case();
    }
    let text = report();
    assert!(text.contains('7'), "{text}");
    assert!(text.contains("no labels"), "{text}");
}
