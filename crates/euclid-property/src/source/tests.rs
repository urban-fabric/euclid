//! The tape's contract: a value is a pure function of it, and editing it is safe.

use super::{Mode, Source};

#[test]
fn a_recorded_tape_replays_identically() {
    let mut generating = Source::generating(99, 64);
    let drawn: Vec<i64> = (0..8)
        .map(|_| generating.in_range(0, 1_000).expect("room on the tape"))
        .collect();
    let tape = generating.consumed();

    let mut replaying = Source::replaying(tape);
    let again: Vec<i64> = (0..8)
        .map(|_| replaying.in_range(0, 1_000).expect("tape has the entries"))
        .collect();

    assert_eq!(drawn, again, "replay must reproduce the draw exactly");
}

/// The property the shrinker depends on: a fixed tape refuses to invent entries. Topping it up
/// would silently change the case under examination, making every shrink unsound.
#[test]
fn a_fixed_tape_overruns_rather_than_inventing() {
    let mut source = Source::replaying(vec![1, 2]);
    assert!(source.below(10).is_ok());
    assert!(source.below(10).is_ok());
    assert!(
        source.below(10).is_err(),
        "a fixed tape must overrun, not extend"
    );
}

#[test]
fn a_growing_tape_extends_up_to_its_limit() {
    let mut source = Source::generating(1, 3);
    assert!(source.below(10).is_ok());
    assert!(source.below(10).is_ok());
    assert!(source.below(10).is_ok());
    assert!(
        source.below(10).is_err(),
        "max_len must bound a growing tape"
    );
}

/// One tape entry per draw, whatever the bound. This is what keeps tape *positions* stable
/// under editing: if a draw's cost depended on its bound, lowering an early entry could shift
/// every later read and turn a local edit into a global reinterpretation.
#[test]
fn every_draw_costs_exactly_one_entry() {
    let mut source = Source::generating(7, 64);
    for bound in [0u64, 1, 2, 3, 1_000, u64::MAX] {
        let before = source.consumed().len();
        let _ = source.below(bound).expect("room on the tape");
        assert_eq!(
            source.consumed().len(),
            before + 1,
            "bound {bound} did not cost exactly one entry"
        );
    }
}

/// A degenerate bound still consumes an entry and still answers zero, for the same reason.
#[test]
fn a_degenerate_bound_is_still_a_draw() {
    let mut source = Source::replaying(vec![12_345]);
    assert_eq!(source.below(1), Ok(0));
    assert!(source.below(1).is_err(), "the entry was consumed");
}

#[test]
fn in_range_stays_within_its_bounds() {
    let mut source = Source::generating(4, 4_096);
    for _ in 0..500 {
        let value = source.in_range(-7, 11).expect("room on the tape");
        assert!((-7..=11).contains(&value), "{value} escaped the range");
    }
}

#[test]
fn in_range_covers_both_endpoints() {
    let mut source = Source::generating(11, 4_096);
    let mut saw_low = false;
    let mut saw_high = false;
    for _ in 0..400 {
        match source.in_range(0, 3) {
            Ok(0) => saw_low = true,
            Ok(3) => saw_high = true,
            _ => {}
        }
    }
    assert!(saw_low && saw_high, "range endpoints were never produced");
}

/// A single-value range is legal and must not consume randomness unpredictably.
#[test]
fn a_single_value_range_is_constant() {
    let mut source = Source::generating(3, 64);
    for _ in 0..8 {
        assert_eq!(source.in_range(5, 5), Ok(5));
    }
}

#[test]
fn weighted_respects_a_zero_weight() {
    let mut source = Source::generating(21, 4_096);
    for _ in 0..200 {
        let pick = source.weighted(&[1, 0, 1]).expect("room on the tape");
        assert_ne!(pick, 1, "a zero-weight option was chosen");
    }
}

#[test]
fn weighted_can_reach_every_positive_option() {
    let mut source = Source::generating(33, 8_192);
    let mut seen = [false; 3];
    for _ in 0..600 {
        let pick = source.weighted(&[1, 1, 1]).expect("room on the tape");
        seen[pick] = true;
    }
    assert_eq!(seen, [true, true, true]);
}

/// `consumed` must not hand the shrinker entries the generator never read — they are not part
/// of the value and would be dead weight to minimise.
#[test]
fn consumed_reports_only_what_was_read() {
    let mut source = Source::replaying(vec![1, 2, 3, 4, 5]);
    let _ = source.below(10).expect("entry present");
    let _ = source.below(10).expect("entry present");
    assert_eq!(source.consumed(), vec![1, 2]);
}

#[test]
fn modes_are_distinguishable() {
    // Guards against the two constructors being accidentally made identical, which would make
    // every shrink unsound and every test above still pass except the overrun ones.
    assert_ne!(Mode::Grow, Mode::Fixed);
}
