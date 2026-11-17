//! The tape's contract: a value is a pure function of it, and editing it is safe.

use super::{Mode, Source};

#[test]
fn weighted_offsets_skip_zero_buckets_and_reach_the_last_bucket() {
    let weights = [0, 2, 0, 3];
    for (pick, index) in [1, 1, 3, 3, 3].into_iter().enumerate() {
        assert_eq!(super::weighted_index(&weights, pick as u64), index);
    }
    assert_eq!(super::weighted_index(&[4, 0], 3), 0);
}

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

/// `below` must be **monotone**: a smaller tape entry can never produce a larger value.
///
/// The invariant the entire shrinker rests on, documented at length on
/// [`below`](super::Source::below) as "the single most expensive mistake available in this file"
/// — and, until this test, not actually checked. `shrink_entries` binary-searches an entry
/// downward assuming that if a smaller value still fails then smaller values are worth trying,
/// and a `%` implementation makes failure *periodic* in the entry rather than monotone, which
/// breaks the search silently: the shrinker reports a tape of full-width garbage and no
/// assertion fires.
///
/// Every other test in this file passes under a modulo implementation, which is exactly why
/// this one is here.
#[test]
fn below_is_monotone_in_the_tape_entry() {
    for bound in [2u64, 3, 8, 17, 100, 1_000, u64::MAX] {
        let mut previous = 0u64;
        for step in 0..=512u64 {
            // A sweep across the whole u64 range, so the coarse and fine ends are both covered.
            let entry = step.saturating_mul(u64::MAX / 512);
            let drawn = Source::replaying(vec![entry])
                .below(bound)
                .expect("one entry is on the tape");
            assert!(
                drawn >= previous,
                "below({bound}) fell from {previous} to {drawn} as the entry rose to {entry}; \
                 the mapping is not monotone and binary-search shrinking is unsound"
            );
            assert!(drawn < bound, "below({bound}) returned {drawn}");
            previous = drawn;
        }
    }
}

/// A zero entry draws zero, whatever the bound. The other half of monotonicity, and the reason
/// `zero_chunks` is a minimisation at all rather than an arbitrary edit.
#[test]
fn a_zero_entry_draws_the_smallest_value() {
    for bound in [1u64, 2, 7, 64, u64::MAX] {
        assert_eq!(
            Source::replaying(vec![0]).below(bound),
            Ok(0),
            "a zeroed entry must draw the smallest value for bound {bound}"
        );
    }
}

/// `boolean` must map a **low** entry to `false`.
///
/// Polarity is not cosmetic. The shrinker lowers entries, so `false` has to be the smaller
/// choice — otherwise a minimised counterexample reports the opposite branch from the one that
/// shrank, and `generate::boolean()` shrinks *away* from its base case. Mutation testing found
/// that inverting this comparison left all 1,169 tests in the workspace green, because a
/// symmetric generator is still a perfectly valid generator and nothing was asserting direction.
#[test]
fn boolean_draws_false_from_a_low_entry_and_true_from_a_high_one() {
    assert_eq!(Source::replaying(vec![0]).boolean(), Ok(false));
    assert_eq!(Source::replaying(vec![u64::MAX]).boolean(), Ok(true));
}

/// `tape` is what the source **recorded**; `consumed` is what the generator **read**. The two
/// differ whenever a replayed tape is longer than the draw needs, and the distinction matters
/// because only `consumed` is the identity of the value — see
/// [`consumed_reports_only_what_was_read`].
#[test]
fn tape_reports_everything_recorded_not_merely_what_was_read() {
    let mut source = Source::replaying(vec![11, 22, 33, 44]);
    let _ = source.below(10).expect("entry present");

    assert_eq!(
        source.tape(),
        &[11, 22, 33, 44],
        "tape must report the whole recorded sequence"
    );
    assert_eq!(
        source.consumed(),
        vec![11],
        "consumed must report only the entry that was read"
    );
}

/// The climber's mode: replay the given prefix, then keep going rather than overrun.
///
/// This is what [`restart_climbing`](super::Source::restart_climbing) exists for. A hill-climbing
/// step perturbs one entry of the best tape, and a perturbation that raises a length makes the
/// generator read further than that tape went — which under a fixed tape is discarded, costing
/// 11% of the case budget at the default `targeted_fraction`.
#[test]
fn a_climbing_tape_replays_its_prefix_then_grows() {
    let prefix = [7u64, 8];
    let mut climbing = Source::generating(0, 0);
    climbing.restart_climbing(&prefix, 42, 64);
    let mut fixed = Source::replaying(prefix.to_vec());

    // Over the prefix the two are indistinguishable: a climbing step must honour the entries it
    // was handed, or the perturbation it is exploring is not the one it chose.
    assert_eq!(climbing.below(100), fixed.below(100));
    assert_eq!(climbing.below(100), fixed.below(100));

    // Past the prefix they diverge, and that divergence is the whole point.
    assert!(
        fixed.below(10).is_err(),
        "precondition: a fixed tape overruns past its end"
    );
    assert!(
        climbing.below(10).is_ok(),
        "a climbing tape must grow past its prefix, not overrun"
    );
    assert_eq!(
        &climbing.consumed()[..2],
        &prefix,
        "the recorded tape must still begin with the prefix it replayed"
    );
}

/// But growth is still bounded, or a runaway generator would allocate without limit.
#[test]
fn a_climbing_tape_still_respects_its_limit() {
    let mut source = Source::generating(0, 0);
    source.restart_climbing(&[1], 42, 3);
    assert!(source.below(10).is_ok(), "the prefix entry");
    assert!(source.below(10).is_ok(), "grown");
    assert!(source.below(10).is_ok(), "grown to the limit");
    assert!(source.below(10).is_err(), "max_len must bound the growth");
}

#[test]
fn modes_are_distinguishable() {
    // Guards against the two constructors being accidentally made identical, which would make
    // every shrink unsound and every test above still pass except the overrun ones.
    assert_ne!(Mode::Grow, Mode::Fixed);
}
