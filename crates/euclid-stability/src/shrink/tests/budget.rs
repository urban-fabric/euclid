//! The probe budget, the memo, and the tape hasher underneath both.
//!
//! Part of the [`shrink`](super) tests. These bound *cost* rather than asserting a result:
//! shrinking is a search, so without a cap and a memo a single unreproducible failure can spend
//! an unbounded number of probes. The hasher is what makes the memo work, so its bulk path and
//! its byte order are pinned directly.

use super::super::{MAX_MEMO, MAX_PROBES, Probe, Prober, minimise};

/// A probe that keeps failing while `predicate` holds of the tape.
fn probe_on(predicate: impl Fn(&[u64]) -> bool) -> impl Fn(&[u64]) -> Probe {
    move |tape| {
        if predicate(tape) {
            Probe::StillFails
        } else {
            Probe::Discard
        }
    }
}

#[test]
fn the_memo_stops_at_its_capacity_and_keeps_existing_entries() {
    let mut prober = Prober::new(|_| Probe::StillFails);
    for index in 0..MAX_MEMO {
        prober.ask(&[index as u64]);
    }
    assert_eq!(prober.memo.len(), MAX_MEMO);
    prober.ask(&[MAX_MEMO as u64]);
    assert_eq!(prober.memo.len(), MAX_MEMO);
    prober.ask(&[0]);
    assert_eq!(prober.cost.reused, 1);
}

/// The probe budget must be respected, so a pathological probe cannot hang a test run.
#[test]
fn the_probe_budget_is_respected() {
    let mut calls = 0usize;
    let huge: Vec<u64> = (0..200).map(|i| u64::MAX - i).collect();
    let (_, cost) = minimise(huge, |_| {
        calls += 1;
        Probe::StillFails
    });
    assert!(cost.probes <= MAX_PROBES, "spent {}", cost.probes);
    assert!(calls <= MAX_PROBES, "probe called {calls} times");
}

/// The probe budget has to actually stop a minimisation that would otherwise run away.
///
/// [`the_probe_budget_is_respected`] looks like it covers this and does not, for the reason
/// spelled out on [`memo_hits_do_not_consume_the_probe_budget`]: an always-failing probe means
/// the widest deletion succeeds immediately, the tape empties in one probe, and the cap is never
/// approached. Mutation testing confirmed it — `Prober::exhausted` could be made to return
/// `false` unconditionally and every test here still passed.
///
/// So this probe resists every pass at once: the length is pinned, zeroing any entry fails, and
/// each of the 200 entries needs a full 64-probe binary search. That is roughly 15,000 probes of
/// work on a 4,096 budget, so the cap must bite and must be reported.
#[test]
fn a_minimisation_that_cannot_converge_stops_at_the_cap() {
    let calls = std::cell::Cell::new(0usize);
    let tape = vec![u64::MAX; 200];

    let (minimal, cost) = minimise(tape, |t| {
        calls.set(calls.get() + 1);
        if t.len() >= 200 && t.iter().all(|v| *v >= 1) {
            Probe::StillFails
        } else {
            Probe::Discard
        }
    });

    assert!(
        calls.get() <= MAX_PROBES,
        "the probe closure ran {} times against a budget of {MAX_PROBES}; the cap is not \
         bounding anything",
        calls.get()
    );
    assert!(cost.probes <= MAX_PROBES, "spent {}", cost.probes);
    assert!(
        cost.hit_the_cap(),
        "this minimisation cannot converge inside the budget, so the report must say the case \
         may not be minimal"
    );
    assert_eq!(minimal.len(), 200, "the required length must be preserved");
}

/// The closure must see each distinct candidate **once**. Everything else is answered from the
/// table, which is the whole point: a repeat re-runs the generator and the property, and for a
/// real suite that is the dominant cost of a failure.
#[test]
fn a_repeated_candidate_never_reaches_the_probe_closure() {
    use std::cell::RefCell;
    use std::collections::HashSet;

    let seen: RefCell<HashSet<Vec<u64>>> = RefCell::new(HashSet::new());
    let repeats = RefCell::new(0usize);

    let tape = vec![500, 400, 300, 200, 100, 50, 25, 12];
    let (_, cost) = minimise(tape, |candidate| {
        if !seen.borrow_mut().insert(candidate.to_vec()) {
            *repeats.borrow_mut() += 1;
        }
        if candidate.len() >= 3 && candidate[0] >= 7 {
            Probe::StillFails
        } else {
            Probe::Discard
        }
    });

    assert_eq!(
        *repeats.borrow(),
        0,
        "the probe closure was handed a candidate it had already answered"
    );
    assert_eq!(
        seen.borrow().len(),
        cost.probes,
        "distinct candidates must equal the reported probe count"
    );
}

/// The fixpoint loop re-tries candidates by design — the final round exists to prove nothing
/// more can be done — so a minimisation with any structure to it must record memo hits. A run
/// reporting zero would mean the memo is never consulted and the whole table is dead weight.
#[test]
fn a_fixpoint_round_produces_memo_hits() {
    let tape = vec![900, 800, 700, 600, 500];
    let (_, cost) = minimise(tape, probe_on(|t| t.len() >= 2 && t[0] >= 5));
    assert!(
        cost.reused > 0,
        "no candidate was answered from the memo, so the fixpoint round is re-probing nothing"
    );
}

/// A memo hit must not spend budget. That is what turns [`MAX_PROBES`] from a bound on loop
/// iterations into a bound on *distinct candidates tried* — the same cap now explores strictly
/// more of the search space, so a case that used to exhaust its budget re-proving a fixpoint can
/// reach an actually smaller tape.
///
/// Tested on the [`Prober`](super::Prober) directly rather than through `minimise`, because a
/// repeat has to be *arranged* to be certain of it. The first attempt at this test drove
/// `minimise` with an always-failing probe over a long tape, assuming churn — but an
/// always-failing probe means the widest deletion succeeds immediately, the tape empties in one
/// probe, and there is nothing left to repeat. The property is about the memo, so it is checked
/// where the memo lives.
#[test]
fn memo_hits_do_not_consume_the_probe_budget() {
    let calls = std::cell::Cell::new(0usize);
    let mut prober = super::Prober::new(|_: &[u64]| {
        calls.set(calls.get() + 1);
        Probe::Discard
    });

    assert_eq!(prober.ask(&[1, 2, 3]), Probe::Discard);
    assert_eq!(prober.ask(&[1, 2, 3]), Probe::Discard);
    assert_eq!(prober.ask(&[1, 2, 3]), Probe::Discard);
    assert_eq!(prober.ask(&[9]), Probe::Discard);

    assert_eq!(
        calls.get(),
        2,
        "only the two distinct candidates may be run"
    );
    assert_eq!(
        prober.cost.probes, 2,
        "budget is spent per distinct candidate"
    );
    assert_eq!(prober.cost.reused, 2, "the two repeats must be memo hits");
}

/// Whatever the tape, the cap bounds the distinct probes actually spent.
#[test]
fn the_cap_bounds_distinct_probes() {
    for tape in [
        vec![u64::MAX; 40],
        (0..60).map(|i| 1_000 - i).collect::<Vec<u64>>(),
        vec![0; 80],
    ] {
        let (_, cost) = minimise(tape, probe_on(|t| t.len() >= 2));
        assert!(
            cost.probes <= MAX_PROBES,
            "distinct probes must respect the cap: {}",
            cost.probes
        );
    }
}

#[test]
fn hitting_the_cap_is_reported() {
    let cheap = super::super::Cost {
        probes: 10,
        reused: 0,
    };
    assert!(!cheap.hit_the_cap());
    let exhausted = super::super::Cost {
        probes: MAX_PROBES,
        reused: 0,
    };
    assert!(exhausted.hit_the_cap());
}

/// The hasher has to consume a `u64` run through its bulk `write` path, because that is how
/// `Hash for [u64]` feeds it — the length through `write_usize`, then every element as one byte
/// slice. A byte-at-a-time implementation would be correct and slow, and a chunked one that
/// dropped the remainder would be fast and wrong.
#[test]
fn the_tape_hasher_distinguishes_tapes_through_its_bulk_path() {
    use std::hash::{BuildHasher, BuildHasherDefault, Hasher};

    let hash_of =
        |tape: &[u64]| BuildHasherDefault::<super::super::TapeHasher>::default().hash_one(tape);

    // Different contents, same length.
    assert_ne!(hash_of(&[1, 2, 3]), hash_of(&[1, 2, 4]));
    // Same contents, different length — the length prefix is what separates these.
    assert_ne!(hash_of(&[1, 2]), hash_of(&[1, 2, 0]));
    // Order matters, or a permutation pass would collide with its own input.
    assert_ne!(hash_of(&[1, 2]), hash_of(&[2, 1]));
    // The empty tape is hashable and does not panic on the remainder path.
    let _ = hash_of(&[]);

    // And a byte slice whose length is not a multiple of eight reaches the remainder branch.
    let mut hasher = BuildHasherDefault::<super::super::TapeHasher>::default().build_hasher();
    hasher.write(&[1, 2, 3, 4, 5]);
    let odd = hasher.finish();
    let mut hasher = BuildHasherDefault::<super::super::TapeHasher>::default().build_hasher();
    hasher.write(&[1, 2, 3, 4, 6]);
    assert_ne!(
        odd,
        hasher.finish(),
        "the remainder bytes must affect the hash"
    );
}

#[test]
fn tape_hashing_keeps_bits_from_each_input_path() {
    use std::hash::Hasher;

    // With a fixed state, each distinct word should lead to a distinct state.
    let word = |value| {
        let mut hasher = super::super::TapeHasher(u64::MAX);
        hasher.write_u64(value);
        hasher.finish()
    };
    assert_ne!(word(0), word(1));
    assert_ne!(word(1), word(u64::MAX));

    let length = |value| {
        let mut hasher = super::super::TapeHasher::default();
        hasher.write_usize(value);
        hasher.finish()
    };
    assert_ne!(length(1), length(2));

    let remainder = |bytes: &[u8]| {
        let mut hasher = super::super::TapeHasher::default();
        hasher.write(bytes);
        hasher.finish()
    };
    assert_ne!(remainder(&[1, 0]), remainder(&[0, 0]));
}

#[test]
fn tape_hashing_preserves_the_byte_order_of_each_tail_length() {
    use std::hash::Hasher;

    let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    for (length, tail) in [
        0,
        0x09,
        0x090a,
        0x09_0a0b,
        0x090a_0b0c,
        0x09_0a0b_0c0d,
        0x090a_0b0c_0d0e,
        0x09_0a0b_0c0d_0e0f,
    ]
    .into_iter()
    .enumerate()
    {
        let mut actual = super::super::TapeHasher::default();
        actual.write(&input[..8 + length]);

        let mut expected = super::super::TapeHasher::default();
        expected.write_u64(0x0807_0605_0403_0201);
        expected.write_u64(tail);
        assert_eq!(actual.finish(), expected.finish());
    }
}
