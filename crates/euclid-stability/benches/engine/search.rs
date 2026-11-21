//! The search and shrink costs — what is paid only when a property *fails*.
//!
//! Shrinking is the asymmetric half of the engine: free on a passing run and the dominant cost on
//! a failing one, so its redundancy is worth measuring separately from the floor. The helpers that
//! build a deliberately failing tape live here too, since nothing else needs them.

use super::*;
use euclid_stability::generate::{self, Gen};
use euclid_stability::runner::{Config, Failure, Outcome};
use euclid_stability::source::Source;
use euclid_stability::{check, shrink};
use std::collections::HashSet;
use std::hint::black_box;

/// The floor: a scalar generator and an empty body, so this is engine overhead and nothing else.
pub(super) fn search_floor() {
    let config = Config::default().cases(CASES);
    let generator = generate::int(0, 1_000_000);

    let timing = measure(|| {
        check("floor", &config, &generator, |x| {
            black_box(x);
            Ok(())
        });
    });
    row("search, scalar gen, empty body", timing, CASES, "cases/s");
}

/// The same with a generator that allocates, which is the common shape — most suites draw a
/// `Vec` of something.
pub(super) fn search_with_allocation() {
    let config = Config::default().cases(CASES);
    let generator = generate::vec(generate::float::finite(), 1, 8);

    let timing = measure(|| {
        check("alloc", &config, &generator, |xs: Vec<f64>| {
            black_box(xs.iter().sum::<f64>());
            Ok(())
        });
    });
    row("search, Vec<f64> generator", timing, CASES, "cases/s");
}

/// How long a minimisation takes, and how many probes it spends.
pub(super) fn shrink_cost() {
    let (generator, tape) = failing_tape();

    let mut calls = 0usize;
    let (minimal, cost) = shrink::minimise(tape.clone(), |candidate| {
        calls += 1;
        probe(&generator, candidate)
    });
    let timing = measure(|| {
        let _ = shrink::minimise(tape.clone(), |candidate| probe(&generator, candidate));
    });

    row("shrink, 20-40 entry tape", timing, cost.probes, "probes/s");
    println!(
        "{:<40} {:>10} {:>16}",
        "  -> tape entries",
        format!("{} -> {}", tape.len(), minimal.len()),
        format!("{} probes", cost.probes)
    );
    assert_eq!(
        calls, cost.probes,
        "the closure must be called exactly once per distinct candidate — a memo hit must not \
         reach it, or the memo is not doing anything"
    );
}

/// How many candidates the memo answered instead of re-running.
///
/// This is the number the fixpoint loop makes interesting. `minimise` runs its four passes
/// repeatedly until a full round leaves the tape unchanged, which it has to — deleting a chunk
/// makes entries newly reducible, and reducing a length makes chunks newly deletable. But the
/// final round exists only to *prove* nothing more can be done, and it proves it by re-probing
/// candidates earlier rounds already rejected.
///
/// Before the memo those were spent probes, each re-running the generator and the property:
/// measured at **194 of 572, or 33.9%**. They are now answered from the table, which both saves
/// the work and stops them consuming the [`shrink::MAX_PROBES`] budget — so the same cap explores
/// strictly more distinct candidates than it used to.
///
/// The external `HashSet` below is the check that the memo is complete rather than a measurement:
/// every candidate the shrinker repeats must be a memo hit, so the two counts have to agree.
pub(super) fn shrink_redundancy() {
    let (generator, tape) = failing_tape();

    let mut seen: HashSet<Vec<u64>> = HashSet::new();
    let mut repeats_reaching_the_closure = 0usize;

    let (_, cost) = shrink::minimise(tape.clone(), |candidate| {
        if !seen.insert(candidate.to_vec()) {
            repeats_reaching_the_closure += 1;
        }
        probe(&generator, candidate)
    });
    let timing = measure(|| {
        let _ = shrink::minimise(tape.clone(), |candidate| probe(&generator, candidate));
    });

    let total = cost.probes + cost.reused;
    #[allow(clippy::cast_precision_loss)]
    let share = if total == 0 {
        0.0
    } else {
        cost.reused as f64 * 100.0 / total as f64
    };
    row(
        "shrink, same work, memo on",
        timing,
        cost.probes,
        "probes/s",
    );
    println!(
        "{:<40} {:>10} {:>16}",
        "  -> answered from the memo",
        format!("{}/{total}", cost.reused),
        format!("{share:.1}%")
    );
    assert_eq!(
        repeats_reaching_the_closure, 0,
        "a repeated candidate reached the probe closure, so the memo missed it"
    );
}

/// A tape that needs real minimisation: the property fails only when some element is large, so
/// the shrinker has to delete the irrelevant elements and binary-search the one that matters.
pub(super) fn failing_tape() -> (Gen<Vec<i64>>, Vec<u64>) {
    let generator = generate::vec(generate::int(0, 1_000_000), 20, 40);
    // Draw until a tape that fails the property turns up, then hand that tape to the shrinker.
    let mut seed = 1u64;
    loop {
        let mut source = Source::generating(seed, 4_096);
        if let Ok(drawn) = generator.draw(&mut source)
            && drawn.iter().any(|v| *v > 500_000)
        {
            return (generator, source.consumed());
        }
        seed = seed.wrapping_add(1);
    }
}

pub(super) fn is_failing(xs: &[i64]) -> bool {
    xs.iter().any(|v| *v > 500_000)
}

/// Unused, but keeps the `Failure`/`Outcome` import honest for readers copying this file as a
/// template for their own benchmark.
#[allow(dead_code)]
pub(super) fn example_failure() -> Outcome {
    Err(Failure::new("example", "a benchmark"))
}
