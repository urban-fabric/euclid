//! Throughput of the engine itself, so "fast" is a measurement rather than a claim.
//!
//! Run with `cargo bench -p euclid-stability`. There is no criterion and no `#[bench]`: the
//! target is `harness = false`, so this is a plain `fn main` that times loops and prints a
//! table. That keeps the workspace's no-third-party property intact — a benchmark dependency is
//! still a dependency, and `deny.toml` would have to grow an exception for it.
//!
//! # What is worth measuring here, and what is not
//!
//! The engine's cost per case is **overhead on top of the property body**, and that ratio is the
//! only thing that matters. A kernel property whose body is three float operations runs in
//! nanoseconds, so engine overhead decides how many cases a suite can afford; a property whose
//! body runs `tensor_laws` over three rank-4 tensors is microseconds, and the engine disappears
//! into the noise. So the first two rows below deliberately use a near-empty body: they measure
//! the floor, which is what caps `cases` for the cheap properties where searching harder is
//! free.
//!
//! The third and fourth rows measure **shrinking**, which is where the time actually goes when a
//! property fails. A search that finds a bug in 300 cases can then spend 4,096 probes minimising
//! it, each probe re-running the generator *and* the property — so shrinking a failure is
//! routinely an order of magnitude more work than finding it. Nobody notices, because it only
//! happens when a test is already failing and the wall clock is competing with a human reading
//! the message. It still sets how long the edit-run-read loop takes while a bug is being fixed,
//! which is exactly when latency is expensive.
//!
//! # Threads are not the answer, and this is where to say why
//!
//! The obvious move for "make it faster" is to run cases in parallel, and it is the wrong one
//! here.
//!
//! libtest **already runs each `#[test]` on its own thread**, and every property is a `#[test]`.
//! A suite with forty properties is therefore already using every core; parallelising *within* a
//! property would compete with that for the same cores rather than adding any. It would also
//! cost the two things this crate is most careful about: `Config::seed` is fixed so a score is a
//! property of the code rather than of the run, and a parallel search would have to either
//! partition the seed space (changing every recorded baseline) or accept a nondeterministic one.
//! And `stats` and `target` are thread-locals read after the loop, which a multi-threaded search
//! would have to collect across workers.
//!
//! So the throughput work is **per-case cost and redundant probes**, not concurrency. That is
//! what the numbers below track.
//!
//! # Comparing across runs does not work, and that is a result in itself
//!
//! Two optimisations went into the search path — the tape is no longer materialised on every
//! case, and the panic-hook guard is a relaxed load rather than a `SeqCst` read-modify-write.
//! The obvious way to report them is a before/after table, and the first attempt at this file
//! did exactly that: 32.3M cases/s before, 62.5M after, a tidy 2×.
//!
//! **That number was noise.** Re-running the unchanged final code minutes later gave 33.7M, with
//! an *internal* spread of 1.0× — so each run was self-consistent and the two disagreed with
//! each other by a factor of two. On this machine the between-run variation (frequency scaling,
//! thermal state, whatever the rest of the box was doing, a `cargo build` that had just
//! finished) is larger than anything a micro-optimisation does to a 16-nanosecond loop. A
//! min-of-seven harness fixes *within*-run noise and cannot touch that.
//!
//! So this file reports two kinds of number and is careful about which is which:
//!
//! - **Counted quantities**, which are exact and reproducible on any machine. The shrinker's
//!   probe count is the one that matters: **572 → 378 distinct probes, 33.9% avoided.** That is
//!   not a timing, it is a count, and it is the same count everywhere.
//! - **Interleaved A/B timings**, where both arms run in the same process a few microseconds
//!   apart, so they see the same machine. Over three consecutive runs:
//!
//!   | what was removed | per iteration | ratio |
//!   |---|---|---|
//!   | `Source::consumed()` on every case | 9.3–9.5 ns | 4.19–4.27× |
//!   | constructing a `Source` per case | 6.0–6.2 ns | 3.06–3.13× |
//!   | constructing a `Source` per shrink probe | 4.4–8.1 ns | 1.10–1.19× |
//!
//!   The first two reproduce to within 2%, which is what a trustworthy measurement looks like
//!   against the 2× disagreement the cross-run comparison produced. The third is looser because
//!   it is a small delta on a larger base — the probe's own `Vec` draw dominates it — so it is
//!   quoted as a range rather than a figure.
//!
//!   Note the ratios are against the *draw loop*, not against a whole case: a bare
//!   `int(0, 1_000_000)` draw is about 3 ns, so a 9.4 ns allocation beside it is 4.2× of very
//!   little. That is the honest reading, and it is why these rows are expressed per iteration as
//!   well as as a ratio.
//!
//!   Absolute throughput rows are reported for scale only — do not read a diff off them between
//!   two runs.
//!
//! The lesson generalises past this file, which is why it is written down rather than quietly
//! fixed: a performance claim derived from two separate runs of a microbenchmark is a claim
//! about the machine. If the arms cannot be interleaved, count something instead.
//!
//! # What the memo does and does not buy
//!
//! The shrink rows barely move in wall clock, because this benchmark's probe is about as cheap
//! as hashing the tape it is keyed on. That is the honest floor of the optimisation and the
//! regime where it matters least; it pays in proportion to probe cost, and for a property whose
//! body runs `tensor_laws` over three rank-4 tensors a probe is microseconds against tens of
//! nanoseconds of hashing.
//!
//! The second effect is unconditional and does not depend on probe cost at all: a memo hit does
//! not spend budget, so the same `MAX_PROBES` cap explores strictly more distinct candidates
//! than it used to. Shrinking got better as well as cheaper.
//!
//! # The optimisations are behaviour-preserving, and that is checked
//!
//! By something stronger than the test suite: `euclid-numeric`'s recorded `target()` baseline is
//! unchanged to the last bit (`5.510272789619652e-1`). The search visits tapes in a fixed order
//! from a fixed seed, so a change that perturbed the trajectory by one tape would have moved
//! that number. An accuracy budget turns out to double as a regression test for the engine.

use std::collections::HashSet;
use std::hint::black_box;
use std::time::Instant;

use euclid_stability::generate::{self, Gen};
use euclid_stability::runner::{Config, Failure, Outcome};
use euclid_stability::source::Source;
use euclid_stability::{check, shrink};

/// Cases per measured loop. Large enough to swamp timer resolution, small enough that the whole
/// benchmark stays under a few seconds.
const CASES: usize = 20_000;

/// Timed repetitions per benchmark, plus one discarded warm-up.
///
/// **A single timed pass is not a measurement.** The first version of this file did exactly one,
/// and reported the `Vec<f64>` row at 3.2M cases/s and then 0.78M on the next run — a 4× swing on
/// code that had not changed. At a couple of milliseconds a pass, the number is mostly allocator
/// warmth, CPU frequency scaling and whatever else the machine was doing, and reading a 34%
/// improvement out of that would have been reading noise.
///
/// So each benchmark runs once to warm up and then `REPEATS` times, and the **minimum** is
/// reported. The minimum rather than the mean because the quantity of interest is the cost of the
/// work itself: noise only ever adds, so the fastest observed pass is the closest estimate of it,
/// and a mean over a contended machine measures the contention. Spread is printed alongside so a
/// run that was too noisy to trust says so instead of looking authoritative.
const REPEATS: usize = 7;

fn main() {
    println!("euclid-stability engine throughput");
    println!("(minimum of {REPEATS} timed passes after one warm-up)\n");
    println!(
        "{:<40} {:>10} {:>16} {:>8}",
        "benchmark", "time", "rate", "spread"
    );
    println!("{}", "-".repeat(78));

    search_floor();
    search_with_allocation();
    tape_materialisation_cost();
    probe_source_cost();
    shrink_cost();
    shrink_redundancy();
}

/// Time two arms **alternately**, in one process, and report the difference per iteration.
///
/// This is the only honest way to attribute a microsecond-scale change on this hardware. Running
/// the cheap arm `REPEATS` times and then the expensive arm `REPEATS` times would let drift land
/// on one of them; alternating puts any frequency or thermal excursion across both. The best of
/// each is kept, for the same reason the rest of the file does: noise only ever adds.
///
/// `cheap` must be the arm *without* the work being measured, so the reported delta is positive
/// and reads as "what the optimisation removed".
fn ab(
    name: &str,
    detail: &str,
    iterations: usize,
    mut cheap: impl FnMut(),
    mut dear: impl FnMut(),
) {
    cheap();
    dear();

    let mut best_cheap = std::time::Duration::MAX;
    let mut best_dear = std::time::Duration::MAX;
    for _ in 0..REPEATS {
        let start = Instant::now();
        cheap();
        best_cheap = best_cheap.min(start.elapsed());

        let start = Instant::now();
        dear();
        best_dear = best_dear.min(start.elapsed());
    }

    let delta = best_dear.saturating_sub(best_cheap);
    #[allow(clippy::cast_precision_loss)]
    let per_iteration = delta.as_secs_f64() / iterations as f64;
    let ratio = if best_cheap.is_zero() {
        0.0
    } else {
        best_dear.as_secs_f64() / best_cheap.as_secs_f64()
    };
    println!(
        "{name:<40} {:>7.2} ms {:>11.1} ns/each {:>5.2}x",
        delta.as_secs_f64() * 1e3,
        per_iteration * 1e9,
        ratio
    );
    println!(
        "{:<40} {:>10} {:>16}",
        format!("  -> {detail}"),
        format!("{:.2} ms", best_cheap.as_secs_f64() * 1e3),
        format!("{:.2} ms", best_dear.as_secs_f64() * 1e3)
    );
}

/// The search-path optimisations, each measured as an interleaved A/B.
///
/// Two separate changes, so two rows:
///
/// - **`Source::consumed` per case.** It allocates a `Vec` and copies the tape in, and the search
///   used to call it on every case while using the result only on a failure or a new best score.
///   For the overwhelming majority of cases it allocated, copied and dropped.
/// - **Constructing a `Source` per case.** A tape grows from empty to a dozen-odd entries, so a
///   fresh `Source` means an allocation plus several reallocations every case. `restart_generating`
///   reuses the buffer, making it once per *property*.
fn tape_materialisation_cost() {
    let generator = generate::int(0, 1_000_000);
    const DRAWS: usize = 200_000;

    ab(
        "per-case consumed(), A/B interleaved",
        "draw only / draw + consumed()",
        DRAWS,
        || {
            let mut source = Source::generating(0, 4_096);
            for seed in 0..DRAWS {
                source.restart_generating(seed as u64, 4_096);
                black_box(generator.draw(&mut source).ok());
            }
        },
        || {
            let mut source = Source::generating(0, 4_096);
            for seed in 0..DRAWS {
                source.restart_generating(seed as u64, 4_096);
                black_box(generator.draw(&mut source).ok());
                black_box(source.consumed());
            }
        },
    );

    ab(
        "per-case Source, A/B interleaved",
        "restart_generating / fresh Source",
        DRAWS,
        || {
            let mut source = Source::generating(0, 4_096);
            for seed in 0..DRAWS {
                source.restart_generating(seed as u64, 4_096);
                black_box(generator.draw(&mut source).ok());
            }
        },
        || {
            for seed in 0..DRAWS {
                let mut source = Source::generating(seed as u64, 4_096);
                black_box(generator.draw(&mut source).ok());
            }
        },
    );
}

/// The shrink-path optimisation: a `Source` per probe, or one reused across all of them.
///
/// `replay` used to take the candidate tape by value, so every probe handed it a fresh `Vec` —
/// one allocation and copy per probe, and a minimisation spends hundreds to thousands of them.
/// This is the row that matters most of the three, because shrinking is where a failure's time
/// actually goes.
fn probe_source_cost() {
    let generator = generate::vec(generate::int(0, 1_000_000), 8, 16);
    let tape: Vec<u64> = (0..24)
        .map(|i| 0x9E37_79B9_7F4A_7C15u64.wrapping_mul(i + 1))
        .collect();
    const PROBES: usize = 200_000;

    ab(
        "per-probe Source, A/B interleaved",
        "restart_replaying / fresh Source",
        PROBES,
        || {
            let mut source = Source::replaying(Vec::new());
            for _ in 0..PROBES {
                source.restart_replaying(&tape);
                black_box(generator.draw(&mut source).ok());
            }
        },
        || {
            for _ in 0..PROBES {
                let mut source = Source::replaying(tape.clone());
                black_box(generator.draw(&mut source).ok());
            }
        },
    );
}

/// Run `body` once unmeasured, then `REPEATS` times, returning the best and worst durations.
fn measure(mut body: impl FnMut()) -> (std::time::Duration, std::time::Duration) {
    body();

    let mut best = std::time::Duration::MAX;
    let mut worst = std::time::Duration::ZERO;
    for _ in 0..REPEATS {
        let start = Instant::now();
        body();
        let elapsed = start.elapsed();
        best = best.min(elapsed);
        worst = worst.max(elapsed);
    }
    (best, worst)
}

fn row(
    name: &str,
    (best, worst): (std::time::Duration, std::time::Duration),
    count: usize,
    unit: &str,
) {
    #[allow(clippy::cast_precision_loss)]
    let rate = count as f64 / best.as_secs_f64();
    let spread = if best.is_zero() {
        0.0
    } else {
        worst.as_secs_f64() / best.as_secs_f64()
    };
    println!(
        "{name:<40} {:>7.2} ms {:>11.0} {unit:<4} {spread:>6.1}x",
        best.as_secs_f64() * 1e3,
        rate
    );
}

/// The floor: a scalar generator and an empty body, so this is engine overhead and nothing else.
fn search_floor() {
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
fn search_with_allocation() {
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

/// A tape that needs real minimisation: the property fails only when some element is large, so
/// the shrinker has to delete the irrelevant elements and binary-search the one that matters.
fn failing_tape() -> (Gen<Vec<i64>>, Vec<u64>) {
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

fn is_failing(xs: &[i64]) -> bool {
    xs.iter().any(|v| *v > 500_000)
}

/// How long a minimisation takes, and how many probes it spends.
fn shrink_cost() {
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
fn shrink_redundancy() {
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

fn probe(generator: &Gen<Vec<i64>>, candidate: &[u64]) -> shrink::Probe {
    let mut source = Source::replaying(candidate.to_vec());
    match generator.draw(&mut source) {
        Ok(drawn) if is_failing(&drawn) => shrink::Probe::StillFails,
        _ => shrink::Probe::Discard,
    }
}

/// Unused, but keeps the `Failure`/`Outcome` import honest for readers copying this file as a
/// template for their own benchmark.
#[allow(dead_code)]
fn example_failure() -> Outcome {
    Err(Failure::new("example", "a benchmark"))
}
