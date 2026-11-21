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
//! By something stronger than the test suite: `euclid-scalar`'s recorded `target()` baseline is
//! unchanged to the last bit (`5.510272789619652e-1`). The search visits tapes in a fixed order
//! from a fixed seed, so a change that perturbed the trajectory by one tape would have moved
//! that number. An accuracy budget turns out to double as a regression test for the engine.

//! # The split
//!
//! This file passed the 400-line budget. The measurements divide by *when* the cost is paid — on
//! every case, or only on failure — and what stays here is the harness they share and `main`.

mod search;
mod tapes;

use search::*;
use tapes::*;

use euclid_stability::generate::Gen;
use euclid_stability::shrink;
use euclid_stability::source::Source;
use std::time::Instant;

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

fn probe(generator: &Gen<Vec<i64>>, candidate: &[u64]) -> shrink::Probe {
    let mut source = Source::replaying(candidate.to_vec());
    match generator.draw(&mut source) {
        Ok(drawn) if is_failing(&drawn) => shrink::Probe::StillFails,
        _ => shrink::Probe::Discard,
    }
}
