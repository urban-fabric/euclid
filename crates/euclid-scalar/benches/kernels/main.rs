//! Cost of the scalar kernels, **per named regime**, so "fast" is a measurement rather than a
//! claim.
//!
//! Run with `cargo bench -p euclid-scalar`. There is no criterion and no `#[bench]`: the target
//! is `harness = false`, so this is a plain `fn main` that times loops and prints a table. That
//! keeps the workspace's dependency-free property intact, for the reason
//! [`euclid-stability`'s own benchmark](../../euclid-stability/benches/engine.rs) gives — a
//! benchmark dependency is still a dependency, and `deny.toml` would have to grow an exception
//! for it.
//!
//! # Why the rows are regimes rather than functions
//!
//! Every kernel in this crate routes between two and seven approximations, and the choice is
//! named: `IncompleteGammaMethod`, `ErfcForm`, `ErfcxDepth`, `JRoute`, `IForm`, `LambertSeed`,
//! `AiryRegime`, `ZetaRoute`. A single ns/call figure for `gamma_lr` is therefore an average
//! over routes that differ by more than an order of magnitude, and it hides exactly the thing
//! worth knowing: *which arm is the expensive one*.
//!
//! So each row below pins one argument into one arm. A regime that costs far more than its
//! neighbours is then visible directly, and — because the arms are named in the source — the row
//! label says where to go and look.
//!
//! # What these numbers are, and what they are not
//!
//! The lesson from the engine benchmark applies here unchanged and is worth restating, because
//! it is the easiest thing to get wrong about this file:
//!
//! **Do not read a diff off two separate runs.** On this machine the between-run variation —
//! frequency scaling, thermal state, whatever else is resident — exceeded a factor of two on a
//! 16-nanosecond loop, while each run was internally self-consistent to 1.0×. A min-of-seven
//! harness fixes *within*-run noise and cannot touch that. Absolute rows here are for **scale
//! and ranking**: they answer "which regime is expensive" and "is this kernel nanoseconds or
//! microseconds", which is what the rows are for, and they do not answer "did my change make it
//! 8% faster".
//!
//! Two kinds of number are trustworthy, and both appear below:
//!
//! - **Counted quantities**, exact and identical on every machine. [`iteration_depths`] reports
//!   how deep each bounded loop actually runs against the cap that bounds it. That is the figure
//!   that governs kernel cost, it is reproducible, and it is the one to put in a commit message.
//! - **Interleaved A/B**, where both arms run in the same process microseconds apart and so see
//!   the same machine. [`ab`] is here for when a specific change needs defending.
//!
//! # The ratios within a run are the result
//!
//! Because the arms of one kernel are measured in the same pass, their *ratio* is far more
//! robust than either absolute. `erfcx` is the clearest case and doubles as a check that the
//! instrument works at all: [`ErfcxDepth`] spends 160 recurrence levels below `x = 2` and 10
//! above `x = 20`, a 16× difference in declared work, so the measured ratio across that ladder
//! should track it. If it ever does not, either the ladder stopped being the cost or this
//! benchmark stopped measuring what it claims to.

//! # The split
//!
//! This file passed the 400-line budget, so the measurements moved out along the section banners that
//! were already here: the regime tables, and the counted quantities. What stays is the harness they
//! share — `measure`, `ab`, and the row printers — and `main`, which is the running order.

mod counted;
mod regimes;

use counted::*;
use regimes::*;

use std::hint::black_box;
use std::time::{Duration, Instant};

/// Calls per measured loop. Large enough to swamp timer resolution — the cheapest row here is a
/// few nanoseconds — small enough that the whole benchmark stays under a few seconds.
const CALLS: usize = 20_000;

/// Timed repetitions per row, plus one discarded warm-up. The **minimum** is reported, because
/// noise only ever adds: the fastest observed pass is the closest estimate of the work itself,
/// while a mean over a contended machine measures the contention. Spread is printed alongside so
/// a pass that was too noisy to trust says so rather than looking authoritative.
const REPEATS: usize = 7;

fn main() {
    println!("euclid-scalar kernel cost, by named regime");
    println!("(minimum of {REPEATS} timed passes after one warm-up; ratios within a kernel are");
    println!(" the trustworthy part — see this file's header before diffing absolutes)\n");
    println!(
        "{:<26} {:<22} {:>11} {:>8}",
        "kernel", "regime", "ns/call", "spread"
    );
    println!("{}", "-".repeat(70));

    incomplete_gamma();
    error_function();
    erfcx_depth_ladder();
    bessel();
    airy();
    transcendental();
    integrals();
    inverses();

    iteration_depths();
}

fn section(title: &str) {
    println!("\n{title}");
}

/// Time a one-argument kernel at a single argument, pinned into one regime.
fn row1(kernel: &str, regime: &str, x: f64, f: impl Fn(f64) -> f64) {
    let (min, spread) = measure(|| {
        black_box(f(black_box(x)));
    });
    print_row(kernel, regime, min, spread);
}

/// Time a two-argument kernel, pinned into one regime.
fn row2(kernel: &str, regime: &str, a: f64, b: f64, f: impl Fn(f64, f64) -> f64) {
    let (min, spread) = measure(|| {
        black_box(f(black_box(a), black_box(b)));
    });
    print_row(kernel, regime, min, spread);
}

fn print_row(kernel: &str, regime: &str, min: Duration, spread: Duration) {
    #[allow(clippy::cast_precision_loss)]
    let per = min.as_secs_f64() / CALLS as f64 * 1e9;
    let ratio = if min.as_nanos() == 0 {
        1.0
    } else {
        #[allow(clippy::cast_precision_loss)]
        let r = (min + spread).as_secs_f64() / min.as_secs_f64();
        r
    };
    println!("{kernel:<26} {regime:<22} {per:>11.1} {ratio:>7.2}x");
}

fn depth_row(what: &str, worst: u32, cap: u32) {
    let headroom = if worst == 0 {
        String::from("-")
    } else {
        #[allow(clippy::cast_precision_loss)]
        let h = f64::from(cap) / f64::from(worst);
        format!("{h:.1}x")
    };
    println!("{what:<34} {worst:>9} {cap:>9} {headroom:>9}");
}

/// Run `body` `CALLS` times per pass, `REPEATS` passes after a warm-up, and return the minimum
/// pass together with the spread between fastest and slowest.
fn measure(mut body: impl FnMut()) -> (Duration, Duration) {
    for _ in 0..CALLS {
        body();
    }
    let mut min = Duration::MAX;
    let mut max = Duration::ZERO;
    for _ in 0..REPEATS {
        let start = Instant::now();
        for _ in 0..CALLS {
            body();
        }
        let elapsed = start.elapsed();
        min = min.min(elapsed);
        max = max.max(elapsed);
    }
    (min, max - min)
}

/// Interleaved A/B, for when a specific change needs defending.
///
/// Both arms run in the same process microseconds apart, so they see the same machine state —
/// which is the only comparison this file trusts. Unused while nothing is under optimisation;
/// kept because the alternative, when the question next comes up, is a cross-run diff.
#[allow(dead_code)]
fn ab(label: &str, mut before: impl FnMut(), mut after: impl FnMut()) {
    let (b, _) = measure(&mut before);
    let (a, _) = measure(&mut after);
    let ratio = b.as_secs_f64() / a.as_secs_f64();
    #[allow(clippy::cast_precision_loss)]
    let delta = (b.as_secs_f64() - a.as_secs_f64()) / CALLS as f64 * 1e9;
    println!("{label:<48} {delta:>8.1} ns/call {ratio:>7.2}x");
}
