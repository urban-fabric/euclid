//! What a tape costs to materialise, and what probing a source costs on top of it.
//!
//! The engine's per-case overhead is almost entirely these two, so they are measured apart from
//! the search: a property that runs a thousand cases pays this a thousand times whether or not
//! it ever fails.

use super::*;
use euclid_stability::generate::{self};
use euclid_stability::source::Source;
use std::hint::black_box;

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
pub(super) fn tape_materialisation_cost() {
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
pub(super) fn probe_source_cost() {
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
