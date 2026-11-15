//! The sequential machinery, and one real application of it.
//!
//! The application is [`persist`](crate::persist) driven as a state machine, which is not a toy
//! chosen to demonstrate the feature — it is the code in this crate with the most genuine
//! multi-entry state, and the code where a bug of exactly this shape was found by hand. `refresh`
//! originally rewrote a regressions file from one property's point of view and deleted its
//! siblings' entries; a regressions file holds a whole suite, so that silently destroyed the only
//! record of cases that had cost a search to find.
//!
//! That bug is a *sequence* bug. No single call to `save` or `refresh` is wrong in isolation; the
//! failure needs a save under one name, a save under another, and then a refresh. A single-shot
//! property cannot express it and the machine below finds it immediately.

use std::fs;

use super::{Machine, check_machine, commands, execute};
use crate::persist::{self, Entry};
use crate::runner::{Config, Failure, Outcome};
use crate::{Gen, generate};

// ===== A real machine over `persist` ======================================

/// Property names a suite might hold. A small fixed set on purpose: the interesting states need
/// *collisions* between names, and drawing identifiers from a wide alphabet would make every
/// entry belong to a different property and never exercise the sibling-preservation path.
const NAMES: [&str; 3] = ["alpha", "beta", "gamma"];

/// A path no other case or test is using.
///
/// A fixed path was the first attempt and it failed immediately, in a way worth recording: the
/// two machine tests in this file each construct a system, libtest runs every `#[test]` on its
/// own thread, and both were truncating and rewriting the same file. The symptom was a file that
/// had lost an entry it had definitely been given — which reads exactly like the `persist` bug
/// the machine was built to hunt, and was the test's own fault.
///
/// A counter rather than a thread id, so each *case* within a run also gets a clean file and no
/// state can leak from one command sequence into the next.
fn scratch_file() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);

    let directory = std::env::temp_dir().join("euclid-stability-machine-persist");
    let _ = fs::create_dir_all(&directory);
    directory
        .join(format!(
            "cases-{}.txt",
            NEXT.fetch_add(1, Ordering::Relaxed)
        ))
        .to_string_lossy()
        .to_string()
}

/// The file under test, at a path owned by this machine.
struct Regressions {
    path: String,
}

impl Drop for Regressions {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

#[derive(Debug, Clone)]
enum Op {
    /// Append a case, unless its tape is already recorded.
    Save {
        tape: Vec<u64>,
        name: &'static str,
        rendering: Option<String>,
    },
    /// Rewrite one property's entries, leaving every other property's alone.
    Refresh {
        name: &'static str,
        entries: Vec<(Vec<u64>, Option<String>)>,
    },
}

struct PersistMachine;

/// A tape of at least one entry.
///
/// Empty tapes are excluded because they are genuinely not round-trippable and that is by
/// design: `encode(&[])` is the empty string and `decode("")` is `None`, so a blank payload must
/// not come back as an empty tape — otherwise a blank line in a hand-edited file would become a
/// phantom regression replayed on every run. A machine that generated them would be asserting the
/// opposite of a documented decision.
fn tape() -> Gen<Vec<u64>> {
    generate::vec(generate::int(0, 64).map(|v| v as u64), 1, 3)
}

/// Renderings from a small set, for the same reason the names are: the interesting question is
/// whether the right rendering survives the right operation, not whether arbitrary text escapes.
fn rendering() -> Gen<Option<String>> {
    generate::one_of(vec![
        generate::just(None),
        generate::just(Some("Len(1.0)".to_string())),
        generate::just(Some("Dense { rank: 2 }".to_string())),
    ])
}

fn name() -> Gen<&'static str> {
    generate::index(0, NAMES.len() - 1).map(|i| NAMES[i])
}

impl Machine for PersistMachine {
    type System = Regressions;
    type Model = Vec<Entry>;
    type Command = Op;

    fn system() -> Self::System {
        Regressions {
            path: scratch_file(),
        }
    }

    fn model() -> Self::Model {
        Vec::new()
    }

    /// Weighted towards `Save`, because a file needs entries in it before a `Refresh` is
    /// interesting — and towards refreshing with entries drawn from the *model's own tapes*,
    /// which is what a recording run actually does.
    fn command(model: &Self::Model) -> Gen<Self::Command> {
        let existing: Vec<Vec<u64>> = model.iter().map(|entry| entry.tape.clone()).collect();

        let refresh_entries = if existing.is_empty() {
            generate::vec(generate::pair(tape(), rendering()), 0, 2)
        } else {
            // Re-render tapes already in the file, which is the realistic shape: a recording run
            // refreshes what it found rather than inventing cases.
            let pool = existing.clone();
            generate::vec(
                generate::pair(
                    generate::index(0, pool.len() - 1).map(move |i| pool[i].clone()),
                    rendering(),
                ),
                0,
                2,
            )
        };

        generate::weighted(vec![
            (
                4,
                generate::triple(tape(), name(), rendering()).map(|(tape, name, rendering)| {
                    Op::Save {
                        tape,
                        name,
                        rendering,
                    }
                }),
            ),
            (
                1,
                generate::pair(name(), refresh_entries)
                    .map(|(name, entries)| Op::Refresh { name, entries }),
            ),
        ])
    }

    /// The model *is* the specification, written the obvious way.
    ///
    /// Reading these two arms is how you check the contract rather than the implementation:
    /// `Save` appends unless the tape is present, and `Refresh` keeps every line that does not
    /// belong to `name` and then appends the new ones. The second arm is the one that was wrong
    /// in the real code.
    fn step_model(model: &mut Self::Model, command: &Self::Command) {
        match command {
            Op::Save {
                tape,
                name,
                rendering,
            } => {
                if model.iter().any(|entry| entry.tape == *tape) {
                    return;
                }
                model.push(Entry {
                    tape: tape.clone(),
                    name: Some((*name).to_string()),
                    rendering: rendering.clone(),
                });
            }
            Op::Refresh { name, entries } => {
                let mut next: Vec<Entry> = model
                    .iter()
                    .filter(|entry| !entry.belongs_to(name))
                    .cloned()
                    .collect();
                for (tape, rendering) in entries {
                    next.push(Entry {
                        tape: tape.clone(),
                        name: Some((*name).to_string()),
                        rendering: rendering.clone(),
                    });
                }
                *model = next;
            }
        }
    }

    fn step_system(
        system: &mut Self::System,
        model: &Self::Model,
        command: &Self::Command,
    ) -> Outcome {
        match command {
            Op::Save {
                tape,
                name,
                rendering,
            } => persist::save(&system.path, tape, name, rendering.as_deref()),
            Op::Refresh { name, entries } => persist::refresh(&system.path, name, entries),
        }

        let loaded = persist::load(&system.path);
        if loaded == *model {
            Ok(())
        } else {
            Err(Failure::new(
                format!(
                    "after {command:?} the file and the model disagree\n    \
                     file:  {loaded:?}\n    model: {model:?}"
                ),
                "a sequential property",
            ))
        }
    }
}

/// The headline: `persist` agrees with an obvious model over any sequence of saves and refreshes.
///
/// Modest case count and a short sequence because every step does real file I/O — the point is
/// the interleaving, and six operations is enough to reach "save under two names, then refresh
/// one of them", which is the state the real bug lived in.
#[test]
fn persist_agrees_with_its_model_over_any_sequence() {
    let config = Config::default().cases(64);
    check_machine::<PersistMachine>(
        "persist_agrees_with_its_model_over_any_sequence",
        &config,
        6,
    );
}

/// The machine must actually be able to express the bug that motivated it — otherwise the suite
/// above is reassurance rather than a test.
///
/// This builds the sequence by hand and runs it against a *deliberately wrong* model: one that
/// refreshes by discarding everything, which is what the real `refresh` used to do. `execute`
/// has to report a disagreement.
#[test]
fn the_machine_detects_a_refresh_that_drops_siblings() {
    struct DroppingModel;

    impl Machine for DroppingModel {
        type System = Regressions;
        /// The model says a refresh keeps siblings. The "system" below does not.
        type Model = Vec<Entry>;
        type Command = Op;

        fn system() -> Self::System {
            PersistMachine::system()
        }
        fn model() -> Self::Model {
            Vec::new()
        }
        fn command(model: &Self::Model) -> Gen<Self::Command> {
            PersistMachine::command(model)
        }
        fn step_model(model: &mut Self::Model, command: &Self::Command) {
            PersistMachine::step_model(model, command);
        }

        /// Stands in for the old implementation: a refresh that rewrites the file from one
        /// property's view and so loses every other property's lines.
        fn step_system(
            system: &mut Self::System,
            model: &Self::Model,
            command: &Self::Command,
        ) -> Outcome {
            match command {
                Op::Save {
                    tape,
                    name,
                    rendering,
                } => persist::save(&system.path, tape, name, rendering.as_deref()),
                Op::Refresh { name, entries } => {
                    let _ = fs::remove_file(&system.path);
                    for (tape, rendering) in entries {
                        persist::save(&system.path, tape, name, rendering.as_deref());
                    }
                }
            }
            let loaded = persist::load(&system.path);
            if loaded == *model {
                Ok(())
            } else {
                Err(Failure::new("disagreement", "a sequential property"))
            }
        }
    }

    // Save under two names, then refresh one. The sibling's line must survive, and the broken
    // implementation drops it.
    let sequence = vec![
        Op::Save {
            tape: vec![1],
            name: "alpha",
            rendering: Some("Len(1.0)".to_string()),
        },
        Op::Save {
            tape: vec![2],
            name: "beta",
            rendering: None,
        },
        Op::Refresh {
            name: "alpha",
            entries: vec![(vec![1], Some("Len(2.0)".to_string()))],
        },
    ];

    execute::<PersistMachine>(&sequence).expect("the real implementation preserves siblings");
    execute::<DroppingModel>(&sequence)
        .expect_err("a refresh that drops siblings must be detected");
}

// ===== The machinery itself ===============================================

/// A machine whose commands are only legal in a specific order, used to check the claim the
/// module header makes: that deleting a command cannot produce an illegal sequence, because the
/// commands after it are re-derived rather than replayed.
struct Gated;

#[derive(Debug, Clone, PartialEq)]
enum Gate {
    Open,
    Use,
}

impl Machine for Gated {
    type System = bool;
    type Model = bool;
    type Command = Gate;

    fn system() -> Self::System {
        false
    }
    fn model() -> Self::Model {
        false
    }

    /// `Use` is only offered once the gate is open. This is the narrowing the module docs ask
    /// for, instead of a precondition that would reject candidates while shrinking.
    fn command(model: &Self::Model) -> Gen<Self::Command> {
        if *model {
            generate::one_of(vec![generate::just(Gate::Open), generate::just(Gate::Use)])
        } else {
            generate::just(Gate::Open)
        }
    }

    fn step_model(model: &mut Self::Model, command: &Self::Command) {
        if *command == Gate::Open {
            *model = true;
        }
    }

    fn step_system(
        system: &mut Self::System,
        _model: &Self::Model,
        command: &Self::Command,
    ) -> Outcome {
        match command {
            Gate::Open => {
                *system = true;
                Ok(())
            }
            Gate::Use if *system => Ok(()),
            Gate::Use => Err(Failure::new("used a closed gate", "a sequential property")),
        }
    }
}

/// Every generated sequence must be legal: no `Use` before an `Open`.
///
/// This is the property the whole design rests on. The generator is a function of the model, so a
/// sequence cannot contain a `Use` the model would not permit — and because the shrinker edits
/// the *tape*, a candidate with a command deleted re-derives the rest against the state that now
/// exists rather than replaying a stored list that has become invalid.
#[test]
fn every_generated_sequence_is_legal() {
    use crate::source::Source;

    let generator = commands::<Gated>(10);
    for seed in 0..200u64 {
        let mut source = Source::generating(seed, 4_096);
        let Ok(sequence) = generator.draw(&mut source) else {
            continue;
        };
        let mut opened = false;
        for command in &sequence {
            assert!(
                !(*command == Gate::Use && !opened),
                "seed {seed} produced a Use before any Open: {sequence:?}"
            );
            if *command == Gate::Open {
                opened = true;
            }
        }
        // And the sequence must actually execute, which is the same claim from the other side.
        execute::<Gated>(&sequence).expect("a generated sequence must be executable");
    }
}

/// The same claim for every tape the *shrinker* might hand back: an arbitrary edit to a tape must
/// still produce a legal sequence, not merely the tapes the search happened to draw.
///
/// This is the test that would fail under a replay-a-stored-list design, where deleting the
/// `Open` leaves the `Use` behind.
#[test]
fn an_arbitrary_tape_edit_still_produces_a_legal_sequence() {
    use crate::source::Source;

    let generator = commands::<Gated>(10);
    for seed in 0..64u64 {
        let mut source = Source::generating(seed, 4_096);
        if generator.draw(&mut source).is_err() {
            continue;
        }
        let tape = source.consumed();

        // Every single-entry deletion, which is the shrinker's most basic move.
        for cut in 0..tape.len() {
            let mut edited = tape.clone();
            edited.remove(cut);
            let mut replaying = Source::replaying(edited);
            if let Ok(sequence) = generator.draw(&mut replaying) {
                execute::<Gated>(&sequence).unwrap_or_else(|failure| {
                    panic!(
                        "deleting entry {cut} of seed {seed}'s tape produced an illegal \
                         sequence {sequence:?}: {}",
                        failure.message
                    )
                });
            }
        }

        // And every single-entry zeroing, the other basic move.
        for position in 0..tape.len() {
            let mut edited = tape.clone();
            edited[position] = 0;
            let mut replaying = Source::replaying(edited);
            if let Ok(sequence) = generator.draw(&mut replaying) {
                execute::<Gated>(&sequence).unwrap_or_else(|failure| {
                    panic!(
                        "zeroing entry {position} of seed {seed}'s tape produced an illegal \
                         sequence {sequence:?}: {}",
                        failure.message
                    )
                });
            }
        }
    }
}

/// The length is drawn first, so the shrinker shortens a sequence by lowering one entry. A
/// zeroed tape must therefore give the empty sequence — which is also the check that `max_steps`
/// of zero is handled rather than panicking.
#[test]
fn a_zeroed_tape_gives_the_empty_sequence() {
    use crate::source::Source;

    let generator = commands::<Gated>(10);
    let mut source = Source::replaying(vec![0; 64]);
    let sequence = generator.draw(&mut source).expect("a zeroed tape draws");
    assert!(
        sequence.is_empty(),
        "the length entry is not drawn first: {sequence:?}"
    );

    assert!(
        commands::<Gated>(0)
            .draw(&mut Source::generating(1, 64))
            .expect("zero steps is legal")
            .is_empty()
    );
}

/// An invariant failure is reported, not just a per-step disagreement. The default implementation
/// holds vacuously, so a machine that relies on one needs it actually called.
#[test]
fn a_broken_invariant_is_reported() {
    struct Leaky;

    impl Machine for Leaky {
        type System = u32;
        type Model = u32;
        type Command = ();

        fn system() -> Self::System {
            0
        }
        fn model() -> Self::Model {
            0
        }
        fn command(_: &Self::Model) -> Gen<Self::Command> {
            generate::just(())
        }
        fn step_model(model: &mut Self::Model, _: &Self::Command) {
            *model += 1;
        }
        fn step_system(system: &mut Self::System, _: &Self::Model, _: &Self::Command) -> Outcome {
            *system += 1;
            Ok(())
        }
        /// Holds for the first two steps and then does not, so the failure is attributable to a
        /// sequence rather than to any single command.
        fn invariant(system: &Self::System, _: &Self::Model) -> Outcome {
            if *system <= 2 {
                Ok(())
            } else {
                Err(Failure::new("ran too long", "a sequential property"))
            }
        }
    }

    execute::<Leaky>(&[(), ()]).expect("two steps are within the invariant");
    execute::<Leaky>(&[(), (), ()]).expect_err("the third step breaks the invariant");
}
