//! Persistence checked against a state-machine model.

use super::*;

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

    let directory = std::env::temp_dir().join(format!(
        "euclid-stability-machine-persist-{}",
        std::process::id()
    ));
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
