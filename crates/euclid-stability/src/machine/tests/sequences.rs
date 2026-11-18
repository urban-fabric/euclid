//! Legal sequence generation, shrinking, and failure detection.

use super::*;

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

/// `check_machine` must **search**, not merely run.
///
/// Mutation testing found that replacing this function's whole body with `()` left every test in
/// this file green, which is a fair summary of what they check: two of them drive `execute`
/// directly with a hand-built sequence, and the third
/// ([`persist_agrees_with_its_model_over_any_sequence`]) hands `check_machine` a *correct*
/// machine and asserts it does not complain. A function that does nothing also does not complain.
///
/// So the missing test is the one that matters most: given a machine that is wrong, the search
/// has to find a sequence that proves it. The divergence here needs three steps, so no
/// single-shot case can witness it and the generator has to produce a sequence long enough —
/// which is the whole claim [`check_machine`] makes.
#[test]
fn check_machine_finds_a_sequence_that_breaks_a_wrong_system() {
    /// Agrees with its model for two steps, then stops incrementing. A counter that saturates
    /// one step early is the smallest honest version of "correct until the third interleaving".
    struct StallsAfterTwo;

    impl Machine for StallsAfterTwo {
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
        fn step_system(
            system: &mut Self::System,
            model: &Self::Model,
            _: &Self::Command,
        ) -> Outcome {
            if *system < 2 {
                *system += 1;
            }
            if system == model {
                Ok(())
            } else {
                Err(Failure::new(
                    format!("system {system} disagrees with model {model}"),
                    "a sequential property",
                ))
            }
        }
    }

    let config = Config::default().cases(256);
    let result = std::panic::catch_unwind(|| {
        check_machine::<StallsAfterTwo>("stalls", &config, 6);
    });

    let message = result
        .expect_err("check_machine must find the sequence that breaks this system")
        .downcast_ref::<String>()
        .cloned()
        .unwrap_or_default();

    assert!(
        message.contains("property failed"),
        "the failure should be reported through the ordinary runner: {message}"
    );
    assert!(
        message.contains("disagrees"),
        "the report should carry the machine's own message: {message}"
    );
}
