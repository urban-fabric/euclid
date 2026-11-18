//! Engine properties over generated tapes, bounds, and dependent shapes.

use crate::{Config, Gen, Source, check, check_eq, generate, persist, property};

fn word() -> Gen<u64> {
    generate::int(i64::MIN, i64::MAX).map(|value| value as u64)
}

fn dependent() -> Gen<(usize, Vec<usize>)> {
    generate::index(0, 16).and_then(|bound| {
        generate::vec(generate::index(0, bound), 0, 16).map(move |values| (bound, values))
    })
}

property! {
    #![config(Config::default().cases(300))]

    /// Nonempty tapes survive persistence without changing their width or bits.
    fn hex_encoding_preserves_every_generated_tape(tape in generate::vec(word(), 1, 24)) {
        let encoded = persist::encode(&tape);
        check_eq!(encoded.len(), tape.len() * 16);
        check_eq!(persist::decode(&encoded), Some(tape));
    }

    /// The bounded-choice mapping must be monotone for binary shrinking to work.
    fn bounded_draws_preserve_order_for_arbitrary_bounds(
        first in word(), second in word(), bound in word(),
    ) {
        let (low, high) = (first.min(second), first.max(second));
        let a = Source::replaying(vec![low]).below(bound).unwrap();
        let b = Source::replaying(vec![high]).below(bound).unwrap();
        check!(a <= b, "bound {bound} reversed {low} <= {high}: {a} > {b}");
        if bound > 0 {
            check!(a < bound && b < bound, "a bounded draw escaped {bound}");
        } else {
            check_eq!((a, b), (0, 0));
        }
    }

    /// A dependent generator re-derives bounds whenever a shrink edits the tape.
    fn dependent_cases_replay_and_remain_legal_after_tape_edits(seed in word()) {
        let generator = dependent();
        let mut source = Source::generating(seed, 32);
        let original = generator.draw(&mut source).unwrap();
        let tape = source.consumed();
        let replayed = generator.draw(&mut Source::replaying(tape.clone())).unwrap();
        check_eq!(replayed, original);

        let zeroed = generator.draw(&mut Source::replaying(vec![0; tape.len()])).unwrap();
        check_eq!(zeroed, (0, Vec::<usize>::new()));

        for position in 0..tape.len() {
            for replacement in [0, u64::MAX] {
                let mut edited = tape.clone();
                edited[position] = replacement;
                let result = generator.draw(&mut Source::replaying(edited));
                if let Ok((bound, values)) = result {
                    check!(bound <= 16 && values.len() <= 16);
                    check!(values.iter().all(|value| *value <= bound));
                }
            }
            let mut deleted = tape.clone();
            deleted.remove(position);
            if let Ok((bound, values)) = generator.draw(&mut Source::replaying(deleted)) {
                check!(values.iter().all(|value| *value <= bound));
            }
        }
    }
}
