//! New domains must reach their boundaries, compose, shrink and replay.

use super::*;
use crate::{Config, run};
use std::rc::Rc;

#[test]
fn full_width_signed_and_unsigned_ranges_reach_every_endpoint() {
    let signed = int(i64::MIN, i64::MAX);
    let unsigned = crate::generate::u64();
    for (raw, want_signed, want_unsigned) in [
        (0, i64::MIN, 0),
        (1 << 63, 0, 1 << 63),
        (u64::MAX, i64::MAX, u64::MAX),
    ] {
        assert_eq!(
            signed.draw(&mut Source::replaying(vec![raw])).unwrap(),
            want_signed
        );
        assert_eq!(
            unsigned.draw(&mut Source::replaying(vec![raw])).unwrap(),
            want_unsigned
        );
    }
    let values = draw_many(&signed, 100, 42);
    assert!(values.iter().any(|v| *v < 0));
    assert!(values.iter().any(|v| *v > 0));
    assert!(
        values.windows(2).any(|pair| pair[0] != pair[1]),
        "full-width range must not collapse to MIN"
    );
    let unsigned = draw_many(&unsigned, 100, 43);
    assert!(unsigned.iter().any(|v| *v > i64::MAX as u64));
    assert!(unsigned.iter().any(|v| *v < i64::MAX as u64));
}

#[test]
fn the_full_usize_range_and_wide_scalars_replay_without_narrowing() {
    let usize_generator = index(0, usize::MAX);
    assert_eq!(
        usize_generator
            .draw(&mut Source::replaying(vec![u64::MAX]))
            .unwrap(),
        usize::MAX
    );
    for (words, unsigned, signed) in [
        (vec![0, 0], 0, i128::MIN),
        (vec![u64::MAX, u64::MAX], u128::MAX, i128::MAX),
        (vec![1 << 63, 0], 1u128 << 127, 0),
    ] {
        assert_eq!(
            crate::generate::u128()
                .draw(&mut Source::replaying(words.clone()))
                .unwrap(),
            unsigned
        );
        assert_eq!(
            crate::generate::i128()
                .draw(&mut Source::replaying(words))
                .unwrap(),
            signed
        );
    }
}

#[test]
fn option_result_and_fixed_arrays_are_complete_small_domains() {
    let generator = crate::generate::option(crate::generate::result(
        crate::generate::array::<_, 2>(boolean()),
        crate::generate::sampled_from(vec!["error"]),
    ));
    let mut seen = std::collections::BTreeSet::new();
    let report = run(
        "composed",
        &Config::default().exhaustive(),
        &generator,
        |value| {
            seen.insert(value);
            Ok(())
        },
    )
    .unwrap();
    assert!(report.exhaustive);
    assert_eq!((report.passed, seen.len()), (6, 6));
    assert!(seen.contains(&None));
    assert!(seen.contains(&Some(Err("error"))));
    assert!(seen.contains(&Some(Ok([true, false]))));
}

#[test]
fn size_is_recorded_and_rederived_after_edits() {
    let generator = crate::generate::sized(8, |size| {
        vec(index(0, size), size, size).map(move |values| (size, values))
    });
    for seed in 0..40 {
        let mut source = Source::generating(seed, 32);
        let first = generator.draw(&mut source).unwrap();
        let tape = source.consumed();
        assert_eq!(
            generator
                .draw(&mut Source::replaying(tape.clone()))
                .unwrap(),
            first
        );
        let (size, values) = generator
            .draw(&mut Source::replaying(vec![0; tape.len()]))
            .unwrap();
        assert_eq!((size, values), (0, Vec::new()));
    }
}

#[test]
fn zero_length_arrays_do_not_need_an_element_value() {
    let generator =
        crate::generate::array::<_, 0>(Gen::<i32>::new(|_| panic!("no elements needed")));
    let report = run(
        "empty arrays",
        &Config::default().exhaustive(),
        &generator,
        |value| {
            assert_eq!(value, []);
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(report.passed, 1);
    assert!(report.exhaustive);
}

#[test]
fn explicit_examples_are_first_and_their_failure_tapes_replay() {
    let generator = int(0, 100).with_examples(vec![-1]);
    let mut values = Vec::new();
    let report = run(
        "examples",
        &Config::default().cases(1).edges_first(),
        &generator,
        |value| {
            values.push(value);
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(values, [-1]);
    assert_eq!(report.edge_cases, 1);
    let error = run(
        "example failure",
        &Config::default().edges_first(),
        &generator,
        |value| {
            if value < 0 {
                Err(crate::Failure::new("negative", "example"))
            } else {
                Ok(())
            }
        },
    )
    .unwrap_err();
    let case = error.counterexample.unwrap();
    assert_eq!(
        generator.draw(&mut Source::replaying(case.tape)).unwrap(),
        -1
    );
}

#[test]
fn ordinary_random_runs_never_construct_exploration_metadata() {
    let touched = Rc::new(std::cell::Cell::new(false));
    let marker = touched.clone();
    let generator = Gen::new(|source| Ok(source.boolean()?)).with_metadata(move || {
        marker.set(true);
        (None, Rc::new(vec![]))
    });
    run(
        "random",
        &Config::default().cases(4),
        &generator,
        |_| Ok(()),
    )
    .unwrap();
    assert!(!touched.get());
    run(
        "edges",
        &Config::default().cases(4).edges_first(),
        &generator,
        |_| Ok(()),
    )
    .unwrap();
    assert!(touched.get());
}
