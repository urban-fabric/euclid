//! Targeting checks for the runner module.

use super::*;

#[test]
fn the_climber_keeps_its_tape_on_ties_and_invalid_scores() {
    let mut climber = crate::runner::Climber::new(&Config::default());
    assert!(climber.would_improve(Some(0.0)));
    assert!(!climber.would_improve(None));
    assert!(!climber.would_improve(Some(f64::INFINITY)));
    climber.observe(&[1], Some(2.0));
    for score in [
        None,
        Some(1.0),
        Some(2.0),
        Some(f64::NAN),
        Some(f64::INFINITY),
    ] {
        assert!(!climber.would_improve(score));
        climber.observe(&[2], score);
        assert_eq!(climber.best_tape.as_deref(), Some(&[1][..]));
    }
    assert!(climber.would_improve(Some(3.0)));
    climber.observe(&[3], Some(3.0));
    assert_eq!(climber.best_tape.as_deref(), Some(&[3][..]));
}

/// Targeting must actually steer. The property scores how far the input is from a hidden point
/// that uniform sampling is very unlikely to approach, and the climb must get closer than a
/// pure random search over the same budget.
///
/// Stated as a comparison against `targeted_fraction(0.0)` rather than against an absolute
/// threshold, because the claim is "the climb helps", and that is the only honest way to say it.
#[test]
fn targeting_finds_a_higher_score_than_uniform_search() {
    fn best_score_over(fraction: f64) -> f64 {
        let config = Config::default().cases(600).targeted_fraction(fraction);
        let observed = std::cell::Cell::new(f64::NEG_INFINITY);
        let mut property = |v: i64| {
            // A smooth score peaking at a specific large value. Smooth is the requirement:
            // a step function gives the climber no gradient to follow.
            #[allow(clippy::cast_precision_loss)]
            let score = -((v - 900_000) as f64).abs();
            crate::target(score);
            observed.set(observed.get().max(score));
            Ok(())
        };
        check("targeting", &config, &int(0, 1_000_000), &mut property);
        observed.get()
    }

    let uniform = best_score_over(0.0);
    let targeted = best_score_over(0.5);

    assert!(
        targeted > uniform,
        "targeting did not improve on uniform search: targeted {targeted:e} vs uniform {uniform:e}"
    );
}

/// A property that reports no score must behave exactly as if targeting were off, so adding the
/// feature cannot change an existing suite.
#[test]
fn a_property_with_no_score_is_unaffected_by_targeting() {
    let run = |fraction: f64| {
        let config = Config::default().cases(128).targeted_fraction(fraction);
        let seen = std::cell::RefCell::new(Vec::new());
        let mut property = |v: i64| {
            seen.borrow_mut().push(v);
            Ok(())
        };
        check("untargeted", &config, &int(0, 1000), &mut property);
        seen.into_inner()
    };
    assert_eq!(
        run(0.0),
        run(1.0),
        "targeting changed the cases of a property that reports no score"
    );
}

/// Climbing must not cost cases.
///
/// The second half of the accounting defect above, and the subtler half. A climbing step
/// perturbs one entry of the best tape, and a perturbation that *raises* a length makes the
/// generator read past the end of that tape — which used to overrun, because the candidate was
/// replayed under `Mode::Fixed`. Those overruns were counted as rejections and, before the fix
/// above, silently consumed cases: measured at **228 of 256** on this very generator.
///
/// The score here deliberately rewards the *shortest* list, so the best tape is one entry long
/// and almost every mutation of it asks for more tape than it has. A generator whose score
/// happens to reward length does not reproduce this at all, which is why the first attempt to
/// pin it came back green.
#[test]
fn climbing_does_not_discard_cases() {
    let generator = gen_vec(int(0, 9), 0, 60);

    for fraction in [0.0, 0.5, 1.0] {
        let ran = std::cell::Cell::new(0usize);
        let config = Config::default().cases(256).targeted_fraction(fraction);
        let mut property = |xs: Vec<i64>| {
            ran.set(ran.get() + 1);
            #[allow(clippy::cast_precision_loss)]
            crate::target(-(xs.len() as f64));
            Ok(())
        };
        check("climbing", &config, &generator, &mut property);

        assert_eq!(
            ran.get(),
            256,
            "targeted_fraction({fraction}) ran {} of 256 cases; a climbing step must extend the \
             tape it perturbs rather than overrun it",
            ran.get()
        );
    }
}

/// Targeting must never make a property more permissive — it changes which cases are tried, not
/// which ones pass.
#[test]
fn targeting_never_hides_a_failure() {
    let config = Config::default().cases(256).targeted_fraction(1.0);
    let mut property = |v: i64| {
        crate::target(f64::from(i32::try_from(v).unwrap_or(0)));
        if v < 50 {
            Ok(())
        } else {
            Err(Failure::new(format!("value was {v}"), "here"))
        }
    };
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        check("targeted", &config, &int(0, 100), &mut property);
    }));
    assert!(
        result.is_err(),
        "a false property passed because targeting was enabled"
    );
}

/// The climb's mutation operator, tested directly rather than through its outcome.
///
/// [`targeting_finds_a_higher_score_than_uniform_search`] asserts that climbing beats uniform
/// sampling, and that is too weak to pin the operator: mutation testing broke the bit flip four
/// different ways — never flipping at all, only ever *setting* bits, masking the entry down to a
/// single bit, and reducing the shift so only bit zero can move — and the climb still beat random
/// search each time, because the other two arms carried it.
///
/// So this asserts the property the docstring on [`Climber::propose`] actually argues for: **local
/// moves at every scale**. That claim has three measurable parts, and each one is a different
/// mutant:
///
/// - *Local*: most proposals differ from the best tape in exactly one bit. Masking the entry
///   instead of flipping a bit destroys this.
/// - *Every scale*: the flipped bit ranges over the whole word, low and high. Shifting the mask
///   right instead of left confines it to bit zero.
/// - *Both directions*: a bit that is set can be cleared. `|` in place of `^` can only ever set.
///
/// The failing first attempt is worth recording: measuring only "does the proposal differ from
/// the original" passes against all four mutants, because all four still produce *different*
/// tapes. It is the shape of the difference that matters.
#[test]
fn the_mutation_operator_makes_local_moves_at_every_scale() {
    let config = Config::default().cases(2).targeted_fraction(1.0);
    let mut climber = crate::runner::Climber::new(&config);
    let original = 0x0F0F_0F0F_0F0F_0F0Fu64;
    climber.observe(&[original], Some(1.0));

    let mut single_bit = 0usize;
    let mut cleared_a_bit = 0usize;
    let mut high_bit_moved = 0usize;
    let mut total = 0usize;

    for case in 0..4_000usize {
        let Some(proposed) = climber.propose(config.cases, case as u64 * 0x9E37_79B9) else {
            continue;
        };
        total += 1;
        let delta = proposed[0] ^ original;
        if delta.count_ones() == 1 {
            single_bit += 1;
        }
        // A bit that was set in the original is now clear: the move went downward.
        if original & delta != 0 {
            cleared_a_bit += 1;
        }
        if delta >= 1u64 << 40 {
            high_bit_moved += 1;
        }
    }

    assert!(total > 3_000, "only {total} of 4000 proposals were made");
    assert!(
        single_bit * 2 > total,
        "only {single_bit} of {total} proposals were single-bit flips; the operator is not \
         making local moves"
    );
    assert!(
        cleared_a_bit * 5 > total,
        "only {cleared_a_bit} of {total} proposals cleared an existing bit; the operator can \
         only move in one direction"
    );
    assert!(
        high_bit_moved * 8 > total,
        "only {high_bit_moved} of {total} proposals moved a bit above 2^40; the operator is \
         confined to small steps"
    );
}

/// Climbing engages only after the exploration budget, and the boundary is exact.
///
/// `targeted_fraction` splits the case budget, so the case at which proposals begin *is* the
/// split. Mutation testing inverted and relaxed that comparison; either way the explore/climb
/// ratio stops being what was configured, which changes every targeted search in the fleet
/// without changing any assertion.
#[test]
fn climbing_begins_exactly_at_the_end_of_exploration() {
    // 100 cases, half explored: the climb starts at case 50.
    let config = Config::default().cases(100).targeted_fraction(0.5);
    let mut climber = crate::runner::Climber::new(&config);
    climber.observe(&[1, 2, 3], Some(1.0));

    assert!(
        climber.propose(0, 1).is_none(),
        "case 0 is exploration, not climbing"
    );
    assert!(
        climber.propose(49, 1).is_none(),
        "case 49 is the last exploration case"
    );
    assert!(
        climber.propose(50, 1).is_some(),
        "case 50 is the first climbing case"
    );
    assert!(climber.propose(99, 1).is_some());

    // With targeting off, no case ever climbs.
    let off = Config::default().cases(100).targeted_fraction(0.0);
    let mut never = crate::runner::Climber::new(&off);
    never.observe(&[1, 2, 3], Some(1.0));
    assert!(never.propose(99, 1).is_none());
}

/// Every position on the tape must be reachable, or entries past the first never get climbed.
///
/// The position is `rng % tape.len()`. Mutation testing replaced that remainder with a division
/// and an addition, both of which index past the end — and nothing failed, because no test ever
/// asked the climber for a proposal over a multi-entry tape.
#[test]
fn every_tape_position_is_reachable_by_a_proposal() {
    let config = Config::default().cases(2).targeted_fraction(1.0);
    let mut climber = crate::runner::Climber::new(&config);
    let original = vec![10u64, 20, 30, 40, 50];
    climber.observe(&original, Some(1.0));

    let mut touched = [false; 5];
    for case in 0..2_000usize {
        let Some(proposed) = climber.propose(config.cases, case as u64 * 0x2545_F491) else {
            continue;
        };
        assert_eq!(proposed.len(), original.len(), "the tape changed length");
        for (i, (before, after)) in original.iter().zip(&proposed).enumerate() {
            if before != after {
                touched[i] = true;
            }
        }
    }

    assert_eq!(
        touched, [true; 5],
        "some tape positions were never proposed for mutation: {touched:?}"
    );
}

/// The nudge arm walks **steadily** in one direction, which a bit flip cannot.
///
/// The operator's third component, and the reason the docstring gives for it: "a flip can only
/// ever toggle, so it cannot walk steadily in one direction". That makes its signature a *small
/// change in magnitude spread across several bits* — a bit flip at position `k` always moves the
/// value by exactly `2^k` and so differs in exactly one bit, and a fresh random value moves it
/// enormously.
///
/// Mutation testing broke this two ways: deleting the arm outright, so the value falls through to
/// a fresh draw, and shifting the step left instead of right, so every nudge is huge. Both leave
/// [`the_mutation_operator_makes_local_moves_at_every_scale`] green, because the bit-flip arm
/// still does its job.
#[test]
fn the_nudge_arm_takes_small_multi_bit_steps() {
    let config = Config::default().cases(2).targeted_fraction(1.0);
    let mut climber = crate::runner::Climber::new(&config);
    let original = 0x0F0F_0F0F_0F0F_0F0Fu64;
    climber.observe(&[original], Some(1.0));

    let mut small_walks = 0usize;
    for case in 0..8_000usize {
        let Some(proposed) = climber.propose(config.cases, case as u64 * 0x9E37_79B9) else {
            continue;
        };
        let delta_bits = (proposed[0] ^ original).count_ones();
        let magnitude = i128::from(proposed[0]).abs_diff(i128::from(original));
        // Small in value but spread across bits: neither a single flip nor a fresh draw.
        if magnitude > 0 && magnitude < (1 << 20) && delta_bits > 1 {
            small_walks += 1;
        }
    }

    assert!(
        small_walks > 10,
        "only {small_walks} of 8000 proposals were small multi-bit steps; the climb cannot walk \
         steadily towards a nearby peak"
    );
}
